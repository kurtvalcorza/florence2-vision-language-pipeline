import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from florence2_vision_language_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_BEAMS,
    REMOTE_CODE_FILES,
    REMOTE_CODE_POLICY,
    TASKS,
    TASKS_WITH_TEXT,
    Florence2Pipeline,
    character_error_rate,
    remote_code_files,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_runner(image, prompt, task, max_new_tokens, num_beams):
    assert image.mode == "RGB"
    parsed = {"bboxes": [[1.0, 2.0, 3.0, 4.0]], "labels": ["box"]} if task == "<OD>" else f"echo:{prompt}"
    return {"text": f"<s>{prompt}</s>", "parsed": parsed}


def _pipeline() -> Florence2Pipeline:
    return Florence2Pipeline(_fake_runner, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights", *, custom_code: bool = False) -> Path:
    files = [("model.safetensors", payload), ("config.json", b'{"model_type": "florence2"}')]
    if custom_code:
        files[1] = ("config.json", b'{"model_type": "florence2", "auto_map": {}}')
        files.append(("modeling_florence2.py", b"# custom code\n"))
    entries = []
    for name, content in files:
        (root / name).write_bytes(content)
        entries.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"modelKey": MODEL_KEY, "modelId": MODEL_ID, "revision": MODEL_REVISION, "files": entries}
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "microsoft/Florence-2-large"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert REMOTE_CODE_POLICY == "refuse"
    assert len(REMOTE_CODE_FILES) == 3


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY
    listed = {entry["path"] for entry in manifest["files"]}
    assert set(REMOTE_CODE_FILES) <= listed  # the pinned snapshot ships custom code


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_remote_code_files_detects_custom_code(tmp_path):
    _write_snapshot(tmp_path)
    assert remote_code_files(tmp_path) == []
    _write_snapshot(tmp_path, custom_code=True)
    assert remote_code_files(tmp_path) == ["modeling_florence2.py", "config.json:auto_map"]


def test_from_pretrained_refuses_remote_code_after_verifying(tmp_path):
    """The verified microsoft snapshot carries custom code; loading is refused, never silently trusted."""
    _write_snapshot(tmp_path, custom_code=True)
    with pytest.raises(RuntimeError, match="refuses") as excinfo:
        Florence2Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    assert "pending" in str(excinfo.value)
    assert MODEL_REVISION in str(excinfo.value)


def test_from_pretrained_refuses_tampered_snapshot_before_remote_code_check(tmp_path):
    manifest_path = _write_snapshot(tmp_path, custom_code=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        Florence2Pipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        Florence2Pipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)
    with pytest.raises(RuntimeError, match="custom code"):
        Florence2Pipeline.from_pretrained(weights_dir=tmp_path, allow_download=True)


def test_run_rejects_bad_inputs():
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32))
    with pytest.raises(TypeError):
        pipe.run("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.run(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 1)))
    with pytest.raises(ValueError, match="task must be one of"):
        pipe.run(image, "<SEGMENT>")
    with pytest.raises(ValueError, match="requires a non-empty text_input"):
        pipe.run(image, "<CAPTION_TO_PHRASE_GROUNDING>")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.run(image, "<CAPTION_TO_PHRASE_GROUNDING>", "x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="takes no text_input"):
        pipe.run(image, "<CAPTION>", "a caption")
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.run(image, max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.run(image, max_new_tokens=0)
    with pytest.raises(TypeError):
        pipe.run(image, max_new_tokens=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.run(image, num_beams=0)


def test_run_output_fields_and_prompt_composition():
    pipe = _pipeline()
    result = pipe.run(Image.new("L", (16, 24)), "<CAPTION_TO_PHRASE_GROUNDING>", "a red square")
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["task"] == "<CAPTION_TO_PHRASE_GROUNDING>"
    assert result["result"] == "echo:<CAPTION_TO_PHRASE_GROUNDING>a red square"
    assert result["image_size"] == [16, 24]
    assert result["generation"] == {"max_new_tokens": 256, "num_beams": NUM_BEAMS, "do_sample": False}
    assert result["device"] == "cpu" and result["source"] == "injected"
    detection = pipe.run(Image.new("RGB", (16, 16)), "<OD>")
    assert detection["result"]["labels"] == ["box"]
    assert all(task in TASKS for task in TASKS_WITH_TEXT)


def test_run_rejects_malformed_runner_output():
    pipe = Florence2Pipeline(lambda *args: {"text": "x"}, "cpu")
    with pytest.raises(RuntimeError, match="runner must return"):
        pipe.run(Image.new("RGB", (8, 8)))


def test_character_error_rate():
    assert character_error_rate("hello", "hello") == 0.0
    assert character_error_rate("hello", "hallo") == 0.2
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "x") == 1.0
    assert character_error_rate("ab", "") == 1.0
    with pytest.raises(TypeError):
        character_error_rate("a", 1)  # type: ignore[arg-type]
