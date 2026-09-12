"""Prompt-driven vision-language tasks with the pinned ``microsoft/Florence-2-large`` snapshot.

The pinned snapshot ships its own ``modeling_florence2.py`` / ``processing_florence2.py`` and can only be
loaded by executing that code (the ``trust_remote_code`` opt-in). This package refuses remote code: the
snapshot is digest-verified, the presence of custom code is detected, and ``from_pretrained`` raises
``RuntimeError`` until the pin decision recorded in ``REMOTE_CODE_POLICY`` is changed by the repository
owner. The task method runs against an injected runner so the contract, validation and tests run offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "microsoft/Florence-2-large"
MODEL_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"
MODEL_LICENSE = "mit"
MODEL_KEY = "florence-2-large"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

# Custom-code files carried by the pinned snapshot. Loading them is refused (see REMOTE_CODE_POLICY).
REMOTE_CODE_FILES = ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py")
REMOTE_CODE_POLICY = "refuse"  # the only value this package implements; "pending owner pin decision"

# Task prompts from the pinned upstream README; the last one needs a caption as text input.
TASKS_WITHOUT_TEXT = (
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR>",
    "<OCR_WITH_REGION>",
)
TASKS_WITH_TEXT = ("<CAPTION_TO_PHRASE_GROUNDING>",)
TASKS = TASKS_WITHOUT_TEXT + TASKS_WITH_TEXT

MAX_IMAGE_SIDE = 4096  # pixels; the processor resizes to 768x768 regardless (preprocessor_config.json)
MAX_TEXT_CHARS = 1000  # characters of caption text accepted for phrase grounding
MAX_NEW_TOKENS = 1024  # upstream README examples use 1024; hard ceiling for `max_new_tokens`
DEFAULT_MAX_NEW_TOKENS = 256
NUM_BEAMS = 3  # snapshot generation_config.json; decoding is deterministic beam search (do_sample=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def remote_code_files(path: str | Path | None = None) -> list[str]:
    """Custom-code files present in the snapshot, plus a marker when ``config.json`` declares ``auto_map``."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    found = [name for name in REMOTE_CODE_FILES if (root / name).is_file()]
    config_path = root / CONFIG_FILE
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as fh:
            if "auto_map" in json.load(fh):
                found.append(f"{CONFIG_FILE}:auto_map")
    return found


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level Levenshtein distance over reference length; for OCR against a known transcript."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be str")
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(reference)


@dataclass
class Florence2Pipeline:
    """``_runner(image, prompt, task, max_new_tokens, num_beams)`` -> ``{"text": raw, "parsed": value}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Florence2Pipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            if allow_download:
                raise RuntimeError(
                    f"Hub loading of {MODEL_ID}@{MODEL_REVISION} requires executing its custom code; "
                    f"REMOTE_CODE_POLICY={REMOTE_CODE_POLICY!r} refuses it (owner pin decision pending)"
                )
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        custom = remote_code_files(root)
        if custom:
            raise RuntimeError(
                f"{MODEL_ID}@{MODEL_REVISION} carries custom code {custom}; loading it needs the "
                "trust_remote_code opt-in, which this package refuses (REMOTE_CODE_POLICY='refuse'). The "
                "native transformers 4.57.6 Florence2ForConditionalGeneration does not match this "
                "checkpoint's tensor layout (918 unused / 920 re-initialised tensors, 2026-09-12). Decision "
                "pending with the repository owner: keep this pin with pinned remote code, or re-pin to the "
                "native port."
            )
        raise RuntimeError("snapshot has no custom code but no native loader is wired; pin decision pending")

    def _validate(
        self, image: Any, task: str, text_input: str | None, max_new_tokens: int, num_beams: int
    ) -> None:
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
        if task not in TASKS:
            raise ValueError(f"task must be one of TASKS {TASKS}, got {task!r}")
        if task in TASKS_WITH_TEXT:
            if not isinstance(text_input, str) or not text_input.strip():
                raise ValueError(f"task {task} requires a non-empty text_input")
            if len(text_input) > MAX_TEXT_CHARS:
                raise ValueError(f"text_input exceeds MAX_TEXT_CHARS={MAX_TEXT_CHARS}: {len(text_input)}")
        elif text_input is not None:
            raise ValueError(f"task {task} takes no text_input")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an int")
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
            raise TypeError("num_beams must be a positive int")

    def run(
        self,
        image: Image.Image,
        task: str = "<CAPTION>",
        text_input: str | None = None,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = NUM_BEAMS,
    ) -> dict[str, Any]:
        """Run one task prompt on one image; ``result`` is the task-parsed value (text, or boxes + labels)."""
        self._validate(image, task, text_input, max_new_tokens, num_beams)
        prompt = task + (text_input or "")
        raw = self._runner(image.convert("RGB"), prompt, task, max_new_tokens, num_beams)
        if not isinstance(raw, dict) or "text" not in raw or "parsed" not in raw:
            raise RuntimeError("runner must return a dict with 'text' and 'parsed'")
        return {
            "task": task,
            "text_input": text_input,
            "result": raw["parsed"],
            "generated_text": str(raw["text"]),
            "image_size": list(image.size),
            "generation": {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False},
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
