"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/florence2_vision_language_colab.ipynb`
(review PR #10: FL-M1..M3, FL-m1..m5).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and inert
stand-ins (an injected runner, a fake `google.colab`, tiny PIL images). Nothing here loads the pinned checkpoint or
needs torch, so the whole file runs under CI's install line.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from PIL import Image, ImageDraw  # noqa: E402

import florence2_vision_language_pipeline as fl  # noqa: E402
from florence2_vision_language_pipeline import (  # noqa: E402
    MAX_TEXT_CHARS,
    MAX_TRANSCRIPT_CHARS,
    Florence2Pipeline,
    byod_record_limits,
    drop_duplicate_images,
    load_byod_dataset,
    split_dataset,
    transcript_overlap,
    validate_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "florence2_vision_language_colab.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _line(k: int, width: int | None = None) -> Image.Image:
    image = Image.new("RGB", (width or 40 + k, 16), (255, 255 - k % 200, 255))
    ImageDraw.Draw(image).rectangle([2, 2, 4 + k % 30, 12], fill=(k % 255, 0, 0))
    return image


def _zip(tmp_path: Path, name: str, n: int, *, duplicates: int = 0, clash: bool = False) -> Path:
    buffer = io.BytesIO()
    rows = []
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(n):
            png = io.BytesIO()
            _line(i).save(png, format="PNG")
            archive.writestr(f"lines/l{i:03d}.png", png.getvalue())
            rows.append((f"l{i:03d}.png", f"ligne {i}"))
        for j in range(duplicates):
            png = io.BytesIO()
            _line(j).save(png, format="PNG")
            archive.writestr(f"lines/dup{j:03d}.png", png.getvalue())
            rows.append((f"dup{j:03d}.png", f"copie {j}"))
        if clash:
            png = io.BytesIO()
            _line(n + 1).save(png, format="PNG")
            archive.writestr("other/l000.png", png.getvalue())
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(["file", "text"])
        writer.writerows(rows)
        archive.writestr("transcripts.csv", text.getvalue())
    path = tmp_path / f"{name}.zip"
    path.write_bytes(buffer.getvalue())
    return path


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _section4(monkeypatch, tmp_path, *, byod_path: str = "") -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. Belfort lines, the transcripts and the split")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns: dict = {name: getattr(fl, name) for name in fl.__all__}
    ns.update({"os": __import__("os"), "Path": Path})
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- FL-m2: BYOD bounds, refusals that name the split, duplicates, file-name clashes, upload and path ---------------


def test_byod_minimum_is_twelve_distinct_images():
    assert byod_record_limits() == (12, 5000)


def test_twelve_lines_pass_section4_and_eleven_are_refused_naming_the_split(monkeypatch, tmp_path):
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "twelve", 12)))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 2, "validation": 2, "train": 8}
    assert ns["byod"]["distinct_images"] == 12 and ns["data_source"] == "BYOD (twelve.zip)"
    with pytest.raises(ValueError, match=r"the train split would hold 7 records \(at least 8 are required\).*at least 12 distinct images"):
        _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "eleven", 11)))


def test_pixel_duplicates_are_reported_before_the_split(monkeypatch, tmp_path, capsys):
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "dups", 14, duplicates=3)))
    out = capsys.readouterr().out
    assert "'pixel_duplicates_dropped': 3" in out and "dup000" in out
    assert ns["byod"]["duplicates_dropped"] == ["dup000", "dup001", "dup002"]
    assert sum(len(v) for v in ns["splits"].values()) == 14


def test_duplicate_file_names_in_two_folders_are_refused(tmp_path):
    with pytest.raises(ValueError, match="two files are named 'l000.png'.*without its folders"):
        load_byod_dataset(_zip(tmp_path, "clash", 12, clash=True))


def test_cancelled_upload_and_no_colab_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match="Upload exactly one .zip file \\(received 0\\).*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` raises ImportError
    with pytest.raises(RuntimeError, match="needs Google Colab.*BYOD_PATH"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))


def test_an_uploaded_zip_goes_through_section4(monkeypatch, tmp_path):
    payload = _zip(tmp_path, "up", 13).read_bytes()
    _fake_colab(monkeypatch, [{"up.zip": payload}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["byod"]["zip_sha256"] and sum(len(v) for v in ns["splits"].values()) == 13


def test_byod_path_reads_a_folder_without_colab(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "folder"
    with zipfile.ZipFile(_zip(tmp_path, "f", 12)) as archive:
        for info in archive.infolist():
            target = folder / Path(info.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    assert ns["byod"]["zip_sha256"] is None and sum(len(v) for v in ns["splits"].values()) == 12


# --- FL-M3: the split is described as what it is, and the transcript overlap is reported -------------------------


def test_transcript_overlap_and_duplicate_helper():
    train = [{"id": "a", "text": "Le  Conseil Municipal"}, {"id": "b", "text": "autre"}]
    other = [{"id": "x", "text": "Le Conseil Municipal"}, {"id": "y", "text": "neuf"}]
    assert transcript_overlap(train, other) == ["x"]
    image = _line(1)
    kept, dropped = drop_duplicate_images([{"id": "p", "image": image}, {"id": "q", "image": image.copy()}])
    assert [r["id"] for r in kept] == ["p"] and dropped == ["q"]


def test_split_wording_and_breakdown():
    markdown = _markdown()
    assert "without leakage" not in markdown and "line-disjoint" not in markdown
    for marker in ("**What the split does and does not protect against.**", "**same-collection estimate**", "**The split is by line image, not by page:**", "135 test lines whose transcript does not occur in training"):
        assert marker in markdown, marker
    assert "overlap = {'validation': transcript_overlap(train_records, val_records)" in _code_after("## 4. Belfort lines")
    section8 = _code_after("## 8. Held-out evaluation")
    assert "rates = ocr_metrics([run['hypotheses'][i] for i in unseen], unseen_records)" in section8


# --- FL-m1: one namespace keeps both text bounds; the generator refuses a name collision -------------------------


def test_carried_modules_in_one_namespace_keep_the_grounding_bound(monkeypatch):
    carried = [c for c in _cells() if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
    assert len(carried) == 3
    module = types.ModuleType("fl_notebook_namespace")  # dataclasses resolve their module through sys.modules
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    for cell in carried:
        exec(compile(_source(cell), cell["metadata"]["dimer"]["embedded_module"], "exec"), ns)
    assert ns["MAX_TEXT_CHARS"] == MAX_TEXT_CHARS == 1000 and ns["MAX_TRANSCRIPT_CHARS"] == MAX_TRANSCRIPT_CHARS == 512
    assert ns["INPUT_SCHEMA"]["text_input_chars"] == [1, 1000]
    ns["validate_inputs"](_line(1, width=64), "<CAPTION_TO_PHRASE_GROUNDING>", "x" * 600)  # accepted, as by the package
    validate_inputs(_line(1, width=64), "<CAPTION_TO_PHRASE_GROUNDING>", "x" * 600)


def test_generator_refuses_a_top_level_name_bound_differently_in_two_modules():
    spec = importlib.util.spec_from_file_location("fl_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_name_collisions({"pipeline.py": "MAX_TEXT_CHARS = 1000\n", "samples.py": "MAX_TRANSCRIPT_CHARS = 512\n"})
    build.check_name_collisions({"a.py": "X = 1\n", "b.py": "X = 1\n"})  # identical bindings are harmless
    with pytest.raises(SystemExit, match="MAX_TEXT_CHARS \\(pipeline.py and samples.py\\)"):
        build.check_name_collisions({"pipeline.py": "MAX_TEXT_CHARS = 1000\n", "samples.py": "MAX_TEXT_CHARS = 512\n"})


def test_ceiling_print_names_both_bounds():
    section5 = _code_after("## 5. Run three capabilities")
    assert "'MAX_TEXT_CHARS (grounding text_input)': MAX_TEXT_CHARS, 'MAX_TRANSCRIPT_CHARS (training transcript)': MAX_TRANSCRIPT_CHARS" in section5


# --- FL-m5: re-runs start from the pretrained model; grounding runs; adapt refuses an adapted pipeline -----------


def _capability_runner(calls: list):
    def runner(image, prompt, task, max_new_tokens, num_beams):
        calls.append(prompt)
        parsed = {
            "<CAPTION>": "a poster",
            "<OD>": {"bboxes": [[0.0, 0.0, 511.0, 511.0]], "labels": ["poster"]},
            "<OCR>": "DIMER 2026",
            "<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [[64.0, 64.0, 224.0, 224.0]], "labels": ["a red square"]},
        }[task]
        return {"text": str(parsed), "parsed": parsed}

    return runner


def test_section5_runs_the_grounding_experiment_with_its_text_input(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    calls: list = []
    ns: dict = {name: getattr(fl, name) for name in fl.__all__}
    import hashlib
    import time

    import numpy as np
    from PIL import ImageFont

    ns.update({"os": __import__("os"), "Path": Path, "json": json, "time": time, "hashlib": hashlib, "np": np, "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont, "WEIGHTS_DIR": "w", "pipe": Florence2Pipeline(_capability_runner(calls), "cpu")})
    source = _code_after("## 5. Run three capabilities")
    exec(compile(source, "<section 5>", "exec"), ns)
    assert ns["frozen_report"]["verdict"] == "sample-sanity" and calls == ["<CAPTION>", "<OD>", "<OCR>"]
    ns["CAPABILITIES"]["<CAPTION_TO_PHRASE_GROUNDING>"] = {"input": "image + text", "output": "boxes for the phrase", "max_new_tokens": ns["DEFAULT_MAX_NEW_TOKENS"], "text_input": "a red square"}
    results, _timings, checks, _report = ns["run_capabilities"](ns["pipe"], "frozen")
    assert all(checks.values()) and calls[-1] == "<CAPTION_TO_PHRASE_GROUNDING>a red square"
    assert results["<CAPTION_TO_PHRASE_GROUNDING>"]["text_input"] == "a red square"


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = Florence2Pipeline(_capability_runner([]), "cpu", _model=object(), _processor=object(), adapter={"best_epoch": 1})
    with pytest.raises(ValueError, match="already adapted.*from_pretrained"):
        pipe.adapt([], None)


def test_sections_5_to_7_reset_and_section_6_refuses_an_adapted_model():
    for heading in ("## 5. Run three capabilities", "## 6. Baselines and the frozen model", "## 7. Bounded fine-tuning"):
        assert "reset_to_pretrained()" in _code_after(heading), heading
    section6 = _code_after("## 6. Baselines and the frozen model")
    assert section6.index("reset_to_pretrained()") < section6.index("pipe.evaluate(")
    assert "if frozen_test['adapted']:" in section6
    parity = _code_after("## 9. Look at the lines")
    assert "raise RuntimeError(f'Reload parity failed: {parity}." in parity and "Re-run from Section 7" in parity


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline():
    source = _code_after("## 5. Run three capabilities")
    block = source[source.index("def reset_to_pretrained():") : source.index("reset_to_pretrained()\n")]
    loads: list = []

    class _Loader:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return Florence2Pipeline(_capability_runner([]), "cpu")

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns = {"gc": __import__("gc"), "torch": fake_torch, "Florence2Pipeline": _Loader, "WEIGHTS_DIR": "w", "pipe": Florence2Pipeline(_capability_runner([]), "cpu")}
    exec(compile(block, "<reset>", "exec"), ns)
    ns["reset_to_pretrained"]()
    assert loads == []
    ns["pipe"] = Florence2Pipeline(_capability_runner([]), "cpu", adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == ["w"] and ns["pipe"].adapter is None


def test_experiments_name_their_cell_and_run_after_scope():
    markdown = _markdown()
    experiments = markdown[markdown.index("**Optional experiments") : markdown.index("## Troubleshooting")]
    for line in [ln for ln in experiments.splitlines() if ln.startswith("- **") and "Section 10" not in ln]:
        assert re.search(r"Section \d", line) and ("Run after" in line or "Section 4, then" in line), line
    assert "'text_input': 'a red square'" in experiments
    assert "**Predict → Change one thing → Run → Observe → Explain**" in markdown


# --- FL-m3: a negative result is reported, not asserted away --------------------------------------------------


def test_section8_reports_a_negative_result_and_writes_the_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    texts = {40 + k: f"ligne numero {k}" for k in range(12)}
    records = [{"id": f"t{k}", "image": _line(k), "text": texts[40 + k]} for k in range(12)]
    test_records, val_records, train_records = records[:6], records[6:9], records[9:] + [{"id": "s", "image": _line(30), "text": texts[40]}]

    def batch(reader):
        return lambda images, max_new_tokens, num_beams: [{"text": reader(im.width), "new_tokens": 3} for im in images]

    frozen = Florence2Pipeline(_capability_runner([]), "cpu", _batch_runner=batch(lambda w: texts.get(w, "")))
    adapted = Florence2Pipeline(_capability_runner([]), "cpu", _batch_runner=batch(lambda w: "zzzz zzzz zzzz"), adapter={"best_epoch": 3})
    ns: dict = {name: getattr(fl, name) for name in fl.__all__}
    ns.update({"os": __import__("os"), "Path": Path, "json": json, "pipe": adapted, "METRICS": ("cer", "wer", "cer_macro", "exact_match"), "LINE_MAX_NEW_TOKENS": 128, "USE_BYOD": True, "EPOCHS": 6, "LEARNING_RATE": 5e-5, "data_source": "BYOD (x.zip)", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {"test": 6}, "adapt_seconds": 1.0, "test_records": test_records, "val_records": val_records})
    ns["baseline_empty"] = fl.empty_baseline(test_records)
    ns["baseline_constant"] = fl.constant_baseline(train_records, test_records)
    ns["frozen_test"] = frozen.evaluate(test_records)
    ns["overlap"] = {"validation": [], "test": transcript_overlap(train_records, test_records)}
    ns["adapt_result"] = {"best_epoch": 3, "history": [], "trainable_names": [], "seed": 0}
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    out = capsys.readouterr().out
    assert ns["adapted_beats_frozen"] is False and "did not help here" in out
    report = json.loads((tmp_path / "outputs" / "florence2_vision_language_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["adapted_beats_frozen"] is False and report["transcript_breakdown"]["transcript_also_in_train"] == 1
    assert report["transcript_breakdown"]["frozen_unseen_transcripts"]["n"] == 5


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = _source(cell)
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]


# --- FL-m4: the stale `flag` expectation is gone; the spread and the tensor count are stated -------------------


def test_expected_outputs_match_the_recorded_run():
    markdown = _markdown()
    assert "`flag`" not in markdown and "`poster`" in markdown
    assert "**Run-to-run spread.**" in markdown and "106 tensors" in markdown
    for doc in ("MODEL_CARD.md", "docs/WEIGHTS.md", "docs/release-verification.md"):
        assert "66 tensors" not in (ROOT / doc).read_text(encoding="utf-8").replace("\n     ", " "), doc


# --- FL-M1: isolated runtime ------------------------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "its restart" not in markdown
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status and "2 passes" in status


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on a spec-less stub."""
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


# --- FL-M2: guided layer and infrastructure labels ------------------------------------------------------------


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 6), ("**What to notice:**", 6), ("<summary>Check your reasoning</summary>", 7), ("> **Infrastructure.**", 3)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Your turn — change one thing", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    code = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code[:7]  # install, router, runtime record, three carried modules, model staging
    assert all(c["metadata"].get("cellView") == "form" for c in setup)
    assert all(_source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert "cellView" not in code[7]["metadata"]  # the learning path starts in Section 4
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    for item in ("GDL1", "GDL2", "GDL3", "GDL4", "GDL5", "GDL6", "GDL7", "GDL9", "GDL10", "GDL11", "GDL12", "GDL14"):
        assert f"| {item} " in registry, item


def test_split_dataset_refusal_message_names_counts():
    records = [{"id": f"r{k}", "image": _line(k), "text": "x"} for k in range(5)]
    with pytest.raises(ValueError, match="the train split would hold 3 records.*5 records, 5 distinct images"):
        split_dataset(records)
