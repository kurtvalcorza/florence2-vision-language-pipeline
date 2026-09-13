"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from florence2_vision_language_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    NUM_BEAMS,
    TASKS,
    TASKS_WITH_TEXT,
    evaluation_report,
    validate_inputs,
)


def _image(width: int = 512, height: int = 512) -> Image.Image:
    return Image.new("RGB", (width, height), (255, 255, 255))


def _result(task: str, value: object, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict:
    return {
        "task": task,
        "text_input": None,
        "result": value,
        "generated_text": f"<s>{value}</s>",
        "image_size": [512, 512],
        "generation": {"max_new_tokens": max_new_tokens, "num_beams": NUM_BEAMS, "do_sample": False},
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), "<OD>", max_new_tokens=128, names=["synthetic_shapes_text_512"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [1, MAX_IMAGE_SIDE]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["schema"]["text_input_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["tasks"] == list(TASKS)
    assert manifest["inputs"] == [{"id": "synthetic_shapes_text_512", "mode": "RGB", "size": [512, 512]}]
    assert manifest["task"] == "<OD>"
    assert manifest["task_requires_text_input"] is False
    assert manifest["generation"] == {"max_new_tokens": 128, "num_beams": NUM_BEAMS, "do_sample": False}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_phrase_grounding_takes_text_and_default_id() -> None:
    manifest = validate_inputs(_image(), TASKS_WITH_TEXT[0], "a red square")
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["task_requires_text_input"] is True
    assert manifest["text_input"] == "a red square"


def test_validate_inputs_rejects_like_run() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 8), "<CAPTION>")
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image", "<CAPTION>")
    with pytest.raises(ValueError, match="task must be one of TASKS"):
        validate_inputs(_image(), "<SEGMENTATION>")
    with pytest.raises(ValueError, match="requires a non-empty text_input"):
        validate_inputs(_image(), TASKS_WITH_TEXT[0])
    with pytest.raises(ValueError, match="takes no text_input"):
        validate_inputs(_image(), "<CAPTION>", "a red square")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(_image(), TASKS_WITH_TEXT[0], "x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), "<CAPTION>", max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="num_beams"):
        validate_inputs(_image(), "<CAPTION>", num_beams=0)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), "<CAPTION>", names=["a", "b"])


def test_evaluation_report_caption_and_detection_are_not_measurable() -> None:
    caption = evaluation_report(_result("<CAPTION>", "a red square with a white background"))
    assert caption["verdict"] == "not-measurable"
    assert caption["metrics"] == []
    assert "caption metric" in caption["needs"]
    detection = evaluation_report(_result("<OD>", {"bboxes": [[63, 63, 193, 193]], "labels": ["flag"]}))
    assert detection["verdict"] == "not-measurable"
    assert "no per-box score" in detection["needs"]
    assert "no probability or confidence" in detection["score_semantics"]
    assert (detection["model_id"], detection["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_ocr_is_sample_sanity_only_with_a_reference() -> None:
    without = evaluation_report(_result("<OCR>", "DIMER 2026"))
    assert without["verdict"] == "not-measurable"
    assert "no reference transcript" in without["reason"]
    with_reference = evaluation_report(_result("<OCR>", "DIMER 2O26"), "DIMER 2026", sample_kind="synthetic")
    assert with_reference["verdict"] == "sample-sanity"
    metric = with_reference["metrics"][0]
    assert metric["id"] == "character_error_rate"
    assert metric["value"] == pytest.approx(0.1)
    assert metric["reference"] == "DIMER 2026"
    assert metric["estimation"]


def test_evaluation_report_accepts_a_sequence_of_capability_results() -> None:
    results = [
        _result("<CAPTION>", "a red square", max_new_tokens=64),
        _result("<OD>", {"bboxes": [], "labels": []}),
        _result("<OCR>", "DIMER 2026", max_new_tokens=128),
    ]
    report = evaluation_report(results, "DIMER 2026")
    assert report["n_capabilities"] == 3
    assert report["verdict"] == "sample-sanity"
    assert [metric["id"] for metric in report["metrics"]] == ["character_error_rate"]
    assert report["metrics"][0]["task"] == "<OCR>"
    assert report["metrics"][0]["value"] == 0.0
    assert [sub["task"] for sub in report["capabilities"]] == ["<CAPTION>", "<OD>", "<OCR>"]
    assert [sub["verdict"] for sub in report["capabilities"]] == [
        "not-measurable",
        "not-measurable",
        "sample-sanity",
    ]
    assert report["baselines"] == []


def test_evaluation_report_multi_capability_without_any_reference_is_not_measurable() -> None:
    report = evaluation_report([_result("<CAPTION>", "a red square"), _result("<OCR>", "DIMER 2026")])
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert "none of the 2 demonstrated capabilities" in report["reason"]
