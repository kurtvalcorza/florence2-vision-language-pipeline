"""Prompt-driven vision-language tasks with the pinned ``florence-community/Florence-2-large`` snapshot.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly
allowed, from the Hugging Face Hub at the pinned revision, through the native ``transformers`` Florence-2
classes with ``trust_remote_code=False``. Pin history: the original ``microsoft/Florence-2-large`` pin was
rejected on 2026-09-12 because loading it requires executing custom code bundled in the model repository.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "florence-community/Florence-2-large"
MODEL_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
MODEL_LICENSE = "mit"
MODEL_KEY = "florence-2-large-community"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

# Task prompts documented upstream for Florence-2; the last one needs a caption as text input.
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
MAX_NEW_TOKENS = 1024  # hard ceiling for `max_new_tokens` (upstream examples use 1024)
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


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one PIL.Image.Image (any mode, converted to RGB) plus one task prompt from TASKS; the tasks "
        "in TASKS_WITH_TEXT additionally require a caption string as text_input"
    ),
    "image_side_px": [1, MAX_IMAGE_SIDE],
    "tasks": list(TASKS),
    "tasks_requiring_text_input": list(TASKS_WITH_TEXT),
    "text_input_chars": [1, MAX_TEXT_CHARS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "num_beams": (
        f"positive int; the snapshot's generation_config default is NUM_BEAMS={NUM_BEAMS} and decoding "
        "is deterministic beam search (do_sample=False)"
    ),
    "preprocessing": (
        "image converted to RGB; the processor resizes it to exactly 768x768 (bicubic, ImageNet "
        "mean/std), so aspect ratio is not preserved and nothing is cropped; region outputs are mapped "
        "back to input pixel coordinates. The prompt sent to the model is the task token followed by "
        "text_input when the task takes one."
    ),
}

# Which capability families have an intrinsic metric in this repository, and what the others need.
_OCR_TASK = "<OCR>"
_NEEDS: dict[str, str] = {
    "caption": (
        "reference captions for the same images plus a caption metric (for example CIDEr or SPICE), or "
        "human adequacy ratings; this repository ships neither the references nor a caption metric"
    ),
    "region": (
        "annotated boxes for the same images and the caller's own matching/mean-average-precision code; "
        "Florence-2 emits no per-box score, so there is also nothing to calibrate or threshold"
    ),
    "ocr": (
        "a known transcript for the image, passed as the reference, so character_error_rate can be "
        "computed"
    ),
    "grounding": (
        "annotated boxes for the phrases in the supplied caption and the caller's own matching code; no "
        "grounding metric ships with this repository"
    ),
}
_TASK_FAMILY: dict[str, str] = {
    "<CAPTION>": "caption",
    "<DETAILED_CAPTION>": "caption",
    "<MORE_DETAILED_CAPTION>": "caption",
    "<OD>": "region",
    "<DENSE_REGION_CAPTION>": "region",
    "<REGION_PROPOSAL>": "region",
    "<OCR>": "ocr",
    "<OCR_WITH_REGION>": "region",
    "<CAPTION_TO_PHRASE_GROUNDING>": "grounding",
}
_SCORE_SEMANTICS = (
    "Florence-2 emits no probability or confidence: captions and OCR are plain generated text, and "
    "region tasks return boxes with labels and no per-box score, so there is nothing to threshold or "
    "calibrate. Decoding is deterministic beam search, not a likelihood estimate."
)


def _check_inputs(
    image: Any, task: Any, text_input: Any, max_new_tokens: Any, num_beams: Any
) -> Image.Image:
    """Raise TypeError/ValueError naming the first violated ceiling; return the RGB image.

    ``Florence2Pipeline.run`` and ``validate_inputs`` both route through this function so their
    acceptance criteria cannot diverge.
    """
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
    return image.convert("RGB")


def validate_inputs(
    image: Image.Image,
    task: str = "<CAPTION>",
    text_input: str | None = None,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = NUM_BEAMS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``run`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _check_inputs(image, task, text_input, max_new_tokens, num_beams)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (run takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
            }
        ],
        "task": task,
        "task_requires_text_input": task in TASKS_WITH_TEXT,
        "text_input": text_input,
        "generation": {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    reference_text: str | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``result`` is one ``run`` result, or a sequence of them for a multi-capability run. The only
    intrinsic metric in this repository is ``character_error_rate``, and it applies to ``<OCR>``
    when a known transcript is supplied as ``reference_text``; every other capability is
    ``not-measurable`` and the report says what labelled data would make it measurable.
    """
    if not isinstance(result, Mapping):
        subreports = [
            evaluation_report(item, reference_text, sample_kind=sample_kind) for item in result
        ]
        metrics = [
            {**metric, "task": sub["task"]} for sub in subreports for metric in sub["metrics"]
        ]
        return {
            "task": "multi-capability: " + ", ".join(sub["task"] for sub in subreports),
            "score_semantics": _SCORE_SEMANTICS,
            "sample_kind": sample_kind,
            "n_capabilities": len(subreports),
            "metrics": metrics,
            "baselines": [],
            "capabilities": subreports,
            "verdict": "sample-sanity" if metrics else "not-measurable",
            "reason": (
                f"{len(metrics)} capability metric(s) over {len(subreports)} capabilities on one "
                "tutorial sample; sanity evidence, not a benchmark"
                if metrics
                else f"none of the {len(subreports)} demonstrated capabilities has an intrinsic metric here"
            ),
            "needs": "; ".join(dict.fromkeys(sub["needs"] for sub in subreports)),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    task = result["task"]
    base = {
        "task": task,
        "score_semantics": _SCORE_SEMANTICS,
        "sample_kind": sample_kind,
        "n_outputs": 1,
        "generation": dict(result.get("generation") or {}),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    family = _TASK_FAMILY.get(task, "region")
    if task == _OCR_TASK and isinstance(reference_text, str) and reference_text.strip():
        reference = reference_text.strip()
        hypothesis = str(result["result"]).strip()
        return {
            **base,
            "metrics": [
                {
                    "id": "character_error_rate",
                    "value": character_error_rate(reference, hypothesis),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "estimation": "one image against a known transcript, no dispersion estimate",
                }
            ],
            "verdict": "sample-sanity",
            "reason": (
                "one image scored against a transcript the caller already knows; on the synthetic "
                "sample that transcript is text the notebook drew itself, so this is a code-path "
                "check on a rendered font, not an OCR benchmark"
            ),
            "needs": (
                "a labelled OCR corpus from the deployment domain for any generalisable "
                "character-error-rate claim"
            ),
        }
    return {
        **base,
        "metrics": [],
        "verdict": "not-measurable",
        "reason": (
            f"no intrinsic metric exists in this repository for the {family} capability {task}"
            if task != _OCR_TASK
            else "no reference transcript was supplied for the evaluated image"
        ),
        "needs": _NEEDS[family],
    }


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
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Florence2ForConditionalGeneration, Florence2Processor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
        processor = Florence2Processor.from_pretrained(location, **common)
        model, info = Florence2ForConditionalGeneration.from_pretrained(
            location, dtype=dtype, output_loading_info=True, **common
        )
        bad = {k: v for k, v in info.items() if v}
        if bad:
            raise RuntimeError(f"checkpoint does not match the native Florence-2 architecture: {bad}")
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, prompt: str, task: str, max_new_tokens: int, num_beams: int) -> dict:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(resolved_device, dtype)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
            text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task=task, image_size=image.size)
            return {"text": text, "parsed": parsed[task]}

        return cls(runner, resolved_device, source)

    def _validate(
        self, image: Any, task: str, text_input: str | None, max_new_tokens: int, num_beams: int
    ) -> None:
        _check_inputs(image, task, text_input, max_new_tokens, num_beams)

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
