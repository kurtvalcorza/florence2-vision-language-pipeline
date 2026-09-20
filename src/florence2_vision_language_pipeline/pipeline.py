"""Prompt-driven vision-language tasks with the pinned ``florence-community/Florence-2-large`` snapshot, plus the
adaptation contract for the ``<OCR>`` task: corpus-level evaluation on labelled text lines, bounded fine-tuning of the
last decoder layers on cached encoder outputs, and a verified adapter artifact.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly
allowed, from the Hugging Face Hub at the pinned revision, through the native ``transformers`` Florence-2
classes with ``trust_remote_code=False``. Pin history: the original ``microsoft/Florence-2-large`` pin was
rejected on 2026-09-12 because loading it requires executing custom code bundled in the model repository.
"""

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

# The processor resizes every image to 768x768 regardless (preprocessor_config.json), so the side and pixel
# ceilings only guard memory while decoding and resizing (a 9,000 px wide text line is fine, a 4096x4096 page is
# the largest area accepted).
MAX_IMAGE_SIDE = 16_384
MAX_IMAGE_PIXELS = 4096 * 4096
MIN_IMAGE_SIDE = 1
MAX_TEXT_CHARS = 1000  # characters of caption text accepted for phrase grounding
MAX_NEW_TOKENS = 1024  # hard ceiling for `max_new_tokens` (upstream examples use 1024)
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_LINE_MAX_NEW_TOKENS = 128  # the corpus stages' budget per text line
NUM_BEAMS = 3  # snapshot generation_config.json; decoding is deterministic beam search (do_sample=False)
OCR_PROMPT_TOKENS = 587  # 577 image tokens + the <OCR> prompt; identical for every image
# Model facts (measured on the pinned snapshot; tests pin them).
PARAMETER_COUNT = 776_505_344
DECODER_LAYERS = 12
TRAINABLE_LAYERS = 4  # the last decoder layers + the decoder's embedding layer norm are the adapter
ADAPTER_PARAMETERS = 67_188_736
# Adaptation contract.
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 8
CACHE_BATCH_SIZE = 4  # images per frozen forward while caching encoder outputs
VISION_BATCH_SIZE = 4  # images per vision-tower forward (bounds its activation memory)
GRAD_CLIP = 1.0
_TRAINABLE_FIRST_LAYER = DECODER_LAYERS - TRAINABLE_LAYERS
_TRAINABLE_PREFIXES = tuple(f"model.language_model.decoder.layers.{i}." for i in range(_TRAINABLE_FIRST_LAYER, DECODER_LAYERS)) + ("model.language_model.decoder.layernorm_embedding.",)


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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHTS_FILE), None)


def normalise_text(text: str) -> str:
    """The transcript form the corpus measures use: whitespace runs collapsed to one space, ends stripped."""
    return " ".join(str(text).split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance (insertions + deletions + substitutions, unit cost) between two sequences."""
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, 1):
        current = [row_index]
        for column_index, hyp_item in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column_index] + 1, previous[column_index - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level Levenshtein distance over reference length; for OCR against a known transcript."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be str")
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def validate_image(image: Any) -> Image.Image:
    """The image contract every task and every dataset record shares; returns the RGB image."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side outside {MIN_IMAGE_SIDE}..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image area {width * height} px > MAX_IMAGE_PIXELS {MAX_IMAGE_PIXELS}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one PIL.Image.Image (any mode, converted to RGB) plus one task prompt from TASKS; the tasks "
        "in TASKS_WITH_TEXT additionally require a caption string as text_input"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "image_pixels_max": MAX_IMAGE_PIXELS,
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
    rgb = validate_image(image)
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
    return rgb


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


def _trainable_names(model: Any) -> list[str]:
    """The last `TRAINABLE_LAYERS` BART decoder layers and the decoder's embedding layer norm; the DaViT vision tower,
    the projector, the BART encoder, the shared embeddings (tied to the output head) and the earlier decoder layers
    stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError(f"artifact tensors must all belong to the last {TRAINABLE_LAYERS} decoder layers or the decoder embedding norm")


@dataclass
class Florence2Pipeline:
    """``_runner(image, prompt, task, max_new_tokens, num_beams)`` -> ``{"text": raw, "parsed": value}``; the optional
    ``_batch_runner(images, max_new_tokens, num_beams)`` -> one ``{"text", "new_tokens"}`` per image for ``<OCR>``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    source: str = "injected"
    dtype: str = "float32"
    _batch_runner: Callable[..., list[dict[str, Any]]] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
        # float32 on every device: the adapter is trained in float32 and overlays without a cast, and CPU,
        # Tesla-class and consumer GPUs then run the same arithmetic.
        dtype = torch.float32
        processor = Florence2Processor.from_pretrained(location, **common)
        model, info = Florence2ForConditionalGeneration.from_pretrained(
            location, dtype=dtype, output_loading_info=True, **common
        )
        bad = {k: v for k, v in info.items() if v}
        if bad:
            raise RuntimeError(f"checkpoint does not match the native Florence-2 architecture: {bad}")
        model = model.eval().to(resolved_device)
        for param in model.parameters():
            param.requires_grad_(False)
        vision_features = model.model.get_image_features

        def chunked_image_features(pixel_values: Any, **kwargs: Any) -> Any:
            """Bound the vision tower's activation memory: `VISION_BATCH_SIZE` images per forward, results concatenated."""
            if pixel_values.shape[0] <= VISION_BATCH_SIZE:
                return vision_features(pixel_values, **kwargs)
            chunks = [vision_features(pixel_values[i : i + VISION_BATCH_SIZE], **kwargs) for i in range(0, pixel_values.shape[0], VISION_BATCH_SIZE)]
            return torch.cat(chunks, dim=0)

        model.model.get_image_features = chunked_image_features
        eos_id = model.config.text_config.eos_token_id
        pad_id = processor.tokenizer.pad_token_id

        def runner(image: Image.Image, prompt: str, task: str, max_new_tokens: int, num_beams: int) -> dict:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(resolved_device, dtype)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
            text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(text, task=task, image_size=image.size)
            return {"text": text, "parsed": parsed[task]}

        def batch_runner(images: Sequence[Image.Image], max_new_tokens: int, num_beams: int) -> list[dict[str, Any]]:
            inputs = processor(text=[_OCR_TASK] * len(images), images=list(images), return_tensors="pt", padding=True)
            if not bool(inputs["attention_mask"].all()):
                raise RuntimeError("prompts in one batch differ in length; batched generation needs identical prompts")
            inputs = inputs.to(resolved_device, dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            out = []
            for row, text in zip(generated, texts, strict=True):
                ids = row.tolist()[2:]  # decoder start + <s>
                n_new = len(ids)
                for position, token in enumerate(ids):
                    if token in (eos_id, pad_id):
                        n_new = position + (token == eos_id)
                        break
                out.append({"text": text, "new_tokens": int(n_new)})
            return out

        return cls(runner, resolved_device, source, str(dtype).removeprefix("torch."), batch_runner, model, processor, _weight_digest(root))

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


    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract (the <OCR> task on transcribed text lines)
    # ------------------------------------------------------------------------------------------------------

    def transcribe(
        self,
        images: Sequence[Image.Image],
        *,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        num_beams: int = NUM_BEAMS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run ``<OCR>`` on many images in batches (every image shares the same prompt, so no padding is involved);
        one ``{text, new_tokens, truncated}`` per image, in order. With an injected runner and no batch runner the
        images are recognised one by one through ``run``."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        checked = [_check_inputs(image, _OCR_TASK, None, max_new_tokens, num_beams) for image in images]
        out: list[dict[str, Any]] = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            if self._batch_runner is not None:
                raw = self._batch_runner(batch, max_new_tokens, num_beams)
            else:
                raw = []
                for image in batch:
                    single = self._runner(image, _OCR_TASK, _OCR_TASK, max_new_tokens, num_beams)
                    raw.append({"text": single["parsed"], "new_tokens": int(single.get("new_tokens", 0))})
            if not isinstance(raw, list) or len(raw) != len(batch) or any(not isinstance(r, dict) or "text" not in r for r in raw):
                raise RuntimeError("batch runner must return one dict with 'text' per image")
            for item in raw:
                new_tokens = int(item.get("new_tokens", 0))
                out.append({"text": normalise_text(str(item["text"])), "new_tokens": new_tokens, "truncated": new_tokens >= max_new_tokens})
            if progress is not None:
                progress(len(out), len(checked))
        return out

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for adapt/save_artifact/load_artifact")
        return self._model, self._processor

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        num_beams: int = NUM_BEAMS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Transcribe every validated record with ``<OCR>`` and score the hypotheses with ``metrics.ocr_metrics``
        (micro and macro CER / WER, exact match). Works with an injected runner too."""
        from .metrics import ocr_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        items = self.transcribe([r["image"] for r in checked], max_new_tokens=max_new_tokens, num_beams=num_beams, batch_size=batch_size, progress=progress)
        hypotheses = [item["text"] for item in items]
        metrics = ocr_metrics(hypotheses, checked)
        metrics.update(
            {
                "hypotheses": hypotheses,
                "truncated": sum(item["truncated"] for item in items),
                "new_tokens": sum(item["new_tokens"] for item in items),
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        *,
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 8,
        seed: int = 0,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        num_beams: int = NUM_BEAMS,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the last `TRAINABLE_LAYERS` BART decoder layers and the decoder's embedding layer norm
        on transcribed lines with the sequence-to-sequence loss over the transcript tokens (``<s>`` … transcript …
        ``</s>``, teacher-forced from the decoder start token) — the model's own objective. The frozen prefix — DaViT
        vision tower, projector and the BART encoder over the ``<OCR>`` prompt — is run once per line under no gradient
        and its encoder output cached, so each step runs only the decoder on those cached states; the loss equals the
        full model's loss exactly. AdamW (no weight decay), gradient clipping at `GRAD_CLIP`, seeded shuffling, no
        scheduler, no augmentation. Epoch 0 records the frozen model's validation metrics; the epoch with the lowest
        validation CER is kept (the final one without a validation split). On any exception the frozen weights are
        restored."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch
        from transformers.modeling_outputs import BaseModelOutput

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        names = _trainable_names(model)
        name_set = set(names)
        device = torch.device(self.device)
        tokenizer = processor.tokenizer
        start_id = int(model.config.text_config.decoder_start_token_id)
        bos_id = int(tokenizer.bos_token_id)
        eos_id = int(tokenizer.eos_token_id)
        pad_id = int(tokenizer.pad_token_id)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            result = self.evaluate(val_checked, max_new_tokens=max_new_tokens, num_beams=num_beams)
            return {"cer": result["cer"], "wer": result["wer"], "cer_macro": result["cer_macro"], "exact_match": result["exact_match"], "n": result["n"]}

        def _targets(batch: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
            seqs = [[start_id, bos_id, *tokenizer(r["text"], add_special_tokens=False)["input_ids"], eos_id] for r in batch]
            length = max(len(seq) for seq in seqs) - 1
            decoder_input = torch.full((len(batch), length), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), length), -100, dtype=torch.long)
            for i, seq in enumerate(seqs):
                decoder_input[i, : len(seq) - 1] = torch.tensor(seq[:-1])
                labels[i, : len(seq) - 1] = torch.tensor(seq[1:])
            return decoder_input, labels

        try:
            # 1. cache the frozen prefix: the encoder output for every training line (identical prompt lengths)
            cache: list[tuple[Any, Any, Any]] = []
            for start in range(0, len(train_checked), CACHE_BATCH_SIZE):
                batch = train_checked[start : start + CACHE_BATCH_SIZE]
                inputs = processor(text=[_OCR_TASK] * len(batch), images=[r["image"] for r in batch], return_tensors="pt", padding=True)
                if not bool(inputs["attention_mask"].all()):
                    raise RuntimeError("prompts in one batch differ in length")
                inputs = inputs.to(device)
                with torch.no_grad():
                    encoded = model.model(**inputs, decoder_input_ids=torch.full((len(batch), 1), start_id, dtype=torch.long, device=device)).encoder_last_hidden_state
                decoder_input, labels = _targets(batch)
                for k in range(len(batch)):
                    cache.append((encoded[k].detach().to("cpu"), decoder_input[k], labels[k]))
                del encoded
            cache_seconds = round(time.perf_counter() - started, 3)
            # 2. train the decoder tail on the cached encoder outputs
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("cer", float("inf"))
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(range(len(cache)))
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    items = [cache[k] for k in order[start : start + batch_size]]
                    encoded = torch.stack([h for h, _, _ in items]).to(device)
                    length = max(int(d.shape[0]) for _, d, _ in items)
                    decoder_input = torch.full((len(items), length), pad_id, dtype=torch.long)
                    labels = torch.full((len(items), length), -100, dtype=torch.long)
                    for k, (_, d, lab) in enumerate(items):
                        decoder_input[k, : d.shape[0]] = d
                        labels[k, : lab.shape[0]] = lab
                    attention = torch.ones((len(items), encoded.shape[1]), dtype=torch.long, device=device)
                    loss = model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded), attention_mask=attention, decoder_input_ids=decoder_input.to(device), labels=labels.to(device)).loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["cer"] < best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("cer", float("inf"))
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "task": _OCR_TASK,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "first_trainable_layer": _TRAINABLE_FIRST_LAYER,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "lowest validation CER" if val_checked is not None else "final epoch (no validation split)",
            "loss": "sequence-to-sequence cross-entropy over the transcript tokens and the end token, teacher-forced from the decoder start token; computed on the cached frozen encoder outputs",
            "lr": float(lr),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests and the training
        configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHTS_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the last decoder layers and the decoder embedding norm."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Florence2Pipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
