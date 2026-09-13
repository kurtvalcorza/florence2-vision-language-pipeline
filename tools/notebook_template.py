"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "florence2_vision_language_pipeline",
    "repo_name": "florence2-vision-language-pipeline",
    "stem": "florence2_vision_language",
    "notebook_name": "florence2_vision_language_colab.ipynb",
    "profile": "MULTI-CAPABILITY",
    "pipeline_class": "Florence2Pipeline",
    "weights_key": "florence-2-large-community",
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "Florence-2-large — DIMER multi-capability vision-language tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/florence2-vision-language-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/florence2-vision-language-pipeline/blob/main/tutorials/florence2_vision_language_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-florence--community%2FFlorence--2--large-ffcc4d?style=flat",
            "https://huggingface.co/florence-community/Florence-2-large",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2FFlorence--2--large-181717?style=flat&logo=huggingface&logoColor=white",
            "https://huggingface.co/microsoft/Florence-2-large",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg", "https://arxiv.org/abs/2311.06242"),
    ],
    "capability": "prompt-selected vision-language tasks — image captioning, object detection and OCR demonstrated — using the pinned Florence-2-large weights",
    "intro": (
        "Florence-2 is one sequence-to-sequence model whose behaviour is selected by a **task prompt token**: the image is "
        "resized to 768 × 768 by the processor (aspect ratio is not preserved), encoded into visual tokens, and the decoder "
        "generates text that the processor then parses per task — plain text for captions and OCR, boxes plus labels in "
        "input-pixel coordinates for region tasks. Decoding is **deterministic beam search** (`num_beams` = 3 from the "
        "snapshot generation config, `do_sample=False`), so a rerun on the same device, dtype and library versions "
        "reproduces the same output. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting — the pinned checkpoint is used as published. The pin is the community \"official "
        "transformers converted checkpoint\" (`florence-community/Florence-2-large`), loadable by native `transformers` "
        "classes with `trust_remote_code=False`; the original `microsoft/Florence-2-large` snapshot needs remote code and "
        "was rejected (see the weight provenance document). What upstream supplies is the model, processor and task-token "
        "convention; what the carried pipeline module adds is manifest verification, input validation and ceilings, a "
        "fixed per-task output contract, and the `character_error_rate`, `validate_inputs` and `evaluation_report` helpers. "
        "The loader also refuses a checkpoint whose weights do not map cleanly onto the native architecture "
        "(`output_loading_info` must report no missing, unexpected or mismatched keys)."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, generate a synthetic image with "
        "drawn shapes and drawn text (or upload your own), stage and digest-verify the immutable upstream snapshot, "
        "surface the ceilings and the task list and validate every demonstrated capability's request into one input "
        "manifest, run three capabilities — `<CAPTION>`, `<OD>` and `<OCR>` — through the public API with explicit "
        "generation settings, read each capability's output contract correctly, produce an evaluation report that is "
        "`sample-sanity` only because OCR can be scored against text the notebook knows and `not-measurable` for the "
        "capabilities this repository ships no metric for, and export machine-readable results plus provenance."
    ),
    "exclusions": (
        "the other task tokens in `TASKS` are exposed but not demonstrated — `<DETAILED_CAPTION>`, "
        "`<MORE_DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR_WITH_REGION>` and "
        "`<CAPTION_TO_PHRASE_GROUNDING>` (the only task that takes a `text_input`). Neither this notebook nor the "
        "repository provides segmentation of any kind (`<REFERRING_EXPRESSION_SEGMENTATION>`, "
        "`<REGION_TO_SEGMENTATION>`), region-to-category or region-to-description prompts, open-vocabulary detection, "
        "visual question answering, batched inference, confidence scores for boxes or text, or fine-tuning. The pipeline "
        "rejects any task token outside `TASKS`."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (float16 there, the dtype the checkpoint ships in); on the model card's CPU smoke the snapshot loaded in 7.1 s, `<CAPTION>` took 4.9 s and `<OD>` 9.0 s on a 256 × 256 drawing, so the three-task default runs in about a minute on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 1.55 GB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python and PIL image handling; what beam search is; what a bounding box in pixel coordinates is; what a character error rate measures.",
        "- **Expected output:** a \"slow image processor\" notice from `transformers` is expected and harmless. A `RuntimeError: checkpoint does not match the native Florence-2 architecture` means the staged weights are not the pinned converted checkpoint.",
        "- **Data:** the default sample is a synthetic image generated in code; BYOD is one image file, gated off by default, any mode (converted to RGB), with both sides between 1 and `MAX_IMAGE_SIDE` = 4096 px, resized to 768 × 768 regardless of aspect ratio. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded images remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a 512 × 512 white canvas drawn in this cell with a filled red square, a "
                "filled blue circle, and the text `DIMER 2026` rendered with Pillow's built-in font — so it needs no download, "
                "contains no personal data, and is reproducible from code (no randomness, no seed; its pixel SHA-256 is printed "
                "and exported). The drawn text is the **known OCR reference**, the only thing in this notebook that can be "
                "scored. A drawing is not a photograph, so every output it produces is smoke/sanity evidence that the code path "
                "works, not a quality measurement and not benchmark evidence: the model card's smoke labelled a plain red square "
                "`flag`.\n\n"
                "BYOD is optional and disabled by default. If your image contains text you know exactly, put it in "
                "`OCR_REFERENCE` and Section 7 scores OCR against it; leave it empty otherwise and the OCR capability is "
                "reported as `not-measurable` like the other two. The upload stays inside this runtime. Nothing is validated in "
                "this cell — the next section hands every demonstrated request to the pipeline's own validation stage, which is "
                "the only checker."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "OCR_REFERENCE = ''  # @param {{type:\"string\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    sample_kind = 'BYOD upload'\n"
                "    ocr_reference = OCR_REFERENCE.strip() or None\n"
                "    drawn_square = None\n"
                "else:\n"
                "    # Deterministic drawing: a red square, a blue circle and one line of text in the built-in font.\n"
                "    image = Image.new('RGB', (512, 512), (255, 255, 255))\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    drawn_square = (64, 64, 224, 224)\n"
                "    draw.rectangle(drawn_square, fill=(220, 30, 30))\n"
                "    draw.ellipse((300, 96, 460, 256), fill=(30, 60, 220))\n"
                "    ocr_reference = 'DIMER 2026'\n"
                "    draw.text((96, 360), ocr_reference, fill=(0, 0, 0), font=ImageFont.load_default(size=48))\n"
                "    image_name = 'synthetic_shapes_text_512'\n"
                "    sample_kind = 'synthetic (drawn in this cell)'\n\n"
                "sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample': image_name, 'sample_kind': sample_kind, 'mode': image.mode, 'size': image.size, 'ocr_reference': ocr_reference, 'drawn_square': drawn_square, 'pixel_sha256': sample_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate every capability's request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `run` applies — "
                "image type and sides 1..`MAX_IMAGE_SIDE` px, a task token inside `TASKS`, a non-empty `text_input` of at most "
                "`MAX_TEXT_CHARS` characters for the tasks in `TASKS_WITH_TEXT` and none for the others, `max_new_tokens` in "
                "1..`MAX_NEW_TOKENS`, and a positive `num_beams` — and returns an **input manifest** naming the schema and "
                "ceilings, the input's observed mode and size, the task, whether that task takes a text input, and the exact "
                "generation settings. This is a multi-capability notebook, so the cell validates **each** demonstrated "
                "capability and writes one combined manifest to `outputs/{stem}_input_manifest.json`: a top-level record listing "
                "the demonstrated tasks, with one per-capability manifest under `capabilities`.\n\n"
                "The per-capability input/output contract is printed next to the ceilings, because the three capabilities do not "
                "share one: `<CAPTION>` and `<OCR>` return a plain string with no score, while `<OD>` returns `{{'bboxes', "
                "'labels'}}` in input-pixel coordinates, also with no per-box score. To show what rejection looks like, the cell "
                "validates an unsupported task token and records the pipeline's own error message as a finding. **What the "
                "pipeline changes about your image:** the processor resizes it to exactly 768 × 768 (bicubic, ImageNet "
                "mean/std) — nothing is cropped, but a non-square image is distorted — and region outputs are mapped back to "
                "your input's pixel coordinates. The notebook itself does not resize, crop, or subsample."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'NUM_BEAMS': NUM_BEAMS}}}})\n"
                "print({{'TASKS_WITHOUT_TEXT': TASKS_WITHOUT_TEXT, 'TASKS_WITH_TEXT': TASKS_WITH_TEXT}})\n"
                "CAPABILITIES = {{\n"
                "    '<CAPTION>': {{'input': 'image only', 'output': 'result: str (one short caption); no score', 'max_new_tokens': 64}},\n"
                "    '<OD>': {{'input': 'image only', 'output': \"result: {{'bboxes': [[x1, y1, x2, y2], ...] in input pixels, 'labels': [str, ...]}}; no per-box score\", 'max_new_tokens': DEFAULT_MAX_NEW_TOKENS}},\n"
                "    '<OCR>': {{'input': 'image only', 'output': 'result: str (transcribed text, reading order chosen by the model); no score', 'max_new_tokens': 128}},\n"
                "}}\n"
                "for task, contract in CAPABILITIES.items():\n"
                "    print(task, contract)\n"
                "manifests = {{task: validate_inputs(image, task, max_new_tokens=contract['max_new_tokens'], num_beams=NUM_BEAMS, names=[image_name]) for task, contract in CAPABILITIES.items()}}\n"
                "input_manifest = {{**manifests['<CAPTION>'], 'task': 'multi-capability: ' + ', '.join(CAPABILITIES), 'tasks': list(CAPABILITIES), 'findings': [], 'capabilities': manifests}}\n"
                "# Demonstrate rejection on a task the pipeline does not expose; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, '<REFERRING_EXPRESSION_SEGMENTATION>')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-task-probe', 'task': '<REFERRING_EXPRESSION_SEGMENTATION>', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Run the three capabilities\n\n"
                "Each call to `run(image, task, *, max_new_tokens=..., num_beams=...)` returns `task`, `result` (the "
                "task-parsed value), `generated_text` (the raw decoder output with its special tokens), `image_size`, the "
                "`generation` settings actually used (`max_new_tokens`, `num_beams`, `do_sample: False`), device, source and "
                "model identity. The settings are printed for every call because they change the output: a caption cut off by "
                "`max_new_tokens` is a truncated caption, and a different beam count is a different search.\n\n"
                "**Capability A — `<CAPTION>`:** input the image; output one short caption string, no score. **Capability B — "
                "`<OD>`:** input the image; output `{{bboxes, labels}}` in input-pixel coordinates with **no confidence "
                "scores** (Florence-2 emits none), so there is no threshold to set and nothing to calibrate. **Capability C — "
                "`<OCR>`:** input the image; output the transcribed text as one string, no score. The structural checks below "
                "assert the output contract of each capability — that the caption and OCR strings are non-empty text, that the "
                "`<OD>` boxes and labels are the same length and lie inside the image, and that every call really used "
                "deterministic settings — and raise if any of them fails. They are contract checks, not quality measurements. "
                "Outputs are fluent even when wrong — the model can name objects that are not there or transpose characters — "
                "and nothing in the output signals it. The printed seconds are measured on this runtime for this one image and "
                "include the first-call warm-up."
            ),
            "code": (
                "import time\n\n"
                "results = {{}}\n"
                "timings = {{}}\n"
                "for task, contract in CAPABILITIES.items():\n"
                "    started = time.perf_counter()\n"
                "    results[task] = pipe.run(image, task, max_new_tokens=contract['max_new_tokens'], num_beams=NUM_BEAMS)\n"
                "    timings[task] = round(time.perf_counter() - started, 3)\n"
                "    print({{'task': task, 'seconds': timings[task], 'generation': results[task]['generation'], 'result': results[task]['result']}})\n"
                "caption = results['<CAPTION>']['result']\n"
                "detections = results['<OD>']['result']\n"
                "ocr_text = results['<OCR>']['result']\n"
                "checks = {{\n"
                "    'caption_is_text': isinstance(caption, str) and bool(caption.strip()),\n"
                "    'od_boxes_and_labels_aligned': isinstance(detections, dict) and len(detections.get('bboxes', [])) == len(detections.get('labels', [])),\n"
                "    'od_boxes_inside_image': all(0 <= x1 <= x2 <= image.width and 0 <= y1 <= y2 <= image.height for x1, y1, x2, y2 in detections.get('bboxes', [])),\n"
                "    'ocr_is_text': isinstance(ocr_text, str),\n"
                "    'deterministic_settings': all(r['generation']['do_sample'] is False and r['generation']['num_beams'] == NUM_BEAMS for r in results.values()),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'capability output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report, here over all three "
                "capabilities at once: the top level carries the combined verdict and every metric found, and `capabilities` "
                "carries one sub-report per task. The **only** intrinsic metric in this repository is "
                "`character_error_rate(reference, hypothesis)` — character-level Levenshtein distance divided by the reference "
                "length — and it applies to `<OCR>` only when a reference transcript is known. On the default sample that "
                "reference is the text the notebook drew, so the figure is a sanity check of the code path on a rendered font, "
                "not an OCR benchmark; with no reference the OCR sub-report is `not-measurable` too. `<CAPTION>` and `<OD>` are "
                "always `not-measurable` here and the report says what each would need: reference captions plus a caption "
                "metric, or annotated boxes plus the caller's own mean-average-precision code. No baseline is reported for any "
                "capability, because none is meaningful without labelled data. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(list(results.values()), ocr_reference, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "for sub in report['capabilities']:\n"
                "    if sub['verdict'] == 'not-measurable':\n"
                "        print({{'task': sub['task'], 'verdict': sub['verdict'], 'needs': sub['needs']}})"
            ),
        },
        {
            "md": (
                "## 8. Preview the detections, export results and provenance\n\n"
                "The preview draws the `<OD>` boxes and labels onto a copy of the input with Pillow and saves it as "
                "`outputs/{stem}_preview.png` so you can see where the detector placed them; it is a visual aid only — the "
                "machine-readable boxes exported alongside it are the outputs intended for downstream use. On the synthetic "
                "drawing expect boxes around the shapes with whatever labels the model chose; the model card's smoke called a "
                "red square `flag`.\n\n"
                "`outputs/{stem}_result.json` records one entry per capability (task token, parsed `result`, raw "
                "`generated_text`, generation settings, seconds), the structural checks, the evaluation report, the input "
                "manifest, the ceilings in force, the full list of exposed tasks, the sample identity (name, kind, size, pixel "
                "digest, OCR reference, drawn square), the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, and the runtime identity "
                "(Python, `torch`, `transformers`, Pillow, device, dtype). No credentials are involved in any step, so none can "
                "reach the export."
            ),
            "code": (
                "preview = image.convert('RGB').copy()\n"
                "draw_preview = ImageDraw.Draw(preview)\n"
                "for (x1, y1, x2, y2), label in zip(detections.get('bboxes', []), detections.get('labels', [])):\n"
                "    draw_preview.rectangle((x1, y1, x2, y2), outline=(0, 160, 0), width=3)\n"
                "    draw_preview.text((x1 + 4, y1 + 4), label, fill=(0, 160, 0))\n"
                "preview.save('outputs/{stem}_preview.png')\n"
                "payload = {{\n"
                "    'capabilities': {{\n"
                "        task: {{'result': r['result'], 'generated_text': r['generated_text'], 'generation': r['generation'], 'seconds': timings[task]}}\n"
                "        for task, r in results.items()\n"
                "    }},\n"
                "    'sanity_checks': checks,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'ceilings': {{'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'NUM_BEAMS': NUM_BEAMS}},\n"
                "    'tasks_exposed': list(TASKS),\n"
                "    'sample': {{'name': image_name, 'kind': sample_kind, 'width': image.width, 'height': image.height, 'pixel_sha256': sample_sha256, 'ocr_reference': ocr_reference, 'drawn_square': drawn_square}},\n"
                "    'preview_file': 'outputs/{stem}_preview.png',\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'pillow': PIL.__version__,\n"
                "        'device': pipe.device,\n"
                "        'source': pipe.source,\n"
                "        'dtype': 'float16' if pipe.device.startswith('cuda') else 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The three outputs are generated text parsed per task: a caption string, boxes with labels and **no confidence "
        "scores**, and an OCR string. None of them carries a probability, and the only quantity the notebook scores is OCR "
        "against text it knows — on the default sample the text it drew itself, which makes the character error rate a "
        "code-path sanity check on a rendered font, not an OCR benchmark; captions and detections are reported "
        "`not-measurable` because this repository ships no metric for them, and they need human judgement or annotated "
        "references plus the caller's own evaluation code. Every output is fluent whether or not it is right: the model can "
        "describe objects that are not there, label a plain square as a `flag`, or transpose characters, and nothing in the "
        "output signals it. The image is squashed to 768 × 768 before encoding, so thin or off-aspect content is distorted; "
        "region outputs are mapped back to input pixels. Decoding is deterministic beam search (3 beams, no sampling) on a "
        "fixed device and dtype; CPU float32 and CUDA float16 can produce different text. The repository exposes the nine "
        "task tokens in `TASKS` — including phrase grounding, which takes a caption as `text_input` — and nothing else: no "
        "segmentation of any kind, no region-to-category or region-to-description prompts, no open-vocabulary detection, no "
        "VQA, no batching.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model snapshot, validate the demonstrated requests against the enforced "
        "ceilings, execute the public pipeline path for three task tokens with explicit generation settings, and emit the "
        "shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, caption, detection or OCR quality on any domain, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/florence-2-large-community/` and rerun Section 3. `RuntimeError: "
        "checkpoint does not match the native Florence-2 architecture` in Section 3: the staged weights are not the pinned "
        "converted checkpoint — re-stage. A `ValueError` naming `MAX_IMAGE_SIDE` in Section 5: resize the BYOD image and "
        "rerun from Section 4. A truncated caption or OCR string: raise that task's `max_new_tokens` in Section 5 (up to "
        "`MAX_NEW_TOKENS`). A \"slow image processor\" notice from `transformers` is expected and harmless.\n\n"
        "**Next experiments.** Upload a photograph with readable signage and supply `OCR_REFERENCE` to see how the character "
        "error rate behaves on real text; add `<CAPTION_TO_PHRASE_GROUNDING>` to `CAPABILITIES` with the caption from "
        "Capability A as its `text_input` and compare its boxes with `<OD>`; run the same image on a CUDA runtime and diff "
        "the float16 outputs against the CPU float32 ones. None of these turns the sample result into evidence of production "
        "fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and pin history: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Pinned converted checkpoint: https://huggingface.co/{MODEL_ID}\n"
        "- Original weights and licence: https://huggingface.co/microsoft/Florence-2-large\n"
        "- Florence-2 paper: https://arxiv.org/abs/2311.06242\n"
        "- Transformers Florence-2 documentation: https://huggingface.co/docs/transformers/model_doc/florence2"
    ),
}
