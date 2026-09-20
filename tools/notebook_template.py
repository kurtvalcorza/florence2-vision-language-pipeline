"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E handwritten-line adaptation workflow for the ``<OCR>`` task: the pinned
florence-community/Florence-2-large snapshot is digest-verified and loaded, 800 MIT-licensed Belfort handwritten line
images with their transcripts are fetched as eight digest-pinned parquet row groups over HTTPS range requests, the
records are validated and split by line, three capabilities (caption, detection, OCR) are run on a synthetic drawing
through the inference contract, the frozen model's character and word error rates over the held-out lines are measured
beside an empty-string and a constant-transcript baseline, a bounded fine-tuning of the last four decoder layers runs
on cached encoder outputs with validation-CER epoch selection, the held-out split is scored again, six held-out lines
and the three capabilities on the drawing are re-run with the adapted model, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "florence2_vision_language_pipeline",
    "repo_name": "florence2-vision-language-pipeline",
    "stem": "florence2_vision_language",
    "notebook_name": "florence2_vision_language_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "Florence2Pipeline",
    "weights_key": "florence-2-large-community",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "Florence-2-large — DIMER E2E handwritten-line OCR adaptation tutorial (standalone)",
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
    "capability": "prompt-selected vision-language tasks (caption, detection and OCR demonstrated) and bounded supervised fine-tuning of the `<OCR>` task's last decoder layers on transcribed text lines, using the pinned Florence-2-large weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `florence-community/Florence-2-large` snapshot (a 1.54 GB `model.safetensors`; no pickle is opened anywhere), "
        "fetches the first eight row groups of the Belfort-line test shard from the Hugging Face Hub at an immutable revision "
        "with HTTPS range requests (about 44 MB; each row group refused on any SHA-256 or byte-total mismatch), validates the "
        "800 line records and splits them by line into 600 / 60 / 140, runs three capabilities (`<CAPTION>`, `<OD>`, "
        "`<OCR>`) on a synthetic drawing through the inference contract with a combined input manifest and a rejection "
        "probe, measures the frozen model's `<OCR>` character and word error rates over the 140 held-out lines beside an "
        "empty-string and a constant-transcript baseline, runs a bounded fine-tuning of the last four BART decoder layers on "
        "cached encoder outputs with validation-CER epoch selection, scores the held-out lines again, re-runs six held-out "
        "lines and the three capabilities on the drawing with the adapted model, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify transcript parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On a Tesla T4 the default path took about 20 minutes of cell time, 22 with the pinned install and its restart (six epochs "
        "822 s, frozen scoring of 140 lines 90 s); a CUDA runtime is used automatically when present, "
        "and **a CPU runtime is not practical for the default path** (beam-search decoding of some 800 lines plus 600 cached "
        "forwards of a 777M-parameter model)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of line images plus a `transcripts.csv` (`file`, `text`, optional `id`; one row per image, at least eight images). The "
        "records pass through the same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact "
        "export and reload-parity cells as the Belfort sample. Uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "intro": (
        "Florence-2 is one sequence-to-sequence model whose behaviour is selected by a **task prompt token**: the image is "
        "resized to 768 × 768 by the processor (aspect ratio is not preserved), encoded by a DaViT vision tower into 577 "
        "visual tokens that a BART encoder reads together with the prompt, and a 12-layer BART decoder generates text that "
        "the processor parses per task — plain text for captions and OCR, boxes plus labels in input-pixel coordinates for "
        "region tasks (776,505,344 parameters in all, published under the **MIT** licence). Decoding is **deterministic "
        "beam search** (`num_beams` = 3 from the snapshot generation config, `do_sample=False`). The output carries **no "
        "score**: captions and transcripts are plain generated text, boxes have no confidence.\n\n"
        "What this notebook adds to inference is **adaptation of one task on transcribed lines**. Florence-2's `<OCR>` was "
        "trained on printed and scene text; nineteenth-century French council minutes in cursive handwriting — the "
        "Belfort-line dataset — are far outside that distribution, and on them the frozen model reads nothing: it emits a "
        "dash or an empty string for almost every line, a character error rate of **0.992** on the 140 held-out "
        "lines (the build record's Tesla T4 figure), the empty baseline's 1.0 in all but name. So the honest question is "
        "narrow: does a bounded fine-tuning of the last four decoder layers on 600 transcribed lines move the held-out "
        "**CER** and **WER** on a line-disjoint test split past two **non-adapted baselines** and the frozen model — and "
        "what does it do to the other tasks the same decoder serves? Nothing here is a claim about your documents or your "
        "script: it is one seeded split of one small labelled set.\n\n"
        "**Snapshot note:** the pin is the community \"official transformers converted checkpoint\" "
        "(`florence-community/Florence-2-large`), loadable by native `transformers` classes with `trust_remote_code=False`; "
        "the original `microsoft/Florence-2-large` snapshot needs remote code and was rejected (see the weight provenance "
        "document). The pinned revision ships `model.safetensors` (a 12-file manifest with the tokenizer files) — no pickle "
        "is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the processor or the "
        "model is constructed, and the loader refuses a checkpoint whose weights do not map cleanly onto the native "
        "architecture. The pipeline loads the checkpoint in **float32 on every device**: the adapter is trained in float32 "
        "and overlays without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled line set with its transcripts, validate it and split it by line "
        "without leakage; run three task tokens on a synthetic drawing through the public API and read each output contract "
        "correctly (generated text, no score, boxes without confidence, error rates against text you drew are not a "
        "benchmark); measure the frozen `<OCR>` corpus CER and WER beside two non-adapted baselines; run a bounded "
        "fine-tuning with the model's own sequence-to-sequence loss, explicit hyperparameters and validation-based epoch "
        "selection; evaluate on a line-disjoint test split; look at the adapted transcripts next to the frozen ones and the "
        "references, and at what the other two capabilities do after the shared decoder was tuned; and export a safetensors "
        "adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "the other task tokens in `TASKS` are exposed but not demonstrated — `<DETAILED_CAPTION>`, "
        "`<MORE_DETAILED_CAPTION>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR_WITH_REGION>` and "
        "`<CAPTION_TO_PHRASE_GROUNDING>` (the only task that takes a `text_input`); segmentation of any kind, open-vocabulary "
        "detection, visual question answering, confidence scores for boxes or text, fine-tuning of the vision tower, the "
        "projector, the encoder, the embeddings or the first eight decoder layers, fine-tuning of any task but `<OCR>`, "
        "evaluation on an OCR benchmark proper (only one seeded 800-line sample is scored here), and any claim that French "
        "cursive minutes stand in for your documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime with a CUDA GPU (Google Colab or Kaggle GPU, Python 3.12). The default path uses CUDA automatically when present. Generation is batched for the corpus stages — every `<OCR>` prompt is the same 587 tokens (577 image tokens plus the task token), so a batch needs no padding — and the build record measured 90 s to score 140 lines with 3-beam search and 822 s for the six epochs (caching the encoder outputs for 600 lines took 371 s) on a Tesla T4, about 20 minutes of cell time for the whole path; a CPU runtime would take hours. The pinned `torch==2.14.0` install and the 1.54 GB checkpoint are the large downloads of the run; the row groups are about 44 MB.",
        "- **Knowledge:** basic Python and PIL; what a sequence-to-sequence model's task prompt and beam search are; what a bounding box in pixel coordinates is; what character and word error rate measure and why they are not capped at 1; why a self-drawn image is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, text}` — `image` a PIL image (or a file decodable by Pillow) with sides within 1..16,384 px and at most 4096² pixels, `text` its transcript (1..512 characters after whitespace runs are collapsed; case and punctuation kept). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `transcripts.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded and every transcript checked for length, but nothing checks that a transcript says what its image shows — a mislabelled set is fine-tuned on without complaint.",
        "- **Expected output:** a \"slow image processor\" notice from `transformers` is expected and harmless. A `RuntimeError: checkpoint does not match the native Florence-2 architecture` means the staged weights are not the pinned converted checkpoint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads eight row groups of `default/test/0000.parquet` from `https://huggingface.co/datasets/Teklia/Belfort-line/resolve/<revision>/` at the immutable parquet-conversion revision `c4a74bbd…` with HTTPS range requests (the parquet footer plus about 44 MB of row-group bytes out of a 210 MB shard), each row group pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. Belfort-line is published under the MIT licence (Teklia; Tarride et al. 2023); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Belfort lines, the transcripts and the split\n\n"
                "`fetch_corpus` returns the eight pinned row groups from the cache under `weights/belfort/` or the Hub at the "
                "pinned parquet-conversion revision — `pyarrow` reads the shard's footer and exactly those row groups over "
                "HTTPS range requests; every cached file is re-hashed and every fetched row group refused on any SHA-256 or "
                "byte-total mismatch — and `read_corpus` turns each row into a record: the line image (128 px tall, 145 to "
                "8,956 px wide) and its crowdsourced transcript with whitespace runs collapsed. `build_sample_dataset` draws a "
                "seeded line-level split (600 / 60 / 140). `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training split's summary "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 800 lines and 33,117 reference characters, three digests, and four refusal probes — a duplicate id, "
                "an empty transcript, an image above the side ceiling, and a dataset too small to use — each rejected before "
                "the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/belfort')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'lines': sum(len(v) for v in corpus_groups.values()), 'bytes': sum(len(r['image']) + len(r['text'].encode('utf-8')) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'chars': manifest['text_chars'], 'words': manifest['text_words']['total'], 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "example['image'].save('outputs/{stem}_example_line.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'text': example['text']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty transcript': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'image above the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Run three capabilities on a synthetic drawing through the inference contract\n\n"
                "The inference contract is exercised as the multi-capability tutorial exercised it: a 512 × 512 white canvas "
                "drawn in code — a filled red square, a filled blue circle and the text `DIMER 2026` in Pillow's bundled font "
                "— whose drawn text is the **known OCR reference**; a different image family from the handwritten lines, and "
                "a drawing the adapted model will see again in Section 9. `validate_inputs` applies exactly the checks `run` "
                "applies (image sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` and at most `MAX_IMAGE_PIXELS`, a task token inside "
                "`TASKS`, a `text_input` only for the tasks that take one, `max_new_tokens` in 1..`MAX_NEW_TOKENS`, a positive "
                "`num_beams`) and one combined manifest records the three requests; an unsupported task token is validated too "
                "and its rejection recorded as a finding. `run` returns the task-parsed `result` — a caption string, `{{bboxes, "
                "labels}}` in input pixels with **no confidence scores**, an OCR string — the raw `generated_text` and the "
                "generation settings; structural checks assert each output's contract. The per-image `evaluation_report` over "
                "the three results scores OCR against the drawn text (`sample-sanity`) and reports the caption and the "
                "detections `not-measurable` — plumbing evidence, not a measurement; whether the model is *good at handwriting* "
                "is what Section 6 measures on 140 lines. The multi-capability card recorded the square labelled `flag` and an "
                "exact OCR of the drawn text."
            ),
            "code": (
                "drawing = Image.new('RGB', (512, 512), (255, 255, 255))\n"
                "draw = ImageDraw.Draw(drawing)\n"
                "drawn_square = (64, 64, 224, 224)\n"
                "draw.rectangle(drawn_square, fill=(220, 30, 30))\n"
                "draw.ellipse((300, 96, 460, 256), fill=(30, 60, 220))\n"
                "drawing_reference = 'DIMER 2026'\n"
                "draw.text((96, 360), drawing_reference, fill=(0, 0, 0), font=ImageFont.load_default(size=48))\n"
                "drawing_name = 'synthetic_shapes_text_512'\n"
                "drawing_sha256 = hashlib.sha256(np.asarray(drawing).tobytes()).hexdigest()\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_IMAGE_PIXELS': MAX_IMAGE_PIXELS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_LINE_MAX_NEW_TOKENS': DEFAULT_LINE_MAX_NEW_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'NUM_BEAMS': NUM_BEAMS, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device, 'dtype': pipe.dtype}}}})\n"
                "print({{'TASKS_WITHOUT_TEXT': TASKS_WITHOUT_TEXT, 'TASKS_WITH_TEXT': TASKS_WITH_TEXT}})\n"
                "CAPABILITIES = {{\n"
                "    '<CAPTION>': {{'input': 'image only', 'output': 'result: str (one short caption); no score', 'max_new_tokens': 64}},\n"
                "    '<OD>': {{'input': 'image only', 'output': \"result: {{'bboxes': [[x1, y1, x2, y2], ...] in input pixels, 'labels': [str, ...]}}; no per-box score\", 'max_new_tokens': DEFAULT_MAX_NEW_TOKENS}},\n"
                "    '<OCR>': {{'input': 'image only', 'output': 'result: str (transcribed text, reading order chosen by the model); no score', 'max_new_tokens': DEFAULT_LINE_MAX_NEW_TOKENS}},\n"
                "}}\n"
                "manifests = {{task: validate_inputs(drawing, task, max_new_tokens=contract['max_new_tokens'], num_beams=NUM_BEAMS, names=[drawing_name]) for task, contract in CAPABILITIES.items()}}\n"
                "input_manifest = {{**manifests['<CAPTION>'], 'task': 'multi-capability: ' + ', '.join(CAPABILITIES), 'tasks': list(CAPABILITIES), 'findings': [], 'capabilities': manifests}}\n"
                "try:\n"
                "    validate_inputs(drawing, '<REFERRING_EXPRESSION_SEGMENTATION>')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-task-probe', 'task': '<REFERRING_EXPRESSION_SEGMENTATION>', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'drawing': drawing_name, 'sha256': drawing_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def run_capabilities(pipeline, label):\n"
                "    results, timings = {{}}, {{}}\n"
                "    for task, contract in CAPABILITIES.items():\n"
                "        started = time.perf_counter()\n"
                "        results[task] = pipeline.run(drawing, task, max_new_tokens=contract['max_new_tokens'], num_beams=NUM_BEAMS)\n"
                "        timings[task] = round(time.perf_counter() - started, 3)\n"
                "    caption, detections, ocr_text = results['<CAPTION>']['result'], results['<OD>']['result'], results['<OCR>']['result']\n"
                "    checks = {{\n"
                "        'caption_is_text': isinstance(caption, str),\n"
                "        'od_boxes_and_labels_aligned': isinstance(detections, dict) and len(detections.get('bboxes', [])) == len(detections.get('labels', [])),\n"
                "        'od_boxes_inside_image': all(0 <= x1 <= x2 <= drawing.width and 0 <= y1 <= y2 <= drawing.height for x1, y1, x2, y2 in detections.get('bboxes', [])),\n"
                "        'ocr_is_text': isinstance(ocr_text, str),\n"
                "        'deterministic_settings': all(r['generation']['do_sample'] is False and r['generation']['num_beams'] == NUM_BEAMS for r in results.values()),\n"
                "        'identity_reported': all(r['model_id'] == MODEL_ID and r['model_revision'] == MODEL_REVISION for r in results.values()),\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'capability output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(list(results.values()), drawing_reference, sample_kind='synthetic (drawn in this notebook)')\n"
                "    preview = drawing.copy()\n"
                "    marker = ImageDraw.Draw(preview)\n"
                "    for (x1, y1, x2, y2), name in zip(detections.get('bboxes', []), detections.get('labels', [])):\n"
                "        marker.rectangle((x1, y1, x2, y2), outline=(0, 160, 0), width=3)\n"
                "        marker.text((x1 + 4, y1 + 4), name, fill=(0, 160, 0))\n"
                "    preview.save(f'outputs/{stem}_preview_{{label}}.png')\n"
                "    print({{label: {{'seconds': timings, 'checks': checks, 'caption': caption, 'detections': list(zip(detections.get('labels', []), [[round(v) for v in b] for b in detections.get('bboxes', [])])), 'ocr': ocr_text, 'ocr_cer': {{m['id']: round(m['value'], 4) for m in report['metrics']}}, 'verdict': report['verdict']}}}})\n"
                "    return results, timings, checks, report\n\n\n"
                "frozen_results, frozen_timings, frozen_checks, frozen_report = run_capabilities(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test lines\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `ocr_metrics` (carried in `metrics.py`): the "
                "**character error rate** and **word error rate** as micro averages — total Levenshtein edits over total "
                "reference characters or words, the corpus CER/WER of the handwriting-recognition literature — beside the "
                "macro (per-line mean) rates and the exact-match rate. Neither rate is capped: a hypothesis longer than its "
                "reference pushes the rate **above 1.0**, the signal that the model is generating text the line does not carry. "
                "The **empty-string** baseline predicts nothing and scores CER 1.0 exactly (every reference character is a "
                "deletion) — the floor any recogniser must beat to do better than silence. The **constant-transcript** baseline "
                "predicts one training transcript — the medoid, the line closest on average to the others — for every test "
                "line: what corpus statistics buy without reading the image. The **frozen model** is scored by `pipe.evaluate`, "
                "which runs `<OCR>` in batches of `EVAL_BATCH_SIZE` with `NUM_BEAMS` beams under a `LINE_MAX_NEW_TOKENS` "
                "budget and returns the hypotheses with the rates. Expect the frozen model **at the empty baseline**: the build "
                "record measured 0.992 — it emits a dash or nothing for cursive it cannot read (hypotheses "
                "0.04 times the reference length); read four of them under the references."
            ),
            "code": (
                "METRICS = ('cer', 'wer', 'cer_macro', 'exact_match')\n"
                "LINE_MAX_NEW_TOKENS = 128  # @param {{type:\"integer\"}}\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_constant = constant_baseline(train_records, test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'constant_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'note': baseline_constant['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'ref_chars': frozen_test['ref_chars'], 'hyp_chars': frozen_test['hyp_chars'], 'truncated': frozen_test['truncated'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record, hypothesis in zip(test_records[:4], frozen_test['hypotheses'][:4], strict=True):\n"
                "    print({{'id': record['id'], 'reference': record['text'], 'frozen': hypothesis[:120]}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder layers\n\n"
                "`pipe.adapt` trains only the last four of the 12 BART decoder layers and the decoder's embedding layer norm — "
                "67,188,736 of 776,505,344 parameters — while the DaViT vision tower, the projector, the BART encoder, the "
                "shared embeddings (tied to the output head) and the first eight decoder layers stay frozen. Each training line "
                "is the `<OCR>` prompt with its 577 visual tokens on the encoder side and, on the decoder side, the transcript's "
                "tokens between the start and end tokens; the loss is the **sequence-to-sequence cross-entropy** over those "
                "target tokens, teacher-forced — the checkpoint's own training objective. Because everything before the decoder "
                "is frozen, the encoder output for every training line is computed once under no gradient and cached (the "
                "**frozen encoder cache**, 587 × 1024 numbers per line), and each step runs only the decoder on those cached "
                "states — the loss equals the full model's loss exactly, at a fraction of the cost. AdamW without weight decay "
                "at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 "
                "records the frozen model's validation rates; every epoch is scored on the 60 validation lines with the same "
                "beam search, and the epoch with the **lowest validation CER** is kept.\n\n"
                "Watch the validation CER fall from 0.969 to 0.799 (epoch 6 in the build "
                "record) while the loss drops from about 6.63 to 3.13 — and then watch the validation curve flatten "
                "near 0.80 while the training loss keeps falling. Four decoder layers move the model from silence to something "
                "that is not yet reading (the adapted model gets 20 % of the characters right and 0 of 140 lines exact): the frozen encoder's representation of a 128-px cursive line "
                "squashed into 768 × 768 is the bottleneck, not the optimiser — the build record's faster rates (1e-4 and 2e-4 "
                "for eight epochs) reached the same plateau (test CER 0.798 and 0.807). The sibling GOT-OCR 2.0 row reached 0.759 on "
                "the same split with the same design; the two models are compared in the card, not here."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'task': adapt_result['task'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'first_trainable_layer': adapt_result['first_trainable_layer'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test lines were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **CER** first (the measure the epoch was selected on — the build record measured "
                "0.992 → **0.797**, past both baselines — and one fifth of the characters right is what that number means), "
                "then **WER** (1.002 → 1.012: it did not move, so the character gain sits inside words and no whole word is "
                "read), then the hypothesis length (from 0.04 times the reference length to 0.58: the adapted model writes "
                "something for every line but still not enough of it), then the exact-match rate (0 of the 140 lines read "
                "perfectly). The cell asserts the adapted CER is below the frozen one and below the empty baseline's 1.0 — the "
                "contract holds; the reading does not. One hundred and forty lines from one seeded split give **no dispersion "
                "estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a "
                "result on one French council's minutes says nothing about other hands, other languages or other scripts until "
                "you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'constant': round(baseline_constant[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['hypothesis_length'] = {{'ref_chars': adapted_test['ref_chars'], 'frozen_hyp_chars': frozen_test['hyp_chars'], 'adapted_hyp_chars': adapted_test['hyp_chars'], 'frozen_truncated': frozen_test['truncated'], 'adapted_truncated': adapted_test['truncated']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'task': '<OCR>',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'constant': {{k: v for k, v in baseline_constant.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_line': [{{**frozen_row, 'frozen_hypothesis': frozen_hyp, 'adapted_cer': adapted_row['cer'], 'adapted_hypothesis': adapted_hyp}} for frozen_row, frozen_hyp, adapted_row, adapted_hyp in zip(frozen_test['rows'], frozen_test['hypotheses'], adapted_test['rows'], adapted_test['hypotheses'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['cer'] < frozen_test['cer']\n"
                "assert adapted_test['cer'] < baseline_empty['cer']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['cer'] < min(baseline_empty['cer'], baseline_constant['cer'])}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the lines, re-run the three capabilities, export the adapter and reload it\n\n"
                "Six held-out lines are written as panels (`outputs/{stem}_examples/`: the line image with the reference, the "
                "frozen transcript and the adapted transcript beneath it) so the numbers can be checked by eye: the adapted "
                "rows should read the cursive the frozen rows left blank. The drawing from Section 5 is then run again through "
                "all three capabilities by the adapted model — the decoder that was tuned serves every task token, so this is "
                "a small look at what the adaptation did *outside* its task and its corpus: the build record measured "
                "before adaptation caption `a red and blue circle with the word dimer 2026 on it`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000); after adaptation caption `Dim 2026 logo with a red and blue circle.`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000) — one drawing of evidence, not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the four decoder layers and the embedding norm, about "
                "269 MB in float32 — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base "
                "model id and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, "
                "the task, the training configuration and the epoch history (OUT8). `Florence2Pipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor outside the last four decoder layers and the embedding norm, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical transcripts on eight test lines (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "for record, frozen_hyp, adapted_hyp in zip(test_records[:6], frozen_test['hypotheses'][:6], adapted_test['hypotheses'][:6], strict=True):\n"
                "    line = record['image']\n"
                "    width = min(1400, line.width)\n"
                "    line = line.resize((width, max(1, round(line.height * width / record['image'].width))))\n"
                "    sheet = Image.new('RGB', (max(width, 1400), line.height + 96), (255, 255, 255))\n"
                "    sheet.paste(line, (0, 0))\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    for i, (tag, text) in enumerate((('REF', record['text']), ('FROZEN', frozen_hyp), ('ADAPTED', adapted_hyp))):\n"
                "        marker.text((8, line.height + 6 + i * 28), f'{{tag}}: {{text[:140]}}', fill=(20, 20, 20) if tag != 'FROZEN' else (150, 40, 40), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'rows': ['reference', 'frozen transcript', 'adapted transcript']}})\n\n"
                "adapted_results, adapted_timings, adapted_checks, adapted_report = run_capabilities(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'task': artifact_manifest['adapter']['task'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = Florence2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['text'] for item in pipe.transcribe([r['image'] for r in test_records[:8]], max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)]\n"
                "after = [item['text'] for item in reloaded.transcribe([r['image'] for r in test_records[:8]], max_new_tokens=LINE_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)]\n"
                "parity = {{'identical_lines': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_lines'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'language': CORPUS_LANGUAGE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'drawing': {{'name': drawing_name, 'sha256': drawing_sha256, 'ocr_reference': drawing_reference, 'drawn_square': drawn_square}}, 'tasks_exposed': list(TASKS), 'frozen': {{'capabilities': {{task: {{'result': r['result'], 'generated_text': r['generated_text'], 'generation': r['generation'], 'seconds': frozen_timings[task]}} for task, r in frozen_results.items()}}, 'checks': frozen_checks, 'report': frozen_report}}, 'adapted': {{'capabilities': {{task: {{'result': r['result'], 'generated_text': r['generated_text'], 'generation': r['generation'], 'seconds': adapted_timings[task]}} for task, r in adapted_results.items()}}, 'checks': adapted_checks, 'report': adapted_report}}, 'output_files': ['outputs/{stem}_preview_frozen.png', 'outputs/{stem}_preview_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'source': pipe.source, 'dtype': pipe.dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A prompt-driven document model reads the print it was trained on, and nineteenth-century cursive French is not among "
        "it: the frozen `<OCR>` scores a character error rate of 0.992 on the Belfort lines — silence, a dash or "
        "nothing for almost every line. A bounded fine-tuning of the last four decoder layers on 600 transcribed lines moves it "
        "off the floor but not far (0.797 CER and 1.012 WER in the build record, 0 of "
        "the held-out lines exact; the constant-transcript baseline scores 0.944) and the validation curve plateaus there at "
        "every rate tried, with a 269 MB adapter that reloads line-for-line. That is the claim: the adaptation contract "
        "works end to end on one task of a multi-task model with a real labelled set, and the numbers it produces are read as "
        "micro and macro rates, against two non-adapted baselines and the frozen model, with the hypothesis length beside "
        "them rather than in isolation — and they say that a decoder-only adapter cannot make this encoder read cursive. "
        "The obvious next experiment, adapting the BART encoder's last layers too (or the vision tower), is not in this "
        "repository.\n\n"
        "The test split is 140 lines from one seeded draw of one 800-line sample, the validation split that picks the epoch is "
        "60, and both rates are corpus edit distances over one crowdsourced transcription — not a benchmark, not a measure of "
        "reading order or layout. So a gain here says the contract works on one council's minutes, not that the adapted model "
        "handles other hands, other languages, other scripts or your scans. The decoder that was tuned serves every task "
        "token: the drawing re-run in Section 9 is one image of evidence about what the tuning did to `<CAPTION>` and `<OD>` "
        "(before adaptation caption `a red and blue circle with the word dimer 2026 on it`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000); after adaptation caption `Dim 2026 logo with a red and blue circle.`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000)), not a measurement, and a deployment that needs the other tasks must measure them after "
        "adapting. The decoder was adapted, not the vision tower: what the encoder cannot resolve in a 128-px line squashed "
        "into a 768 × 768 square stays unread.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and constant-transcript rates on *your* transcripts, "
        "and the frozen model's hypothesis length, are the numbers to read before any adapted one. **Rates above 1.0:** an "
        "uncapped CER tells you the model is generating, not reading; a capped one would hide it. **Leakage:** keep every image "
        "in one split (the contract de-duplicates by decoded pixels) and split by page, writer or volume when your lines come "
        "from few sources — lines cut from the same page share a hand.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled line set, validate the "
        "demonstrated dataset contract without leakage, execute the inference contract for three task tokens and a bounded "
        "fine-tuning of one of them with the model's own objective, evaluate against two non-adapted baselines and the frozen "
        "model on a line-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, recognition quality on any other hand, language or "
        "document family, caption or detection quality after adaptation, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `LEARNING_RATE` to `1e-4` or `2e-4` with `EPOCHS = 8` "
        "and read the same plateau the build record found (0.798 and 0.807); raise `EPOCHS` and watch the validation CER pick the epoch; set `NUM_BEAMS` to `1` in "
        "Sections 6–9 and read what greedy decoding costs; lower `LINE_MAX_NEW_TOKENS` to `64` and read how the truncation "
        "count changes; add `<CAPTION_TO_PHRASE_GROUNDING>` to `CAPABILITIES` with the frozen caption as its `text_input`; or "
        "bring your own transcribed lines through BYOD and read the two baselines before the adapted number.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/florence-2-large-community/` and rerun Section 3. `RuntimeError: "
        "checkpoint does not match the native Florence-2 architecture` in Section 3: the staged weights are not the pinned "
        "converted checkpoint — re-stage. A \"slow image processor\" notice from `transformers` is expected and harmless.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and pin history: https://github.com/kurtvalcorza/florence2-vision-language-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Pinned converted checkpoint: https://huggingface.co/{MODEL_ID}\n"
        "- Original weights and licence: https://huggingface.co/microsoft/Florence-2-large\n"
        "- Florence-2 paper (Xiao et al., 2023): https://arxiv.org/abs/2311.06242\n"
        "- Transformers Florence-2 documentation: https://huggingface.co/docs/transformers/model_doc/florence2\n"
        "- Belfort-line dataset (Teklia, MIT): https://huggingface.co/datasets/Teklia/Belfort-line — Tarride et al., Handwritten Text Recognition from Crowdsourced Annotations (HIP 2023): https://doi.org/10.1145/3604951.3605517\n"
        "- Sibling row on the same split: https://github.com/kurtvalcorza/got-ocr2-pipeline\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
