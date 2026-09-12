---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: florence-community/Florence-2-large
---

# Florence-2-large (DIMER package v0.1.0) — Prompt-driven Vision-Language Model (Caption, OCR, Detection, Grounding)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-florence--community%2FFlorence--2--large-ffcc4d?style=flat)](https://huggingface.co/florence-community/Florence-2-large)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg)](https://arxiv.org/abs/2311.06242)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://huggingface.co/microsoft/Florence-2-large/resolve/main/LICENSE)
[![Pipeline](https://img.shields.io/badge/Pipeline-florence2--vision--language--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/florence2-vision-language-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`florence-community/Florence-2-large` is the "official transformers converted checkpoint" (pinned README note) of Microsoft's 0.77 B-parameter Florence-2-large (Xiao et al., arXiv:2311.06242), pinned here to revision `4271c66b88cdbc05735372ec13b2360108de5317`; the README carries over Microsoft's statement that this is a continued-pretrained variant with a 4k context length, trained on a further 0.1 B samples, with the OCR task updated to emit line separators. Florence-2 is a sequence-to-sequence vision-language model: a DaViT vision encoder produces 577 visual tokens per 768×768 image (`preprocessor_config.json` `image_seq_length`) that are projected into a BART-style encoder-decoder language model; at inference the decoder generates text conditioned on the image tokens and a task prompt such as `<CAPTION>`, `<OD>` or `<OCR_WITH_REGION>`, and region outputs are emitted as quantised `<loc_N>` tokens that the processor converts back to pixel boxes. Adaptation is by prompt only — no training or in-context examples happen in this repository. The converted checkpoint loads through the native `transformers` `Florence2ForConditionalGeneration` and `Florence2Processor` classes with no custom code (`config.json` `model_type: florence2`, `transformers_version: 4.56.1`); 776505344 parameters were instantiated with zero missing, unexpected or mismatched keys on the smoke run. What this repository adds is packaging: `Florence2Pipeline` in `src/florence2_vision_language_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`, `stage_missing_files`), a loading-info check that refuses a checkpoint the native classes do not fully consume, input validation, a fixed output contract, a `character_error_rate` helper and a CPU smoke run.

#### Intended Use and Limitations

###### Primary Intended Uses

The task family is image-plus-prompt to text: input one PIL image and one of the nine task prompts in `TASKS` (optionally a caption string for `<CAPTION_TO_PHRASE_GROUNDING>`); output a caption string, an OCR transcript, or a dict of boxes and labels (`<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR_WITH_REGION>`, grounding), as the pinned README documents per task. Envisioned applications are photograph and document captioning for indexing, OCR of scanned pages and signage with line layout, zero-shot object localisation on everyday scenes, and grounding of a caption's noun phrases to image regions — one model replacing several single-task services in a research or internal-tooling setting. In a larger system the pipeline is an inference component whose outputs feed a human-reviewed index or a downstream detector-evaluation harness, not a decision engine.

###### Primary Intended Users

The intended users are machine-learning engineers, computer-vision researchers and application developers integrating a multi-task vision-language model into research prototypes, internal enterprise document or image tooling, or the DIMER model workbench. The pipeline assumes its users understand that generated text is not a verified fact about the image (captions and OCR can be fluent and wrong), that box coordinates come from quantised location tokens with roughly 1-px precision at 256 px and coarser on larger images (the smoke run returned 63–193 for a square drawn at 64–192), that detection labels are open-vocabulary guesses (the same run labelled a plain red square "flag"), and that the model's behaviour is defined by the task-prompt vocabulary rather than free-form questions. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not visual question answering with free-form questions (the pinned snapshot is the pretrained model, not the `-ft` variant; `smolvlm-vision-language-pipeline` covers chat-style VQA); not segmentation masks (`<REFERRING_EXPRESSION_SEGMENTATION>` and region-to-segmentation prompts are not in `TASKS`); not video, multi-image, or text-only input; not identity or face recognition. Anything outside the nine `TASKS` prompts raises `ValueError`.
2. **Input boundary:** only a single `PIL.Image.Image` is accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; the processor resizes every image to 768×768 (`preprocessor_config.json`), so aspect ratio is not preserved and small text in large pages is lost; grounding captions above `MAX_TEXT_CHARS = 1000` characters are rejected; `max_new_tokens` is capped at `MAX_NEW_TOKENS = 1024` (default 256); a `text_input` supplied to a task that takes none is rejected.
3. **Decision boundary:** not for autonomous or high-impact decisions — evidentiary OCR, safety interlocks on detections, moderation takedowns, medical or forensic reading of images — without a human reviewing each output and a locally measured error rate.

#### Factors

###### Groups

The pipeline is not human-centric by design — its prompts name objects, regions and text, not people — but the model can and will caption, detect and ground people in an image, and the upstream training corpus contains people. FLD-5B was built by automatic annotation of 126 M web-sourced images (pinned README and the paper), and neither the paper nor the README reports any group-level breakdown of caption quality, detection recall or OCR accuracy by age, gender, skin type or region; the corpus is not group-audited, and the community conversion changed the weights' storage layout only, not their content. The obligation transfers to the operator: before deployment, measure task accuracy on a labelled sample of their own images stratified by the groups relevant to their application (for example caption quality across skin types or OCR accuracy across scripts) and treat a material gap as a blocker. This repository measures nothing of the kind.

###### Instrumentation

FLD-5B's images were collected from web sources (the paper names ImageNet-22k, Object 365, Open Images, Conceptual Captions and LAION among its image sources) and are consumer and stock photographs of undocumented camera, lens, compression and colour-management provenance; the 5.4 B annotations were generated by specialist models and iteratively refined, so label noise is itself an instrument characteristic. The pipeline consumes decoded pixel arrays and resizes them to 768×768 with bicubic resampling (`resample: 3`) and ImageNet mean/std normalisation (`preprocessor_config.json`), so resolution, JPEG artefacts, rotation and scan skew all reach the model as changed pixel statistics. The pipeline does not detect blur, skew, low contrast or a changed capture device; it only rejects non-image types and sides outside 1–4096 px. OCR on scans should be validated on the scanner and DPI actually in use.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional: `from_pretrained` picks `cuda:0` when available with float16 (the dtype the checkpoint ships in, `config.json` `dtype`), else CPU with float32; the DIMER build environment is CPU-only (`CUDA_VISIBLE_DEVICES=-1`). On this repository's smoke run (Windows venv, CPU, float32, `HF_HUB_OFFLINE=1`, one 256×256 synthetic white image with a centred red square) loading the verified snapshot took 7.11 s, `<CAPTION>` with 3 beams took 4.91 s and `<OD>` 9.02 s (21.03 s total); a repeat `<CAPTION>` call returned identical text. The CUDA/float16 path is not executed in this repository. Data environment: inputs are assumed to be natural photographs or document images resembling the web-sourced FLD-5B distribution, upright, with the subject or text legible at 768 px; line drawings, medical or satellite imagery, rotated scans and dense small print fall outside that assumption and degrade in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `character_error_rate(reference, hypothesis)` in `pipeline.py`: the character-level Levenshtein distance divided by the reference length, for use against `<OCR>` output when a ground-truth transcript exists. It captures reconstruction error of transcribed text, which is the operational question for OCR, and is preferred over word error rate because Florence-2's OCR output has no guaranteed word tokenisation across scripts. It says nothing about captioning or detection: caption quality needs CIDEr/BLEU against multiple references and detection needs COCO-style mAP, both of which require external tooling and labelled sets, so the pipeline reports neither. Upstream reports for Florence-2-large, zero-shot, COCO caption CIDEr 135.6, NoCaps CIDEr 120.8, TextCaps CIDEr 72.8 and COCO val2017 detection mAP 37.5 (pinned README table; the README summary states COCO OD AP 39.8 for the continued-pretrained weights); this pipeline has not reproduced any of them.

###### Decision thresholds

The default decision rule is deterministic beam search: `num_beams = NUM_BEAMS = 3` (snapshot `generation_config.json`) with `do_sample=False`, echoed in every result's `generation` field — the returned text is the highest-scoring beam, an implicit "best sequence wins" rule with no minimum score. For detection and grounding tasks the model emits boxes without confidence scores; the processor's post-processing returns every parsed `<loc_N>` quadruple, so no score threshold exists to tune and every box the decoder emits is returned (the smoke run returned one box for one square). No acceptance threshold was set during development and none is shipped. A deployment that needs to suppress low-quality outputs must add its own filter — an OCR post-check against a lexicon, a box-size floor, or a label whitelist — trading the cost of a wrong accepted output (false positive) against the cost of a suppressed correct one (false negative) for its application.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; the upstream figures cited above are single-evaluation numbers from the pinned README with no reported interval. Decoding is deterministic given the same weights, device, dtype and library versions: beam search with `do_sample=False` and a fixed `max_new_tokens`, no seed needed — confirmed on the smoke run, where two consecutive `<CAPTION>` calls on the same image returned identical text; float16 on GPU versus float32 on CPU can change near-tied beams and hence the text. The `generated_text` field is raw decoder output with no probability attached; box coordinates are dequantised location tokens (1000 bins per axis, so about 0.26 px at 256 px and 4 px at 4096 px), not calibrated estimates, and no confidence is emitted. A caller who needs calibrated confidence must expose the library's `output_scores` path and calibrate it on labelled data; nothing in this repository does so.

#### Ethical considerations and biases

###### Data

Upstream states the model was pretrained on FLD-5B, "5.4 billion annotations across 126 million images" (pinned README; the paper describes the images as drawn from public web datasets and the annotations as machine-generated), and that these weights were continued-pretrained on a further 0.1 B samples; the disclosure stops there — no per-image licensing, consent status or demographic composition is given, and web-sourced image corpora are known to contain photographs of identifiable people and text with personal information, so the presence of personal data is not ruled out. The community repository converted Microsoft's checkpoint to the native `transformers` layout and states no additional training. This repository distributes code, tests and documentation; the 1.55 GB `model.safetensors` snapshot is git-ignored and staged locally under `weights/florence-2-large-community/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content — OCR in particular will transcribe whatever text is present — and the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite (15 tests) and one CPU smoke run (a caption and a detection on a synthetic image) in this repository. Where a sensitive use is foreseeable — reading medication labels, transcribing identity documents, locating people in security footage — it is admissible only with a human reviewer on every consequential output, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/florence2_vision_language_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/florence-2-large-community/dimer-base-manifest.json` (12 entries) and raises on the first size or SHA-256 mismatch before any weight is loaded; `stage_missing_files` fetches only manifest-listed files at the pinned revision and refuses a manifest naming another model; the local path is loaded with `local_files_only=True` and the Hub path only with `allow_download=True` and `revision=MODEL_REVISION`; `trust_remote_code=False` on both loaders, and the explicit native classes `Florence2ForConditionalGeneration` / `Florence2Processor` are used so the `auto_map` line left in the snapshot's `preprocessor_config.json` is never consulted; the original `microsoft/Florence-2-large` pin was rejected because it can only be loaded by executing code bundled in the model repository (see `docs/WEIGHTS.md`). (2) Architecture match — `from_pretrained` requests `output_loading_info=True` and raises `RuntimeError` if any key is missing, unexpected or mismatched, so a checkpoint the native classes would silently re-initialise cannot load (all four counts were 0 on the smoke run). (3) Input integrity — `_validate` rejects non-PIL input, sides outside 1–4096 px, unknown task prompts, missing or over-long grounding text, text on tasks that take none, `max_new_tokens` outside 1–1024 and non-positive `num_beams` before the model runs; malformed runner output raises `RuntimeError`. (4) Reproducibility — exact `==` pins, `model.eval()`, deterministic decoding settings echoed in the result, `model_id`/`model_revision` in every result. (5) Refusals — no training, fine-tuning or score-output API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation is applied because the pipeline does not train.

###### Risks and harms

Fluent hallucination: captions and OCR transcripts read as confident prose even when wrong — objects that are not there, digits transposed in a serial number — and the operator or a downstream reader bears the harm when the text is trusted; likelihood is moderate on in-distribution photographs and high on dense or degraded documents. Wrong open-vocabulary labels: the smoke run labelled a plain red square "flag" with a correctly placed box, so a box being right does not make its label right. Detection without confidence: every emitted box is returned with no score, so spurious boxes cannot be filtered without an operator-built rule; harm falls on whoever acts on a phantom detection. Bias from a web-scraped corpus: captions may describe people with stereotyped or demeaning terms and detection recall may vary by demographic group; data subjects and third parties bear that harm. Automation bias: reviewers presented with a fluent caption check the image less carefully. Privacy: OCR transcribes any personal data visible in an image. Latency: beam search on CPU took 5–9 s per call on a 256-px image, so a deployment that assumes real-time behaviour will queue.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — including locating or tracking individuals via `<OD>`, `<DENSE_REGION_CAPTION>` or phrase grounding, or bulk-transcribing identity documents and licence plates via `<OCR>`. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating captions or transcripts presented as evidence of what an image contains. Any use that violates the MIT terms of the upstream weights (the copyright and permission notice must be preserved; the community README points to Microsoft's licence file) or the DIMER deployment terms is prohibited. The developers identify these because the model's outputs — free text and localisation — are precisely the primitives such misuses need; no further prohibited use is identified beyond them.

## Immutable provenance

- Model: `florence-community/Florence-2-large`
- Revision: `4271c66b88cdbc05735372ec13b2360108de5317`
- Snapshot manifest: `weights/florence-2-large-community/dimer-base-manifest.json`, 12 files, `totalBytes` 1558929750
- `model.safetensors` SHA-256: `7715423d6549bf1e71188bdd84f4ac960cc0597886af24a5ef7b66f128660685` (1553541016 bytes)
- `config.json` SHA-256: `8412483f687f2f71587328a38c6fa70a68d9488f28e90607be3c182641f60f2c` (2396 bytes)
- Weight format: SafeTensors (float16 as shipped); loader `Florence2ForConditionalGeneration.from_pretrained(<snapshot dir>, local_files_only=True, trust_remote_code=False, output_loading_info=True)` and `Florence2Processor.from_pretrained(...)`, both native to `transformers==4.57.6`; no custom code in the snapshot.
- Pin history: the first pin, `microsoft/Florence-2-large`, was rejected on 2026-09-12 (custom-code requirement); its revision is recorded once in `docs/WEIGHTS.md`.

## Input/output contract

- `Florence2Pipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `run(image, task="<CAPTION>", text_input=None, *, max_new_tokens=256, num_beams=3)` — `image`: one `PIL.Image.Image`, sides 1–4096 px, any mode (converted to RGB); `task` in `TASKS`; `text_input` required for `<CAPTION_TO_PHRASE_GROUNDING>` (≤ 1000 chars) and forbidden otherwise. Returns `{"task", "text_input", "result", "generated_text", "image_size", "generation": {"max_new_tokens", "num_beams", "do_sample"}, "device", "source", "model_id", "model_revision"}`; `result` is a string for caption/OCR tasks and `{"bboxes": [[x1, y1, x2, y2], ...], "labels": [...]}` (or `quad_boxes` for `<OCR_WITH_REGION>`) for region tasks, in pixel coordinates of the input image.
- `character_error_rate(reference, hypothesis)` — float in [0, ∞); 0.0 for an exact match.
- `verify_snapshot(path=None)`, `stage_missing_files(path=None, *, allow_download=False, downloader=None)`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12. The build venv carries `torch 2.14.0+cu130`.
- Precision: float32 on CPU (the measured path); float16 on CUDA (not executed here). Preprocessing per `preprocessor_config.json`: resize to 768×768, bicubic, ImageNet mean/std; 577 visual tokens.
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, 2026-09-12): device `cpu`, source `local-snapshot`; load 7.11 s with loading info `missing_keys 0 / unexpected_keys 0 / mismatched_keys 0 / error_msgs 0` (776505344 parameters); `<CAPTION>` (`max_new_tokens=64`, 3 beams) 4.91 s → `"a red square with a white background"`; `<OD>` 9.02 s → `{"bboxes": [[63, 63, 193, 193]], "labels": ["flag"]}` for a square drawn at 64–192 px; total 21.03 s; repeat `<CAPTION>` identical; exit 0. The only stderr line was the library's slow-image-processor notice.
- Tests: `pytest -q -o addopts= tests` — 15 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Xiao, Wu, Xu, Dai, Hu, Lu, Zeng, Liu, Yuan. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. arXiv:2311.06242 (2023). https://arxiv.org/abs/2311.06242
- Pinned upstream card (converted checkpoint): https://huggingface.co/florence-community/Florence-2-large; original weights and licence: https://huggingface.co/microsoft/Florence-2-large (no separate upstream source repository is published)
- Transformers 4.57.6 native implementation: `transformers/models/florence2/`
