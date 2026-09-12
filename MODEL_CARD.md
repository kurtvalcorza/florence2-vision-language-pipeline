---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: microsoft/Florence-2-large
---

# Florence-2-large (DIMER package v0.1.0) — Prompt-driven Vision-Language Model (Caption, OCR, Detection, Grounding)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2FFlorence--2--large-ffcc4d?style=flat)](https://huggingface.co/microsoft/Florence-2-large)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg)](https://arxiv.org/abs/2311.06242)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://huggingface.co/microsoft/Florence-2-large/resolve/main/LICENSE)
[![Pipeline](https://img.shields.io/badge/Pipeline-florence2--vision--language--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/florence2-vision-language-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent), and none can be written yet: the pinned snapshot cannot be loaded without executing its bundled custom code, which this package refuses (see Mitigations). The package is exercised through its offline test suite (`tests/`) and the refusal smoke in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up that depends on the owner's pin decision recorded in `STATUS.md`, not a claim this card makes.

---

###### Description

`microsoft/Florence-2-large` is the 0.77 B-parameter Florence-2 model from Microsoft (Xiao et al., arXiv:2311.06242), pinned here to revision `21a599d414c4d928c9032694c424fb94458e3594`; the pinned README states this revision is a continued-pretrained variant with a 4k context length, trained on a further 0.1 B samples, with the OCR task updated to emit line separators. Florence-2 is a sequence-to-sequence vision-language model: a DaViT vision encoder (snapshot `config.json` `vision_config`: four stages, embedding dims 256/512/1024/2048, depths 1/1/9/1, 768×768 input) produces 577 visual tokens that are projected into a 1024-d BART-style encoder-decoder language model (12 encoder + 12 decoder layers, vocabulary 51289). At inference the decoder generates text conditioned on the image tokens and a task prompt such as `<CAPTION>`, `<OD>` or `<OCR_WITH_REGION>`; region outputs are emitted as quantised location tokens that the upstream processor converts back to pixel boxes. Adaptation is by prompt only — no training or in-context examples happen in this repository. The boundary with upstream is sharp here: this repository adds packaging only — `Florence2Pipeline` in `src/florence2_vision_language_pipeline/pipeline.py`, manifest verification (`verify_snapshot`, `stage_missing_files`), custom-code detection (`remote_code_files`), input validation, a fixed output contract and a `character_error_rate` helper — and it does **not** yet execute the model, because the pinned snapshot's modeling code is refused (Mitigations).

#### Intended Use and Limitations

###### Primary Intended Uses

The task family is image-plus-prompt to text: input one PIL image and one of the nine task prompts in `TASKS` (optionally a caption string for `<CAPTION_TO_PHRASE_GROUNDING>`); output a caption string, an OCR transcript, or a dict of boxes and labels (`<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR_WITH_REGION>`, grounding), as the pinned README documents per task. Envisioned applications are photograph and document captioning for indexing, OCR of scanned pages and signage with line layout, zero-shot object localisation on everyday scenes, and grounding of a caption's noun phrases to image regions — one model replacing several single-task services in a research or internal-tooling setting. In a larger system the pipeline is an inference component whose outputs feed a human-reviewed index or a downstream detector-evaluation harness, not a decision engine. Until the pin decision in `STATUS.md` is taken, the intended use is limited to exercising the contract with an injected runner.

###### Primary Intended Users

The intended users are machine-learning engineers, computer-vision researchers and application developers integrating a multi-task vision-language model into research prototypes, internal enterprise document or image tooling, or the DIMER model workbench. The pipeline assumes its users understand that generated text is not a verified fact about the image (captions and OCR can be fluent and wrong), that box coordinates come from quantised location tokens with limited precision, that the model's behaviour is defined by the task prompt vocabulary rather than free-form questions, and that executing third-party modeling code from a model repository is a supply-chain decision they must take deliberately — which is why this package refuses it by default and leaves the decision to the repository owner.

###### Out-of-scope use cases

1. **Capability boundary:** not visual question answering with free-form questions (the pinned snapshot is the pretrained model, not the `-ft` variant; `smolvlm-vision-language-pipeline` covers chat-style VQA); not segmentation masks (`<REFERRING_EXPRESSION_SEGMENTATION>` and region-to-segmentation prompts are not in `TASKS`); not video, multi-image, or text-only input; not identity or face recognition. Anything outside the nine `TASKS` prompts raises `ValueError`.
2. **Input boundary:** only a single `PIL.Image.Image` is accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; the upstream processor resizes every image to 768×768 (`preprocessor_config.json`), so aspect ratio is not preserved and small text in large pages is lost; grounding captions above `MAX_TEXT_CHARS = 1000` characters are rejected; `max_new_tokens` is capped at `MAX_NEW_TOKENS = 1024`; a `text_input` supplied to a task that takes none is rejected.
3. **Decision boundary:** not for autonomous or high-impact decisions — evidentiary OCR, safety interlocks on detections, moderation takedowns, medical or forensic reading of images — without a human reviewing each output and a locally measured error rate.
4. **Loading boundary:** `from_pretrained` on the pinned snapshot raises `RuntimeError` and the Hub path raises the same; no path in v0.1.0 runs the model weights.

#### Factors

###### Groups

The pipeline is not human-centric by design — its prompts name objects, regions and text, not people — but the model can and will caption, detect and ground people in an image, and the upstream training corpus contains people. FLD-5B was built by automatic annotation of 126 M web-sourced images (upstream README and the paper), and neither the paper nor the pinned README reports any group-level breakdown of caption quality, detection recall or OCR accuracy by age, gender, skin type or region; the corpus is not group-audited. The obligation transfers to the operator: before deployment, measure task accuracy on a labelled sample of their own images stratified by the groups relevant to their application (for example caption quality across skin types or OCR accuracy across scripts) and treat a material gap as a blocker. This repository measures nothing of the kind.

###### Instrumentation

FLD-5B's images were collected from web sources (the paper names ImageNet-22k, Object 365, Open Images, Conceptual Captions and LAION among its image sources) and are consumer and stock photographs of undocumented camera, lens, compression and colour-management provenance; the 5.4 B annotations were generated by specialist models and iteratively refined, so label noise is itself an instrument characteristic. The pipeline consumes decoded pixel arrays and resizes them to 768×768 with bicubic resampling and ImageNet mean/std normalisation (`preprocessor_config.json`), so resolution, JPEG artefacts, rotation and scan skew all reach the model as changed pixel statistics. The pipeline does not detect blur, skew, low contrast or a changed capture device; it only rejects non-image types and sides outside 1–4096 px. OCR on scans should be validated on the scanner and DPI actually in use.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `pillow==11.3.0` (exact pins in `pyproject.toml`); the DIMER build environment is CPU-only with `CUDA_VISIBLE_DEVICES=-1`, and upstream trained and ships the weights in float16 (`config.json` `torch_dtype`), so a CPU run would use float32. On this repository's smoke run (Windows venv, CPU, 2026-09-12) `verify_snapshot()` re-hashed all 12 manifest entries (1556231245 bytes) in 1.11 s and `from_pretrained(device="cpu")` raised the documented `RuntimeError` after 1.06 s; no forward pass has been executed, so no inference timing or memory figure exists. Data environment: inputs are assumed to be natural photographs or document images resembling the web-sourced FLD-5B distribution, upright, with the subject or text legible at 768 px; line drawings, medical or satellite imagery, rotated scans and dense small print fall outside that assumption and degrade in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `character_error_rate(reference, hypothesis)` in `pipeline.py`: the character-level Levenshtein distance divided by the reference length, for use against `<OCR>` output when a ground-truth transcript exists. It captures reconstruction error of transcribed text, which is the operational question for OCR, and is preferred over word error rate because Florence-2's OCR output has no guaranteed word tokenisation across scripts. It says nothing about captioning or detection: caption quality needs CIDEr/BLEU against multiple references and detection needs COCO-style mAP, both of which require external tooling and labelled sets, so the pipeline reports neither. Upstream reports for Florence-2-large, zero-shot, COCO caption CIDEr 135.6, NoCaps CIDEr 120.8, TextCaps CIDEr 72.8 and COCO val2017 detection mAP 37.5 (pinned README table; the README summary states COCO OD AP 39.8 for this continued-pretrained revision); this pipeline has not reproduced any of them and, in v0.1.0, cannot.

###### Decision thresholds

The default decision rule is deterministic beam search: `num_beams = NUM_BEAMS = 3` with `do_sample=False`, echoed in every result's `generation` field — the returned text is the highest-scoring beam, an implicit "best sequence wins" rule with no minimum score. For detection and grounding tasks the model emits boxes without confidence scores by default; the upstream README's optional transition-score path is not exposed, so no score threshold exists to tune, and every box the decoder emits is returned. No acceptance threshold was set during development and none is shipped. A deployment that needs to suppress low-quality outputs must add its own filter — for example an OCR post-check against a lexicon, or a box-size floor — trading the cost of a wrong accepted output (false positive) against the cost of a suppressed correct one (false negative) for its application, and must recalibrate it if the pin decision changes the loader.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; the upstream figures cited above are single-evaluation numbers from the pinned README with no reported interval. Decoding is deterministic given the same weights, device, dtype and library versions: beam search with `do_sample=False` and a fixed `max_new_tokens`, no seed needed; float16 on GPU versus float32 on CPU can change near-tied beams and hence the text. The `generated_text` field is raw decoder output with no probability attached; box coordinates are dequantised location tokens, not calibrated estimates, and no confidence is emitted. A caller who needs calibrated confidence must expose the upstream transition-score path and calibrate it on labelled data; nothing in this repository does so. Because no forward pass has run, even the determinism statement is inferred from the decoding settings, not measured.

#### Ethical considerations and biases

###### Data

Upstream states the model was pretrained on FLD-5B, "5.4 billion annotations across 126 million images" (pinned README; the paper describes the images as drawn from public web datasets and the annotations as machine-generated), and that this revision was continued-pretrained on a further 0.1 B samples; the disclosure stops there — no per-image licensing, consent status or demographic composition is given, and web-sourced image corpora are known to contain photographs of identifiable people and text with personal information, so the presence of personal data is not ruled out. This repository distributes code, tests and documentation; the 1.55 GB `model.safetensors` snapshot and the three custom-code files are git-ignored or untracked and staged locally under `weights/florence-2-large/` with a manifest; no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content — OCR in particular will transcribe whatever text is present — and the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite (16 tests) and the refusal smoke in this repository; no output of the model has been produced here. Where a sensitive use is foreseeable — reading medication labels, transcribing identity documents, locating people in security footage — it is admissible only with a human reviewer on every consequential output, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/florence2_vision_language_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/florence-2-large/dimer-base-manifest.json` (12 entries) and raises on the first size or SHA-256 mismatch; `stage_missing_files` fetches only manifest-listed files at the pinned revision and refuses a manifest naming another model. (2) Refusal of remote code — `REMOTE_CODE_POLICY = "refuse"`; `remote_code_files` detects the snapshot's `configuration_florence2.py`, `modeling_florence2.py`, `processing_florence2.py` and the `auto_map` in `config.json`, and `from_pretrained` raises `RuntimeError` after verification, naming the pending owner decision; the Hub path (`allow_download=True` with no manifest) raises the same; the string `trust_remote_code` is never set true anywhere in the package. The refusal is grounded in an executed check: with `transformers==4.57.6` the native `Florence2ForConditionalGeneration` accepts the snapshot directory but reports 918 checkpoint tensors unused and 920 parameters newly initialised (key layout `language_model.model.decoder.*` versus native `model.language_model.decoder.*`), and the native processor fails on the snapshot's tokenizer — a "successful" native load would be a randomly initialised model. (3) Input integrity — `_validate` rejects non-PIL input, sides outside 1–4096 px, unknown task prompts, missing or over-long grounding text, text on tasks that take none, `max_new_tokens` outside 1–1024 and non-positive `num_beams` before any runner call; malformed runner output raises `RuntimeError`. (4) Reproducibility — exact `==` pins, deterministic decoding settings echoed in the result, `model_id`/`model_revision` in every result. No statistical mitigation is applied because the pipeline does not train.

###### Risks and harms

Fluent hallucination: captions and OCR transcripts read as confident prose even when wrong — objects that are not there, digits transposed in a serial number — and the operator or a downstream reader bears the harm when the text is trusted; likelihood is moderate on in-distribution photographs and high on dense or degraded documents. Detection without confidence: every emitted box is returned with no score, so spurious boxes cannot be filtered without an operator-built rule; harm falls on whoever acts on a phantom detection. Bias from a web-scraped corpus: captions may describe people with stereotyped or demeaning terms and detection recall may vary by demographic group; data subjects and third parties bear that harm. Automation bias: reviewers presented with a fluent caption check the image less carefully. Supply-chain risk specific to this pin: loading the model as upstream documents requires executing 190 KB of Python from the model repository; this package refuses it, and an operator who overrides that refusal outside this package takes on arbitrary-code-execution risk at the pinned revision. Privacy: OCR transcribes any personal data visible in an image.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — including locating or tracking individuals via `<OD>`, `<DENSE_REGION_CAPTION>` or phrase grounding, or bulk-transcribing identity documents and licence plates via `<OCR>`. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating captions or transcripts presented as evidence of what an image contains. Any use that violates the MIT terms of the upstream weights (the copyright and permission notice must be preserved) or the DIMER deployment terms is prohibited. The developers identify these because the model's outputs — free text and localisation — are precisely the primitives such misuses need; no further prohibited use is identified beyond them.

## Immutable provenance

- Model: `microsoft/Florence-2-large`
- Revision: `21a599d414c4d928c9032694c424fb94458e3594`
- Snapshot manifest: `weights/florence-2-large/dimer-base-manifest.json`, 12 files, `totalBytes` 1556231245
- `model.safetensors` SHA-256: `4f38ce741c6b71188fe2b3419a55e11917a8a7b321ae2e63c61da0191b0ebad7` (1553563458 bytes)
- `config.json` SHA-256: `6f8a8f92a74ce18b5c1e5646b4a8222477dd1ac49f31b953e75d6fc3e0f8583a` (2445 bytes)
- Custom code in the snapshot (refused): `modeling_florence2.py` SHA-256 `5162bf465e61b6e29cc113a467630ec3cb56ed8e4d46eb6207157f10fb9b8a24` (127455 bytes), `processing_florence2.py` `c655782a9e4347965c735ea54cbc4e98fdbc02155ffd1ce2ecd61f42c45eda28` (48674 bytes), `configuration_florence2.py` `de2e45a975b3582de05d2f4d963a3e9f9a3d20dccf78d28e0052932a0be93bdf` (15119 bytes)
- Weight format: SafeTensors (float16 as shipped); loader: none wired in v0.1.0 — `from_pretrained` verifies and refuses
- Alternative pin under consideration (read-only Hub metadata, 2026-09-12, not downloaded): `florence-community/Florence-2-large`, license `mit`, 776.7 M parameters, no `custom_code` tag, loadable by the native `transformers` Florence-2 classes; 13 files including `model.safetensors` (1553541016 bytes), `processor_config.json`, `added_tokens.json`, `merges.txt`. Its revision sha is recorded in `STATUS.md` so that this card carries exactly one pinned revision.

## Input/output contract

- `Florence2Pipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — in v0.1.0 always raises: `FileNotFoundError` (no manifest, downloads disallowed), `ValueError` (digest mismatch), or `RuntimeError` (custom code refused).
- `Florence2Pipeline(_runner, device="cpu", source="injected")` — constructible with an injected runner `(image, prompt, task, max_new_tokens, num_beams) -> {"text": str, "parsed": Any}`.
- `run(image, task="<CAPTION>", text_input=None, *, max_new_tokens=256, num_beams=3)` — `image`: one `PIL.Image.Image`, sides 1–4096 px, any mode (converted to RGB); `task` in `TASKS`; `text_input` required for `<CAPTION_TO_PHRASE_GROUNDING>` (≤ 1000 chars) and forbidden otherwise. Returns `{"task", "text_input", "result", "generated_text", "image_size", "generation": {"max_new_tokens", "num_beams", "do_sample"}, "device", "source", "model_id", "model_revision"}`.
- `character_error_rate(reference, hypothesis)` — float in [0, ∞); 0.0 for an exact match.
- `verify_snapshot(path=None)`, `stage_missing_files(path=None, *, allow_download=False, downloader=None)`, `remote_code_files(path=None)`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12. The build venv carries `torch 2.14.0+cu130`.
- Precision: upstream ships float16 weights; a CPU run would be float32. Not measured — no forward pass has run in this repository.
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, 2026-09-12): `verify_snapshot()` 12 files / 1556231245 bytes in 1.11 s; `from_pretrained(device="cpu")` → `RuntimeError` (custom code refused) after 1.06 s; total 2.18 s, exit 0.
- Investigation record (same venv): `from transformers import Florence2ForConditionalGeneration` succeeds; `Florence2ForConditionalGeneration.from_pretrained(<snapshot>, trust_remote_code=False)` returns in 13.08 s with 918 unused checkpoint tensors and 920 newly initialised parameters (505101312 parameters instantiated versus 776.7 M upstream); `AutoProcessor.from_pretrained(<snapshot>, trust_remote_code=False)` fails with `AttributeError: BartTokenizerFast has no attribute image_token`; `AutoModelForCausalLM.from_pretrained(<snapshot>, trust_remote_code=False)` fails with `ValueError: ... contains custom code which must be executed`.
- Tests: `pytest -q -o addopts= tests` — 16 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Xiao, Wu, Xu, Dai, Hu, Lu, Zeng, Liu, Yuan. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. arXiv:2311.06242 (2023). https://arxiv.org/abs/2311.06242
- Upstream card and inline modeling code: https://huggingface.co/microsoft/Florence-2-large (no separate upstream source repository is published; the modeling code lives in the Hub repository)
- Native port under consideration: https://huggingface.co/florence-community/Florence-2-large
- Transformers 4.57.6 native implementation: `transformers/models/florence2/`
