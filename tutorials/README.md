# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/florence2-vision-language-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/florence2-vision-language-pipeline/blob/main/tutorials/florence2_vision_language_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-florence--community%2FFlorence--2--large-ffcc4d?style=flat)](https://huggingface.co/florence-community/Florence-2-large)
[![Upstream](https://img.shields.io/badge/Upstream-microsoft%2FFlorence--2--large-181717?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/microsoft/Florence-2-large)
[![arXiv](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg)](https://arxiv.org/abs/2311.06242)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `florence2_vision_language_colab.ipynb` | `MULTI-CAPABILITY` | Florence-2-large task-prompted inference: `<CAPTION>`, `<OD>` (boxes + labels, no scores) and `<OCR>` on one synthetic drawing with drawn text; deterministic 3-beam decoding with explicit `max_new_tokens`; `character_error_rate` for OCR against the drawn text (sanity only); no metric for captions or detections | CPU float32 (CUDA float16 used automatically when available) | one image file plus an optional OCR reference form field, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `Florence2Pipeline` from the repository public API rather than reimplementing model inference; the pipeline pins the immutable community-converted revision (`florence-community/Florence-2-large`, loadable with `trust_remote_code=False`), stages the missing weight file through `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`), and refuses a checkpoint that does not map cleanly onto the native architecture. The notebook never calls `transformers`, `huggingface_hub` or `.generate(` directly.
- MULTI-CAPABILITY obligations (INF10): three task tokens are demonstrated with their own stated input/output contracts (`<CAPTION>` -> str; `<OD>` -> boxes + labels in input pixels, no scores; `<OCR>` -> str); generation settings (`max_new_tokens`, `num_beams` = `NUM_BEAMS`, `do_sample=False`) are explicit per call (INF8/INF9); the six other exposed task tokens are listed as exposed-but-not-demonstrated, and segmentation, region-to-category/description, open-vocabulary detection, VQA and batching are stated as not provided (G7/ID6).
- Metrics (EVAL1/EVAL7/EVAL9): only OCR has a helper (`character_error_rate`), applied to the drawn text on the default sample as a sanity check and to `OCR_REFERENCE` on BYOD; captions and detections have no metric and the notebook says what labelled data they would need. The IoU between the best `<OD>` box and the drawn square is printed as an observation, not a metric. Recorded `SHOULD` deviation: EVAL11 (no baselines — none is meaningful without labelled data).
- Ceilings `MAX_IMAGE_SIDE`, `MAX_NEW_TOKENS`/`DEFAULT_MAX_NEW_TOKENS`, `MAX_TEXT_CHARS`, `NUM_BEAMS` and the full `TASKS` list are surfaced before the model runs; the processor's 768 x 768 squash is stated (DAT22/DAT23).
- The default sample is a synthetic drawing generated in code (shapes plus text in Pillow's built-in font); `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- The validator whitelists the rejected `microsoft/Florence-2-large` revision that `docs/WEIGHTS.md` records once in its pin history.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
