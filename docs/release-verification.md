# Release verification

`tutorials/florence2_vision_language_colab.ipynb` (`MULTI-CAPABILITY`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.0. This file is the durable release-gate record for
the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `MULTI-CAPABILITY`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Florence2Pipeline.from_pretrained`, `run` for `<CAPTION>`, `<OD>` and `<OCR>` with explicit
  `max_new_tokens` and `num_beams=NUM_BEAMS`, `character_error_rate` for OCR), the ceiling and task
  constants imported from the package, the per-capability contracts and sanity checks, the exports,
  the learner-facing statements (task-token selection, deterministic beam search, 768 x 768 squash,
  no box confidence scores, OCR CER sanity-only, exposed-but-not-demonstrated tasks, no
  segmentation) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, direct `transformers` or `huggingface_hub` calls that bypass the pipeline,
  `.generate(`, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA float16 used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and `DIMER_TUTORIAL_REF` set to the candidate commit | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit, an empty Hugging Face
   cache, and no pre-staged weight files under `weights/florence-2-large-community/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `OCR_REFERENCE = ''`);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins (`torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`);
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - synthetic 512 x 512 drawing (red square, blue circle, the text `DIMER 2026`) generated in code with its pixel SHA-256 printed;
   - ceilings `MAX_IMAGE_SIDE = 4096`, `MAX_NEW_TOKENS = 1024`, `DEFAULT_MAX_NEW_TOKENS = 256`, `MAX_TEXT_CHARS = 1000`, `NUM_BEAMS = 3`, the `TASKS` split and the three capability contracts printed, and the image accepted before model execution;
   - `stage_missing_files(..., allow_download=True)` reporting `['model.safetensors']` fetched from `florence-community/Florence-2-large` at the immutable revision, `verify_snapshot` reporting 12 files, and `Florence2Pipeline.from_pretrained` reporting `source: local-snapshot` (loading info clean);
   - `run` completing for `<CAPTION>`, `<OD>` and `<OCR>` with `do_sample: False` and `num_beams: 3` in every `generation` block, all five sanity checks true, the IoU observation printed, and the OCR `character_error_rate` against `DIMER 2026` printed as sanity evidence;
   - the detection preview rendered;
   - `outputs/florence2_vision_language_result.json` and `outputs/florence2_vision_language_preview.png` written, the JSON carrying one entry per capability with parsed result, raw text and generation settings, the repository SHA, model identifier, immutable model revision, snapshot summary, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, Pillow, device, dtype), model
   identifier and immutable revision, whether the model cache and weights directory were clean,
   outcome, produced outputs, the caption, the boxes with labels, the OCR string and its character error rate (as observations), and any warning or applicable `SHOULD`
   deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tutorials/florence2_vision_language_colab.ipynb` | | | | pending — queued to the GPU lane |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/florence2_vision_language_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/florence2_vision_language_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

No execution of the notebook has been recorded. The only runtime measurements that exist for this repository are the pipeline smoke run documented in `MODEL_CARD.md` (CPU float32, load 7.11 s, `<CAPTION>` 4.91 s and `<OD>` 9.02 s on a 256 x 256 drawing of a red square, labelled `flag`). That run exercised the
package, not this notebook, and is not notebook execution evidence.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | — | — | Default sample path | — | pending — queued to the GPU lane |

## Current status

The notebook source is complete and passes the static checks above; **no clean-runtime execution
has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row.
