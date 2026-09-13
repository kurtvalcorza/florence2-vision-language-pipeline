# Release verification

`tutorials/florence2_vision_language_colab.ipynb` (`MULTI-CAPABILITY`, **standalone** carrier) remains a **release candidate**. A clean Python 3.12 GPU execution of the exact notebook blob was recorded on 2026-09-13; the result and retained artifacts are below. Static checks are not runtime evidence, and promotion still requires a reviewer to accept the recorded run.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `MULTI-CAPABILITY`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `1.1`, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/florence2_vision_language_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Florence2Pipeline.from_pretrained(weights_dir=...)`, `validate_inputs` per capability, `run` for
  `<CAPTION>`, `<OD>` and `<OCR>` with explicit `max_new_tokens` and `num_beams=NUM_BEAMS`, and
  `evaluation_report` over all three), the ceiling and task constants carried by the module, the
  per-capability contracts and structural checks, the exports,
  the learner-facing statements (task-token selection, deterministic beam search, 768 x 768 squash,
  no box confidence scores, OCR CER sanity-only, exposed-but-not-demonstrated tasks, no
  segmentation) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `transformers` or `huggingface_hub` use **outside the carried
  module cell**, `.generate(`, `post_process_generation(`, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA float16 used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or Pillow that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with **no repository checkout**, an empty Hugging Face cache, and no pre-staged
   weight files under `weights/florence-2-large-community/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `OCR_REFERENCE = ''`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`) (`torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `Florence2Pipeline`, `validate_inputs`,
     `evaluation_report`, `character_error_rate`, `verify_snapshot`, `stage_missing_files`) with no
     import of the repository package;
   - synthetic 512 x 512 drawing (red square, blue circle, the text `DIMER 2026`) generated in code with its pixel SHA-256 printed;
   - ceilings `MAX_IMAGE_SIDE = 4096`, `MAX_NEW_TOKENS = 1024`, `DEFAULT_MAX_NEW_TOKENS = 256`, `MAX_TEXT_CHARS = 1000`, `NUM_BEAMS = 3`, the `TASKS` split and the three capability contracts printed, and `validate_inputs` writing `outputs/florence2_vision_language_input_manifest.json` with verdict `accepted`, one per-capability sub-manifest, and one recorded rejection finding from the unsupported-task probe;
   - `stage_missing_files(..., allow_download=True)` reporting all 12 manifest entries fetched into an empty standalone weights directory from `florence-community/Florence-2-large` at the immutable revision, `verify_snapshot` reporting 12 files, and `Florence2Pipeline.from_pretrained` reporting `source: local-snapshot` (loading info clean);
   - `run` completing for `<CAPTION>`, `<OD>` and `<OCR>` with `do_sample: False` and `num_beams: 3` in every `generation` block and all five structural checks true;
   - `evaluation_report` writing `outputs/florence2_vision_language_evaluation_report.json` with the combined verdict `sample-sanity`, exactly one `character_error_rate` metric tagged `<OCR>` scored against `DIMER 2026`, and `not-measurable` sub-reports for `<CAPTION>` and `<OD>` naming what each would need;
   - the detection preview rendered;
   - `outputs/florence2_vision_language_result.json` and `outputs/florence2_vision_language_preview.png` written, the JSON carrying one entry per capability with parsed result, raw text and generation settings, plus `NOTEBOOK_SOURCE`, the model identifier, the immutable model revision, the model licence, the runtime versions, device and dtype;
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
| `tutorials/florence2_vision_language_colab.ipynb` | `ece9e7f72e5915c7e84208415c0e803955e1b774` / `e807053d55c0849b2436ff85bfc0e8c194f1eca4` | 2026-09-13 | Colab CLI → isolated Python 3.12.3, T4 | PASS — 8/8 cells; evidence review pending; [Retained run](verification/2026-09-13/README.md) |

## Recorded executions

Notebook identity is the Git blob of `tutorials/florence2_vision_language_colab.ipynb` at the source commit in the row below. The documentation commit recording the run does not change that notebook blob. Cell wall time is the sum of recorded code-cell times, including installation and model downloads; total time additionally includes environment setup and bookkeeping. These measurements describe this one run.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Cell wall / total | Outcome |
|---|---|---|---|---|---|
| 2026-09-13 | `ece9e7f72e5915c7e84208415c0e803955e1b774` / `e807053d55c0849b2436ff85bfc0e8c194f1eca4` | Colab CLI → fresh Python 3.12.3 venv/interpreter; Tesla T4, 15,360 MiB | Unchanged default sample, no repository checkout, empty per-model cache and weights | 159.258 s / 163.882 s | PASS — 8/8 cells; evidence review pending; [Retained run](verification/2026-09-13/README.md) |

The run used PyTorch `2.14.0+cu130`, `cuda:0` and `float16`. All eight code cells completed, runtime pins matched, every snapshot file was SHA-256 verified, inputs were accepted, and the negative validation probe was recorded. Results, model identity/revision, observed output, warnings, package versions, notebook outputs, executor source and cleanup evidence are retained in [the run record](verification/2026-09-13/README.md).

The native hosted kernel was Python 3.13.15; its direct notebook attempt was aborted in installation after the Python-version mismatch was confirmed. The successful result above uses the repository-supported Python 3.12 interpreter on the Colab GPU. No completed native hosted-kernel run is claimed.

## Current status

Clean GPU execution evidence is now recorded for the exact notebook blob above. The registry status remains **Candidate** pending a reviewer’s acceptance of the evidence and an integrator’s promotion. This documentation change performs no promotion. The run is default-sample inference/contract evidence; it does not establish model quality or a benchmark result. CPU and BYOD paths were not exercised by this GPU run.

Current source update: snapshot validation now runs before model-library imports (Kokoro also validates the language first), so rejected requests fail with the intended validation error even when model libraries are absent. The standalone notebook was regenerated from this source. The retained 2026-09-13 GPU run identifies the earlier notebook blob; the regenerated notebook has not had a fresh GPU execution. Status remains **Candidate**.
