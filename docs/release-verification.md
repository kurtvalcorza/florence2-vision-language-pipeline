# Release verification

`tutorials/florence2_vision_language_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook. The earlier `MULTI-CAPABILITY` notebook's Colab Tesla T4 run (2026-09-13, retained under
`verification/2026-09-13/`) is history for a superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 12-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the Belfort-line
  parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb` and the rejected `microsoft/Florence-2-large`
  pin `21a599d414c4d928c9032694c424fb94458e3594` are the two other 40-hex revisions the documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Florence2Pipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the ceiling print,
  `validate_inputs` per capability with the unsupported-task refusal probe, `run` for the three task tokens with the
  structural checks and the multi-capability `evaluation_report` on the drawing, `empty_baseline`, `constant_baseline`,
  `pipe.evaluate` on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with the two CER assertions, the three capabilities on the drawing after
  adaptation, the example panels, `pipe.save_artifact`, `Florence2Pipeline.from_artifact` and the transcript-parity
  assertion, and the result fields `weight_file` / `weight_format` / `weight_sha256`, the `corpus` block and
  `tasks_exposed`), the seven expected `outputs/` paths, the learner-facing statements (MIT weights, the task prompt
  token, deterministic beam search, adaptation of one task on transcribed lines, the two non-adapted baselines, CER and
  WER, rates above 1.0, the empty-string and constant-transcript baselines, the sequence-to-sequence loss, the frozen
  encoder cache, lowest validation CER, no dispersion estimate, no confidence scores, float32 on every device, the
  leakage guidance, the excluded tasks, the snapshot note, the troubleshooting block) and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `Florence2ForConditionalGeneration` /
  `Florence2Processor` / `post_process_generation(` / `model.generate(` / `torch.inference_mode(` / `from
  huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors`, `numpy`,
`pillow` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (CUDA; a CPU runtime is not practical for the default path) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA runtime (Colab GPU, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/florence-2-large-community/` or the row-group cache `weights/belfort/` (the standalone path writes
   the manifest itself, stages the missing files from the Hub and reads the pinned row groups over range requests, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `LINE_MAX_NEW_TOKENS = 128`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`,
   `BATCH_SIZE = 8`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1` (an interpreter restart after the install is expected where
   the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `Florence2Pipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `character_error_rate`, `edit_distance`, `normalise_text`, `ocr_metrics`,
     `empty_baseline`, `constant_baseline`, `medoid_transcript`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, the task lists and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 12 manifest entries fetched from `florence-community/Florence-2-large` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (12 files, the 1.55 GB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory on `cuda:0` in float32 with a clean `output_loading_info`;
   - Section 4: `fetch_corpus` reading the eight pinned row groups over HTTPS range requests with every SHA-256 and
     byte total matching (800 lines, about 44 MB); the seeded split into 600 / 60 / 140 with `check_split_disjoint`
     reporting no shared image and the three dataset digests printed; `outputs/…_train.csv` and
     `outputs/…_example_line.png` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the drawing rendered; the combined input manifest written to
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the unsupported-task
     probe); `<CAPTION>`, `<OD>` and `<OCR>` run with every structural check `True`, `outputs/…_preview_frozen.png`
     written and the multi-capability `evaluation_report` verdict `sample-sanity` (OCR of the drawn text; the
     multi-capability card recorded the square labelled `flag` and an exact OCR — observations, not assertions);
   - Section 6: the empty baseline (CER 1.0 exactly), the constant-transcript baseline (≈ 0.94) and the frozen model's
     test rates (≈ @P:FROZEN_CER@ CER / @P:FROZEN_WER@ WER in the Tesla T4 build record — the model emits a dash or
     nothing for cursive) with four hypotheses printed under their references;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 67,188,736 trainable of 776,505,344 parameters,
     `first_trainable_layer` 8, and a six-epoch history with the validation CER falling (build record:
     @P:VAL_CURVE@, `best_epoch` @P:BEST_EPOCH@);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the hypothesis
     lengths and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test CER is below the frozen
     one and below 1.0 — @P:ADAPTED_CER@ against @P:FROZEN_CER@ in the build record, WER @P:FROZEN_WER@ →
     @P:ADAPTED_WER@; the adapted model also clears the constant baseline, reported, not asserted);
   - Section 9: six example panels under `outputs/…_examples/`; the three capabilities re-run on the drawing by the
     adapted model with `outputs/…_preview_adapted.png` (build record: @P:DRAWING_AFTER@ — a recorded observation,
     not an assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (66
     tensors, about 269 MB) and `Florence2Pipeline.from_artifact` reloading it with 8/8 identical transcripts on
     eight test lines (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block,
     the inference-contract records before and after adaptation, the comparison, the artifact digest, the reload
     parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `florence2_vision_language_colab.ipynb` (`E2E`) | pending | — | — | pending: the supported-runtime run of the exact candidate blob is not yet recorded |
| `florence2_vision_language_colab.ipynb` (`MULTI-CAPABILITY`, superseded) | `ece9e7f72e5915c7e84208415c0e803955e1b774` / `e807053d55c0849b2436ff85bfc0e8c194f1eca4` | 2026-09-13 | Colab CLI → isolated Python 3.12.3, Tesla T4 | PASS — 8/8 cells; [retained run](verification/2026-09-13/README.md); evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/florence2_vision_language_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/florence2_vision_language_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | package API at `@P:PROBE_SHA@` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-florence2-e2e` v1; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (@P:MB_RESULT@) and the recipe probe: the eight pinned row groups read over range requests (800 lines, digest match), empty and constant baselines, frozen `<OCR>` on the 140 test lines, `adapt(epochs=6, lr=5e-5, batch_size=8)` with validation-CER selection, adapted evaluation, artifact round trip; a second kernel (v2, 2574 s) repeated the model-backed suite (7/7 in 128 s) and swept lr 1e-4 and 2e-4 for eight epochs | @P:PROBE_WALL@ | @P:PROBE_OUTCOME@ The sweep: lr 1e-4 × 8 → validation 0.969 → 0.929 / 0.815 / 0.852 / 0.809 / 0.801 / 0.807 / 0.828 / 0.804 (epoch 5), test 0.798 / 1.032; lr 2e-4 × 8 → 0.835 / 0.813 / 0.831 / 0.826 / 0.817 / 0.882 / 0.830 / 0.817 (epoch 2), test 0.807 / 1.051 — the same plateau |
| 2026-09-13 | `ece9e7f72e5915c7e84208415c0e803955e1b774` / `e807053d55c0849b2436ff85bfc0e8c194f1eca4` (`MULTI-CAPABILITY`, superseded) | Colab CLI → fresh Python 3.12.3 venv/interpreter; Tesla T4, 15,360 MiB | Unchanged default sample, no repository checkout, empty per-model cache and weights | 159.3 s / 163.9 s | PASS — 8/8 cells; [retained run](verification/2026-09-13/README.md); not evidence for the `E2E` blob |

## Current status

**Candidate.** The `E2E` notebook has not yet executed in a supported runtime; the registry stays **Candidate** until a
clean run of the exact candidate blob is recorded above and an integrator promotes it. What exists: static validation
(`tools/validate_release_assets.py`), the generator parity checks (`--check` OK), the offline unit suite, the CPU run of
the model-backed suite (6 of 7, the CUDA test skipped) and the Tesla T4 pre-flight of the package API (table above).

Facts a reviewer should weigh: the sample is nineteenth-century French cursive, far outside the checkpoint's printed and
scene-text training distribution, which is why the frozen `<OCR>` sits at the empty baseline (it emits a dash or nothing)
and why the gain is a repair of a domain gap, not evidence about other hands or scripts; the rates are uncapped micro
CER/WER over one crowdsourced transcription and the notebook says so; the 60-line validation split selects the epoch;
the vision tower and the encoder are frozen, so what they cannot resolve in a 128-px line squashed into 768 × 768
stays unread; the decoder that was tuned serves every task token, and the drawing re-run after adaptation is the only
evidence about what happened to `<CAPTION>` and `<OD>`. Beam search is deterministic on a fixed device and dtype, but the
training of four decoder layers is not bit-reproducible across GPUs, so a Kaggle number a few hundredths off the build
record is the expected spread, not a finding. The adapted rate plateaus near 0.80 at every learning rate tried (5e-5, 1e-4, 2e-4), so a decoder-only adapter
cannot make this encoder read cursive — the docs say so rather than tuning further. The sibling GOT-OCR 2.0 row scored
0.759 CER on the same split with the same design; the two are compared in the card.
