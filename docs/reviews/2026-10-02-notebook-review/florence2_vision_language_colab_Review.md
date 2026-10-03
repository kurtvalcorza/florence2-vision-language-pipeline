# Florence-2-large Handwritten-Line OCR Adaptation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/florence2-vision-language-pipeline`  
**Notebook:** `tutorials/florence2_vision_language_colab.ipynb`  
**Reviewed commit:** `af3e164cbf683aed741904b9e33396a600f19790` (`main`, confirmed with `gh api repos/kurtvalcorza/florence2-vision-language-pipeline/commits/main`)  
**Notebook Git blob:** `035c2c027cd47caf0c54dae9850a23bdd0c73274`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `bc7e9b8`); the notebook last changed in `517503c`.  
**Finding prefix:** `FL`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The engineering is careful. The notebook carries its three modules verbatim, digest-verifies the 1.55 GB snapshot, reads eight digest-pinned Belfort row groups over HTTPS range requests, validates and splits them, compares the frozen and adapted `<OCR>` against an empty and a constant-transcript baseline with uncapped micro/macro CER and WER, fine-tunes only the last four decoder layers with stated hyperparameters and validation-CER epoch selection, and reloads a safetensors adapter with an 8/8 transcript-parity assertion. The prose is unusually honest about how little the adaptation achieves (CER 0.992 → 0.797, WER unchanged, 0 lines exact). A CPU run of the data stage in this review reproduced the recorded split exactly (600 / 60 / 140, same three digests, `SAMPLE_DIGEST` match).

| Measure | This review (CPU, data stages only) | Kaggle T4 record (blob `035c2c02`) |
|---|---|---|
| Code cells completed | cells 3 (install skipped), 5, 7, 9, 13; model cells not run | 11/11 on pass 2; pass 1 stopped at the install guard |
| Belfort fetch | 800 lines, 42,403,899 bytes, digests match, 35 s | identical counts, 6.3 s |
| Split digests | `64186dce…` / `5768e17f…` / `0e815da4…` | identical |
| Test lines with a neighbouring corpus line in train | **131 / 140** | not reported |
| BYOD smallest accepted dataset | **50 images** (8, 20, 38, 49 refused) | not run |
| Frozen → adapted test CER / WER | not run | 0.992 → 0.797 / 1.002 → 1.012 |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (FL-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart; the opening cell itself budgets "22 [minutes] with the pinned install and its restart", and the repository marks the blob `Release-grade` on that run.
2. **Guided layer largely absent (FL-M2).** Declared `GUIDED`, but there is no audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint or conclusion template, and 1,387 lines of carried modules sit in three unlabelled, uncollapsed cells.
3. **The held-out split is not leakage-free in the sense the notebook teaches (FL-M3).** Lines are shuffled individually, so 131 of 140 test lines have the line just before or after them in the corpus in the training split. The notebook calls the split "without leakage" and then advises learners to split by page because "lines cut from the same page share a hand".
4. Unresolved MUSTs outside those Majors: the BYOD contract states a minimum of 8 records while the branch needs 50 (DAT12, DAT19, FL-m2), and the carried cells silently change a pipeline ceiling (VAL6, FL-m1).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (cell 1): basic Python and PIL; task prompts and beam search; pixel bounding boxes; what CER/WER measure and why they are not capped; self-drawn image vs held-out split |
| Supported runtime | "a fresh supported runtime with a CUDA GPU (Google Colab or Kaggle GPU, Python 3.12)"; "a CPU runtime would take hours" |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-pinned Belfort fetch, validation and "split … by line without leakage"; three task tokens on a drawing with their output contracts; frozen `<OCR>` CER/WER beside two baselines; bounded fine-tuning with explicit hyperparameters and validation epoch selection; line-disjoint test evaluation; panels and the three capabilities re-run after adaptation; safetensors adapter with reload parity; BYOD zip through the same cells |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `8fd45ec3` |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7, 9 are the carried `pipeline.py` 901 lines, `metrics.py` 99 lines, `samples.py` 387 lines). Also read: the generator, template and `tools/validate_release_assets.py` (relevant parts); `src/…/pipeline.py` (`_check_inputs`, `validate_inputs`, `INPUT_SCHEMA`, `evaluate`, `adapt`), `samples.py` (`fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `load_byod_dataset`), `metrics.py`; `README.md`, `STATUS.md`, `MODEL_CARD.md` (build record), `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` (evidence lives in `docs/release-verification.md` and `docs/verification/2026-09-13/`, the latter for a superseded blob).
- **Documented execution evidence.** `docs/release-verification.md` row 2026-09-20 and the workspace run summary (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-florence2-vision-language/v3/evidence/run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`): Kaggle Tesla T4, **the reviewed blob** (SHA-1 verified before execution), clean HF cache; pass 1 `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime…`, `restarted_after_install_cell: true`, pass 2 11/11 in 1143 s. No Colab run of this blob; no BYOD run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.10, torch 2.14.0+cpu, transformers 4.57.6, pillow 11.3.0, numpy 2.5.3, pyarrow 25.0.1 — the notebook's pins, taken read-only from another workspace `.venv`. Nothing was installed. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Probes (61 s total):** P0 static checks; P1 the three carried cells executed in one namespace and compared with the installed package; P2 cell 13 at defaults from an empty working directory (real Hub fetch of the eight row groups); P3 cell 13's BYOD branch through a shimmed `google.colab.files.upload` with eleven archives; P4 metric sanity.
- **Not verified:** Sections 3 and 5–9 (model load, inference, baselines on the model, fine-tuning, evaluation, export, reload) beyond the Kaggle record; any Colab run; the real upload dialog; BYOD beyond the validation/split stage.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong supply-chain and data-integrity handling (row groups refused on any digest mismatch; P2 digests identical to the Kaggle record). Defects: the install pattern forces a restart (FL-M1); carrying the three modules into one kernel namespace changes the phrase-grounding text ceiling from 1000 to 512 while the exported schema still says 1000 (FL-m1); a re-run of Section 7 continues training from the adapted weights and labels them "frozen model" (FL-m5).
- **Scientific validity:** baselines, uncapped micro/macro rates, hypothesis length, validation-only epoch selection and an untouched test split are all sound, and the conclusion is appropriately modest. The weak point is the split unit: shuffled individual lines from 800 consecutive corpus lines put almost every test line next to training lines from the same page and hand (FL-M3); the notebook does not state run-to-run variability of the GPU fine-tune while quoting build-record numbers to three decimals (FL-m4).
- **Promise fulfilment:** default-path promises are met on the documented run, except one-pass `Run all` (FL-M1) and "without leakage" (FL-M3). BYOD's stated minimum is wrong (FL-m2) and a legitimate negative BYOD result would stop at an assertion before export (FL-m3). The phrase-grounding experiment cannot run as written (FL-m5).
- **Learner experience:** precise prose, "Look for"/"Watch" notes at the main stages, a troubleshooting paragraph and a careful closing interpretation; but no guided layer (FL-M2), and one expected-output note contradicts the run (FL-m4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2, REL11 (FL-M1); SPL3, SPL5 (FL-M3); DAT12, DAT19, VAL7 (FL-m2); VAL6 (FL-m1); DAT13 (FL-m3); ENV8 (FL-m4); UX7 (FL-m5). SHOULD deviations: GDL1–GDL7, GDL9–GDL12, GDL14, UX8 (FL-M2); EXE2, UX10 (FL-m2); GDL8 (FL-m4); GDL10, UX5 (FL-m5); EXE5 (FL-S3).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | cell 0 budgets "its restart"; Troubleshooting says restart | **Not met** (FL-M1) |
| Digest-verified pinned snapshot | cell 11 | 12/12 fetched and verified, `cuda:0`, float32 (Kaggle) | clear | Met (documented) |
| Digest-pinned Belfort fetch and validation | cell 13 | P2: 800 lines, digests match, four refusal probes rejected with clear messages | "Look for" note matches | Met |
| "Split … by line without leakage" | `build_sample_dataset` (line-level shuffle) + `check_split_disjoint` (pixel digest) | P2: 0 shared images, but 131/140 test lines adjacent to a training line; 5 test transcripts occur verbatim in train | cell 24 advises page/writer splits for *your* data | **Partly met** (FL-M3) |
| Three capabilities with output contracts | cell 15 | Kaggle: checks all `True`; caption, 1 `poster` box `[0,0,511,511]`, OCR exact | cell 14 sets up "`flag`" square | Met; expectation stale (FL-m4) |
| Frozen CER/WER beside two baselines | cell 17 | Kaggle: empty 1.0, constant 0.944, frozen 0.992 | well explained | Met (documented) |
| Bounded fine-tune, explicit hyperparameters, validation selection | cell 19 → `adapt` | Kaggle: 67,188,736 trainable, best epoch 6 by val CER | well explained; seed (0) not printed | Met (documented) |
| Held-out evaluation, four-way comparison | cell 21 | Kaggle: CER 0.797, WER 1.012, 0 exact; assertions pass | careful reading order, "no dispersion estimate" | Met; assertions unsafe for BYOD (FL-m3) |
| Adapter export and fresh reload with parity | cell 23 | Kaggle: 106 tensors, 268,769,008 bytes, parity 8/8 | explained; docs say 66 tensors (FL-m4) | Met (documented) |
| BYOD zip through the same cells, "at least eight images" | cell 13 BYOD branch | P3: 8–49 images refused with "N records; 8..5000 are required"; 50 accepted | contract states 8 | **Not met as stated** (FL-m2) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried package, stage and verify the snapshot | run cells | printed identity and verified-file count |
| Fetch, validate and split without leakage | run cell 13 | digests and refusal probes printed; no prompt to check the split unit |
| Read each output contract on the drawing | read output | prose explains no score / no confidence; no prediction or checkpoint |
| Measure frozen CER/WER beside baselines | read output | "Expect the frozen model at the empty baseline" — a stated expectation, not a learner prediction |
| Fine-tune, evaluate, look at the lines and the other capabilities | run, read panels | no question asks the learner to explain the plateau or the WER |
| Export and reload with parity | run | parity printed and asserted |

The objectives are operations the code performs rather than learner actions with a check (GDL5); there is no learner-controlled Predict → Change one thing → Run → Observe → Explain activity with rerun scope (GDL10). The "Optional experiments" paragraph is the only transfer prompt.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 25 cells | Each section opens with an accurate explanation and a "Look for"/"Watch"/"Expect" note, and the closing interpretation is careful. Missing: audience statement, how-to-use, roadmap, Input → Model → Output contract, glossary (task prompt token, DaViT, BART, beam search, teacher forcing, Levenshtein, micro/macro, medoid, encoder cache, safetensors), predictions, checkpoints with sample answers, conclusion template; the three carried cells (1,387 lines) are unlabelled and uncollapsed (FL-M2). Cell 14 tells the learner to expect a `flag` square that the run does not produce (FL-m4). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, data stages, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 11/11 after a restart (FL-M1); all outputs and numbers match the prose. Direct: cells 3/5/7/9/13 from an empty working directory, Belfort fetch over range requests 35 s, split and digests identical to Kaggle, `SAMPLE_DIGEST` match, refusals identical. Sections 3, 5–9 not executed. No Colab run. |
| **Active learning** | Source inspection | No documented exercise states a prediction or rerun scope. Rerunning cell 19 after changing `EPOCHS`/`LEARNING_RATE` (the first suggested experiment) continues from the already adapted weights and labels epoch 0 "frozen model" (FL-m5, inferred from `adapt`). The phrase-grounding experiment would raise `task <CAPTION_TO_PHRASE_GROUNDING> requires a non-empty text_input` because `run_capabilities` passes no `text_input` (FL-m5). Not executed. |
| **Reuse and recovery** | Direct (P3, cell 13 with shimmed upload) + source; real upload dialog and BYOD downstream stages not verified | 50 synthetic lines → split 32/8/10, all splits validated. Refused: 8, 20, 38, 49 lines with "2/4/6/7 records; 8..5000 are required" (the split is not named); non-image member (clear); missing `transcripts.csv` (clear); 513-character transcript (clear); not a zip (clear); same basename in two folders → "duplicate id 'x'" with no mention that the archive was flattened; cancelled upload → bare `StopIteration`. 70 records with 10 pixel duplicates → 60 split, nothing reported. |

## 5. Findings

### Major

#### FL-M1 — `Run all` needs a manual restart after the install cell, and the blob is marked `Release-grade` on that run

- **Cell/section:** cell 3, Section 1 prose (cell 2), opening cell 0 ("22 with the pinned install and its restart"), Troubleshooting (cell 24). Generator: `tools/build_notebook.py` lines 48–70 (install-cell body, guard at line 69) and line 453 (Section 1 prose); `tools/notebook_template.py` line 501 (Troubleshooting). Records: `docs/release-verification.md` procedure step 4 ("an interpreter restart after the install is expected"), Manual evidence and Current status; `STATUS.md`; `tutorials/README.md` registry row.
- **Observed issue:** the cell `pip install`s nine pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The opening cell promises that **Run all** in a fresh runtime completes every stage.
- **Consequence:** on a stock Kaggle (and, by the same mechanism, Colab) image the learner's **Run all** stops in the first code cell and must be restarted and re-run; RUN1, RUN10 and ENV6 forbid this, and a restart-dependent run is not REL2 evidence. The repository discloses the restart but still records `PASSED` and promotes the blob to `Release-grade`; the procedure even declares the restart expected.
- **Evidence:** documented — Kaggle T4 run of blob `035c2c02`: pass 1 `RuntimeError` naming `cuda-bindings` 12.9.4 → 13.4.2 and `numpy` 2.0.2 → 2.5.3, `restarted_after_install_cell: true`, pass 2 11/11. Source — P0: `pip_install_in_kernel: true`, `restart_instruction_in_install_cell: true`, `uses_uv: false`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py` (and the validator's install-cell expectations in `tools/validate_release_assets.py`), regenerate, re-qualify with a one-pass hosted Run all, and return the registry to `Candidate` until then; correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab GPU runtime completes every code cell in one pass with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/florence2_vision_language_colab.ipynb` and `grep -n "its restart" tutorials/florence2_vision_language_colab.ipynb` return nothing; no document marks a blob `Release-grade` on a run with `restarted_after_install_cell: true`.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11; §27 ("MUST NOT be marked release-grade when its default Run all path is known to fail").

#### FL-M2 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9 and 11, end of notebook. Generator: `tools/notebook_template.py` section texts and `tools/build_notebook.py` section assembly.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output task contract, no glossary, no prediction before the frozen scoring, the fine-tune or the held-out comparison, no interpretation checkpoint with a sample answer, no conclusion template. Cells 5 (901 lines), 9 (387), 7 (99), 3 and 11 carry no **Infrastructure** label and no `cellView: form`. The "Expect"/"Look for"/"Watch" notes state the build record's numbers rather than asking the learner to predict.
- **Consequence:** a self-paced learner gets precise explanations but no help predicting, checking their reading of the four-way comparison, or writing a bounded conclusion, and the carried modules dominate the scroll before any model runs.
- **Evidence:** source inspection; P0 `guided_markers` (how_to_use, roadmap, glossary, check_your_reasoning, what_to_notice, conclusion_template, infrastructure_label, intended_learner, predict_prompt all false; troubleshooting and look_for true), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template per NOTEBOOK_SPEC 2.2 (reference notebook §25.13): audience and how-to-use, roadmap, task contract, glossary, a learner prediction before Sections 6, 7 and 8, "What to notice" after each principal stage, collapsible "Check your reasoning" answers (for example: why WER rose while CER fell; why the validation curve plateaus), one Predict → Change one thing → Run → Observe → Explain activity with its rerun scope (see FL-m5), and a conclusion scaffold; title cells 3, 5, 7, 9 and 11 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL7, GDL9–GDL12 and GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3, 5, 7, 9 and 11 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL7, GDL9–GDL12, GDL14, UX8.

#### FL-M3 — The "without leakage" split shuffles individual lines, so test lines sit beside their own page's training lines

- **Cell/section:** cell 13 (`build_sample_dataset(corpus, seed=SPLIT_SEED)`), cell 12 prose ("seeded line-level split"), cell 0 objective ("split it by line without leakage"), cell 20 ("line-disjoint"), cell 24 Leakage advice. Generator: `samples.py` `build_sample_dataset` / `split_dataset`; `tools/notebook_template.py` lines 108, 142 and 491.
- **Observed issue:** the 800 lines are the first eight row groups of the test shard in corpus order, so neighbouring lines come from the same page and hand. `build_sample_dataset` shuffles single lines; `check_split_disjoint` only rejects pixel-identical images. The closing cell then tells the learner that "lines cut from the same page share a hand" and to split by page, writer or volume — advice the notebook's own split does not follow, and it is not disclosed that the reported held-out CER is a same-page estimate.
- **Consequence:** the learner is taught that a pixel-disjoint line split is "without leakage" and is given a held-out number whose independence assumption is not stated (SPL3) and whose grouping boundary is not preserved (SPL5). The effect on this sample is probably small because the adapted model barely reads, but the method is what the learner is told to carry to real data.
- **Evidence:** direct (P2): 0 shared images; **131 of 140** test lines have the corpus line immediately before or after them in the training split; test lines come from all eight row groups (11–23 each); 5 test transcripts and 2 validation transcripts occur verbatim in training (formulaic lines such as "Le Conseil Municipal", "Après en avoir délibéré, à l'unanimité :"). Source: cell 0 and cell 24 text (P0 `cell0_without_leakage_claim`, `cell24_split_page_advice`).
- **Recommended correction:** split by contiguous blocks (for example whole row groups or fixed runs of consecutive lines as a page proxy) so that no test line has a training neighbour, assert that property in the notebook, and state the independence assumption; or, if the line-level split is kept for size reasons, drop "without leakage" and say plainly that test lines share pages and hands with training lines, so the test CER is an optimistic same-document estimate. Re-record the build-record numbers after a change of split.
- **Acceptance check:** either the notebook asserts that no test line id is within ±1 (or the chosen block size) of a training line id and the prose describes the grouping, or the words "without leakage" no longer describe the line split and cell 20/24 state the same-page limitation explicitly.
- **Spec:** SPL3, SPL5 (MUST), SPL10.

### Minor

#### FL-m1 — Carrying the modules into one namespace silently lowers the phrase-grounding ceiling from 1000 to 512

- **Cell/section:** cell 5 (`MAX_TEXT_CHARS = 1000  # characters of caption text accepted for phrase grounding`), cell 9 (`MAX_TEXT_CHARS = 512`, the transcript bound), cell 15 ceiling print; cell 4 prose ("what you run here is what the repository tests"). Generator: `tools/build_notebook.py` module carrying (`apply_rewrites`, line 197) and the parity checks in `tests/test_notebook_parity.py` / `tools/validate_release_assets.py`.
- **Observed issue:** in the package the two constants live in separate modules. In the notebook, cell 9 rebinds the kernel global that `_check_inputs` reads at call time, while `INPUT_SCHEMA` (built in cell 5) still records `[1, 1000]`. The Section 5 ceiling print shows `MAX_TEXT_CHARS: 512` with no indication which bound it is.
- **Consequence:** in the notebook a 600-character grounding caption is rejected while the exported input manifest's schema says up to 1000 are accepted, and the package accepts it; the byte-for-byte parity guarantee does not cover cross-module name collisions.
- **Evidence:** direct (P1): only colliding top-level name across the three cells is `MAX_TEXT_CHARS`; notebook namespace `MAX_TEXT_CHARS == 512`, `INPUT_SCHEMA['text_input_chars'] == [1, 1000]`; `validate_inputs(img, '<CAPTION_TO_PHRASE_GROUNDING>', 'x'*600)` → `ValueError: text_input exceeds MAX_TEXT_CHARS=512: 600` in the notebook, accepted by the package. Documented: Kaggle cell 15 printed `'MAX_TEXT_CHARS': 512`.
- **Recommended correction:** rename one constant (for example `MAX_TRANSCRIPT_CHARS` in `samples.py`), and make the generator fail when two carried modules bind the same top-level name to different values.
- **Acceptance check:** after regeneration, executing cells 5, 7 and 9 in one namespace leaves every module constant equal to its package value; the generator check exits non-zero on a deliberately introduced collision; the ceiling print names both bounds.
- **Spec:** VAL6 (MUST), SRC2, ST5 (parity).

#### FL-m2 — BYOD: the stated minimum is 8 records but the branch needs 50, and several refusals are not actionable

- **Cell/section:** cell 0 ("at least eight images"), cell 1 ("a dataset needs 8..5,000 records"), cell 13 BYOD branch (`split_dataset` then `validate_dataset` per split with the default `min_records=8`). Generator: `tools/notebook_template.py` lines 74, 128 and 156–170; `samples.py` `split_dataset`, `load_byod_dataset`.
- **Observed issue:** `split_dataset`'s default fractions (0.15 / 0.2) give splits below 8 records for any dataset under 50 distinct images, and each split is then validated with the 8-record minimum; the error names a count but not the split. Pixel-duplicate images are dropped by `split_dataset` without a message. Two archive members with the same basename in different folders overwrite each other (the archive is flattened) and surface as "duplicate id 'x'". A cancelled upload raises a bare `StopIteration`. There is no location field, so an executor or a user with a mounted file must use the upload dialog.
- **Consequence:** a user who follows the stated contract with 8–49 lines is refused with "2 records; 8..5000 are required" for a dataset of 8 and cannot tell why; dropped duplicates change the split silently.
- **Evidence:** direct (P3, shimmed upload through cell 13): 8 → "2 records; 8..5000 are required", 20 → 4, 38 → 6, 49 → 7, 50 → accepted (32/8/10); 70 records with 10 pixel duplicates → 60 split, nothing reported; same basename in `a/` and `b/` → "duplicate id 'x'"; empty upload → `StopIteration`. Clear refusals: non-image member, missing `transcripts.csv`, 513-character transcript, non-zip.
- **Recommended correction:** state the real minimum (or size the validation/test minimums to the split, for example `min_records=1` for validation and test with a warning below a stated size), name the split in the error, report dropped duplicates (count and ids), reject duplicate basenames with a message about flattening, check for an empty upload, and add a `BYOD_PATH = ''  # @param` location field that bypasses the upload dialog.
- **Acceptance check:** a BYOD zip of the stated minimum size passes cell 13; a smaller one is refused before splitting with a message naming the minimum; a zip with duplicates prints the number dropped; a cancelled upload raises a `ValueError` naming the next action; setting `BYOD_PATH` reads the file without importing `google.colab`.
- **Spec:** DAT12, DAT19, VAL7 (MUST); EXE2, UX10.

#### FL-m3 — The held-out assertions turn a legitimate negative BYOD result into a crash before export

- **Cell/section:** cell 21 (`assert adapted_test['cer'] < frozen_test['cer']`, `assert adapted_test['cer'] < baseline_empty['cer']`), cell 22 prose ("the adapted rows should read the cursive the frozen rows left blank"). Generator: `tools/notebook_template.py` lines 389–390 and 399.
- **Observed issue:** the assertions encode the expected result on the Belfort sample, but the same cell runs for BYOD. On a set where the frozen model already reads well (printed text, for instance), a bounded decoder fine-tune can legitimately match or worsen the frozen CER; the cell then raises a bare `AssertionError` after writing the evaluation report but before the panels, the export and the reload.
- **Consequence:** the BYOD branch promised to reach "artifact export and reload-parity cells" stops on an honest negative result with no explanation; a negative result is treated as a failure instead of being reported.
- **Evidence:** source inspection (P0 `cell21_hard_asserts`); not executed (no model run in this review).
- **Recommended correction:** keep the assertions for the default sample only (or turn them into a reported verdict such as `adapted_beats_frozen: false` with an explanatory message), and soften the cell 22 wording to what the panels can show.
- **Acceptance check:** with `USE_BYOD = True` and a dataset on which the adapted CER is not lower than the frozen CER, cells 21–23 complete, the report and result JSON record the negative comparison, and the adapter is still exported and reloaded.
- **Spec:** DAT13 (MUST), RUN9, UX10.

#### FL-m4 — Expected outputs that do not match the run, and no statement of run-to-run variability

- **Cell/section:** cell 14 ("The multi-capability card recorded the square labelled `flag` and an exact OCR"), cells 18 and 20 (build-record numbers to three decimals), `docs/release-verification.md` step 5 and `MODEL_CARD.md` build record ("66 tensors"). Generator: `tools/notebook_template.py` lines 217 and 322.
- **Observed issue:** the float32 pipeline yields one `poster` box covering the whole drawing (`[0, 0, 511, 511]`) before and after adaptation, as cell 22 itself says; cell 14 still sets up a `flag` square from the superseded float16 notebook. Cells 18 and 20 quote the build record's epoch-by-epoch CER and loss without saying that the GPU fine-tune is not bit-reproducible (the release record says "a Kaggle number a few hundredths off the build record is the expected spread"). The artifact has 106 tensors, not 66.
- **Consequence:** a learner checking their output against cell 14 sees a mismatch and cannot tell whether something is wrong; a learner whose fine-tune lands a few hundredths away has no way to know that is expected.
- **Evidence:** documented (Kaggle executed notebook: frozen and adapted detections `('poster', [0, 0, 511, 511])`; artifact `tensors: 106`); source (P0 `cell14_mentions_flag: true`, `run_to_run_variability: false`). 106 = 4 BART decoder layers × 26 tensors + 2 for the embedding layer norm.
- **Recommended correction:** describe the current `<OD>` behaviour in cell 14; add one sentence to Sections 7 and 8 that the fine-tune's numbers vary by GPU and that beam search is deterministic only for fixed weights, device and dtype; correct the tensor count in the docs.
- **Acceptance check:** no cell mentions `flag`; Sections 7 or 8 state the expected spread; `grep -n "66 tensors" docs MODEL_CARD.md` returns nothing.
- **Spec:** ENV8 (MUST), GDL8.

#### FL-m5 — The suggested experiments lack rerun scope, and one cannot run as written

- **Cell/section:** cell 24 "Optional experiments"; cell 19 (`pipe.adapt` on the shared `pipe`), cell 15 `run_capabilities`. Generator: `tools/notebook_template.py` lines 496–499.
- **Observed issue:** changing `LEARNING_RATE`/`EPOCHS` and rerunning cell 19 adapts the already adapted `pipe`: `adapt` snapshots the current trainable weights as its restore point and labels epoch 0 "frozen model", so the second run starts from the first run's weights. The experiment "add `<CAPTION_TO_PHRASE_GROUNDING>` to `CAPABILITIES` with the frozen caption as its `text_input`" cannot work, because `run_capabilities` never passes a `text_input` and `run` then raises "requires a non-empty text_input". No experiment says which cells to rerun or asks for a prediction.
- **Consequence:** the first suggested experiment, done the obvious way, compounds two fine-tunes and mislabels the starting point; the grounding experiment ends in an error.
- **Evidence:** source inspection (`pipeline.py` `adapt`: `frozen_state` cloned from the current weights, `"note": "frozen model"` on epoch 0, no refusal when `self.adapter` is set; `_check_inputs` requires `text_input` for `TASKS_WITH_TEXT`). Not executed.
- **Recommended correction:** give each experiment its rerun scope (for example "rerun from Section 3" to reload the frozen model, or refuse `adapt` on an already adapted pipeline with a message), pass `text_input` through `run_capabilities` when a task needs one, and frame one experiment as Predict → Change one thing → Run → Observe → Explain.
- **Acceptance check:** following an experiment's written instructions exactly produces a valid run whose epoch 0 is the pinned frozen model; the grounding experiment completes and records its `text_input`.
- **Spec:** UX7 (MUST: exercises must not leave the notebook inconsistent), GDL10, UX5.

### Suggestions

- **FL-S1 — Declare the current spec.** The notebook, `tutorials/README.md` and the validator declare NOTEBOOK_SPEC 2.0; regenerate against 2.2 when the template is revised.
- **FL-S2 — Print the fine-tune seed.** `adapt` uses `seed=0` and records it in the report, but cell 19's form and summary do not show it; expose it beside `EPOCHS`/`LEARNING_RATE`/`BATCH_SIZE`.
- **FL-S3 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it to skip the install, but no markdown mentions it (EXE5).
- **FL-S4 — Show the per-line CER distribution** (or the worst and best panels) beside the corpus rates, so the learner can see where the 0.797 comes from.

## 6. Readiness

**Needs revision.** Open Majors FL-M1 to FL-M3; unresolved MUSTs RUN1, RUN10, ENV6, REL2, REL11 (FL-M1), SPL3, SPL5 (FL-M3), VAL6 (FL-m1), DAT12, DAT19, VAL7 (FL-m2), DAT13 (FL-m3), ENV8 (FL-m4), UX7 (FL-m5). The repository's `Release-grade` status rests on a restart-dependent run and should return to `Candidate`. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob, a re-recorded held-out result on a grouped (or explicitly qualified) split, and hosted BYOD positive and negative runs that reach export.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, data stages only):** the Belfort fetch, validation, refusal probes and split (identical to Kaggle), the split-adjacency and duplicate-transcript counts (P2), the `MAX_TEXT_CHARS` collision and its effect (P1), the BYOD branch of cell 13 with eleven archives (P3), and metric sanity (P4).
- **Verified from documented execution:** the install-cell restart and every model-stage number quoted here (Kaggle T4, reviewed blob).
- **Inferred from source:** the Section 7 rerun behaviour and the grounding experiment error (FL-m5), the BYOD negative-result crash (FL-m3), Colab behaviour and the real upload dialog.
- **Most likely to be wrong:** FL-M3's severity. Adjacent line ids are a proxy for "same page"; the parquet carries no page field, and the adapted model reads so little that the leakage may not move the test CER at all. The finding rests on the "without leakage" claim and the notebook's own page advice, which hold whatever the effect size; it may fairly be judged Minor.

Probe ZIP: `florence2_vision_language_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
