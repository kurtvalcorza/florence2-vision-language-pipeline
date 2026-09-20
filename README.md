# Florence-2 Vision-Language Pipeline

DIMER pipeline for **`florence-community/Florence-2-large`** — the native-`transformers` conversion of Microsoft's Florence-2-large — for prompt-driven captioning, OCR, object detection, dense region captioning and phrase grounding, pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot with no remote code. On top of inference it carries the **adaptation contract for the `<OCR>` task**: corpus-level character and word error rates over transcribed lines, two non-adapted baselines, a bounded fine-tuning of the last four BART decoder layers on cached encoder outputs, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `florence-community/Florence-2-large`
- Revision: `4271c66b88cdbc05735372ec13b2360108de5317`
- Upstream weight license: MIT (Microsoft; the community card links Microsoft's licence file)
- Upstream task: image + task prompt → text / boxes (`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>`)
- Repository adaptation: **bounded supervised fine-tuning of the `<OCR>` task** — the last four of the 12 BART decoder layers and the decoder's embedding layer norm (67,188,736 of 776,505,344 parameters) on `{id, image, text}` line records with the sequence-to-sequence loss over the transcript tokens; the DaViT vision tower, the projector, the BART encoder, the shared embeddings and the first eight decoder layers stay frozen. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base. The other eight task tokens are inference-only; the tuned decoder serves them too, which the tutorial shows on one drawing and the card records.

## Quick start

```python
from PIL import Image
from florence2_vision_language_pipeline import Florence2Pipeline, character_error_rate

pipe = Florence2Pipeline.from_pretrained()                # cuda:0 when visible, else cpu; float32 everywhere
caption = pipe.run(Image.open("photo.jpg"), "<CAPTION>")
boxes = pipe.run(Image.open("photo.jpg"), "<OD>")
ocr = pipe.run(Image.open("page.png"), "<OCR>", max_new_tokens=1024)
print(caption["result"], boxes["result"]["bboxes"], boxes["result"]["labels"])
print(character_error_rate("expected text", ocr["result"]))

from florence2_vision_language_pipeline import fetch_sample_dataset
splits = fetch_sample_dataset()                    # 800 digest-pinned Belfort handwritten lines, 600 / 60 / 140
print(pipe.evaluate(splits["test"])["cer"])        # frozen <OCR> corpus CER (about 1.0: the model emits nothing)
pipe.adapt(splits["train"], splits["validation"])  # last four decoder layers, lowest-validation-CER epoch kept
print(pipe.evaluate(splits["test"])["cer"])
pipe.save_artifact("outputs/adapter")
again = Florence2Pipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

`result` is a string for caption/OCR tasks and `{"bboxes", "labels"}` in input-image pixel coordinates for region tasks; no confidence scores are emitted. `transcribe` and `evaluate` run `<OCR>` in batches (every `<OCR>` prompt is the same 587 tokens, so a batch needs no padding). Image ceilings: sides within 1..16,384 px and at most 4096² pixels (the processor resizes everything to 768×768, so the ceilings bound decode and resize memory, not model cost).

## Adaptation contract

- **Records:** `{id, image, text}` — a PIL image (sides within the ceilings) and its transcript (1..512 characters after whitespace runs are collapsed); `validate_dataset` checks the structure, `split_dataset` de-duplicates by decoded pixels and `check_split_disjoint` asserts no image is shared. The default sample (`samples.py`) is the first eight parquet row groups of the Belfort-line test shard (`Teklia/Belfort-line`, MIT; nineteenth-century French council minutes in cursive) read over HTTPS range requests at an immutable Hub revision, each row group refused on any SHA-256 or byte-total mismatch — the same digest-pinned sample and split as the sibling `got-ocr2-pipeline` row; `load_byod_dataset` reads a zip or directory of line images plus `transcripts.csv`.
- **Measures (`metrics.py`):** `ocr_metrics` — micro CER and WER (total edits over total reference characters or words), macro rates, exact match, and the hypothesis length; uncapped, so a rate above 1.0 means the model generates text the line does not carry. `empty_baseline` (CER 1.0 by construction) and `constant_baseline` (the medoid training transcript for every line).
- **Fine-tuning:** `adapt(train, val, *, epochs=6, lr=5e-5, batch_size=8, seed=0)` caches the encoder output (the DaViT features and the `<OCR>` prompt through the BART encoder, 587 × 1024 per line) for every training line, then trains decoder layers 8–11 and `layernorm_embedding` on those states with the model's own sequence-to-sequence loss, AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the loss equals the full model's loss exactly. Epoch 0 records the frozen validation rates; the epoch with the lowest validation CER is kept; on any exception the frozen weights are restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 269 MB) + `manifest.json` (`org.valcorza.florence-2-large-community.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, task, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen `<OCR>` CER 0.992 / WER 1.002 on the 140 held-out lines (the model emits a dash or nothing for cursive), adapted **0.797** / **1.012** (epoch 6 of 6, 0 lines exact), reload parity 8/8; the drawing's three capabilities after adaptation: before adaptation caption `a red and blue circle with the word dimer 2026 on it`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000); after adaptation caption `Dim 2026 logo with a red and blue circle.`, 1 box(es) labelled `poster`, OCR `DIMER 2026` (CER 0.000). One seeded split of one 800-line sample; no dispersion estimate. The validation curve plateaus near 0.80 at every rate tried (1e-4 and 2e-4 for eight epochs: 0.798 and 0.807), so the frozen encoder, not the optimiser, bounds a decoder-only adapter here; the sibling GOT-OCR 2.0 row reached 0.759 on the same split.

## Weights layout

```
weights/florence-2-large-community/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # Florence2ForConditionalGeneration, model_type florence2 (native classes)
  preprocessor_config.json   # 768x768 resize, ImageNet mean/std, 577 image tokens
  processor_config.json, generation_config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
  added_tokens.json, special_tokens_map.json, README.md
  model.safetensors          # 1553541016 bytes, git-ignored
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any file is missing or its SHA-256 differs from the manifest; the snapshot is then loaded through the native `Florence2ForConditionalGeneration` / `Florence2Processor` classes with `local_files_only=True` and `trust_remote_code=False`, and the load is refused if any tensor is missing, unexpected or mismatched. Without a snapshot, `allow_download=True` loads from the Hub at `revision=4271c66b88cdbc05735372ec13b2360108de5317`; the default is to refuse. To stage the snapshot: `hf download florence-community/Florence-2-large --revision 4271c66b88cdbc05735372ec13b2360108de5317 --local-dir weights/florence-2-large-community`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed (tests/test_model_backed.py runs only where the snapshot is staged)
python -c "from PIL import Image, ImageDraw; from florence2_vision_language_pipeline import Florence2Pipeline; im = Image.new('RGB', (256, 256), 'white'); ImageDraw.Draw(im).rectangle([64, 64, 192, 192], fill='red'); p = Florence2Pipeline.from_pretrained(device='cpu'); print(p.run(im, '<CAPTION>')['result'], p.run(im, '<OD>')['result'])"
```

Measured on CPU (float32, Windows venv, 2026-09-12): load 7.11 s; `<CAPTION>` 4.91 s → "a red square with a white background"; `<OD>` 9.02 s → one box `[63, 63, 193, 193]` labelled "flag".

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/florence2-vision-language-pipeline/blob/main/tutorials/florence2_vision_language_colab.ipynb)

`tutorials/florence2_vision_language_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the eight pinned Belfort row groups and splits the 800 lines 600 / 60 / 140, runs `<CAPTION>`, `<OD>` and `<OCR>` on a synthetic drawing through the inference contract, measures the frozen `<OCR>` CER and WER on the held-out lines beside the empty and constant baselines, runs `adapt` with validation-CER epoch selection, scores the held-out lines again, writes six line panels and re-runs the three capabilities on the drawing with the adapted model, and exports the adapter and reloads it with verified transcript parity. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `035c2c02` (committed at `bc7e9b8`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 1298.6 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance, trust boundary and pin history
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; the Belfort-line sample is MIT and is not redistributed; see `docs/WEIGHTS.md`.

Current source update: snapshot validation now runs before model-library imports (Kokoro also validates the language first), so rejected requests fail with the intended validation error even when model libraries are absent. The standalone notebook was regenerated from this source. The retained 2026-09-13 GPU run identifies the earlier notebook blob; the regenerated notebook has not had a fresh GPU execution. Status remains **Candidate**.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
