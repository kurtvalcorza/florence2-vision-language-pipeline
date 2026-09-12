# Florence-2 Vision-Language Pipeline

DIMER inference wrapper for **`microsoft/Florence-2-large`** — prompt-driven captioning, OCR, object detection, dense region captioning and phrase grounding — pinned to an immutable Hugging Face revision and verified against a digest manifest before any load.

**Status: Blocked — upstream pin decision.** The pinned snapshot can only be loaded by executing its bundled modeling code (`trust_remote_code`), which this package refuses; `from_pretrained` verifies the snapshot and then raises `RuntimeError`. See `STATUS.md` for the two options and `MODEL_CARD.md` for the measured evidence.

## Upstream alignment

- Model: `microsoft/Florence-2-large`
- Revision: `21a599d414c4d928c9032694c424fb94458e3594`
- Upstream weight license: MIT
- Upstream task: image + task prompt → text / boxes (`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<OCR>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>`)
- Repository adaptation: **none**; inference contract only, no forward pass executed in v0.1.0

## Quick start

```python
from PIL import Image
from florence2_vision_language_pipeline import Florence2Pipeline, TASKS, character_error_rate

pipe = Florence2Pipeline.from_pretrained(device="cpu")   # v0.1.0: verifies the snapshot, then raises RuntimeError
result = pipe.run(Image.open("page.png"), "<OCR>")
print(result["result"], result["generation"])
print(character_error_rate("expected text", result["result"]))
```

The contract (`run`, validation, output fields) is exercised today only through an injected runner: `Florence2Pipeline(runner, "cpu")` where `runner(image, prompt, task, max_new_tokens, num_beams)` returns `{"text": ..., "parsed": ...}`.

## Weights layout

```
weights/florence-2-large/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # declares auto_map -> custom code (refused)
  configuration_florence2.py / modeling_florence2.py / processing_florence2.py   # custom code, refused
  preprocessor_config.json, generation_config.json, tokenizer.json, tokenizer_config.json, vocab.json, LICENSE, README.md
  model.safetensors          # 1553563458 bytes, git-ignored
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to continue if any file is missing or its SHA-256 differs from the manifest. With a verified snapshot it then detects the custom code (`remote_code_files()`) and raises `RuntimeError` naming the pending owner decision; `allow_download=True` without a manifest raises the same. To stage the snapshot: `hf download microsoft/Florence-2-large --revision 21a599d414c4d928c9032694c424fb94458e3594 --local-dir weights/florence-2-large`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed; 16 tests incl. the remote-code refusal
python -c "from florence2_vision_language_pipeline import *; print(len(verify_snapshot()['files']), remote_code_files())"
# -> 12 ['configuration_florence2.py', 'modeling_florence2.py', 'processing_florence2.py', 'config.json:auto_map']
```

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card, including the native-load investigation record
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and trust boundary
- [`STATUS.md`](STATUS.md) — release status and the pin decision

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights and bundled code are MIT; see `docs/WEIGHTS.md`.
