# Weight provenance and DIMER hosting

- Upstream: `florence-community/Florence-2-large` — the "official transformers converted checkpoint" (pinned README) of `microsoft/Florence-2-large`
- Immutable revision: `4271c66b88cdbc05735372ec13b2360108de5317`
- Weight format: SafeTensors (`model.safetensors`, 1553541016 bytes, float16 as shipped)
- Upstream weight license: MIT (the community card's `license_link` points to Microsoft's licence file; the snapshot itself carries no `LICENSE` file)
- Local snapshot: `weights/florence-2-large-community/` with `dimer-base-manifest.json` (12 entries, per-file bytes + SHA-256, `totalBytes` 1558929750); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/florence2_vision_language_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `stage_missing_files()` fetches only manifest-listed files at the pinned revision; `from_pretrained` additionally refuses a checkpoint whose keys are missing, unexpected or mismatched against the native architecture (`output_loading_info`).
- DIMER hosting: MIT permits use, modification, distribution, sublicensing and commercial use subject to preservation of the copyright and permission notice; DIMER may mirror the pinned checkpoint under the upstream license.
- Loader trust boundary: `transformers==4.57.6` native `Florence2ForConditionalGeneration` and `Florence2Processor`, `trust_remote_code=False`, `local_files_only=True`; no custom code in the snapshot (the `auto_map` line left in `preprocessor_config.json` is never consulted because the explicit native classes are used). Hub download is opt-in and pinned to the revision above.

## Pin history

- 2026-09-12, rejected: `microsoft/Florence-2-large` @ `21a599d414c4d928c9032694c424fb94458e3594`. That snapshot ships `configuration_florence2.py` / `modeling_florence2.py` / `processing_florence2.py` and can only be loaded with the `trust_remote_code` opt-in; the native classes accept its directory but leave 918 tensors unused and re-initialise 920 (measured), so it was re-pinned to the community conversion above by the repository owner.

## Adaptation artifacts and the sample corpus

- Dtype: the pipeline loads the float16 checkpoint in float32 on every device (the adapter is trained in float32 and overlays without a cast); the vision tower runs at most `VISION_BATCH_SIZE = 4` images per forward.
- Adapter artifacts: `save_artifact` writes the trained tensors (the last four BART decoder layers and the decoder's embedding layer norm, 66 tensors, about 269 MB in float32) as `adapter.safetensors` beside a `manifest.json` (`org.valcorza.florence-2-large-community.adapter.v1`) naming the base model id and revision, the base `model.safetensors` SHA-256, the tensor names, the file size and SHA-256, the task (`<OCR>`), the training configuration and the epoch history. `from_artifact` re-verifies the base snapshot and checks the manifest, the digest and the exact tensor set before deserialising; an adapter is derived from the lines it was trained on and is not redistributed by this repository.
- Adaptation sample: `Teklia/Belfort-line` (MIT), parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb`, `default/test/0000.parquet` row groups 0–7 (800 lines, about 44 MB) read over HTTPS range requests, each row group pinned by SHA-256 and byte total in `samples.py`; cached git-ignored under `weights/belfort/`, never vendored.
