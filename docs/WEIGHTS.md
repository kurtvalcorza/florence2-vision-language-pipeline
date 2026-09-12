# Weight provenance and DIMER hosting

- Upstream: `microsoft/Florence-2-large`
- Immutable revision: `21a599d414c4d928c9032694c424fb94458e3594`
- Weight format: SafeTensors (`model.safetensors`, 1553563458 bytes, float16 as shipped)
- Upstream weight license: MIT (snapshot `LICENSE`, Microsoft Corporation)
- Local snapshot: `weights/florence-2-large/` with `dimer-base-manifest.json` (12 entries, per-file bytes + SHA-256, `totalBytes` 1556231245); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/florence2_vision_language_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `stage_missing_files()` fetches only manifest-listed files at the pinned revision.
- DIMER hosting: MIT permits use, modification, distribution, sublicensing and commercial use subject to preservation of the copyright and permission notice; DIMER may mirror the pinned checkpoint under the upstream license once a loader is wired.
- Loader trust boundary: **none wired in v0.1.0.** The snapshot carries `configuration_florence2.py`, `modeling_florence2.py` and `processing_florence2.py` (each digest-listed in the manifest) and `config.json` declares an `auto_map`; loading requires the `trust_remote_code` opt-in, which this package refuses (`REMOTE_CODE_POLICY = "refuse"`). `from_pretrained` verifies the snapshot, then raises `RuntimeError`. See `STATUS.md` for the two options before the repository owner.
