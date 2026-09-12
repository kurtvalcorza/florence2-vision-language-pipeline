# Release status

Current status: **Blocked — upstream pin decision**. The package, offline unit tests (16), `MODEL_CARD.md` (MODEL_CARD_SPEC 1.1) and provenance documents exist and pass the static gate, but no forward pass of the model has been executed: the pinned `microsoft/Florence-2-large` snapshot (`21a599d414c4d928c9032694c424fb94458e3594`) can only be loaded by executing its bundled `modeling_florence2.py` / `processing_florence2.py` (`trust_remote_code`), which this package refuses (`REMOTE_CODE_POLICY = "refuse"`), and the native `transformers==4.57.6` Florence-2 classes do not match the checkpoint's tensor layout (918 unused / 920 re-initialised tensors, measured 2026-09-12).

Decision pending with the repository owner:

- (a) keep the `microsoft/Florence-2-large` pin and allow the snapshot's custom code at the pinned, digest-verified revision (each `.py` file's SHA-256 is in the manifest and in `MODEL_CARD.md`), adding a `REMOTE_CODE_JUSTIFICATION` constant; or
- (b) re-pin to the native port `florence-community/Florence-2-large` (Hub revision `4271c66b88cdbc05735372ec13b2360108de5317` as read on 2026-09-12, license `mit`, 776.7 M parameters, no `custom_code` tag, not downloaded), which the native classes load without remote code; this changes `MODEL_ID`, `MODEL_REVISION`, the weights key and every digest.

No tutorial notebook exists yet; the card pass only is complete.
