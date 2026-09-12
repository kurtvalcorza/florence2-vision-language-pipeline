# florence2-vision-language-pipeline

DIMER pipeline scaffold for **microsoft/Florence-2-large** — OCR / Captioning / Grounding.

| | |
|---|---|
| Upstream model | [`microsoft/Florence-2-large`](https://huggingface.co/microsoft/Florence-2-large) |
| Pinned revision | `21a599d414c4d928c9032694c424fb94458e3594` (resolved 2026-09-12) |
| Upstream license | `mit` (verified on the Hub 2026-09-12; re-check at the pinned revision before release) |
| Weight files to stage | `model.safetensors + *.py custom code` |
| Status | scaffold only — no weights downloaded, no pipeline code yet |

Weights are staged under `weights/` and are git-ignored. This repository follows the
MODEL_CARD_SPEC 1.0 / NOTEBOOK_SPEC 1.0 conventions used by the other `*-pipeline` repos.
