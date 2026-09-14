# Florence-2 Large — Colab T4 execution evidence

Execution date (UTC): 2026-09-13. Outcome: **PASS — 8/8 unchanged code cells**, with GPU device `cuda:0`.

| Provenance | Value |
|---|---|
| Tested repository commit | `ece9e7f72e5915c7e84208415c0e803955e1b774` |
| Source notebook Git blob | `e807053d55c0849b2436ff85bfc0e8c194f1eca4` |
| Source notebook SHA-256 | `4f2058c965b8900efd02ed9b5ffee0e00e76e9cfb6f3d4a62c47a2c0b47a8d60` |
| Embedded source revision | `a5a7c2a73f65acfc053bc3ee73ecefa8c40fa32b` |
| Model | `florence-community/Florence-2-large@4271c66b88cdbc05735372ec13b2360108de5317` |
| GPU / driver | `Tesla T4, 15360 MiB, 580.82.07` |
| Runtime | `{"device": "cuda:0", "dtype": "float16", "pillow": "11.3.0", "python": "3.12.3", "source": "local-snapshot", "torch": "2.14.0+cu130", "transformers": "4.57.6"}` |
| Sum of code-cell wall times | 159.258 s |
| Total with environment setup and bookkeeping | 163.882 s |

## Execution method

Colab CLI 0.6.0 ran a driver from WSL `claude-science` on one Tesla T4 VM. The driver created a separate Python 3.12.3 virtual environment for this notebook and launched a new interpreter. Each original code cell was executed sequentially with `exec(compile(...))`; source cells and default form parameters were unchanged. The executor source is retained as [executor-source.txt](executor-source.txt), an evidence artifact rather than repository tooling. The VM had no repository checkout. The weights directory and isolated Hugging Face cache were empty before this notebook ran; all snapshot entries were downloaded and SHA-256 verified by the embedded pipeline.

The hosted Colab kernel used Python 3.13.15. An earlier direct `.ipynb` attempt was aborted during installation after that mismatch was confirmed. The successful result here uses the repository-supported Python 3.12 interpreter. A completed native hosted-kernel run is not claimed.

CLI transport prerequisite: PyPI `jupyter-kernel-client==1.0.2` lacked `KernelClient`; the CLI environment used Google’s fork at `f18e982c3265df5e923aa9def101ab3fd737e139` (distribution 0.8.0). This affects the CLI host, not the notebook runtime pins.

## Observations

- `<CAPTION>`: `"a red and blue circle with the word dimer 2026 on it"` (3.764 s).
- `<OD>`: `{"bboxes": [[0, 0, 511, 511]], "labels": ["poster"]}` (0.371 s).
- `<OCR>`: `"DIMER 2026"` (0.31 s).

All five structural sanity checks passed. OCR exactly reproduced `DIMER 2026` (sample character error rate 0.0). The caption omitted the red square; object detection labelled the entire image as `poster`. These are observations, not caption/detection accuracy evidence. The evaluation verdict is `sample-sanity` for OCR and `not-measurable` for captioning and detection.

## Verification and retained files

The read-back checks confirmed the exact source hash and Git blob, unchanged code cells, eight error-free executed cells, runtime pins, CUDA inference, accepted inputs, and the recorded rejection probe. Model-specific sanity checks and exported artifacts were checked after download. See [execution-record.json](execution-record.json) for the individual checks and per-cell timings.

- [Executed notebook](florence2_vision_language_colab_output.ipynb)
- [Execution log](execution.log)
- [Installed distributions](packages-after.json)
- [Artifact SHA-256 manifest](artifacts.sha256)
- [florence2_vision_language_evaluation_report.json](outputs/florence2_vision_language_evaluation_report.json)
- [florence2_vision_language_input_manifest.json](outputs/florence2_vision_language_input_manifest.json)
- [florence2_vision_language_preview.png](outputs/florence2_vision_language_preview.png)
- [florence2_vision_language_result.json](outputs/florence2_vision_language_result.json)

The CLI stopped the shared runtime after all four notebook tests; a subsequent `colab sessions` call returned no active sessions. Both cleanup outputs are retained in the execution record. Repository release status remains **Candidate** pending evidence review; this record does not promote it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
