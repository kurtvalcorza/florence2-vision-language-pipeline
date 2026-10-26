"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
florence2: Prerequisites build-record timing and Section 7 spread expectation. Follow-up: md0 and
Troubleshooting decimal-spread expectations removed.
Fails on PR head b1e6e9e.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/florence2_vision_language_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'the build record measured',
    'about 20 minutes of cell time',
    'expect your epoch rates to differ',
    'the kept epoch to move by one',
    "your GPU's numbers can differ in the second or third decimal",
    'differences in the second or third decimal are expected',
]
PRESENT = [
    'the header gives the recorded Colab T4 timings',
    'can differ from the recorded runs; read your own epoch history.',
    'so your numbers can differ from the recorded runs',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
