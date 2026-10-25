"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
florence2: Prerequisites build-record timing and Section 7 spread expectation.
Fails on PR head 24c2dce.
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
]
PRESENT = [
    'the header gives the recorded Colab T4 timings',
    'can differ from the recorded runs; read your own epoch history.',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
