"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Training time: Colab T4 `ccd411d4` 1,100.7 s (2026-10-08) and Colab T4 `c79997b9` 1,135.1 s (2026-10-25);
session walls 2,221.4 s and 2,294.3 s, inside the stated 35-45 minute estimate.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "phi4_multimodal_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_training_time_is_approximate_and_names_the_runs():
    md = _markdown()
    assert "1,100.7 s" not in md and "1135.1" not in md and "1,135.1" not in md
    assert "about 18–19 minutes on a Colab T4" in md and "`c79997b9` on 2026-10-25" in md


def test_stated_wall_estimate_contains_the_recorded_runs():
    md = _markdown()
    assert "35–45 minutes on a T4" in md  # 37.0 and 38.2 minutes recorded
