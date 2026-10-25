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
    # Historical wording (2026-10-26): the exact recorded times are allowed again, labelled with their runs.
    assert "between 18.3 and 19.2 minutes on a Colab T4" in md
    assert "`c79997b9` and `99878ae5` on 2026-10-25" in md


def test_stated_wall_estimate_contains_the_recorded_runs():
    md = _markdown()
    assert "between 37.0 and 38.3 minutes" in md  # 2,221.4 s, 2,294.3 s, 2,287.5 s recorded


def test_historical_wording_records_runs_and_drops_expectations():
    """Historical wording (2026-10-26): timings and run numbers are records of named runs, never estimates or
    expectations; unprinted GPU-memory figures name their true source. Fails on PR #12 head e23f5fe."""
    md = _markdown()
    for stale in ("35–45 minutes", "about 18–19 minutes", "quarter of an hour", "Expect the frozen model",
                  "about 9–10 GB", "(about 6.8 GB) and peaks near 9.2 GB"):
        assert stale not in md, stale
    for record in ("1,153.2 s", "2,287.5 s for `99878ae5`", "`MODEL_CARD.md` gives about 6.8 GB",
                   "which epoch is kept can vary between runs", "this can vary between runs"):
        assert record in md, record
