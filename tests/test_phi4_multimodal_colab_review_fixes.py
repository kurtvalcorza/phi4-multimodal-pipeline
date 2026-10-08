"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings on phi4_multimodal_colab (PHI-M1..M5,
PHI-m1..m5). CI dependencies only: notebook fragments are executed with stand-ins, never the model."""
# ruff: noqa: E501  -- notebook source fragments are kept on single lines

from __future__ import annotations

import contextlib
import io
import json
import textwrap
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from phi4_multimodal_pipeline.samples import MIN_RECORDS, split_dataset, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "phi4_multimodal_colab.ipynb"
GUARD_HEAD = "if globals().get('pipe') is None or pipe.adapter is not None:"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [s for s in _code(nb) if marker in s]
    assert len(found) == 1, marker
    return found[0]


def test_phi_m1_no_kernel_install_and_spec_22(nb: dict) -> None:
    """PHI-M1 / PHI-m5: nothing is pip-installed into the kernel, no restart instruction, spec 2.2 declared."""
    code = "\n".join(_code(nb))
    assert "sys.executable, '-m', 'pip'" not in code and "Restart the runtime" not in code
    assert "--require-hashes" in code and "--only-binary" in code
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"


def test_phi_m2_frozen_is_not_described_as_close_to_the_baselines(nb: dict) -> None:
    """PHI-M2: no learner-facing cell says the frozen score is close to the baselines; the wide-margin reading is stated."""
    md = _markdown(nb)
    for gone in ("close to the baselines", "close to a caption", "fluently but at length", "fluently and at length"):
        assert gone not in md, gone
    assert md.count("wide margin") >= 3  # Section 6 expectation, interpretation, Baselines-first rule


@pytest.mark.parametrize("marker", ["sign = synthetic_sign()", "baseline_constant = constant_caption_baseline(", "adapt_result = pipe.adapt("])
def test_phi_m3_reruns_start_from_the_frozen_snapshot_and_free_the_reloaded_copy(nb: dict, marker: str) -> None:
    """PHI-M3: Sections 5, 6 and 7 reload the frozen pipeline when `pipe` was adapted or freed, and drop Section 9's
    `reloaded` copy first (two 4-bit copies do not fit a 16 GB card)."""
    source = _cell_with(nb, marker)
    rest = source.split(GUARD_HEAD, 1)[1]
    end = rest.index("print({'reloaded_frozen_pipeline'")
    block = GUARD_HEAD + rest[: rest.index("\n", end)]
    loads = []

    class _Pipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            loads.append(kwargs)
            return types.SimpleNamespace(adapter=None, fresh=True)

    torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"Phi4MultimodalPipeline": _Pipeline, "WEIGHTS_DIR": "w", "torch": torch_stub, "reloaded": object()}  # after Section 9: pipe deleted
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(block, "<guard>", "exec"), ns)
    assert ns["pipe"].fresh and "reloaded" not in ns and loads[0]["quantization"] == "nf4"
    ns = {"pipe": types.SimpleNamespace(adapter=None), "Phi4MultimodalPipeline": _Pipeline, "WEIGHTS_DIR": "w", "torch": torch_stub}
    exec(compile(block, "<guard>", "exec"), ns)
    assert len(loads) == 1, "a frozen pipeline on the default first pass is kept"


def test_phi_m3_rerun_instructions_name_the_sections(nb: dict) -> None:
    md = _markdown(nb)
    assert "run Sections 7, 8 and 9 again in order" in md
    assert "re-run from that cell **through to the end**" in md


def _jpegs(directory: Path, n: int) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        path = directory / f"p{i:03d}.jpg"
        Image.new("RGB", (32, 24), (i * 5 % 255, 40, 90)).save(path, "JPEG")
        records.append({"id": f"p{i:03d}", "image": str(path), "captions": [f"a photo number {i}"]})
    return records


def test_phi_m4_stated_byod_minimum_is_the_real_edge(nb: dict, tmp_path: Path) -> None:
    """PHI-M4: the stated minimum (50) passes every split's validation at the default fractions; 49 does not."""
    cell = _cell_with(nb, "BYOD_PATH = ''")
    assert "BYOD_MIN_PHOTOGRAPHS = 50" in cell
    records = _jpegs(tmp_path / "img", 50)
    splits = split_dataset(records, seed=42)
    for part in splits.values():
        validate_dataset(part)
    short = split_dataset(records[:49], seed=42)
    assert min(len(p) for p in short.values()) < MIN_RECORDS
    md = _markdown(nb)
    assert "at least 50 photographs" in md and "at least eight photographs" not in md


def test_phi_m4_split_failure_names_the_split(nb: dict, tmp_path: Path) -> None:
    cell = _cell_with(nb, "BYOD_PATH = ''")
    start = cell.index("dataset_manifests = {}")
    loop = cell[start : cell.index("splits = {name: manifest['records']", start)]
    records = _jpegs(tmp_path / "img", 49)
    ns = {"splits": split_dataset(records, seed=42), "validate_dataset": validate_dataset}
    with pytest.raises(ValueError, match=r"^validation split \(7 of 49 records\): 7 records; 8\.\.5000"):
        exec(compile(loop, "<splits>", "exec"), ns)
    assert "photographs; at least {BYOD_MIN_PHOTOGRAPHS} are needed" in cell


def _extract(nb: dict, tmp_path: Path, entries: dict[str, bytes], limit: int = 2 * 1024**3) -> Path:
    cell = _cell_with(nb, "BYOD_PATH = ''")
    start = cell.index("    with zipfile.ZipFile(io.BytesIO(payload)) as archive:")
    block = textwrap.dedent(cell[start : cell.index("    records_file = ", start)])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    byod_dir = tmp_path / "byod"
    byod_dir.mkdir(parents=True)
    ns = {"zipfile": zipfile, "io": io, "payload": buf.getvalue(), "file_name": "mine.zip", "byod_dir": byod_dir, "BYOD_MAX_EXPANDED_BYTES": limit}
    exec(compile(block, "<extract>", "exec"), ns)
    return byod_dir


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"../../evil.txt": b"x", "records.jsonl": b""}, "climbs out"),
        ({"/abs/evil.txt": b"x"}, "absolute path"),
        ({"a/p.jpg": b"1", "b/p.jpg": b"2"}, "share the file name"),
    ],
)
def test_phi_m3_bad_archives_are_rejected_by_name(nb: dict, tmp_path: Path, entries: dict, message: str) -> None:
    """PHI-m3: traversal, absolute and duplicate-basename entries are refused with named errors."""
    with pytest.raises(ValueError, match=message):
        _extract(nb, tmp_path, entries)


def test_phi_m3_expanded_size_is_capped_and_good_zip_extracts(nb: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expands to"):
        _extract(nb, tmp_path / "a", {"p.jpg": b"x" * 100}, limit=10)
    out = _extract(nb, tmp_path / "b", {"photos/p.jpg": b"1", "records.jsonl": b"{}", "__MACOSX/._p.jpg": b"", ".hidden": b""})
    assert sorted(p.name for p in out.iterdir()) == ["p.jpg", "records.jsonl"]


def test_phi_m5_no_bare_assert_in_learner_cells(nb: dict) -> None:
    """PHI-M5: the outcome comparisons are verdicts; the one hard check (reload parity) raises with a message."""
    learner = [s for s in _code(nb) if "# dimer: kernel cell" not in s and "def verify_snapshot" not in s and "def caption_metrics" not in s and "def validate_dataset" not in s]
    for source in learner:
        assert not any(line.lstrip().startswith("assert ") for line in source.splitlines()), source[:80]
    cell = _cell_with(nb, "reloaded = Phi4MultimodalPipeline.from_artifact(")
    assert "Reload parity failed" in cell


def test_phi_m1_m2_stale_statements_gone(nb: dict) -> None:
    """PHI-m1 / PHI-m2: the five stale statements and the 'more literal' reading are gone; hallucination is named."""
    md = _markdown(nb)
    for gone in ("caption for caption", "400 M parameters", "about 45 minutes", "more literal"):
        assert gone not in md, gone
    assert "369,098,752" in md and "What to notice:** check every adapted caption" in md
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "none; inference only" not in readme
    for doc in ("STATUS.md", "docs/release-verification.md"):
        assert "53 tests" not in (ROOT / doc).read_text(encoding="utf-8"), doc
