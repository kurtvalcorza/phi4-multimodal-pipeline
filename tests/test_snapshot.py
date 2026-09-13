"""Offline tests for the fleet snapshot scheme: manifest-driven verification, staging, loading."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from phi4_multimodal_pipeline import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    REMOTE_CODE_FILES,
    Phi4MultimodalPipeline,
    stage_missing_files,
    verify_snapshot,
)
from phi4_multimodal_pipeline.pipeline import MANIFEST_NAME

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "weights" / MODEL_KEY / MANIFEST_NAME
LOADER_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    *REMOTE_CODE_FILES,
}
FILES = {
    "config.json": b'{"model_type": "phi4mm"}\n',
    **{name: f"# {name}\n".encode() for name in REMOTE_CODE_FILES},
    "model-00001-of-00003.safetensors": b"\x00" * 64,
}


def _write_snapshot(root: Path, *, files: dict[str, bytes] | None = None, model_id: str = MODEL_ID) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    files = FILES if files is None else files
    entries = []
    for name, payload in files.items():
        entries.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _materialise(root: Path, files: dict[str, bytes] | None = None) -> None:
    for name, payload in (FILES if files is None else files).items():
        (root / name).write_bytes(payload)


class _Calls:
    def __init__(self) -> None:
        self.processor: list = []
        self.model: list = []
        self.generation: list = []


def _stub_stack(monkeypatch, calls: _Calls, cuda: bool = True) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.processor.append((args, kwargs))
            return object()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.model.append((args, kwargs))
            return types.SimpleNamespace(eval=lambda: "model")

    class GenerationConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.generation.append((args, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=AutoProcessor,
            AutoModelForCausalLM=AutoModelForCausalLM,
            GenerationConfig=GenerationConfig,
        ),
    )


def test_committed_manifest_pins_identity_loader_files_and_remote_code() -> None:
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    identity = (manifest["modelId"], manifest["revision"], manifest["modelKey"])
    assert identity == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths >= LOADER_FILES, "every file the remote-code loader reads must be pinned"
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    _materialise(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["path"] == str(tmp_path)
    assert result["files"] == manifest["files"]


def test_verify_snapshot_rejects_tampered_remote_code(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "modeling_phi4mm.py").write_bytes(b"# modeling_phi4mm.pz\n")  # same size, different bytes
    with pytest.raises(ValueError, match="modeling_phi4mm.py: sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_manifest_without_remote_code_pins(tmp_path: Path) -> None:
    files = {"config.json": FILES["config.json"], "model-00001-of-00003.safetensors": b"\x00" * 64}
    _write_snapshot(tmp_path, files=files)
    _materialise(tmp_path, files)
    with pytest.raises(ValueError, match="does not pin the remote-code files"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, model_id="someone/else")
    _materialise(tmp_path)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    present = {k: v for k, v in FILES.items() if k != "model-00001-of-00003.safetensors"}
    _materialise(tmp_path, present)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched: list[str] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(FILES[relative_path])

    shard = "model-00001-of-00003.safetensors"
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == [shard]
    assert fetched == [shard]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_loads_the_verified_directory(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    calls = _Calls()
    _stub_stack(monkeypatch, calls)
    pipe = Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=tmp_path)
    assert (pipe.source, pipe.device, pipe.attention_implementation) == ("local-snapshot", "cuda", "eager")
    location = {"pretrained_model_name_or_path": str(tmp_path), "trust_remote_code": True}
    assert calls.processor == [((), location)]
    assert calls.model[0][1]["pretrained_model_name_or_path"] == str(tmp_path)
    assert calls.model[0][1]["_attn_implementation"] == "eager"
    assert calls.generation == [((), {"pretrained_model_name_or_path": str(tmp_path)})]


def test_from_pretrained_hub_path_only_with_allow_download(monkeypatch, tmp_path: Path) -> None:
    calls = _Calls()
    _stub_stack(monkeypatch, calls)
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=tmp_path)
    pipe = Phi4MultimodalPipeline.from_pretrained(
        allow_remote_code=True, weights_dir=tmp_path, allow_download=True
    )
    assert pipe.source == "hf-hub"
    assert calls.model[0][1]["revision"] == MODEL_REVISION
    assert calls.processor[0][1]["pretrained_model_name_or_path"] == MODEL_ID


def test_from_pretrained_still_refuses_without_remote_code_opt_in(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    _stub_stack(monkeypatch, _Calls())
    with pytest.raises(RuntimeError, match="allow_remote_code=True"):
        Phi4MultimodalPipeline.from_pretrained(weights_dir=tmp_path)


def test_from_pretrained_refuses_a_tampered_snapshot(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "config.json").write_bytes(b'{"model_type": "other!"}\n')
    _stub_stack(monkeypatch, _Calls())
    with pytest.raises(ValueError, match="config.json"):
        Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=tmp_path)
