"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from phi4_multimodal_pipeline import (
    INPUT_SCHEMA,
    MAX_IMAGES,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    Phi4MultimodalPipeline,
    evaluation_report,
    validate_inputs,
)

AUDIO = ([0.0] * 16_000, 16_000)


def _result(text: str, images: int = 0, audios: int = 0) -> dict:
    return {"text": text, "image_count": images, "audio_count": audios, "temperature": 0.0}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    image = Image.new("RGB", (64, 48))
    manifest = validate_inputs("describe", images=[image], audios=[AUDIO], names=["img", "clip"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["inputs"] == [
        {"id": "img", "kind": "image", "mode": "RGB", "size": [64, 48]},
        {"id": "clip", "kind": "audio", "samples": 16_000, "sampling_rate": 16_000, "seconds": 1.0},
    ]
    assert (manifest["image_count"], manifest["audio_count"]) == (1, 1)
    assert manifest["prompt_chars"] > manifest["instruction_chars"]
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_text_only_default_ids() -> None:
    manifest = validate_inputs("hello")
    assert manifest["inputs"] == []
    manifest = validate_inputs("hello", images=[Image.new("L", (8, 8))])
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]


def test_validate_inputs_rejects_like_generate() -> None:
    pipe = Phi4MultimodalPipeline(lambda *a: "x", "cpu", "eager")
    too_many = [Image.new("RGB", (8, 8))] * (MAX_IMAGES + 1)
    with pytest.raises(ValueError, match="at most 4 images"):
        validate_inputs("describe", images=too_many)
    with pytest.raises(ValueError, match="at most 4 images"):
        pipe.generate("describe", images=too_many)
    with pytest.raises(ValueError, match="instruction must be non-empty"):
        validate_inputs("   ")
    with pytest.raises(ValueError, match="max_new_tokens must be between 1 and 2048"):
        validate_inputs("hi", max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        pipe.generate("hi", temperature=2.5)
    with pytest.raises(ValueError, match="names must have one entry per media input"):
        validate_inputs("hi", images=[Image.new("RGB", (8, 8))], names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(
        {
            "text": _result("An answer."),
            "image": _result("A sign.", images=1),
            "audio": _result("", audios=1),
        },
        sample_kind="synthetic",
    )
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_capabilities"] == 3
    assert [c["non_empty"] for c in report["capabilities"]] == [True, True, False]
    assert "word error rate" in report["needs"]
    assert "VQA" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_empty_results() -> None:
    report = evaluation_report({})
    assert report["verdict"] == "not-measurable"
    assert report["capabilities"] == []
