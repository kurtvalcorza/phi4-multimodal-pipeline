import pytest

from phi4_multimodal_pipeline import Phi4MultimodalPipeline


def test_remote_code_requires_opt_in():
    with pytest.raises(RuntimeError):
        Phi4MultimodalPipeline.from_pretrained()


def test_invalid_attention_rejected_before_model_imports():
    with pytest.raises(ValueError, match="attention_implementation"):
        Phi4MultimodalPipeline.from_pretrained(
            allow_remote_code=True,
            attention_implementation="auto",
        )


def test_prompt_contract():
    seen = {}

    def fake(prompt, images, audios, max_new_tokens, temperature):
        seen["prompt"] = prompt
        return "answer"

    pipeline = Phi4MultimodalPipeline(fake, "cuda", "eager")
    result = pipeline.generate("describe", images=[object()], audios=[object()])
    assert "<|image_1|><|audio_1|>describe" in seen["prompt"]
    assert result["image_count"] == 1
    assert result["audio_count"] == 1
    assert result["attention_implementation"] == "eager"
