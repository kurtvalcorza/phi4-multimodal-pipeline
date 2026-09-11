from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
MODEL_LICENSE = "MIT"


def _build_prompt(instruction: str, image_count: int, audio_count: int) -> str:
    if not instruction or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if image_count > 4 or audio_count > 4:
        raise ValueError(
            "DIMER reference path allows at most 4 images and 4 audio clips per request"
        )
    media = "".join(f"<|image_{index}|>" for index in range(1, image_count + 1))
    media += "".join(f"<|audio_{index}|>" for index in range(1, audio_count + 1))
    return f"<|user|>{media}{instruction.strip()}<|end|><|assistant|>"


@dataclass
class Phi4MultimodalPipeline:
    _generate: Callable[..., str]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        *,
        allow_remote_code: bool = False,
        device: str = "cuda",
    ) -> Phi4MultimodalPipeline:
        if not allow_remote_code:
            raise RuntimeError(
                "Phi-4-multimodal requires upstream custom Python model code. "
                "Set allow_remote_code=True only after reviewing the pinned revision trust boundary."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; choose an appropriate GPU runtime"
            )

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device if device != "cuda" else "cuda:0",
        ).eval()
        generation_config = GenerationConfig.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )

        def runner(
            prompt: str,
            images: Sequence[Any] | None,
            audios: Sequence[Any] | None,
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            processor_kwargs: dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = list(images)
            if audios:
                processor_kwargs["audios"] = list(audios)
            inputs = processor(**processor_kwargs).to(model.device)

            do_sample = temperature > 0
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "generation_config": generation_config,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
            generated = model.generate(**inputs, **generation_kwargs)
            generated = generated[:, inputs["input_ids"].shape[1] :]
            return processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        return cls(runner, device)

    def generate(
        self,
        instruction: str,
        *,
        images: Sequence[Any] | None = None,
        audios: Sequence[Any] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        image_list = list(images or [])
        audio_list = list(audios or [])
        prompt = _build_prompt(instruction, len(image_list), len(audio_list))
        text = self._generate(
            prompt,
            image_list or None,
            audio_list or None,
            max_new_tokens,
            temperature,
        )
        return {
            "text": text,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "image_count": len(image_list),
            "audio_count": len(audio_list),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
