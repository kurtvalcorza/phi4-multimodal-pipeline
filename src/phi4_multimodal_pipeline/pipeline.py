"""Text-, image- and audio-conditioned generation with the pinned ``microsoft/Phi-4-multimodal-instruct``.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision. **Trust boundary
(MOD9/MOD10):** this upstream release ships its model, configuration and processor as custom Python
files (``configuration_phi4mm.py``, ``modeling_phi4mm.py``, ``processing_phi4mm.py``,
``speech_conformer_encoder.py``, ``vision_siglip_navit.py``) that transformers must execute, so the
loader passes ``trust_remote_code=True``. It refuses to do so unless the caller opts in with
``allow_remote_code=True``, and on the snapshot path every one of those files is pinned by SHA-256 in
the manifest and re-hashed by ``verify_snapshot`` before it is imported; the weights are SafeTensors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
MODEL_LICENSE = "MIT"
MODEL_KEY = "phi4-multimodal-instruct"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
CONFIG_FILE = "config.json"
# Upstream Python files that transformers executes when trust_remote_code=True (pinned by digest).
REMOTE_CODE_FILES = (
    "configuration_phi4mm.py",
    "modeling_phi4mm.py",
    "processing_phi4mm.py",
    "speech_conformer_encoder.py",
    "vision_siglip_navit.py",
)
# MOD9/MOD10: why the loader enables remote code, and what bounds it (the fleet card gate keys on this name).
REMOTE_CODE_JUSTIFICATION = (
    "upstream ships Phi4MMForCausalLM / Phi4MMProcessor / Phi4MMConfig only as custom Python files in the "
    "model repository (config.json auto_map); transformers==4.48.2 has no built-in implementation, so "
    "trust_remote_code=True is unavoidable. Bounded by: the explicit allow_remote_code=True opt-in, the "
    "immutable MODEL_REVISION on every path, and on the snapshot path the per-file SHA-256 pins of "
    "REMOTE_CODE_FILES re-hashed by verify_snapshot before transformers imports them."
)
SUPPORTED_ATTENTION = {"eager", "flash_attention_2"}
MAX_IMAGES = 4  # images per request on the DIMER reference path
MAX_AUDIOS = 4  # audio clips per request on the DIMER reference path
MAX_NEW_TOKENS = 2048
MAX_TEMPERATURE = 2.0
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.0  # greedy decoding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    listed = {entry["path"] for entry in manifest.get("files", [])}
    missing_code = [name for name in REMOTE_CODE_FILES if name not in listed]
    if missing_code:
        raise ValueError(f"manifest does not pin the remote-code files {missing_code}; refusing")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest and the
    small config/tokenizer files but git-ignores the shards, the remote code and the sample media).
    Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _build_prompt(instruction: str, image_count: int, audio_count: int) -> str:
    if not instruction or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if image_count > MAX_IMAGES or audio_count > MAX_AUDIOS:
        raise ValueError(
            "DIMER reference path allows at most 4 images and 4 audio clips per request"
        )
    media = "".join(f"<|image_{index}|>" for index in range(1, image_count + 1))
    media += "".join(f"<|audio_{index}|>" for index in range(1, audio_count + 1))
    return f"<|user|>{media}{instruction.strip()}<|end|><|assistant|>"


INPUT_SCHEMA: dict[str, Any] = {
    "instruction": "non-empty text; the user turn of the chat template",
    "images": f"optional sequence of PIL images, at most {MAX_IMAGES}",
    "audios": f"optional sequence of (float waveform, sampling_rate) tuples, at most {MAX_AUDIOS}",
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "temperature": [0.0, MAX_TEMPERATURE],
    "decoding": "temperature 0 → greedy (do_sample=False); otherwise sampling at that temperature",
    "preprocessing": (
        "the upstream processor builds the <|user|>…<|end|><|assistant|> prompt, tiles images for the SigLIP "
        "encoder and computes speech features; nothing is altered by this module"
    ),
}


def _check_inputs(
    instruction: str,
    images: Sequence[Any] | None,
    audios: Sequence[Any] | None,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[Any], list[Any], str]:
    """Raise ValueError naming the first violated ceiling; return the media lists and the prompt."""
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError("max_new_tokens must be between 1 and 2048")
    if not 0.0 <= temperature <= MAX_TEMPERATURE:
        raise ValueError("temperature must be between 0 and 2")
    image_list = list(images or [])
    audio_list = list(audios or [])
    prompt = _build_prompt(instruction, len(image_list), len(audio_list))
    return image_list, audio_list, prompt


def _observe_image(image: Any) -> dict[str, Any]:
    size = getattr(image, "size", None)
    return {"kind": "image", "mode": getattr(image, "mode", None), "size": list(size) if size else None}


def _observe_audio(audio: Any) -> dict[str, Any]:
    if isinstance(audio, tuple | list) and len(audio) == 2:
        waveform, rate = audio
        samples = len(waveform) if hasattr(waveform, "__len__") else None
        has_rate = isinstance(rate, int | float) and bool(rate)
        seconds = round(samples / rate, 3) if samples is not None and has_rate else None
        return {"kind": "audio", "samples": samples, "sampling_rate": rate, "seconds": seconds}
    return {"kind": "audio", "samples": None, "sampling_rate": None, "seconds": None}


def validate_inputs(
    instruction: str,
    *,
    images: Sequence[Any] | None = None,
    audios: Sequence[Any] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed media, request, verdict).

    Rejection is reported by raising exactly as ``generate`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    image_list, audio_list, prompt = _check_inputs(
        instruction, images, audios, max_new_tokens, temperature
    )
    observed = [_observe_image(image) for image in image_list]
    observed += [_observe_audio(audio) for audio in audio_list]
    if names is not None and len(names) != len(observed):
        raise ValueError("names must have one entry per media input")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"{item['kind']}-{i}", **item} for i, item in enumerate(observed)
        ],
        "instruction_chars": len(instruction.strip()),
        "prompt_chars": len(prompt),
        "image_count": len(image_list),
        "audio_count": len(audio_list),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Mapping[str, Mapping[str, Any]], *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though nothing is measurable.

    ``results`` maps a capability label (``text``, ``image``, ``audio``) to a ``generate`` result. The
    repository ships no metric helper for open-ended generation, so the verdict is always
    ``not-measurable`` (EVAL9) and the report states, per capability, what labelled data would make it
    measurable; the generated texts are sanity evidence that each path executed.
    """
    needs = {
        "text": "reference answers and a task metric (exact match or a rubric) on instruction-answer pairs",
        "image": "image-question pairs with reference answers (VQA accuracy) or reference captions (CIDEr)",
        "audio": "reference transcripts scored with a word error rate after a stated normalisation",
    }
    capabilities = []
    for label, result in results.items():
        text = str(result.get("text", ""))
        capabilities.append(
            {
                "capability": label,
                "generated_chars": len(text),
                "non_empty": bool(text.strip()),
                "image_count": int(result.get("image_count", 0)),
                "audio_count": int(result.get("audio_count", 0)),
                "temperature": result.get("temperature"),
                "needs": needs.get(label, "labelled references for this capability"),
            }
        )
    return {
        "task": "multimodal instruction-following text generation",
        "score_semantics": "generated text; the pipeline exposes no likelihood, confidence or threshold",
        "sample_kind": sample_kind,
        "n_capabilities": len(capabilities),
        "capabilities": capabilities,
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "no reference answers, captions or transcripts were supplied and the repository ships no metric "
            "helper for open-ended generation; non-empty output only shows that each capability path executed"
        ),
        "needs": "; ".join(f"{c['capability']}: {c['needs']}" for c in capabilities) or "labelled references",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class Phi4MultimodalPipeline:
    _generate: Callable[..., str]
    device: str
    attention_implementation: str = "eager"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        *,
        allow_remote_code: bool = False,
        device: str = "cuda",
        attention_implementation: str = "eager",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Phi4MultimodalPipeline:
        if not allow_remote_code:
            raise RuntimeError(
                "Phi-4-multimodal requires upstream custom Python model code. "
                "Set allow_remote_code=True only after reviewing the pinned revision trust boundary."
            )
        if attention_implementation not in SUPPORTED_ATTENTION:
            raise ValueError(
                "attention_implementation must be 'eager' or 'flash_attention_2'"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; choose an appropriate GPU runtime"
            )
        if attention_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "flash_attention_2 was requested but flash-attn is not installed; "
                    "install a compatible flash-attn build or use attention_implementation='eager'"
                ) from exc

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            # A directory argument makes transformers read the config, the (digest-verified) remote
            # code, the tokenizer and the SafeTensors shards from it directly — no Hub resolution.
            location: dict[str, Any] = {"pretrained_model_name_or_path": str(root)}
            source = "local-snapshot"
        elif allow_download:
            location = {"pretrained_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION}
            source = "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )

        processor = AutoProcessor.from_pretrained(**location, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            **location,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
            _attn_implementation=attention_implementation,
        ).eval()
        generation_config = GenerationConfig.from_pretrained(**location)

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

        return cls(runner, device, attention_implementation, source)

    def generate(
        self,
        instruction: str,
        *,
        images: Sequence[Any] | None = None,
        audios: Sequence[Any] | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict[str, Any]:
        image_list, audio_list, prompt = _check_inputs(
            instruction, images, audios, max_new_tokens, temperature
        )
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
            "attention_implementation": self.attention_implementation,
            "device": self.device,
            "source": self.source,
        }
