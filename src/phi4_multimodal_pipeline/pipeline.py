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

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

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
WEIGHT_FILE = "model-00001-of-00003.safetensors"  # the shard whose digest names the base in adapter manifests
QUANTIZATIONS = (None, "nf4")
# Modules kept in fp16 under NF4: the output head, the token embeddings and the vision/audio towers and projectors.
NF4_SKIP_MODULES = ("lm_head", "embed_tokens", "embed_tokens_extend", "image_embed", "audio_embed", "img_processor", "encoder")
VISION_LORA_PATTERN = r"^model\.layers\.(\d+)\..*\.lora_(A|B)\.vision\.weight$"
DEFAULT_TRAINED_LAYERS = 8  # the built-in vision-LoRA tensors of the last N decoder layers are what `adapt` trains
CAPTION_MAX_NEW_TOKENS = 40
MIN_SCORED_RECORDS = 50
MAX_EVAL_RECORDS = 5_000
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
# Captioning adaptation contract (shared with the fleet's captioning rows).
MAX_PREFIX_CHARS = 128
MIN_IMAGE_SIDE = 16
MAX_IMAGE_SIDE = 4096
CAPTION_INSTRUCTION = "Describe this photo in one short sentence."
_PUNCT_RE = re.compile(r"[^\w\s]")


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


def normalize_caption(text: str) -> str:
    """COCO-caption-style normalisation: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def caption_tokens(text: str) -> list[str]:
    return normalize_caption(text).split()


def unigram_f1(prediction: str, references: Sequence[str]) -> float:
    """Bag-of-words F1 between the normalised prediction and the best-matching reference (a plumbing
    check, not a captioning metric; `metrics.py` has BLEU-4 / ROUGE-L / CIDEr-D)."""
    if not references:
        raise ValueError("references must contain at least one caption")
    pred = caption_tokens(prediction)
    best = 0.0
    for reference in references:
        ref = caption_tokens(reference)
        if not pred or not ref:
            continue
        ref_counts: dict[str, int] = {}
        for token in ref:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def validate_image(image: Any) -> Image.Image:
    """A PIL image within the side limits, converted to RGB (the captioning records' image check)."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


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
    quantization: str | None = None
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        allow_remote_code: bool = False,
        device: str = "cuda",
        attention_implementation: str = "eager",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        quantization: str | None = None,
    ) -> Phi4MultimodalPipeline:
        """Load the processor and model. ``quantization="nf4"`` loads the language model's linear layers as
        4-bit NF4 (bitsandbytes, fp16 compute) so the 5.6 B model fits a 16 GB accelerator for adaptation;
        the vision encoder, its projector and the embeddings stay in fp16."""
        if quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}")
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

        weight_sha256 = None
        if source == "local-snapshot":
            with open(root / MANIFEST_NAME, encoding="utf-8") as handle:
                entries = json.load(handle).get("files", [])
            weight_sha256 = next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)
        processor = AutoProcessor.from_pretrained(**location, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if quantization == "nf4":
            import torch
            from transformers import BitsAndBytesConfig

            load_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=list(NF4_SKIP_MODULES),
                ),
            }
        model = AutoModelForCausalLM.from_pretrained(
            **location,
            trust_remote_code=True,
            device_map=device,
            _attn_implementation=attention_implementation,
            **load_kwargs,
        ).eval()
        for param in (model.parameters() if hasattr(model, "parameters") else ()):  # tests stub the model
            param.requires_grad_(False)
        model_name = location.get("pretrained_model_name_or_path", MODEL_ID)
        gen_kwargs = {k: v for k, v in location.items() if k != "pretrained_model_name_or_path"}
        try:
            generation_config = GenerationConfig.from_pretrained(model_name, **gen_kwargs)
        except Exception:
            generation_config = getattr(model, "generation_config", None)

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

        return cls(runner, device, attention_implementation, source, quantization, model, processor, weight_sha256)

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

    # ------------------------------------------------------------------ captioning adaptation contract

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("no model loaded: construct with from_pretrained or from_artifact")
        return self._model, self._processor

    def caption(
        self,
        image: Image.Image,
        *,
        instruction: str = CAPTION_INSTRUCTION,
        max_new_tokens: int = CAPTION_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """One greedy caption for one image through the vision LoRA (the model's image path)."""
        checked = validate_image(image)
        model, _processor = self._require_model()
        model.set_lora_adapter("vision")
        result = self.generate(instruction, images=[checked], max_new_tokens=max_new_tokens, temperature=0.0)
        return {"caption": result["text"], "instruction": instruction, "max_new_tokens": max_new_tokens, "adapted": self.adapter is not None, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = CAPTION_MAX_NEW_TOKENS,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Caption every record's image (greedy) and score the predictions against its reference captions
        (BLEU-4, ROUGE-L, CIDEr-D, unigram F1)."""
        from .metrics import caption_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = []
        for index, record in enumerate(checked):
            with Image.open(record["image"]) as image:
                image.load()
                predictions.append(self.caption(image, max_new_tokens=max_new_tokens)["caption"])
            if progress is not None:
                progress(index + 1, len(checked))
        metrics = caption_metrics(predictions, [[str(c) for c in r["captions"]] for r in checked])
        metrics.update(
            {
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trained_layers: int) -> list[str]:
        """The built-in vision-LoRA tensors (A and B of every LoRA-wrapped projection) of the last
        `trained_layers` decoder layers — the checkpoint's own adaptation parameters, nothing new is added."""
        model, _processor = self._require_model()
        n_layers = len(model.model.layers)
        if isinstance(trained_layers, bool) or not isinstance(trained_layers, int) or not 1 <= trained_layers <= n_layers:
            raise ValueError(f"trained_layers must be an int in 1..{n_layers}")
        pattern = re.compile(VISION_LORA_PATTERN)
        names = []
        for name, _param in model.named_parameters():
            match = pattern.match(name)
            if match and int(match.group(1)) >= n_layers - trained_layers:
                names.append(name)
        if not names:
            raise RuntimeError("no vision-LoRA tensors found; the loaded model is not the pinned Phi-4-multimodal")
        return names

    def _encode_pair(self, image: Image.Image, instruction: str, answer: str | None) -> tuple[Any, int]:
        """Processor inputs for the chat-formatted prompt (and answer, when training) plus the prompt length in tokens."""
        _model, processor = self._require_model()
        prompt = _build_prompt(instruction, 1, 0)
        prompt_len = processor(text=prompt, images=[image], return_tensors="pt")["input_ids"].shape[1]
        text = prompt if answer is None else prompt + answer + "<|end|>"
        return processor(text=text, images=[image], return_tensors="pt"), prompt_len

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 4e-5,
        trained_layers: int = DEFAULT_TRAINED_LAYERS,
        grad_accumulation: int = 4,
        instruction: str = CAPTION_INSTRUCTION,
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the checkpoint's **vision LoRA** in the last `trained_layers` decoder layers
        (the upstream vision recipe, narrowed): the 4-bit or fp16 base, the vision tower, the projector and the
        other LoRA tensors stay frozen. One (image, reference caption) sample per image per epoch — the
        reference rotates with the epoch — as the chat-formatted `<|user|><|image_1|>instruction<|end|>
        <|assistant|>caption<|end|>` with the loss on the caption tokens only; AdamW (no weight decay) on fp32
        masters of the trained tensors, gradient accumulation over `grad_accumulation` images, clipping at 1.0,
        seeded shuffling, no scheduler. Epoch 0 records the frozen model's validation metrics; the epoch with
        the highest validation CIDEr-D is kept (the final one without a validation split). On any exception the
        frozen tensors are restored."""
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-3:
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(grad_accumulation, bool) or not isinstance(grad_accumulation, int) or not 1 <= grad_accumulation <= 64:
            raise ValueError("grad_accumulation must be an int in 1..64")
        model, _processor = self._require_model()  # refuse before importing torch
        from .samples import validate_dataset

        names = self._trainable_names(trained_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import torch

        name_set = set(names)
        params_by_name = {n: p for n, p in model.named_parameters() if n in name_set}
        frozen_state = {n: p.detach().clone() for n, p in params_by_name.items()}
        frozen_dtypes = {n: p.dtype for n, p in params_by_name.items()}
        previous_adapter = self.adapter
        history: list[dict[str, Any]] = []
        started = time.perf_counter()
        device = model.device

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            scored = self.evaluate(val_checked)
            return {k: scored[k] for k in ("bleu4", "rouge_l", "cider_d", "unigram_f1", "n") if k in scored}

        def _restore(state: Mapping[str, Any]) -> None:
            with torch.no_grad():
                for name, param in params_by_name.items():
                    param.data = state[name].to(device=param.device, dtype=frozen_dtypes[name]).clone()
                    param.requires_grad_(False)

        try:
            model.set_lora_adapter("vision")
            for param in model.parameters():
                param.requires_grad_(False)
            params = []
            for param in params_by_name.values():
                param.data = param.data.float()  # fp32 master for the trained tensors
                param.requires_grad_(True)
                params.append(param)
            n_trainable = sum(p.numel() for p in params)
            use_cache = getattr(model.config, "use_cache", None)
            model.config.use_cache = False
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (entry["val"] or {}).get("cider_d", -1.0)
            best_state = {n: p.detach().clone() for n, p in params_by_name.items()}
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(train_checked)
                rng.shuffle(order)
                losses: list[float] = []
                optimizer.zero_grad(set_to_none=True)
                for index, record in enumerate(order):
                    references = [str(c) for c in record["captions"]]
                    answer = references[(epoch - 1) % len(references)]
                    with Image.open(record["image"]) as image:
                        image.load()
                        encoded, prompt_len = self._encode_pair(image.convert("RGB"), instruction, answer)
                    encoded = encoded.to(device)
                    labels = encoded["input_ids"].clone()
                    labels[:, :prompt_len] = -100
                    output = model(**encoded, labels=labels)
                    loss = output.loss / grad_accumulation
                    loss.backward()
                    losses.append(float(output.loss.detach()))
                    if (index + 1) % grad_accumulation == 0 or index + 1 == len(order):
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                model.eval()
                entry = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 6), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["cider_d"] > best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("cider_d", -1.0)
                    best_state = {n: p.detach().clone() for n, p in params_by_name.items()}
            _restore(best_state)
            model.eval()
            if use_cache is not None:
                model.config.use_cache = use_cache
        except BaseException:
            _restore(frozen_state)
            model.eval()
            self.adapter = previous_adapter
            raise
        self.adapter = {
            "task": "image captioning",
            "trained": "built-in vision LoRA of the last decoder layers (A/B tensors)",
            "trained_layers": trained_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "quantization": self.quantization,
            "instruction": instruction,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation CIDEr-D" if val_checked is not None else "final epoch (no validation split)",
            "loss": "causal cross-entropy on the caption tokens only (prompt tokens masked)",
            "lr": float(lr),
            "grad_accumulation": grad_accumulation,
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained vision-LoRA tensors as safetensors plus a manifest naming the base, the digests,
        the licence and the training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()
        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        import torch
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        params = dict(model.named_parameters())
        tensors = {name: params[name].detach().to(torch.float16).cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "license": MODEL_LICENSE,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256, "remote_code_files": list(REMOTE_CODE_FILES), "quantization": self.quantization},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "tensor_dtype": "float16",
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest,
        licence and exact tensor set. Refuses tensors outside the recorded vision-LoRA scope."""
        model, _processor = self._require_model()
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        layers = manifest.get("adapter", {}).get("trained_layers")
        expected = sorted(self._trainable_names(layers))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the manifest")
        params = dict(model.named_parameters())
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(params[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(params[name].shape)}")
        import torch

        with torch.no_grad():
            for name, tensor in tensors.items():
                params[name].data = tensor.to(device=params[name].device, dtype=params[name].dtype).clone()
                params[name].requires_grad_(False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        allow_remote_code: bool = False,
        device: str = "cuda",
        attention_implementation: str = "eager",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        quantization: str | None = None,
    ) -> Phi4MultimodalPipeline:
        """Load the verified base snapshot, then overlay the adapter (verified before deserialising)."""
        pipe = cls.from_pretrained(allow_remote_code=allow_remote_code, device=device, attention_implementation=attention_implementation, weights_dir=weights_dir, allow_download=allow_download, quantization=quantization)
        pipe.load_artifact(artifact_dir)
        return pipe


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if (base.get("model_id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("artifact was trained on a different base model or revision")
    if base_sha256 and base.get("weight_sha256") and base["weight_sha256"] != base_sha256:
        raise ValueError("artifact base weight digest does not match the loaded snapshot")
    if manifest.get("license") != MODEL_LICENSE:
        raise ValueError(f"artifact licence {manifest.get('license')!r} != {MODEL_LICENSE!r}")
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    if ADAPTER_WEIGHTS not in files:
        raise ValueError(f"artifact manifest does not list {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    if weights.stat().st_size != files[ADAPTER_WEIGHTS]["bytes"]:
        raise ValueError("artifact weights size does not match the manifest")
    if _sha256(weights) != files[ADAPTER_WEIGHTS]["sha256"]:
        raise ValueError("artifact weights digest does not match the manifest")
