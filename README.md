# Phi-4 Multimodal Pipeline

DIMER-oriented inference wrapper for **Microsoft Phi-4-multimodal-instruct**. It exposes text generation, image-conditioned generation, audio-conditioned generation, and combined image+audio prompting through one normalized public API while pinning the exact upstream snapshot.

## Upstream alignment

- Model: `microsoft/Phi-4-multimodal-instruct`
- Revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight license: MIT
- Inputs: text, image, audio; output: generated text
- Upstream size/context: approximately 5.6B parameters; 128K-token context reported upstream
- Adaptation in this repository: none; inference only

## Trust and runtime boundaries

The upstream model requires custom Python code. This wrapper therefore refuses `from_pretrained()` unless the caller explicitly passes `allow_remote_code=True`; the executed code is pinned to the immutable model revision above. This is materially different from the standard-code Whisper path.

The DIMER reference loader uses `attention_implementation="eager"` by default so a clean runtime does not silently depend on FlashAttention 2. Upstream also supports `flash_attention_2`; this wrapper exposes it only as an explicit opt-in and fails clearly when the compatible `flash-attn` package is absent. The release-reference tutorial requires CUDA because the full model snapshot is heavyweight.

## Public API

```python
from phi4_multimodal_pipeline import Phi4MultimodalPipeline

pipe = Phi4MultimodalPipeline.from_pretrained(
    allow_remote_code=True,
    device="cuda",
    attention_implementation="eager",
)
print(pipe.generate("Explain why the sky appears blue.")["text"])
```

For an explicitly optimized environment with a compatible FlashAttention 2 installation, set `attention_implementation="flash_attention_2"`.

## Tutorial

`tutorials/phi4_multimodal_colab.ipynb` is `MULTI-CAPABILITY`: text, image+text, and audio+text are demonstrated separately so their input/output contracts remain explicit. It self-bootstraps the repository in a fresh runtime and records the tested repository SHA in exported provenance.

## Release status

**Candidate.** The full snapshot is heavyweight and GPU-oriented; clean GPU notebook execution must be recorded against the exact release commit before promotion. Static CI is not runtime evidence.
