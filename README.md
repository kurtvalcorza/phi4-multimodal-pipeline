# Phi-4 Multimodal Pipeline

DIMER-oriented inference wrapper for **Microsoft Phi-4-multimodal-instruct**. It exposes text generation, image-conditioned generation, audio-conditioned generation, and combined image+audio prompting through one normalized public API while pinning the exact upstream snapshot.

## Upstream alignment

- Model: `microsoft/Phi-4-multimodal-instruct`
- Revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight license: MIT
- Inputs: text, image, audio; output: generated text
- Upstream size/context: approximately 5.6B parameters; 128K-token context reported upstream
- Adaptation in this repository: none; inference only

## Trust boundary

The upstream model requires custom Python code. This wrapper therefore refuses `from_pretrained()` unless the caller explicitly passes `allow_remote_code=True`; the executed code is pinned to the immutable model revision above. This is materially different from the standard-code Whisper path.

## Public API

```python
from phi4_multimodal_pipeline import Phi4MultimodalPipeline
pipe = Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, device="cuda")
print(pipe.generate("Explain why the sky appears blue.")["text"])
```

## Tutorial

`tutorials/phi4_multimodal_colab.ipynb` is `MULTI-CAPABILITY`: text, image+text, and audio+text are demonstrated separately so their input/output contracts remain explicit.

## Release status

**Candidate.** The full snapshot is heavyweight and GPU-oriented; clean GPU notebook execution must be recorded against the exact release commit before promotion. Static CI is not runtime evidence.
