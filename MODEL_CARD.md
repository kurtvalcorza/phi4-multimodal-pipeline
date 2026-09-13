---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: microsoft/Phi-4-multimodal-instruct
---

# Phi-4 Multimodal Instruct (DIMER package v0.1.0) — Multimodal Language Model (Text, Image & Audio)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2FPhi--4--multimodal--instruct-ffcc4d?style=flat)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2503.01743-b31b1b.svg)](https://arxiv.org/abs/2503.01743)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/LICENSE)
[![Pipeline](https://img.shields.io/badge/Pipeline-phi4--multimodal--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/phi4-multimodal-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Multi-Capability Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb) [`phi4_multimodal_colab.ipynb`](https://github.com/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb)  
  *Text-, image-, and audio-conditioned text generation with one pinned Phi-4 Multimodal checkpoint, including the custom-code trust-boundary acknowledgement; no adaptation occurs.*

> [!NOTE]
> Requires a fresh CUDA runtime with enough memory for the roughly 13 GB model snapshot; the notebook does not run on CPU.

---

###### Description

Phi-4-multimodal-instruct is Microsoft's approximately 5.6B-parameter multimodal transformer that accepts text, image, and audio inputs and generates text, packaged here from `microsoft/Phi-4-multimodal-instruct` at immutable revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`. Upstream reports a Phi-4-Mini-Instruct language backbone with vision and speech encoders/adapters and a 128K-token context. This repository adds a normalized multimodal request contract, explicit remote-code opt-in, explicit attention-backend selection, provenance, validation, and DIMER tutorial packaging without weight adaptation.

#### Intended Use and Limitations

###### Primary Intended Uses

The supported machine-learning task is instruction-conditioned text generation from text alone or from text paired with images, audio, or both. Intended applications include document/image question answering, visual description, authorized speech transcription or summarization, multimodal assistants, and research prototypes where generated text is reviewed. The repository is an inference component and capability demonstrator; it does not expose the upstream speech- or vision-LoRA training examples as a DIMER fine-tuning contract.

###### Primary Intended Users

Primary users are ML engineers, multimodal researchers, application developers, data scientists, and public-sector technical teams with experience in generative-model prompting, GPU runtime management, image/audio preprocessing, model-output evaluation, and the security implications of executing pinned third-party model code. Users are expected to understand that generated text can be incorrect even when fluent, and that each modality needs domain-specific validation before operational use.

###### Out-of-scope use cases

1. **Capability boundary:** this repository does not expose parameter fine-tuning, image generation, speech synthesis, biometric recognition, or a calibrated classifier.
2. **Input boundary:** the DIMER reference wrapper caps one request at four images and four audio clips and caps generated output at 2,048 new tokens; larger requests require a separately reviewed serving contract.
3. **Runtime boundary:** the release-reference tutorial requires a suitable CUDA GPU. FlashAttention 2 is not an implicit dependency; callers selecting it must provide a compatible `flash-attn` installation. The default DIMER attention implementation is `eager`.
4. **Decision boundary:** autonomous high-consequence decisions based solely on generated multimodal text are outside scope, regardless of model fluency.

#### Factors

###### Groups

The pipeline is human-centric whenever images, speech, or prompts describe people. Upstream training spans multilingual text and speech plus image-text data, but this repository has not independently audited demographic representation or group-level error rates. Downstream operators must evaluate relevant language, accent, dialect, visual phenotype, disability/accessibility, and other context-specific groups on representative data when people can be affected by outputs, and must investigate materially unequal failure rates.

###### Instrumentation

Inputs may come from cameras, scanners, screenshots, microphones, telephony systems, meeting software, recorders, or pre-existing digital media. Resolution, cropping, compression, color processing, audio sampling, clipping, reverberation, codec artifacts, and capture context can change the evidence available to the encoders. The wrapper validates request structure and media counts but cannot infer whether a camera or microphone was calibrated, whether content was manipulated, or whether crucial context was omitted upstream.

###### Environment

The reference runtime pins Python 3.12 and the model-facing stack around PyTorch 2.6 and Transformers 4.48.2. Practical release-reference execution requires a suitable CUDA GPU because the model snapshot is roughly 13 GB. The DIMER loader explicitly uses `eager` attention by default so a clean runtime does not silently depend on FlashAttention 2; `flash_attention_2` is an explicit optional path that fails clearly when a compatible `flash-attn` build is absent. Data-environment assumptions vary by modality: text-language coverage, image-domain shift, audio language/noise, and cross-modal alignment can all change behavior. The repository does not claim equivalent quality across modalities, languages, hardware, or deployment domains.

#### Metrics

###### Performance Measures

This inference wrapper does not report a universal scalar quality metric because its outputs span open-ended text generation, visual question answering, and audio-conditioned generation. The public `evaluation_report` stage therefore writes a machine-readable report whose verdict is always `not-measurable`, recording per capability whether the output was non-empty and what labelled data (reference answers, VQA pairs or captions, reference transcripts) would make it measurable. The tutorial verifies that each demonstrated capability returns machine-readable text through the same public API, but that is functional evidence rather than quality measurement. Deployment evaluation must provide task-specific labeled references or human scoring, such as exact-match/F1 for bounded QA, WER for ASR, or rubric-based generation evaluation.

###### Decision thresholds

No classification, acceptance, or safety threshold is shipped by this wrapper. Generation uses explicit decoding parameters, with `temperature=0.0` as the deterministic tutorial default and stochastic sampling enabled only when a positive temperature is requested. Generated text must not be interpreted as a calibrated probability or confidence score. Applications requiring acceptance/rejection rules must define and validate those rules on representative labeled data with costs appropriate to false acceptance and false rejection.

###### Approaches to uncertainty and variability

The tutorial performs one functional run per demonstrated capability and does not report dispersion or confidence intervals. With temperature zero, sampling randomness is suppressed, but GPU kernels, library versions, media decoding, attention implementation, and upstream custom code can still affect exact outputs. Positive temperature intentionally introduces stochastic decoding. The model does not provide a calibrated probability for the correctness of generated text through this wrapper, so downstream systems must establish task-specific uncertainty or review procedures.

#### Ethical considerations and biases

###### Data

Microsoft's upstream card reports training on large text, image-text, and speech collections, but this repository does not enumerate every source item and therefore does not assert that sensitive or proprietary material is absent from pretraining. This repository distributes code, tests, documentation, and tutorial logic; it does not commit the upstream weights or user media. Operators must review prompts, images, and audio for personal, confidential, copyrighted, classified, or restricted information before processing.

###### Human Life

This pipeline is not intended, certified, or independently validated for autonomous decisions in medicine, public safety, criminal justice, employment, credit, housing, education access, or other domains central to human flourishing. No external review board has cleared this repository for those uses. If multimodal assistance is considered in a sensitive domain, deployment requires qualified human oversight, representative local validation, security/privacy review, and any regulatory or institutional approval relevant to the task and data.

###### Mitigations

Implemented controls include an immutable upstream revision; a committed `dimer-base-manifest.json` whose per-file SHA-256 digests — including those of the five executed remote-code files — `verify_snapshot` re-checks before every load; the public `validate_inputs` stage, which applies the same instruction, media-count and decoding checks as `generate` and writes an input manifest with any rejection recorded as a finding; an explicit `allow_remote_code=True` opt-in before executing the model repository's custom Python code; exact model-facing dependency pins; explicit `eager` attention as the portable default; refusal of `flash_attention_2` when its compatible package is absent; CUDA availability failure when GPU is requested; non-empty prompt validation; media-count and generation-length ceilings; explicit decoding parameters; normalized provenance fields on every result; unit tests for prompt/media and attention contracts; model-card/notebook source validation; and documentation that functional execution does not establish answer correctness or safety.

###### Risks and harms

The model can hallucinate facts, misread images, mistranscribe audio, follow misleading context, or generate biased or harmful text. Cross-modal mistakes may appear persuasive because image or audio evidence is hidden from downstream readers. Executing third-party custom model code creates a software supply-chain risk despite revision pinning. User media may contain sensitive information. Automation bias, prompt injection in multimodal content, attention-backend/runtime differences, and domain shift can magnify harms when outputs are accepted without review.

###### Use cases

The pipeline must not be used for unlawful surveillance, biometric or demographic profiling, social scoring, discriminatory eligibility decisions, deceptive impersonation, manipulative targeting, or generation intended to facilitate abuse. It must not be used to bypass privacy, consent, copyright, security, or data-governance obligations, nor in any manner prohibited by the upstream MIT license or DIMER terms. Generated content must not be represented as independently verified evidence when it has not been checked.

## Immutable provenance and trust boundary

- Model: `microsoft/Phi-4-multimodal-instruct`
- Revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight format: sharded SafeTensors
- Weight license: MIT
- Repository code license: Apache-2.0
- Remote code: **required upstream** and explicitly opt-in in this repository; pinned by digest in `weights/phi4-multimodal-instruct/dimer-base-manifest.json` (`modeling_phi4mm.py` SHA-256 `e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d`) and re-hashed before import
- Snapshot manifest: `model-00001-of-00003.safetensors` SHA-256 `c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094` (from the Hub LFS metadata at the pinned revision; 26 files, 11172645832 bytes)
- Default attention implementation: `eager`
- Optional optimized attention: `flash_attention_2`, only with a compatible separately installed `flash-attn`
- Upstream model card: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- Technical report: https://arxiv.org/abs/2503.01743
