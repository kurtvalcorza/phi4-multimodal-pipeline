---
license: mit
model_card_spec: "1.0"
pipeline_tag: image-text-to-text
base_model: microsoft/Phi-4-multimodal-instruct
---

# Phi-4 Multimodal Instruct (DIMER package v0.1.0)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-microsoft%2FPhi--4--multimodal--instruct-ffcc4d)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
[![Weight license](https://img.shields.io/badge/weights-mit-blue)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/LICENSE)

###### Description

Phi-4-multimodal-instruct is Microsoft's approximately 5.6B-parameter multimodal transformer that accepts text, image, and audio inputs and generates text, packaged here from `microsoft/Phi-4-multimodal-instruct` at immutable revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`. Upstream reports a Phi-4-Mini-Instruct language backbone with vision and speech encoders/adapters and a 128K-token context. This repository adds a normalized multimodal request contract, explicit remote-code opt-in, provenance, validation, and DIMER tutorial packaging without weight adaptation.

#### Intended Use and Limitations

###### Primary Intended Uses

The supported machine-learning task is instruction-conditioned text generation from text alone or from text paired with images, audio, or both. Intended applications include document/image question answering, visual description, authorized speech transcription or summarization, multimodal assistants, and research prototypes where generated text is reviewed. The repository is an inference component and capability demonstrator; it does not expose the upstream speech- or vision-LoRA training examples as a DIMER fine-tuning contract.

###### Primary Intended Users

Primary users are ML engineers, multimodal researchers, application developers, data scientists, and public-sector technical teams with experience in generative-model prompting, GPU runtime management, image/audio preprocessing, model-output evaluation, and the security implications of executing pinned third-party model code. Users are expected to understand that generated text can be incorrect even when fluent, and that each modality needs domain-specific validation before operational use.

###### Out-of-scope use cases

1. **Capability boundary:** this repository does not expose parameter fine-tuning, image generation, speech synthesis, biometric recognition, or a calibrated classifier.
2. **Input boundary:** the DIMER reference wrapper caps one request at four images and four audio clips and caps generated output at 2,048 new tokens; larger requests require a separately reviewed serving contract.
3. **Decision boundary:** autonomous high-consequence decisions based solely on generated multimodal text are outside scope, regardless of model fluency.

#### Factors

###### Groups

The pipeline is human-centric whenever images, speech, or prompts describe people. Upstream training spans multilingual text and speech plus image-text data, but this repository has not independently audited demographic representation or group-level error rates. Downstream operators must evaluate relevant language, accent, dialect, visual phenotype, disability/accessibility, and other context-specific groups on representative data when people can be affected by outputs, and must investigate materially unequal failure rates.

###### Instrumentation

Inputs may come from cameras, scanners, screenshots, microphones, telephony systems, meeting software, recorders, or pre-existing digital media. Resolution, cropping, compression, color processing, audio sampling, clipping, reverberation, codec artifacts, and capture context can change the evidence available to the encoders. The wrapper validates request structure and media counts but cannot infer whether a camera or microphone was calibrated, whether content was manipulated, or whether crucial context was omitted upstream.

###### Environment

The reference runtime pins Python 3.12 and the upstream-recommended generation stack around PyTorch 2.6 and Transformers 4.48.2. Practical execution requires a suitable GPU because the model snapshot is roughly 13 GB and upstream examples use CUDA. Data-environment assumptions vary by modality: text-language coverage, image-domain shift, audio language/noise, and cross-modal alignment can all change behavior. The repository does not claim equivalent quality across modalities, languages, hardware, or deployment domains.

#### Metrics

###### Performance Measures

This inference wrapper does not report a universal scalar quality metric because its outputs span open-ended text generation, visual question answering, and audio-conditioned generation. The tutorial verifies that each demonstrated capability returns machine-readable text through the same public API, but that is functional evidence rather than quality measurement. Deployment evaluation must provide task-specific labeled references or human scoring, such as exact-match/F1 for bounded QA, WER for ASR, or rubric-based generation evaluation.

###### Decision thresholds

No classification, acceptance, or safety threshold is shipped by this wrapper. Generation uses explicit decoding parameters, with `temperature=0.0` as the deterministic tutorial default and stochastic sampling enabled only when a positive temperature is requested. Generated text must not be interpreted as a calibrated probability or confidence score. Applications requiring acceptance/rejection rules must define and validate those rules on representative labeled data with costs appropriate to false acceptance and false rejection.

###### Approaches to uncertainty and variability

The tutorial performs one functional run per demonstrated capability and does not report dispersion or confidence intervals. With temperature zero, sampling randomness is suppressed, but GPU kernels, library versions, media decoding, and upstream custom code can still affect exact outputs. Positive temperature intentionally introduces stochastic decoding. The model does not provide a calibrated probability for the correctness of generated text through this wrapper, so downstream systems must establish task-specific uncertainty or review procedures.

#### Ethical considerations and biases

###### Data

Microsoft's upstream card reports training on large text, image-text, and speech collections, but this repository does not enumerate every source item and therefore does not assert that sensitive or proprietary material is absent from pretraining. This repository distributes code, tests, documentation, and tutorial logic; it does not commit the upstream weights or user media. Operators must review prompts, images, and audio for personal, confidential, copyrighted, classified, or restricted information before processing.

###### Human Life

This pipeline is not intended, certified, or independently validated for autonomous decisions in medicine, public safety, criminal justice, employment, credit, housing, education access, or other domains central to human flourishing. No external review board has cleared this repository for those uses. If multimodal assistance is considered in a sensitive domain, deployment requires qualified human oversight, representative local validation, security/privacy review, and any regulatory or institutional approval relevant to the task and data.

###### Mitigations

Implemented controls include an immutable upstream revision; an explicit `allow_remote_code=True` opt-in before executing the model repository's custom Python code; exact model-facing dependency pins; CUDA availability failure when GPU is requested; non-empty prompt validation; media-count and generation-length ceilings; explicit decoding parameters; normalized provenance fields on every result; unit tests for the prompt/media contract; model-card/notebook source validation; and documentation that functional execution does not establish answer correctness or safety.

###### Risks and harms

The model can hallucinate facts, misread images, mistranscribe audio, follow misleading context, or generate biased or harmful text. Cross-modal mistakes may appear persuasive because image or audio evidence is hidden from downstream readers. Executing third-party custom model code creates a software supply-chain risk despite revision pinning. User media may contain sensitive information. Automation bias, prompt injection in multimodal content, and domain shift can magnify harms when outputs are accepted without review.

###### Use cases

The pipeline must not be used for unlawful surveillance, biometric or demographic profiling, social scoring, discriminatory eligibility decisions, deceptive impersonation, manipulative targeting, or generation intended to facilitate abuse. It must not be used to bypass privacy, consent, copyright, security, or data-governance obligations, nor in any manner prohibited by the upstream MIT license or DIMER terms. Generated content must not be represented as independently verified evidence when it has not been checked.

## Immutable provenance and trust boundary

- Model: `microsoft/Phi-4-multimodal-instruct`
- Revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight format: sharded SafeTensors
- Remote code: **required upstream** and explicitly opt-in in this repository
- Upstream model card: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- Technical report: https://arxiv.org/abs/2503.01743
