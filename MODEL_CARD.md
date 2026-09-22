---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
task: "Others - Multimodal Generation & Image Captioning Fine-Tuning"
base_model: microsoft/Phi-4-multimodal-instruct
date_published: "2025-02-24"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/microsoft/Phi-4-multimodal-instruct)"
---

# Phi-4 Multimodal Instruct — Multimodal Language Model (Text, Image & Audio) with Bounded Captioning Adaptation

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2FPhi--4--multimodal--instruct-ffcc4d?style=flat)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2503.01743-b31b1b.svg)](https://arxiv.org/abs/2503.01743)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/93f923e1a7727d1c4f446756212d9d3e8fcc5d81/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **E2E captioning fine-tuning tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb) [`phi4_multimodal_colab.ipynb`](https://github.com/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb)  
  *Standalone `E2E` tutorial (DIMER Notebook Specification 2.0 §4): the pinned checkpoint loaded in 4-bit NF4 with its five model-code files digest-verified before import; the three inference capabilities (text, image, audio) through the contract; a digest-pinned VizWiz-Captions sample (208 / 40 / 70 photographs); constant-caption and colour-neighbour baselines and the frozen model scored with BLEU-4 / ROUGE-L / CIDEr-D; bounded fine-tuning of the checkpoint's own vision LoRA in the last eight decoder layers with validation-based epoch selection; held-out evaluation; a safetensors adapter exported and reloaded into a fresh pipeline with verified parity (verbatim captions reported; the reloaded model's held-out CIDEr-D asserted within 0.02 of the adapted model's).*

> [!NOTE]
> Requires a fresh CUDA runtime of about 16 GB (a Tesla T4 is the reference device): the 4-bit load takes about 6.8 GB and training peaks near 9.2 GB. The notebook does not run on CPU.

---

#### Description

Phi-4-multimodal-instruct is Microsoft's approximately 5.6B-parameter multimodal transformer that accepts text, image, and audio inputs and generates text, packaged here from `microsoft/Phi-4-multimodal-instruct` at immutable revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`. Upstream reports a Phi-4-Mini-Instruct language backbone with vision and speech encoders/adapters and a 128K-token context. This repository adds a normalized multimodal request contract, explicit remote-code opt-in, explicit attention-backend selection, provenance, validation, DIMER tutorial packaging, and — since 2026-09-20 — a bounded adaptation contract: an optional 4-bit NF4 load (`quantization="nf4"`, bitsandbytes, fp16 compute; the vision tower, projector, embeddings, output head and the checkpoint's own LoRA tensors stay fp16) and `adapt`, which fine-tunes the checkpoint's **own vision LoRA** in the last `trained_layers` decoder layers (eight by default: 64 tensors, 92,274,688 parameters) for image captioning, with `caption` / `evaluate` for the task, metrics from `metrics.py` (BLEU-4, ROUGE-L, CIDEr-D) and a safetensors adapter round trip (`save_artifact` / `from_artifact`). The demonstrated task is the VizWiz-Captions sample pinned in `samples.py`; the DIMER upload of the weights remains on HOLD by decision.

#### Intended Use and Limitations

###### Primary Intended Uses

The supported machine-learning task is instruction-conditioned text generation from text alone or from text paired with images, audio, or both. Intended applications include document/image question answering, visual description, authorized speech transcription or summarization, multimodal assistants, and research prototypes where generated text is reviewed. The repository is an inference component and capability demonstrator with one bounded adaptation route: supervised image captioning that trains the checkpoint's own vision LoRA (last eight decoder layers) on labelled `{id, image, captions}` records, exported as a safetensors adapter. The speech LoRA, the vision tower, the projector, the embeddings and the 4-bit base are never trained here.

###### Primary Intended Users

Primary users are ML engineers, multimodal researchers, application developers, data scientists, and public-sector technical teams with experience in generative-model prompting, GPU runtime management, image/audio preprocessing, model-output evaluation, and the security implications of executing pinned third-party model code. Users are expected to understand that generated text can be incorrect even when fluent, and that each modality needs domain-specific validation before operational use.

###### Out-of-scope use cases

1. **Capability boundary:** this repository exposes exactly one adaptation — captioning fine-tuning of the vision LoRA's last eight layers on a 4-bit base — and no image generation, speech synthesis, biometric recognition, calibrated classifier, speech-LoRA training, full fine-tuning, LoRA ranks other than the checkpoint's own, or multi-task adaptation. An adapter learns its training set's captioning convention (window of what is named, length, whether printed text is read out) and nothing else.
2. **Input boundary:** the DIMER reference wrapper caps one request at four images and four audio clips and caps generated output at 2,048 new tokens; larger requests require a separately reviewed serving contract.
3. **Runtime boundary:** the release-reference tutorial requires a CUDA GPU of about 16 GB; the 4-bit load (`quantization="nf4"`) needs CUDA and bitsandbytes, and CPU is not supported for the default path. FlashAttention 2 is not an implicit dependency; callers selecting it must provide a compatible `flash-attn` installation. The default DIMER attention implementation is `eager`. For the captioning contract every photograph is downscaled to at most 448 px on its long side before the processor (one or two tiles, about 550 tokens), identically for training and inference.
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

The multimodal inference wrapper does not report a universal scalar quality metric because its outputs span open-ended text generation, visual question answering, and audio-conditioned generation; for those capabilities the public `evaluation_report` stage writes a machine-readable report whose verdict is `not-measurable`, recording per capability whether the output was non-empty and what labelled data would make it measurable. The **captioning adaptation contract is measured**: `evaluate` captions every record of a labelled set through the contract and scores the predictions with corpus-level **CIDEr-D** (the headline; TF-IDF-weighted n-gram agreement with the references, comparable only across systems scored on the same records), **BLEU-4**, **ROUGE-L** and the per-caption unigram-F1 plumbing check, and the tutorial reads the adapted model against the frozen model and two non-neural baselines on the same 70 held-out photographs — the **constant caption** (the training-corpus medoid; what a captioner that never looks at the image reaches) and the **colour nearest neighbour** (the caption of the training photograph with the closest 3×3 mean-colour grid). The build record is in *Runtime* below and in `docs/release-verification.md`. The tutorial verifies that each demonstrated capability returns machine-readable text through the same public API, but that is functional evidence rather than quality measurement. Deployment evaluation must provide task-specific labeled references or human scoring, such as exact-match/F1 for bounded QA, WER for ASR, or rubric-based generation evaluation.

###### Decision thresholds

No classification, acceptance, or safety threshold is shipped by this wrapper. Generation uses explicit decoding parameters, with `temperature=0.0` as the deterministic tutorial default and stochastic sampling enabled only when a positive temperature is requested; the captioning contract fixes one instruction, forty new tokens and greedy decoding. Epoch selection during `adapt` uses one stated rule — the epoch with the highest validation CIDEr-D (the final one without a validation split) — and the tutorial asserts, rather than assumes, that the adapted CIDEr-D exceeds the frozen model's and both baselines'. Generated text must not be interpreted as a calibrated probability or confidence score. Applications requiring acceptance/rejection rules must define and validate those rules on representative labeled data with costs appropriate to false acceptance and false rejection.

###### Approaches to uncertainty and variability

The tutorial performs one functional run per demonstrated capability and one seeded adaptation (split seed 42, training seed 0) and does not report dispersion or confidence intervals: seventy held-out photographs from one draw of one row group give none, and the 40-photograph validation split that picks the epoch is noisy. Training on a 4-bit base with fp16 compute is not bit-reproducible across devices, and the frozen numbers are those of the quantised model, not of the bf16 checkpoint the upstream card evaluates (that difference is not measured here). With temperature zero, sampling randomness is suppressed, but GPU kernels, library versions, media decoding, attention implementation, and upstream custom code can still affect exact outputs. Positive temperature intentionally introduces stochastic decoding. The model does not provide a calibrated probability for the correctness of generated text through this wrapper, so downstream systems must establish task-specific uncertainty or review procedures.

#### Ethical considerations and biases

###### Data

Microsoft's upstream card reports training on large text, image-text, and speech collections, but this repository does not enumerate every source item and therefore does not assert that sensitive or proprietary material is absent from pretraining. This repository distributes code, tests, documentation, and tutorial logic; it does not commit the upstream weights or user media. Operators must review prompts, images, and audio for personal, confidential, copyrighted, classified, or restricted information before processing.

###### Human Life

This pipeline is not intended, certified, or independently validated for autonomous decisions in medicine, public safety, criminal justice, employment, credit, housing, education access, or other domains central to human flourishing. No external review board has cleared this repository for those uses. If multimodal assistance is considered in a sensitive domain, deployment requires qualified human oversight, representative local validation, security/privacy review, and any regulatory or institutional approval relevant to the task and data.

###### Mitigations

Implemented controls include an immutable upstream revision; a committed `dimer-base-manifest.json` whose per-file SHA-256 digests — including those of the five executed remote-code files — `verify_snapshot` re-checks before every load; the public `validate_inputs` stage, which applies the same instruction, media-count and decoding checks as `generate` and writes an input manifest with any rejection recorded as a finding; for the adaptation contract, `validate_dataset` (record shape, decodable photographs within the side limits, 1..5 bounded captions, unique ids, dataset bounds) before any model work, `check_split_disjoint` by photograph, a digest-pinned sample refused on any mismatch, a trainable set restricted by name to the vision-LoRA tensors of the last `trained_layers` layers with a transactional restore on failure, and an adapter manifest (format, base identity and shard digest, remote-code files, quantization, licence, tensor set, file digest, history) that `load_artifact` verifies before deserialising and whose tensor set it refuses if it strays outside the recorded scope; an explicit `allow_remote_code=True` opt-in before executing the model repository's custom Python code; exact model-facing dependency pins; explicit `eager` attention as the portable default; refusal of `flash_attention_2` when its compatible package is absent; CUDA availability failure when GPU is requested; non-empty prompt validation; media-count and generation-length ceilings; explicit decoding parameters; normalized provenance fields on every result; unit tests for prompt/media and attention contracts; model-card/notebook source validation; and documentation that functional execution does not establish answer correctness or safety.

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
- Adaptation: the checkpoint's own `vision_lora` (r = 256, alpha 512, on every decoder layer's `qkv_proj` / `o_proj` / `gate_up_proj` / `down_proj`) trained in the last 8 of 32 layers = 64 tensors / 92,274,688 parameters; adapter format `org.valcorza.phi4-multimodal-instruct.adapter.v1`, safetensors fp16 (about 185 MB) + `manifest.json`; the base's own `set_lora_adapter` is replaced in this pipeline by a switch that changes only peft's active adapter (upstream's flips `requires_grad` on every forward — see `docs/WEIGHTS.md`)
- Quantized load: `quantization="nf4"` = bitsandbytes NF4 with double quantization and fp16 compute on the language model's 128 base projections; `lm_head`, embeddings, the vision/audio towers and projectors and all LoRA tensors stay fp16 (`NF4_SKIP_MODULES`)
- Fine-tuning sample: VizWiz-Captions (CC BY 4.0) from the Hub mirror `mm-eval/VizWiz-Captions` at revision `c4a6d897836e7885d0095134f92d392e4e770539`, `data/val-00004-of-00005.parquet` (392,245,504 bytes, SHA-256 `4492465a…`), text columns of all 1,550 rows and the 336 photographs of row group 0 pinned by size and SHA-256 in `samples.py`; seeded split 208 / 40 / 70
- Default attention implementation: `eager`
- Optional optimized attention: `flash_attention_2`, only with a compatible separately installed `flash-attn`
- Upstream model card: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- Technical report: https://arxiv.org/abs/2503.01743

## Runtime

- Pins (`pyproject.toml`): `torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`, `transformers==4.48.2`, `accelerate==1.3.0`, `huggingface-hub==0.36.2`, `soundfile==0.13.1`, `pillow==11.1.0`, `scipy==1.15.2`, `backoff==2.2.1`, `peft==0.13.2`, `bitsandbytes==0.45.5`, `safetensors==0.5.3`, `numpy==1.26.4`, `pyarrow==19.0.1`; dev `pytest==9.0.2`, `ruff==0.12.10`. Python 3.12; offline tests in a Windows venv (53 passed; stubbed model, no weights, no GPU). No local GPU pre-flight exists for this profile: the workstation runs no GPU jobs by decision (2026-09-20), so every execution below is a hosted clean-runtime run.
- Executed 2026-09-20, the **committed notebook blob** (`1e348e2` / `75ab9bb4`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-phi4-multimodal` v5; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.6.0+cu124` / `transformers 4.48.2` / `bitsandbytes 0.45.5` after; Python 3.12.13; Tesla T4 15,360 MiB; empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 11/11 code cells ok (1 restart after the install cell), 2032.3 s, 391 files / 11,257 MB staged and digest-verified inside the notebook (the 26-entry snapshot including the five remote-code files, the 392 MB VizWiz parquet slice and the 336 pinned photographs); three capabilities exercised (`not-measurable` report); baselines constant-caption CIDEr-D 0.052 / colour-neighbour 0.042; **frozen 4-bit model on the 70 test photographs: BLEU-4 0.171 / ROUGE-L 0.439 / CIDEr-D 0.722 / unigram-F1 0.483** (mean 14.1 words); `adapt` 2 epochs, lr 5e-5, last 8 layers, grad-accumulation 4, 1140.8 s: validation CIDEr-D 0.835 → 0.910 → 1.010 (best epoch 2; train loss 2.607 → 2.382); **adapted model: BLEU-4 0.262 / ROUGE-L 0.489 / CIDEr-D 0.890 / unigram-F1 0.522** (mean 11.2 words against 12.1 in the references; +0.168 CIDEr-D over frozen, +0.838 over the best baseline); adapter 64 tensors / 184,557,520 bytes fp16 (SHA-256 `79aab939…`); model freed to 16 MiB; fresh 4-bit reload with `best_epoch 2`; reload parity 7 of 8 captions verbatim (the eighth differs by a trailing full stop) and reloaded test CIDEr-D 0.8893 vs adapted 0.8900 (difference 0.0007, asserted ≤ 0.02). Sample-sanity evidence from one seeded run, not a benchmark. Recorded in `docs/release-verification.md`.
- The two earlier runs of the E2E carrier (kernel v3 at `66e3f1a`, v4 at `40ea48f`) read identical frozen and adapted numbers and failed only on a training-mode defect of the remote vision encoder (fixed in `40ea48f`) and on the exact-caption reload rule (replaced in `1e348e2`); both are recorded in `docs/release-verification.md`.
- Not executed: the CPU path (unsupported for the 4-bit load), BYOD, any published benchmark, any comparison of the 4-bit model against the bfloat16 checkpoint, and any measurement of latency or memory beyond the cell times above.
