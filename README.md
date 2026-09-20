# Phi-4 Multimodal Pipeline

DIMER pipeline for `microsoft/Phi-4-multimodal-instruct`: text-, image- and audio-conditioned generation behind an explicit remote-code opt-in and a digest-verified snapshot, plus a bounded adaptation contract — the checkpoint's own vision LoRA fine-tuned for image captioning on a 4-bit NF4 base and exported as a safetensors adapter.

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

Inference (fp16/bf16 as stored):

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

## Adaptation contract

```python
from phi4_multimodal_pipeline import (
    Phi4MultimodalPipeline, constant_caption_baseline, colour_neighbour_baseline, fetch_sample_dataset,
)

splits = fetch_sample_dataset(cache_dir="weights/vizwiz-captions")   # pinned VizWiz-Captions slice: train 208 / validation 40 / test 70
pipe = Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, quantization="nf4")   # 4-bit base, ~6.8 GB on a T4

constant = constant_caption_baseline(splits["train"], splits["test"])       # corpus-medoid caption for every image
neighbour = colour_neighbour_baseline(splits["train"], splits["test"])     # caption of the nearest 3x3 mean-colour grid
frozen = pipe.evaluate(splits["test"])                                       # BLEU-4 / ROUGE-L / CIDEr-D of the frozen model
result = pipe.adapt(splits["train"], splits["validation"])                 # vision LoRA of the last 8 layers; 2 epochs, lr 5e-5
adapted = pipe.evaluate(splits["test"])

pipe.save_artifact("outputs/adapter")                                       # adapter.safetensors (~185 MB fp16) + manifest.json
reloaded = Phi4MultimodalPipeline.from_artifact("outputs/adapter", allow_remote_code=True, quantization="nf4")
```

Records are `{id, image, captions}` (a photograph path, one to five reference captions, an optional `category`); 8..5,000 records, unique ids; `split_dataset` splits by image and `check_split_disjoint` asserts no leakage. `caption` feeds every photograph downscaled to a 448-px long side with one fixed instruction and greedy decoding (40 new tokens). `adapt` trains the checkpoint's **own vision LoRA** tensors (`lora_A.vision` / `lora_B.vision`) of the last `trained_layers` decoder layers — eight by default, 64 tensors, 92,274,688 parameters as fp32 masters — with the causal cross-entropy on the caption tokens, AdamW without weight decay, gradient accumulation over four photographs, clipping at 1.0 and epoch selection on validation CIDEr-D; on any exception the frozen tensors are restored. `save_artifact` writes the trained tensors as fp16 safetensors with a manifest naming the base identity and first-shard digest, the five remote-code files, the quantization, the MIT licence, the tensor set, the file digest and the training history; `load_artifact` refuses anything that disagrees before deserialising.

Two implementation facts matter: under NF4 the checkpoint's LoRA tensors are kept out of quantisation (`NF4_SKIP_MODULES`) because bitsandbytes would otherwise quantise them too; and the upstream `set_lora_adapter`, which the model calls on every forward with image or audio inputs and which flips `requires_grad` on all 32 layers' vision LoRA, is replaced by a switch that changes only peft's active adapter (`docs/WEIGHTS.md`).

The build record (Kaggle Tesla T4) is in `docs/release-verification.md` and the model card's *Runtime* section.

## Weights layout

```
weights/phi4-multimodal-instruct/
  dimer-base-manifest.json          # modelId, revision, per-file bytes + sha256 for all 26 files (verified on every load)
  config.json, generation_config.json, preprocessor_config.json, processor_config.json
  tokenizer.json (git-ignored, 15 MB), tokenizer_config.json, vocab.json, merges.txt, added_tokens.json, special_tokens_map.json
  *.py                              # the 8 upstream Python files, git-ignored: pinned by digest, staged at run time, verified before import
  model-0000{1,2,3}-of-00003.safetensors + model.safetensors.index.json   # git-ignored, 11.15 GB
  examples/*.wav                    # the checkpoint's two example clips, git-ignored
  README.md, LICENSE
```

`from_pretrained(allow_remote_code=True)` calls `stage_missing_files()` then `verify_snapshot()` — which also refuses a manifest that does not pin the five remote-code files — and only then hands the directory to `transformers` with `trust_remote_code=True`; the remote-code opt-in stays mandatory on every path. `allow_download=True` fetches only the absent manifest entries at revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; without a snapshot directory the same flag falls back to the pinned Hub revision, and the default is to refuse. The manifest was written from the Hub API (LFS size + SHA-256 for the shards and the tokenizer; the small files downloaded and hashed); the shards are not kept locally. To stage by hand: `hf download microsoft/Phi-4-multimodal-instruct --revision 93f923e1a7727d1c4f446756212d9d3e8fcc5d81 --local-dir weights/phi4-multimodal-instruct`.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/phi4-multimodal-pipeline/blob/main/tutorials/phi4_multimodal_colab.ipynb)

`tutorials/phi4_multimodal_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three package modules, model identity, manifest digests (weights **and** remote code) and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default path needs a CUDA GPU of about 16 GB: the model is loaded in 4-bit NF4, the three inference capabilities are exercised, the VizWiz sample is fetched and split, the baselines and the frozen model are scored, the vision LoRA of the last eight layers is fine-tuned, the held-out photographs are scored again, and the adapter is exported and reloaded into a fresh pipeline. Text, image + text and audio + text are demonstrated separately so their input/output contracts remain explicit; the default samples are a synthetic sign image drawn in code and the checkpoint's own example clip, `validate_inputs` writes one input manifest per capability, `evaluation_report` is honestly `not-measurable`, and JSON plus the generated texts are exported. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `75ab9bb4` (committed at `1e348e2`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 code cells ok (1 restart after the install cell), 2032.3 s, 391 files / 11,257 MB staged and digest-verified inside the notebook (the 26-entry snapshot including the five remote-code files, the 392 MB VizWiz parquet slice and the 336 pinned photographs)); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified. The DIMER upload of the weights stays on HOLD by decision (2026-09-20). Static and unit checks are necessary but are never the evidence.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
