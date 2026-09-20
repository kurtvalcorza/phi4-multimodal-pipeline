# Release verification

`tutorials/phi4_multimodal_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, generating revision, modules, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` (26 entries: the three weight shards, the configs, the
  tokenizer files, the example clip **and the eight upstream `.py` files**) equal to the committed snapshot manifest
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE`
  recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same identity
  string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the VizWiz-Captions dataset
  revision `c4a6d897…` is the only other 40-hex commit the documents may name);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=WEIGHTS_DIR, quantization='nf4')`,
  `fetch_annotations` / `fetch_images` / `build_sample_dataset` from the pinned VizWiz slice, `load_byod_dataset` /
  `split_dataset`, `validate_dataset` per split with `check_split_disjoint` and `write_dataset_jsonl`, the refusal
  probe, `validate_inputs` once per capability plus the five-image probe, `generate` three times, `evaluation_report`,
  `pipe.caption` on the synthetic sign, `constant_caption_baseline` and `colour_neighbour_baseline`, `pipe.evaluate`
  on the frozen model with the assertion that it beats the constant caption, `pipe.adapt` with its explicit
  hyperparameters, `pipe.evaluate` after adaptation with the assertions that the adapted CIDEr-D exceeds the frozen
  model's and both baselines', `pipe.save_artifact`, `Phi4MultimodalPipeline.from_artifact(..., quantization='nf4')`,
  `reloaded.evaluate` on the test records and the reload-parity assertion on the held-out CIDEr-D, and the provenance
  fields (remote-code files, model revision and licence, weight digest, quantization, the VizWiz corpus record,
  runtime versions, device and attention backend)), the five expected `outputs/` paths, the learner-facing statements
  (the checkpoint ships its own model code behind the `allow_remote_code=True` opt-in, MIT, adaptation with labelled
  photographs, the checkpoint's own vision LoRA in the last eight decoder layers, the snapshot note, the section
  headings, the two baselines, validation CIDEr-D, no dispersion estimate, labelling convention, leakage, the 4-bit
  load needs CUDA) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct `huggingface_hub` /
  `transformers` / `datasets` / `peft` / `safetensors` / `BitsAndBytesConfig` / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` use **outside the carried module cells**, `pickle.load`, `torch.load(`,
  `extractall(`; `trust_remote_code=True` is forbidden in every notebook cell and permitted only inside the carried
  module cell, where it sits behind the `allow_remote_code` opt-in and the digest check);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and immutable provenance.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_snapshot.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_notebook_parity.py`, `tests/test_release_assets.py`; stubbed `transformers` / `torch` where a model is
needed, no weights, 53 tests). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CUDA runtime with a 16 GB GPU (T4 or better) and `bitsandbytes` support | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel, fresh-interpreter harness | Kaggle `NvidiaTeslaT4` kernel, Python 3.12 image; the committed notebook (blob SHA-1 verified against GitHub) executed verbatim by the fleet executor in a fresh interpreter with a `google.colab` shim and **no repository checkout**; the restart the install cell requests is honoured by re-running from the top | Reproducible clean-room executor of the same class; promotion evidence |

No local pre-flight path exists for this row any more: the workstation runs no GPU jobs by decision (2026-09-20), and
the 4-bit path needs CUDA. Every execution below is a hosted clean-runtime run.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA GPU runtime (Colab, or the Kaggle executor above) with **no
   repository checkout**, an empty Hugging Face cache and nothing pre-staged under `weights/` (the standalone path
   writes the manifest itself and stages all 26 entries — about 11.2 GB — plus the VizWiz slice: the 392 MB parquet
   for the text columns and the 336 pinned photographs); the runtime needs about 16 GB of GPU memory;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 2`, `LEARNING_RATE = 5e-5`, `TRAINED_LAYERS = 8`, `GRAD_ACCUMULATION = 4`, `SEED = 0`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from`, `cuda: True`, and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`): `torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`, `transformers==4.48.2`,
   `accelerate==1.3.0`, `huggingface-hub==0.36.2`, `soundfile==0.13.1`, `pillow==11.1.0`, `scipy==1.15.2`,
   `backoff==2.2.1`, `peft==0.13.2`, `bitsandbytes==0.45.5`, `safetensors==0.5.3`, `numpy==1.26.4`,
   `pyarrow==19.0.1` (an interpreter restart after the install is expected where the image's preinstalled numpy or
   Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `Phi4MultimodalPipeline`, `validate_inputs`,
     `evaluation_report`, `REMOTE_CODE_FILES`, the ceilings, the captioning helpers and the metrics and sample
     helpers) with no import of the repository package;
   - the inline `MANIFEST` asserted against the module identity and written to `weights/phi4-multimodal-instruct/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reporting all 26 entries on a clean runtime,
     `verify_snapshot` returning after hashing the three shards and the eight `.py` files, and
     `from_pretrained(allow_remote_code=True, weights_dir=WEIGHTS_DIR, quantization='nf4')` reporting
     `source == 'local-snapshot'`, `attention_implementation == 'eager'` and the 4-bit load;
   - the VizWiz sample fetched (1,550 annotation rows, 336 pinned photographs digest-verified), split 208 / 40 / 70 by
     image, `check_split_disjoint` clean, `outputs/phi4_multimodal_train.jsonl` written, the dataset refusal probe
     rejected;
   - `validate_inputs` writing `outputs/phi4_multimodal_input_manifest.json` with three accepted manifests and one
     recorded rejection finding (the five-image probe); three generations through `generate` (text, image + text,
     audio + text), each non-empty; `evaluation_report` writing `outputs/phi4_multimodal_evaluation_report.json`
     with verdict `not-measurable` for the three capabilities; the synthetic sign captioned;
   - the two baselines and the frozen model scored on the 70 test photographs (BLEU-4, ROUGE-L, CIDEr-D, unigram-F1),
     the frozen model asserted above the constant caption;
   - `pipe.adapt` printing epoch 0 as the frozen model and one row per epoch with the validation CIDEr-D, then the
     summary (64 trained tensors, 92,274,688 parameters, `quantization nf4`, the kept epoch);
   - `pipe.evaluate` on the test records with the adapted numbers beside the frozen and the baselines, the assertions
     that the adapted CIDEr-D exceeds the frozen model's and both baselines', and the evaluation report rewritten;
   - six before/after captions printed, `pipe.save_artifact` writing `outputs/phi4_multimodal_adapter/{adapter.safetensors,manifest.json}`
     (64 tensors, fp16), eight adapted captions recorded, the model freed, `from_artifact` loading a fresh 4-bit base
     with the adapter, the eight captions re-generated with every mismatch printed, `reloaded.evaluate` on the 70 test
     photographs, and the assertion that the reloaded model's CIDEr-D is within 0.02 of the adapted model's;
   - `outputs/phi4_multimodal_result.json` written with `NOTEBOOK_SOURCE`, the model revision and licence, the
     remote-code files, the weight digest and quantization, the corpus record, the comparison, the artifact digest,
     the reload parity, runtime versions, device and attention backend;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, GPU), model identifier
   and immutable revision, whether the model cache was clean, outcome, produced outputs, the observed metrics (as
   observations, not a benchmark) and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Recorded executions

Notebook identity is the Git blob id of `tutorials/phi4_multimodal_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/phi4_multimodal_colab.ipynb`). Wall times are the sum of per-cell times reported
by the executor and include installs and the model download; they are measurements for the stated runtime, not
general estimates.

### `E2E` standalone carrier (NOTEBOOK_SPEC 2.0) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `1e348e2` / `75ab9bb4` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-phi4-multimodal` v5; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.6.0+cu124` / `transformers 4.48.2` / `bitsandbytes 0.45.5` after; Python 3.12.13; Tesla T4 15,360 MiB) | Default sample path, `Run all` from a fresh interpreter (1 restart after the install cell), empty Hugging Face cache, no repository checkout (blob SHA-1 verified against GitHub before execution) | 2032.3 s | **PASSED** — 11/11 code cells ok (1 restart after the install cell), 2032.3 s, 391 files / 11,257 MB staged and digest-verified inside the notebook (the 26-entry snapshot including the five remote-code files, the 392 MB VizWiz parquet slice and the 336 pinned photographs); per-cell: stage/verify/4-bit load 160 s, VizWiz sample 45 s, three capabilities 11 s, baselines + frozen evaluation 199 s, `adapt` 1141 s, adapted evaluation 230 s, export + reload + parity 238 s; baselines constant-caption CIDEr-D 0.052 / colour-neighbour 0.042; **frozen 4-bit model on the 70 test photographs: BLEU-4 0.171 / ROUGE-L 0.439 / CIDEr-D 0.722 / unigram-F1 0.483** (mean 14.1 words); `adapt` 2 epochs, lr 5e-5, last 8 layers, grad-accumulation 4, 1140.8 s: validation CIDEr-D 0.835 → 0.910 → 1.010 (best epoch 2; train loss 2.607 → 2.382); **adapted model: BLEU-4 0.262 / ROUGE-L 0.489 / CIDEr-D 0.890 / unigram-F1 0.522** (mean 11.2 words against 12.1 in the references; +0.168 CIDEr-D over frozen, +0.838 over the best baseline); adapter 64 tensors / 184,557,520 bytes fp16 (SHA-256 `79aab939…`); model freed to 16 MiB; fresh 4-bit reload with `best_epoch 2`; reload parity 7 of 8 captions verbatim (the eighth differs by a trailing full stop) and reloaded test CIDEr-D 0.8893 vs adapted 0.8900 (difference 0.0007, asserted ≤ 0.02). Evidence: `.agent/backups/tier-c-build-2026-09-20/kaggle-phi4/dimer-nb2-phi4-multimodal/v5/` in the workspace |
| 2026-09-20 | `40ea48f` / `22dbbfa8` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-phi4-multimodal` v4; image `torch 2.10.0+cu128` before the pinned install, `torch 2.6.0` / `transformers 4.48.2` / `bitsandbytes 0.45.5` after; Tesla T4 15,360 MiB, driver 580.159.04) | Default sample path, `Run all` from a fresh interpreter (1 restart after the install cell), empty Hugging Face cache, no repository checkout; 391 files / 11,257 MB staged and digest-verified inside the notebook | 2002.8 s | **FAILED at the last cell only** — 10/11 code cells ok. Everything the tutorial claims executed: 26-entry snapshot staged and verified, nf4 load, VizWiz 208 / 40 / 70, three capabilities, baselines constant-caption CIDEr-D 0.052 / colour-neighbour 0.042, **frozen model test BLEU-4 0.171 / ROUGE-L 0.439 / CIDEr-D 0.722 / unigram-F1 0.483** (167.7 s for 70 photographs); `adapt` 2 epochs, 1093 s: validation CIDEr-D 0.835 → 0.910 → 1.010 (best epoch 2; train loss 2.607 → 2.382); **adapted model test BLEU-4 0.262 / ROUGE-L 0.489 / CIDEr-D 0.890 / unigram-F1 0.522** (+0.168 CIDEr-D over frozen, +0.838 over the best baseline; mean words 14.1 → 11.2 against 12.1 in the references); adapter 64 tensors / 184,557,520 bytes (sha `79aab939…`); model freed to 16 MiB; fresh 4-bit reload with `best_epoch 2`; **7 of 8 held-out captions reproduced verbatim, and the cell's exact-identity assertion failed on the eighth**. Diagnosis: greedy decoding turns a sub-ULP logit difference between the trained-in-memory model and a freshly loaded one into a different word; the fix (`1e348e2`) reports verbatim agreement and asserts the reloaded model's CIDEr-D on the 70 test photographs within 0.02 of the adapted model's. Evidence: `.agent/backups/tier-c-build-2026-09-20/kaggle-phi4/dimer-nb2-phi4-multimodal/v4/` in the workspace |
| 2026-09-20 | `66e3f1a` / `38f52b77` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-phi4-multimodal` v3) | Default sample path, `Run all` from a fresh interpreter (1 restart after the install cell), empty Hugging Face cache, no repository checkout | 717.5 s | **FAILED at cell 19 (`adapt`)** — 8/11 ok; the frozen-model numbers above were first read here (identical to run v4); `AttributeError: 'SiglipEncoder' object has no attribute '_gradient_checkpointing_func'` — the remote vision encoder is built with `gradient_checkpointing=True` and in train mode calls the function that `model.gradient_checkpointing_enable()` installs; fixed in `40ea48f` (`adapt` enables it before training and disables it after) |

### Previous `MULTI-CAPABILITY` standalone carrier (NOTEBOOK_SPEC 1.1) — audit trail, does not cover the E2E path

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `813c993` / `efdde43e6259` | Kaggle T4 (`kurtvalcorza/dimer-nb2-phi4-multimodal` v2) | Default sample path (three capabilities, bf16 weights, no adaptation) | 378.4 s | **PASSED** — 10/10 ok code cells, 54 files, 11173 MB staged; the Release-grade record of the inference-only carrier this E2E carrier replaces |
| 2026-09-11 | `2c80d53acf4e` / `31bbfce8250c` and `05a7e0c487e5` / `8fd63008f8b0` | Kaggle T4 (`dimer-phi4-multimodal-t4-verify-v2` v1, `dimer-phi4-multimodal-t4-verify` v1) | The earlier repository-installing carrier (NOTEBOOK_SPEC 1.0), default path, 6 code cells | 363.3 s / ≈5.5 min | **PASS** — bf16 weights on sm_75, peak CUDA allocation 12.74 GiB, eager attention; text / image / audio outputs sensible (the audio transcript matched the LibriSpeech reference); superseded by the standalone carriers above |

## Current status

**Release-grade.** The `E2E` notebook blob `75ab9bb4` (committed at `1e348e2`) executed top-to-bottom in a clean Kaggle
Tesla T4 runtime on 2026-09-20 (11/11 ok, 2032.3 s, 391 files / 11,257 MB staged and digest-verified inside the notebook,
remote code hashed before import) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates
on. The two earlier E2E runs (v3, v4) are history: identical frozen and adapted numbers, each stopped by one defect that the
next revision fixed. Any later change to the carried modules or to the notebook produces a new blob, and the registry
returns to **Candidate** until a clean run of that blob is recorded here. The DIMER upload of the weights is on HOLD by
the maintainer's decision (2026-09-20) independently of this gate.
