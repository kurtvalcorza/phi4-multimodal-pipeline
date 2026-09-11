# Release verification

`tutorials/phi4_multimodal_colab.ipynb` (`MULTI-CAPABILITY`) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `MULTI-CAPABILITY`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls, exports, learner-facing statements and gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, direct `transformers`
  loading that bypasses the pipeline, `trust_remote_code=True` outside the pipeline boundary,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CUDA runtime with ≥16 GB GPU memory | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle `NvidiaTeslaT4` kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used when a repository pins a package that the Kaggle kernel pre-imports (numpy 2.0.2): the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA GPU runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - pinned `microsoft/Phi-4-multimodal-instruct` model and custom-code acquisition at the immutable revision with `allow_remote_code=True` acknowledged explicitly;
   - `eager` attention selected (no FlashAttention dependency);
   - text-only generation, image+text generation and audio+text generation through `Phi4MultimodalPipeline.generate`;
   - `outputs/phi4_multimodal_results.json` written with repository SHA, model revision, runtime versions, GPU device and attention implementation;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/phi4_multimodal_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/phi4_multimodal_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `05a7e0c487e5` / blob `8fd63008f8b0` (this revision) | Kaggle kernel `kurtvalcorza/dimer-phi4-multimodal-t4-verify` v1 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; the committed notebook pushed verbatim plus one leading shim cell (`DIMER_TUTORIAL_REF`, scratch cwd, `google.colab` shim); the kernel pre-imports numpy 2.0.2 and Pillow 11.3.0, neither of which this repository pins | Default path, all 6 code cells, clean HF cache (`hf cache clean: True`) | ≈5.5 min (05:37:25 → 05:42:54 UTC, including the ~13 GB snapshot) | **PASS** — `repository_revision` equals the candidate commit; `microsoft/Phi-4-multimodal-instruct` acquired at the pinned revision with `allow_remote_code=True` and `attention_implementation='eager'` (upstream warns "not running the flash-attention implementation"); torch 2.6.0+cu124, transformers 4.48.2, accelerate 1.3.0, datasets 4.4.0, soundfile-decoded audio; bf16 weights executed on sm_75 with peak CUDA allocation 12.74 GiB of 14.56; text: "Automatic speech recognition (ASR) is a technology that enables computers to interpret and process human speech into text…"; image: "…a red octagonal stop sign with the word 'STOP' written in white capital letters."; audio: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel" (matches the LibriSpeech reference); `outputs/phi4_multimodal_results.json` sha256 `0e063a95…ae1c95`; no cross-repository tokenizer fetch appears in the log; warnings: upstream `speech_conformer_encoder.py` checkpoint FutureWarning and a transformers slow-image-processor notice |

## Current status

A clean supported-class GPU execution of this exact revision is recorded above (Kaggle T4, fresh container, clean cache, all stages of the default path). Static CI is green on the same commit. The registry status remains **Candidate** until a reviewer confirms the recorded run against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder. The commit that adds a recorded-execution row changes documentation only; the executed source is the commit named in the row.
