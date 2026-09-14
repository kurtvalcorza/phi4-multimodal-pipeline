# Release verification

`tutorials/phi4_multimodal_colab.ipynb` (`MULTI-CAPABILITY`, **standalone** carrier) is a **release candidate**
until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the
notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `MULTI-CAPABILITY`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, generating revision, module, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/phi4_multimodal_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` (26 entries, weights **and** the eight upstream
  `.py` files) equal to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime
  pins; the notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the
  same identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Phi4MultimodalPipeline.from_pretrained(allow_remote_code=True, weights_dir=...)`, `validate_inputs` once per
  capability, `generate` three times, `evaluation_report`), the ceiling print (`MAX_IMAGES`, `MAX_AUDIOS`,
  `MAX_NEW_TOKENS`, `MAX_TEMPERATURE`), the synthetic octagon and the checkpoint example clip as samples, the exports,
  the learner-facing statements (trust boundary, per-capability input/output contracts, generated text is not a
  calibrated statement, `not-measurable` by construction) and the gated-off BYOD default listed in the validator;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `transformers` / `datasets` / `urlopen` / `huggingface_hub` use **outside the
  carried module cell**, any worker process or subprocess outside the generator-owned install cell, `pickle.load`,
  `torch.load(`, `extractall(`; `trust_remote_code=True` is forbidden in every notebook cell and permitted only inside
  the carried module cell, where it sits behind the `allow_remote_code` opt-in and the digest check);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_snapshot.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`, `tests/test_release_assets.py`;
stubbed `transformers`/`torch`, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CUDA runtime with ≥16 GB GPU memory | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle `NvidiaTeslaT4` kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used when the kernel pre-imports a distribution the pinned install replaces (Pillow 11.3.0 vs the pinned 11.1.0, numpy): the stale-import guard correctly halts the in-kernel path, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA GPU runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from`, `cuda: True`, and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `Phi4MultimodalPipeline`, `validate_inputs`, `evaluation_report`,
     `REMOTE_CODE_FILES` and the ceilings) with no import of the repository package;
   - pinned `microsoft/Phi-4-multimodal-instruct` acquisition at the immutable revision through the package: the
     inline `MANIFEST` is asserted against the module identity and written to `weights/phi4-multimodal-instruct/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all 26 manifest entries on a clean runtime,
     `verify_snapshot` returns the manifest dict **after hashing the three shards against the Hub-derived digests and
     the eight `.py` files against their pinned digests — the first time either is checked against bytes anywhere**,
     and `from_pretrained(allow_remote_code=True, weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'`
     with `attention_implementation == 'eager'` (upstream warns that flash-attention is not used);
   - the synthetic octagon image and the checkpoint example clip prepared with their digests printed;
   - `validate_inputs` writes `outputs/phi4_multimodal_input_manifest.json` with three accepted manifests and one
     recorded rejection finding (the five-image probe);
   - three generations through `generate` — text-only, image + text, audio + text — each non-empty (the previous
     notebook's runs produced an ASR explanation, a stop-sign description of a photograph, and a LibriSpeech
     transcript; the samples changed, so the standalone outputs must be read afresh, not compared);
   - `evaluation_report` writes `outputs/phi4_multimodal_evaluation_report.json` with verdict `not-measurable` and
     `non_empty: true` for all three capabilities;
   - `outputs/phi4_multimodal_result.json` and `outputs/phi4_multimodal_generations.txt` written with
     `NOTEBOOK_SOURCE`, model revision, model licence, the trust-boundary record, runtime versions, device and
     attention backend;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, GPU), model
   identifier and immutable revision, whether the model cache was clean, outcome, produced outputs, peak CUDA
   allocation, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/phi4_multimodal_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/phi4_multimodal_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `813c993` / `efdde43e6259` | Kaggle T4 (`kurtvalcorza/dimer-nb2-phi4-multimodal` v2) | Default sample path | 378.4 s | **PASSED** — 10/10 ok code cells executed cleanly, 54 files, 11173 MB staged |

### Previous repository-installing notebook (NOTEBOOK_SPEC 1.0) — audit trail, does not cover the standalone carrier

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `2c80d53acf4e` / blob `31bbfce8250c` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-phi4-multimodal-t4-verify-v2` v1 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default path, all 6 code cells, clean HF cache (afterwards: `models--microsoft--Phi-4-multimodal-instruct` and the LibriSpeech dummy dataset only — no other repository was fetched) | 363.3 s (246 s install, 96 s snapshot fetch + load, 21 s for the three generations) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0, torchvision 0.21.0, transformers 4.48.2, **pillow 11.1.0 (the pin, no longer a mixed state)**, accelerate 1.3.0, datasets 4.4.0, soundfile 0.13.1, scipy 1.15.2, peft 0.13.2; `allow_remote_code=True` at the pinned revision, `attention_implementation='eager'`, bf16 weights, peak CUDA allocation 12.74 GiB; the default image digest matched the pinned sha256 `2070aff9…c1552` and the export records both sample identities; text/image/audio outputs identical in substance to the v1 run (audio transcript matches the LibriSpeech reference); `outputs/phi4_multimodal_results.json` sha256 `605cad6c…90937` |
| 2026-09-11 | `05a7e0c487e5` / blob `8fd63008f8b0` (this revision) | Kaggle kernel `kurtvalcorza/dimer-phi4-multimodal-t4-verify` v1 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; the committed notebook pushed verbatim plus one leading shim cell (`DIMER_TUTORIAL_REF`, scratch cwd, `google.colab` shim); the kernel pre-imports numpy 2.0.2 (not pinned here) and Pillow 11.3.0 — **Pillow is pinned (`pillow==11.1.0`)**, so this run executed with Pillow 11.3.0 in memory and 11.1.0 on disk, a mixed state the two-name stale-import guard of that revision could not detect (reviewer finding A-01); superseded by the fresh-interpreter run recorded above it | Default path, all 6 code cells, clean HF cache (`hf cache clean: True`) | ≈5.5 min (05:37:25 → 05:42:54 UTC, including the ~13 GB snapshot) | **PASS** — `repository_revision` equals the candidate commit; `microsoft/Phi-4-multimodal-instruct` acquired at the pinned revision with `allow_remote_code=True` and `attention_implementation='eager'` (upstream warns "not running the flash-attention implementation"); torch 2.6.0+cu124, transformers 4.48.2, accelerate 1.3.0, datasets 4.4.0, soundfile-decoded audio; bf16 weights executed on sm_75 with peak CUDA allocation 12.74 GiB of 14.56; text: "Automatic speech recognition (ASR) is a technology that enables computers to interpret and process human speech into text…"; image: "…a red octagonal stop sign with the word 'STOP' written in white capital letters."; audio: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel" (matches the LibriSpeech reference); `outputs/phi4_multimodal_results.json` sha256 `0e063a95…ae1c95`; the kernel log cannot show Hugging Face fetches, so the absence of a cross-repository tokenizer fetch is established by code inspection only (the processor loads its tokenizer files from the pinned repository) — the fresh-interpreter re-run records the cache listing; warnings: upstream `speech_conformer_encoder.py` checkpoint FutureWarning and a transformers slow-image-processor notice |

## Current status

**No clean-runtime execution of the standalone notebook has been recorded yet**; clean GPU execution evidence is now recorded below. The rows above under the previous notebook prove that the pipeline's three capability paths,
the pinned ~13 GB snapshot fetch through the Hub loader with `allow_remote_code=True`, and eager attention on a T4
produced sensible outputs in a clean Kaggle container, but they executed the earlier repository-installing carrier
with different samples (a third-party stop-sign URL and a LibriSpeech dataset row, both dropped because the standalone
primary path may reach only the Hugging Face Hub). The standalone path (carried module cell, inline 26-entry manifest,
`stage_missing_files` through `hf_hub_download`, `verify_snapshot` over the real shards **and the remote code**, and
`transformers` loading from the verified directory) has been validated statically only (parity PASS, carrier probe
with the repository package blocked) and never run. Three facts a reviewer should weigh: the shard digests were taken
from the Hub's LFS metadata at the pinned revision and have never been checked against downloaded bytes; the eight
`.py` digests were computed from the files fetched to the build machine at the pinned revision and are the only
locally verified entries; and `from_pretrained(weights_dir=...)` was exercised only with stubbed `transformers`
classes — whether the upstream remote code resolves everything it needs from a local directory (rather than a Hub id)
is inspected, not executed. The registry status remains **Candidate** until a reviewer confirms a recorded run against
the notebook blob under review and an integrator promotes it; promotion is not performed by the builder.