# phi4_multimodal_colab.ipynb — Notebook Review (Framework v1)

**Readiness: Needs revision.** The notebook is a long, carefully engineered `E2E` lesson. It loads a digest-verified,
remote-code Phi-4-multimodal snapshot in 4-bit, exercises the text, image and audio paths, and fine-tunes the
checkpoint's own vision LoRA on a pinned VizWiz-Captions slice against two baselines. It then exports a safetensors
adapter and reloads it with a parity check. The reviewed blob `75ab9bb4` is the exact blob recorded as PASSED on a
Kaggle Tesla T4 (2026-09-20). The default path is therefore documented as working, apart from the restart. Five
problems hold it back:

- Run all needs a manual restart after the in-kernel install. The hosted record shows it (PHI-M1).
- The notebook tells the learner to expect the frozen model's CIDEr-D "close to the baselines", and its conclusion
  repeats that. The recorded run shows 0.722 against 0.052, 14 times higher (PHI-M2).
- The documented reruns (BYOD "after the tutorial workflow completes", the optional experiments) cannot work in the
  same session. Section 9 runs `del pipe`. Before Section 9, `pipe` already carries the adapted LoRA, so a rerun
  would score an adapted model as "frozen" (PHI-M3).
- BYOD promises "at least eight photographs". Cell 13 validates every split against an 8-record minimum, so BYOD
  needs at least 50 photographs. Direct execution rejected 8, 12, 20, 30, 40 and 49 (PHI-M4).
- Three bare `assert`s require a predetermined winner (frozen > constant, adapted > frozen, adapted > baselines). A
  BYOD set or experiment with a negative result stops before the adapter is exported and gets no explanation
  (PHI-M5).

Findings: 0 Blocker, 5 Major, 5 Minor, 4 Suggestion. Prefix `PHI`.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/phi4-multimodal-pipeline` |
| Notebook | `tutorials/phi4_multimodal_colab.ipynb` (25 cells: 11 code, 14 markdown; no outputs saved) |
| Reviewed revision | `origin/main` = `2fe7cbd51ff63b302118a3f6ddea61762d520c6e` (GitHub API `commits/main`, 2026-10-04) |
| Notebook blob | `75ab9bb4ec13fa42002fec7c175065936508d09d`, identical to the blob recorded at `1e348e2` in `docs/release-verification.md` |
| Spec baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main` `b1cfe13`). The notebook declares spec `2.0` in `metadata.dimer` |
| Profile / mode | `E2E` / `GUIDED`, standalone carrier (3 modules carried verbatim; generator `tools/build_notebook.py` + `tools/notebook_template.py`; `--check` passes) |
| Audience / prerequisites | Basic Python and PIL; chat prompts, greedy decoding, `trust_remote_code`, LoRA; what BLEU / ROUGE-L / CIDEr-D measure (cell 1) |
| Supported runtime | "Google Colab or Kaggle, Python 3.12" with a CUDA GPU of about 16 GB; Tesla T4 reference; no CPU path (cell 1) |
| Promised outcomes | Pinned install; 26-file snapshot staged and digest-verified (including 5 remote-code files) and loaded in NF4; VizWiz slice fetched, validated and split 208 / 40 / 70 by image with four refusal probes; three capabilities with an input manifest and a rejection probe; two baselines and the frozen model scored; bounded vision-LoRA fine-tuning with validation-CIDEr-D epoch selection; held-out evaluation; before/after captions; safetensors adapter export, model freed, fresh reload with parity; five exports; optional BYOD zip through the same stages; optional experiments (`TRAINED_LAYERS`, `EPOCHS`) |
| Status | Release-grade (README, STATUS, tutorials/README, release-verification), on the Kaggle T4 run of `1e348e2` / `75ab9bb4` |
| Open PRs | None. PRs #1–#9 are merged |

### Evidence actually obtained

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Strong orientation: the contract, the trust boundary, the 4-bit memory envelope, the data provenance and an honest "does not demonstrate" list. Two expectations contradict the recorded run: frozen "close to the baselines" (PHI-M2) and the by-eye reading of adapted captions (PHI-m2). Stale prose (PHI-m1). The 2.2 guided layer is partial, and 108 KB of carried code is not labelled as infrastructure (PHI-m5) |
| Clean default | Documented execution evidence (not re-run here) | Kaggle Tesla T4, 2026-09-20, exact blob `75ab9bb4` (`run_summary.json`, `fetched_blob_verified: true`, HF cache clean): **PASSED**, 11/11 code cells, 2032.3 s. **Pass 1 stopped in the install cell** with `Core dependencies changed while older modules were loaded: numpy: loaded=2.0.2, installed=1.26.4; pillow: loaded=11.3.0, installed=11.1.0`. Pass 2 ran after a restart (PHI-M1). Recorded: frozen CIDEr-D 0.722, adapted 0.890, baselines 0.052 / 0.042; reload parity 7/8 verbatim, ΔCIDEr-D 0.0007. No Colab record exists (PHI-m4). The model was **not** loaded in this review (5.6 B, CPU-only host) |
| Active learning | Source inspection + namespace simulation; **not verified by execution** | The experiments in cell 24 (`TRAINED_LAYERS`, `EPOCHS`) give no rerun cell. A rerun after Section 9 fails: `del pipe` (cell 23) leaves the name undefined, and a simulation gives `NameError: name 'pipe' is not defined`. Before Section 9, `adapt` starts from the current tensors, not the checkpoint's (`frozen_state` is a clone of the live parameters), so a rerun compounds the adaptation (PHI-M3) |
| Reuse and recovery | Direct execution, CPU, of the carried `samples.py` helpers with synthetic JPEG stand-ins (no model); the Colab upload widget was not exercised | Cell 13's BYOD split-and-validate sequence: 8 → `split leaves 5 training records`; 12–49 → `N records; 8..5000 are required` without naming the split; 50, 53, 54, 60 accepted (PHI-M4). A zip without `records.jsonl` gives a bare `StopIteration`. Same-name files in two folders collide silently. `../../evil.txt` is flattened into the root, not rejected (PHI-m3). `google.colab` is imported unconditionally, so BYOD fails on Kaggle (PHI-m3). The default path's four dataset refusal probes and the image-count probe are recorded as rejected in the hosted run |

Limitations: no GPU, no Colab and no model load, so every model-dependent behaviour comes from the hosted record of
the same blob. Local env: `eo-notebook-test` Python with torch 2.13.0+cpu (not the pinned 2.6.0) and Pillow 12.3.0
(not 11.1.0). Only PIL-level helpers were executed, with `CUDA_VISIBLE_DEVICES=-1`. The repository's own static
checks pass: `pytest` 58 passed (the docs say 53), `tools/validate_release_assets.py` PASS, and
`tools/build_notebook.py --check` up to date. These are source checks, not execution evidence (REL8).

## 2. Separate judgments

- **Technical correctness:** the default path is well engineered. Remote code is digest-pinned and re-hashed before
  import, behind an explicit opt-in. The LoRA tensors are kept out of NF4. The upstream `set_lora_adapter` side
  effect on `requires_grad` is neutralised. Frozen tensors are restored on any exception. The adapter manifest is
  checked before deserialisation, and reload parity uses an explicit tolerance. Defects:
  - The install forces a restart (PHI-M1).
  - The in-place model state breaks every documented rerun (PHI-M3).
  - The BYOD minimum is wrong (PHI-M4).
  - Bare outcome assertions (PHI-M5).
  - Archive and upload edge cases (PHI-m3).
- **Promise fulfilment:** the default-path promises are delivered on the hosted record. The BYOD promise ("at least
  eight photographs … re-run from that cell") and the optional experiments are not deliverable as written
  (PHI-M3, PHI-M4, PHI-M5).
- **Learner experience:** the prose is unusually thorough about trust, memory and the limits of the claim. Its central
  interpretive expectation is wrong: the frozen model is not close to the baselines, and the frozen captions are 14.1
  words against 12.1 in the references, so they are not "long" in any material sense. A learner is therefore taught
  to read the result as the opposite of what the run shows (PHI-M2). The before/after guidance does not mention the
  hallucinations visible in the recorded examples (PHI-m2).
- **Spec conformance (2.2):** fails RUN1/RUN10/ENV6 (restart) and DAT12/DAT19 (wrong BYOD limit, a split-blind
  message). REL12 (BYOD) is unverified. §20 says `..` and absolute archive entries MUST be rejected; the notebook
  flattens them instead. Guided-layer SHOULDs are partial, and the declared spec is 2.0.

## 3. Findings

### PHI-M1 — Major: Run all needs a manual restart after the in-kernel pinned install

- **Cell/section:** cell 3 (Section 1, *Install the pinned runtime*); generator `tools/build_notebook.py:61` (install
  line) and the stale-import guard emitted around it.
- **Observed issue:** the cell `pip install`s 15 exact pins (`torch==2.6.0`, `numpy==1.26.4`, `pillow==11.1.0`, …)
  into the running kernel. When a preloaded distribution changes, it raises `Restart the runtime, then rerun from the
  top`. Every current hosted image preloads newer numpy and Pillow, so the guard always fires.
- **Consequence:** `Run all` stops after the first code cell. The learner must restart by hand and run again, which
  violates the one-pass contract. The 2.6.0 torch install is also the largest download of the run, before any lesson
  content.
- **Evidence (documented execution):** Kaggle v5 `run_summary.json` for blob `75ab9bb4`: pass 1 `ok: false`,
  `RuntimeError: Core dependencies changed while older modules were loaded: numpy: loaded=2.0.2, installed=1.26.4;
  pillow: loaded=11.3.0, installed=11.1.0`. Pass 2 ran after a restart. `docs/release-verification.md` records "(1
  restart after the install cell)", and step 4 calls it "expected".
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern** instead of installing into the
  kernel. A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a
  hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in
  that environment, so the kernel's NumPy, Pillow and torch are never replaced. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Make
  the change in the generator (`tools/build_notebook.py` install block, `tools/notebook_template.py`), not by hand.
  Update `docs/release-verification.md` step 4 so it no longer calls a restart "expected".
- **Acceptance check:** a fresh Colab T4 (or Kaggle T4) Run all of the regenerated blob completes in one pass with no
  error output in any cell and no restart. The recorded run states "no restart".
- **Spec:** RUN1, RUN10, ENV6, REL2.

### PHI-M2 — Major: the expected result and the conclusion say the frozen model scores "close to the baselines"; the recorded run shows it 14× higher

- **Cell/section:** cell 16 (Section 6 intro: "Expect the frozen model fluent but long — its CIDEr-D sits close to the
  baselines because the references are short"); cell 24 (*Interpretation*: "CIDEr-D scores that style close to a
  caption that never looks at the image"); cell 0 ("captions these photographs fluently but at length").
  Generator: `tools/notebook_template.py:242`, `:430`, and the opening text near `:51`.
- **Observed issue:** in the hosted run of this blob, the frozen model's test CIDEr-D is **0.722**. The constant
  caption scores **0.052** and the colour neighbour **0.042**, a ratio of 14.0. BLEU-4 is 0.171 against 0.041, and
  ROUGE-L 0.439 against 0.336. The frozen mean caption length is 14.09 words against 12.07 in the references, so the
  "long captions" premise is a 2-word gap. The adaptation gain, +0.168 CIDEr-D, is about a fifth of the frozen
  model's lead over the baselines.
- **Consequence:** the learner is told what to see, sees the opposite, and is then given a conclusion that repeats
  the wrong reading. The notebook's own rule in cell 24 ("if the frozen model already clears them by a wide margin,
  adaptation buys style, not competence") applies to this exact run, but the text never says so. The lesson teaches
  the wrong conclusion about what the adaptation achieved.
- **Evidence (documented execution + source inspection):** Kaggle v5 `executed.ipynb` cells 17 and 21 and
  `outputs/phi4_multimodal_result.json` `comparison`. Probe P7 confirms that the three phrases are in the reviewed
  blob.
- **Recommended correction:** rewrite the Section 6 expectation and the first paragraph of the interpretation in the
  template to match the measured pattern, without hard-coding numbers. The frozen model is already far above both
  non-neural baselines. Adaptation moves CIDEr-D and BLEU-4 further, mainly by matching the references' length and
  phrasing. Point the learner to cell 24's "wide margin" rule as the right reading. Drop "at length" or quantify it
  against the references' mean length, which the notebook already prints.
- **Acceptance check:** no learner-facing cell says the frozen score is "close to" the baselines or to the constant
  caption. The Section 6 expectation is consistent with the recorded frozen/baseline ratio. The interpretation
  states that the frozen model clears the baselines by a wide margin and what adaptation adds on top.
- **Spec:** UX4, GDL8, GDL14, ENV9.

### PHI-M3 — Major: the documented reruns (BYOD after completion, optional experiments) cannot work in the same session

- **Cell/section:** cell 0 (BYOD: "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and
  re-run from that cell"); cell 24 (*Optional experiments*: `TRAINED_LAYERS = 4 / 32`, raise `EPOCHS`, BYOD); cell 23
  (`del pipe`); `pipeline.py` `adapt`. Generator: `tools/notebook_template.py:54` and `:387`.
- **Observed issue:**
  1. Section 9 runs `del pipe` to free memory for the reload. Every later rerun of Sections 5–8 uses `pipe` (P2: load
     sites in cells 15, 17, 19 and 21; the only assignment is in cell 11). Rerunning from Section 4 or Section 7 after
     completion raises `NameError: name 'pipe' is not defined`.
  2. Before Section 9, `pipe` holds the adapted vision LoRA (`adapt` restores the *best* epoch in place). `adapt`
     snapshots the current tensors as its "frozen" state, not the checkpoint's own. Rerunning Section 6 would
     therefore label the adapted model "frozen", and rerunning Section 7 would fine-tune on top of the previous run.
  3. None of the experiments says which cell to rerun from.
- **Consequence:** the BYOD instruction as written fails on its first model call. An experiment rerun either crashes
  or, without Section 9, silently produces an invalid comparison: "frozen" is adapted, and epoch 0 is not the
  checkpoint. Learners lose a long GPU run, or draw conclusions from contaminated numbers.
- **Evidence (source inspection + namespace simulation):** P2. `cell23.del = [25]`; `adapt_frozen_state_is_current_tensors:
  true`; `adapt_resets_to_checkpoint_lora: false`; simulated rerun → `NameError`. Not executed with the model.
- **Recommended correction:** tell learners to rerun from **Section 3** (fresh base load) for BYOD and for every
  experiment, or run BYOD as a fresh Run all with `USE_BYOD = True`. Add a guard at the top of Sections 5–7 that
  raises an actionable error when `pipe` is missing or `pipe.adapter is not None`, for example "the base model was
  freed or already adapted — rerun from Section 3". Optionally make `adapt` refuse an already-adapted pipeline.
- **Acceptance check:** following the BYOD and experiment instructions word for word in one session reaches a fresh
  frozen evaluation and adaptation. If the instructions are followed wrongly, the learner gets a named error that
  says which section to rerun, not a `NameError` or a silently adapted "frozen" row.
- **Spec:** RUN9, UX7, GDL10, DAT13, UX10.

### PHI-M4 — Major: BYOD promises "at least eight photographs"; cell 13 needs at least 50, and its rejection does not name the split

- **Cell/section:** cell 0 (BYOD paragraph: "at least eight photographs"); cell 1 ("A dataset needs 8..5,000
  records"); cell 13 (`dataset_manifests = {name: validate_dataset(part) …}` with the default `min_records=8` for every
  split after `split_dataset` with 15% / 20% fractions). Generator: `tools/notebook_template.py:56`, and the cell-13
  block near `:124–160`.
- **Observed issue:** `split_dataset` puts 20% into test and 15% into validation. Cell 13 then requires **each** split
  to hold 8..5,000 records, so validation needs 8, which takes about 50 photographs.
- **Evidence (direct execution, CPU, synthetic 64×48 JPEGs; cell 13's BYOD sequence with `seed=42`):**

  | Photographs | Split sizes (test/val/train) | Verdict |
  |---|---|---|
  | 8 | — | rejected: `split leaves 5 training records; at least 8 are required` |
  | 12 | 2 / 2 / 8 | rejected: `2 records; 8..5000 are required` |
  | 20 | 4 / 3 / 13 | rejected: `4 records; 8..5000 are required` |
  | 30 | 6 / 4 / 20 | rejected: `6 records; 8..5000 are required` |
  | 40 | 8 / 6 / 26 | rejected: `6 records; 8..5000 are required` |
  | 49 | 10 / 7 / 32 | rejected: `7 records; 8..5000 are required` |
  | 50 / 53 / 54 / 60 | 10/8/32 · 11/8/34 · 11/8/35 · 12/9/39 | accepted |

- **Consequence:** a learner who prepares 8–49 photographs, as the stated contract allows, is rejected after
  uploading. The message names a count that matches nothing they uploaded and does not name the split. The DAT12
  limits stated before upload are wrong by a factor of about 6.
- **Recommended correction:** either state the real minimum (about 50 photographs at the default fractions, or the
  formula), or validate the splits with the minima `adapt` actually needs (`adapt` uses `min_records=1` for
  validation). In both cases, prefix the per-split error with the split name and the uploaded total. Make the change
  in the template and, if the minimum changes, in `samples.py`.
- **Acceptance check:** the stated BYOD minimum is accepted and one record fewer is rejected, with a message that
  names the split and the requirement. A regression test covers both edges.
- **Spec:** DAT12, DAT19, VAL6, UX10.

### PHI-M5 — Major: bare outcome assertions require a predetermined winner and stop BYOD/experiments before export

- **Cell/section:** cell 17 (`assert frozen_test['cider_d'] > baseline_constant['cider_d']`); cell 21 (`assert
  adapted > frozen`, `assert adapted > max(baselines)`); cell 23 (`assert parity … <= 0.02`). Generator:
  `tools/notebook_template.py:264`, `:344–345`, `:403`.
- **Observed issue:** all four `assert`s have no message (P6). Three of them require a particular experimental
  outcome. They run on the BYOD and experiment paths too, where a negative result is legitimate. Cell 24 even
  anticipates one: "if the frozen model already clears them … adaptation buys style".
- **Consequence:** suppose a BYOD dataset where adaptation does not beat the frozen model, or an experiment such as
  `TRAINED_LAYERS = 4` with one epoch. The notebook stops with a bare `AssertionError` before the evaluation report
  of Section 8 is written, and before the adapter export, reload and `result.json` of Section 9. The learner loses
  every promised downstream stage and gets no explanation. The notebook also teaches that a negative result is an
  error rather than a finding. The 2.2 reference notebook preserves negative held-out results.
- **Evidence (source inspection):** P6 lists the four asserts. Not executed with the model.
- **Recommended correction:** keep the hard checks for the default sample path only (`if not USE_BYOD`). Give each
  check a message that says what failed and what it means. On BYOD and experiment paths, record the comparison
  outcome as a named verdict (e.g. `adapted_beats_frozen: false`) in the report and continue to export and reload.
  Keep the parity tolerance as a real check, with a message.
- **Acceptance check:** with `USE_BYOD = True` and a set on which adaptation does not win (or with the outcome
  checks forced false in a test), Sections 8–9 still write the evaluation report, the adapter, the reload parity and
  `result.json`, and the verdict is visible. On the default path, a failing check prints a message, not a bare
  `AssertionError`.
- **Spec:** RUN9, DAT14, UX10, GDL14.

### PHI-m1 — Minor: stale or imprecise statements

- **Cell/section and evidence (source inspection, compared with the hosted record):**
  - Cell 24 says "the adapter reloads caption for caption into a fresh pipeline". The recorded run reproduced **7 of
    8** captions; the eighth lost a trailing full stop. Cell 22's own text says this correctly.
  - Cell 24 says `TRAINED_LAYERS = 32` trains "400 M parameters". 32 layers × 64/8 tensors = 4 × 92,274,688 =
    **369,098,752**.
  - Cells 0 and 1 say "about 45 minutes on a T4", but the recorded wall time is 2032.3 s (≈34 min). The figure is
    neither labelled as an estimate nor tied to the record.
  - README line 14 says "Adaptation in this repository: none; inference only", which contradicts the E2E adaptation
    contract described in the same README.
  - The docs give "53 tests"; `pytest` collects 58.
- **Consequence:** small credibility losses. The reload statement overclaims what VER4 measured.
- **Recommended correction:** fix in `tools/notebook_template.py` (`:51`, `:430`, `:457`), README and STATUS.
- **Acceptance check:** none of the five statements survives. The runtime figure cites the recorded environment or
  is labelled as an estimate.
- **Spec:** SRC3, UX12, VER5.

### PHI-m2 — Minor: before/after guidance reads the adapted captions as "more literal" while the shown examples introduce unsupported content

- **Cell/section:** cell 22 ("shorter, more literal captions that name the held object … are what the VizWiz
  convention rewards"); cell 23's six examples.
- **Observed issue (documented execution):** in the recorded run, two of the six adapted captions add content the
  frozen caption and the reference do not support:
  - test-0001 → "A black and white keyboard with **a Dell monitor on the screen**". The reference is "a keyboard with
    a lot of numbers".
  - test-0004 → "A **black iPhone**". The frozen caption says "flip phone"; the reference says "smartphone with a full
    qwerty keyboard".
- **Consequence:** the learner is primed to read a higher CIDEr-D as more faithful captions. The metric rewards
  length and phrasing match, not faithfulness, and the notebook never names hallucination as an error mode to check.
- **Recommended correction:** add a *What to notice* prompt asking the learner to check each adapted caption against
  the photograph for invented objects or brands. State that CIDEr-D / BLEU do not penalise a fluent wrong noun much.
- **Acceptance check:** Section 9 names hallucination or unsupported content as a failure mode to inspect, and its
  expectation no longer asserts that adapted captions are "more literal".
- **Spec:** EVAL15, UX4, GDL8.

### PHI-m3 — Minor: the BYOD upload works only on Colab, and archive edge cases are unhandled

- **Cell/section:** cell 13 (`if USE_BYOD: from google.colab import files`, flatten-extract,
  `next(p for p in (records.jsonl, records.json) if p.is_file())`). Generator: `tools/notebook_template.py:124–140`.
- **Observed issue (direct execution of cell 13's extraction logic, P4/P5):**
  1. Kaggle is a stated supported runtime, but `google.colab` is not importable there, so `ModuleNotFoundError`. There
     is no location field for a file already in the runtime (EXE2).
  2. A zip without `records.jsonl` raises a bare `StopIteration`.
  3. Two files with the same name in different folders are silently collapsed into one.
  4. `../../evil.txt` and absolute entries are flattened into the root rather than rejected. Root containment holds,
     but §20 says such entries MUST be rejected.
  5. There is no limit on expanded size.
- **Recommended correction:** add a `BYOD_ZIP_PATH` form field, read it when set, and import `google.colab` only when
  it is empty. Reject traversal and absolute entries and duplicate basenames with named errors. Raise
  `FileNotFoundError('records.jsonl not found in the zip')`. Cap the expanded size.
- **Acceptance check:** a zip passed via `BYOD_ZIP_PATH` on Kaggle or Jupyter reaches validation. The four bad
  archives are rejected with messages that name the problem.
- **Spec:** DAT16, DAT19, EXE2, UX10, §20.

### PHI-m4 — Minor: execution-evidence gaps

- **Observed issue (documented evidence):**
  - All E2E records are Kaggle T4 runs. There is no Colab run, although the Colab badge is the primary entry point,
    the notebook is named `_colab`, and REL11 names Colab/Jupyter.
  - REL12 (BYOD accepted, one incompatible input rejected, downstream stages reached) is not recorded.
  - No active-learning experiment is recorded.
- **Recommended correction:** after PHI-M1, record one Colab T4 run of the new blob, one BYOD run (≥ the corrected
  minimum) with a recorded rejection, and one experiment rerun per PHI-M3's instructions.
- **Acceptance check:** `docs/release-verification.md` has a Colab row and a REL12 row for the same blob.
- **Spec:** REL1, REL11, REL12.

### PHI-m5 — Minor: the spec 2.2 guided layer is partial and the notebook declares spec 2.0

- **Observed issue (source inspection):** present: audience/prerequisites (GDL1), Input → Output contract (GDL4),
  section *Look for* notes, the interpretation and limits. Absent:
  - **How to use this notebook** (GDL2) and a roadmap (GDL3).
  - A glossary for CIDEr-D, NF4, LoRA and remote code (GDL6).
  - Predictions before results (GDL7).
  - Checkpoints with sample answers (GDL9).
  - A Predict → Change → Run → Observe → Explain activity (GDL10).
  - **Infrastructure** labels and collapsed views on the 52 KB, 44 KB and 12 KB carried cells (GDL11).
  - Troubleshooting for OOM, download failure or digest mismatch (GDL13).
  - A conclusion template (GDL14).

  `metadata.dimer.notebook_spec` is `2.0`.
- **Recommended correction:** adopt the 2.2 layer in the template, following the §25.13 reference notebook, and bump
  the declared spec.
- **Acceptance check:** each listed GDL item is present, the carried cells are titled `Infrastructure: …` with
  `cellView: form`, and the metadata declares 2.2.
- **Spec:** GDL2, GDL3, GDL6, GDL7, GDL9, GDL10, GDL11, GDL13, GDL14.

### PHI-S1 — Suggestion: demonstrate the combined image + audio prompt the README advertises

The README offers "combined image+audio prompting", but the notebook shows each modality alone. The example clip
asks "What is the traffic sign in the image?", and the notebook already draws a sign, so one `generate(images=[sign],
audios=[audio])` call would show the combined path at almost no cost.

### PHI-S2 — Suggestion: show the example clip's known utterance beside its transcription

The recorded audio output is exactly the clip's file-name sentence. Printing that expected text lets the learner
judge the transcription, which the `not-measurable` verdict otherwise leaves unexamined.

### PHI-S3 — Suggestion: make `TRAINED_LAYERS = 4` a Predict → Change → Run → Observe → Explain activity

Pair it with PHI-M3's rerun instruction. Ask for a prediction of the CIDEr-D gain relative to 8 layers, and add a collapsible sample answer.

### PHI-S4 — Suggestion: report CIDEr-D per category (`text` / `no-text`)

The split already carries the category (test: 40 text, 30 no-text). Reading printed text is the VizWiz skill users
most need, so a per-category row would show where adaptation helps.

## 4. Readiness

**Needs revision.** No Blocker remains open. The default path has hosted evidence for the exact blob, apart from the
restart. Five Majors are open:

- the restart (PHI-M1);
- the wrong central expectation and conclusion (PHI-M2);
- reruns that cannot work (PHI-M3);
- the wrong BYOD minimum (PHI-M4);
- outcome assertions (PHI-M5).

Unmet applicable MUSTs: RUN1, RUN10, ENV6, DAT12, DAT19, REL12 (unverified) and the §20 rejection rule. Remaining
gates after the fixes: a one-pass Colab T4 run of the regenerated blob, a REL12 BYOD record, and one experiment rerun
recorded per the corrected instructions. The registry's **Release-grade** label describes the 2026-09-20 run of
this blob. It does not cover these journeys.

## 5. Verified versus inferred

- **Verified by documented execution (Kaggle v5, blob `75ab9bb4`):** the restart (PHI-M1), all metric values,
  example captions and parity figures cited in PHI-M2, PHI-m1 and PHI-m2.
- **Verified by direct execution (CPU, stand-ins, no model):** the BYOD minimum table (PHI-M4) and the zip edge cases
  (PHI-m3), using the carried `samples.py` code and cell 13's verbatim extraction logic. `google.colab` is not
  importable outside Colab. The repository's `pytest` (58), validator and generator `--check` all exit 0.
- **Source inspection only:** the `del pipe` rerun failure and the in-place adaptation contamination (PHI-M3,
  backed by a namespace simulation, not a model run). The bare assertions (PHI-M5). The guided-layer gaps.
- **Not verified:** any Colab run; BYOD through the upload widget; any experiment rerun; memory claims (9.2 GB peak).
- **Most likely to be wrong:** PHI-M5's severity. On the default sample path the three outcome checks passed with
  wide margins. If Kurt treats BYOD and experiments as best-effort extras, PHI-M5 could fairly be Minor. Its BYOD
  consequence (no export after a negative result) is inferred from source, not observed.

Probe ZIP: `phi4_multimodal_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
Re-run from the repository root with
`CUDA_VISIBLE_DEVICES=-1 python run_probes.py --evidence <Kaggle v5 evidence dir>`.
