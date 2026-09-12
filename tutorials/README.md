# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `phi4_multimodal_colab.ipynb` | `MULTI-CAPABILITY` | text generation; image+text generation; audio+text transcription/generation | CUDA GPU (`device='cuda'`, ≥16 GB) | single image, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence is recorded in `../docs/release-verification.md` and must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook executes pinned upstream custom code. `Phi4MultimodalPipeline.from_pretrained` refuses implicit trust; the tutorial opts in with `allow_remote_code=True` only after printing the immutable revision, and never calls `transformers` directly.
- `eager` attention is the portable default; `flash_attention_2` is an explicit opt-in that the tutorial does not use.
- `USE_BYOD_IMAGE` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
