# Weight provenance and DIMER hosting

- Upstream: `microsoft/Phi-4-multimodal-instruct`
- Immutable revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight format: sharded SafeTensors
- Upstream weight/code license: MIT
- DIMER hosting: MIT permits redistribution and hosted use with the license/copyright notice preserved. The Git repository does not vendor the ~13 GB snapshot; a DIMER model store may mirror the pinned snapshot separately.
- Critical custom-code trust boundary: this upstream release uses custom Python model code and requires `trust_remote_code=True`. DIMER must pin and review the same revision; the public loader refuses to proceed until `allow_remote_code=True` is explicitly supplied.
- Attention backend: DIMER selects `eager` by default for the portable release-reference path. `flash_attention_2` is available only by explicit caller choice and requires a separately installed compatible `flash-attn` build; the loader fails clearly rather than silently changing attention implementations when that dependency is missing.

A mirrored DIMER snapshot should record the exact upstream revision and per-file digests in its offline manifest before it is treated as a distributable DIMER model package. The repository-level revision pin establishes identity but is not a substitute for validating a future offline archive under the DIMER manifest contract.
