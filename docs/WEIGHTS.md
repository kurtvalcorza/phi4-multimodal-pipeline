# Weight provenance and DIMER hosting

- Upstream: `microsoft/Phi-4-multimodal-instruct`
- Immutable revision: `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- Weight format: sharded SafeTensors
- Upstream weight/code license: MIT
- DIMER hosting: MIT permits redistribution and hosted use with the license/copyright notice preserved. The Git repository does not vendor the ~13 GB snapshot; a DIMER model store may mirror the pinned snapshot separately.
- Critical trust boundary: this upstream release uses custom Python model code and requires `trust_remote_code=True`. DIMER must pin and review the same revision; the public loader refuses to proceed until `allow_remote_code=True` is explicitly supplied.
