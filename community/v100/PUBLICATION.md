# Publication validation — 2026-10-19

This initial public snapshot is based on upstream commit
`477ac13cbe8e582c20484871ce459d20d82df09d` and the exact deployed source patch.

Checks completed while preparing this publication:

- Three modified source files compared byte-for-byte with the tested deployment sources.
- 20 Python unit tests passed: 11 startup guard positive/negative cases, six configuration cases and three request-association cases. Fixtures are synthetic; tests need no GPU.
- The portable startup checker passed against the existing live 256K deployment, checking its actual hugepage allocation, full experts/graphs, context and configured switches.
- One live synthetic arithmetic API smoke passed through the public runner: SSE parsing, token usage, metrics association and expected answer hash checked.
- The public configuration generator reproduced the selected profile's engine flags and environment from the optimized128K base configuration.
- Public short prompts and normalized fresh-input corpus matched the reference benchmark construction.
- Curated files were scanned for the production API key, GPU UUIDs, private deployment/development paths, host names and private network domain. No matches.
- Upstream LICENSE and third-party notices retained. Private research directories, model weights, runtime binaries, keys and conversation histories excluded.

The prior CUDA numerical/behavior/performance/soak tests are reported in
BENCHMARKS.md and results/, with their original scope. They were not rerun or
relabeled as new during publication. This preparation does not establish HIP,
SYCL, Windows or different-hardware compatibility.

The included deployed-source.patch has the source_patch_sha256 recorded in
PROVENANCE.json and applies to the stated upstream baseline. Runtime binary
hashes identify the measured build; compiled binaries are not distributed here.
