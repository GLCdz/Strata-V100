# Changes

## Initial public source snapshot — 2026-10-19

- Based on Strata v0.1.45, upstream commit 477ac13cbe8e582c20484871ce459d20d82df09d.
- Added the optional length-based prefill chunk policy and checkpoint-geometry retention, as deployed on the reference four-V100 machine.
- Included Volta in the pointer-table MMQ comparison test's device gate.
- Published the 256K/full-KV/64-GiB-cache measured profile, sanitized request statistics, power comparison, build provenance and acceptance summary.
- Added portable configuration, benchmark and startup-check tools; bilingual entry points and reproduction instructions.
- Kept upstream and third-party license notices and original README content.

No native concurrent decoding, new mid-prefill preemption, new quantization, or rewritten GPU kernel is claimed. CUDA12.6/sm70 only; other backends unvalidated.
