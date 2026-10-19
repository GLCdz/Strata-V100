# Strata-V100: a measured four-V100 community fork

[中文](README.md) · [Results and limits](BENCHMARKS.md) · [Reproduction](REPRODUCE.md)

Based on Niko1221/Strata v0.1.45, commit `477ac13cbe8e582c20484871ce459d20d82df09d`.
Reference hardware: four V100-SXM2-16GB cards, EPYC 7402, about 252 GiB system RAM,
Linux, CUDA 12.6, Qwen3.8-Flash-Next IQ3_S, 300 W power cap per card.

The measured profile holds all 24,576 experts and eight verifier graphs, uses
262,144-token context with full INT8 KV residency, and allows up to 64 GiB / 16
RAM conversation-cache entries. Final medians: 4,100.6 tok/s on fresh 130K input,
4,337.1 tok/s on fresh 240K input, and 148.4 tok/s for the short-code decode task.
A fixed 240K-background code task on the same configuration decoded at
94.7 / 90.9 tok/s cold / warm during candidate selection.
These are workload-specific measurements, not universal speed guarantees.

The local C++ patch adds an opt-in length-based prefill chunk policy and preserves
that geometry through checkpoint restoration. Existing kernels, MTP, MMQ direct,
pipeline windows and cache implementation are upstream work, not new algorithms
introduced by this fork. The related pointer-table test now includes Volta.

This is a single-sequence deployment: multiple model requests queue. Cache slots
are not concurrent generation slots. A long fresh prompt can delay other users;
no new mid-prefill preemption was implemented. Only CUDA 12.6 / sm70 was tested;
HIP, SYCL, Windows and different GPU layouts were not validated.

Start with REPRODUCE.md. The config generator retains your setup-generated model
paths and writes a separate file; no private keys, model weights, conversation
history, runtime binaries or machine-specific deployment files are included.
The root MIT license and all third-party notices remain in place. See NOTICE.md.
