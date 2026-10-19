# Attribution

This fork is based on [Niko1221/Strata](https://github.com/Niko1221/Strata),
MIT License, copyright 2026 Niko1221 and the Strata contributors.
The root LICENSE and third-party license notices are retained.

GLCdz maintains this four-V100 deployment profile, local length-chunk patch,
benchmark harness and measurement report. Preparation and engineering used
ZCode and Codex; all published performance figures came from the stated hardware.
Existing SM70 kernels, MMQ direct, MTP, pipeline windows, checkpoint/cache
mechanisms and serving APIs belong to the upstream implementation.
The local patch does not introduce those algorithms or native concurrent decoding.

corpora/strata-paper.txt is an extracted text form of the upstream
MIT-licensed docs/paper/Strata-Paper.pdf, used for the reference fresh-input probes.
No model weights or third-party model licenses are replaced by this fork's MIT license.
