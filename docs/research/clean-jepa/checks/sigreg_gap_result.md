# Does pooled SIGReg see a modality gap? (numerical check, 2026-10-10, this VM, numpy)

Reference SIGReg (galilai-group/lejepa MINIMAL.md: 17 knots on [0,3], 256 random unit slices, Epps-Pulley statistic x N), N=2048 samples, 20 draws of slices averaged.
Mixture: two components with signs +/-1 (a two-sensor or two-availability-pattern gap), each coordinate renormalised to unit variance.

| embedding dim D | pure Gaussian | gap along 1 coord (mu=1..5) | gap in 8 dims | gap in 32 dims | gap in 128 dims |
|---|---|---|---|---|---|
| 16 | 1.28 | 1.00-1.31 | 141 | - | - |
| 128 (LeJEPA proj_dim) | 1.07 | 1.02-1.10 | 2.7 | 30.8 | 745 |
| 512 | 1.09 | 1.01-1.06 | 1.1 | 3.0 | 29.7 |

A slice aligned with the gap coordinate gives statistic 210.6 for the same two-point mixture.

Reading: with random slices, a gap confined to one coordinate (or a few) in a 128-d projector space is invisible to the pooled statistic (within noise of a Gaussian), so pooled SIGReg can be satisfied by a two-component mixture whose components are separated by sensor or availability pattern, provided the separation occupies a low-dimensional subspace. It is detected only when the gap spans a large fraction of the dimensions. This is the quantitative basis for per-modality / availability-stratified SIGReg (plan v2, sections Q3 and M). Not a theorem; a check with the reference statistic.
