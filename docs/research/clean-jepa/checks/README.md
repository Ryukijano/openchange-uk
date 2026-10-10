# Synthetic checks (provenance only)

`sigreg_gap.py` / `sigreg_gap_result.md`: a NumPy-only check that pooled SIGReg can
miss a single-coordinate modality gap while a stratified statistic detects it. Toy
synthetic data, one statistic — motivation for the stratified-SIGReg experiment,
not an EO result and not part of `make check`.
