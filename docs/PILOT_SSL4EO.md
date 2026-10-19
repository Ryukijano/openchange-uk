# SSL4EO-S12 pilot dataset (local-fit)

- Source: embed2scale/SSL4EO-S12-v1.1, revision 97a91733, CC-BY-4.0, `train/` shard_000001 (S1GRD + S2L2A). Approved in `configs/sources.yaml` as `use: imagery`, `splits: [pilot]`.
- Build: `PILOT_DATA_ROOT=<dir> python scripts/build_ssl4eo_pilot.py` after downloading the two tars, extracting to `zarr_s1/`, `zarr_s2/`, and placing `train_metadata.parquet` beside them.
- Output: `pilot_v1/{manifest,split,normalization}.json`, `geo_audit.txt`, `records/*/s1.npy|s2.npy|s2_valid.npy` (SHA-256 checksummed).
- Selection: <=500 locations; geographic groups are 10-degree lon/lat cells, the two smallest cells are validation, next two test; records are train-only. One pair per location, season chosen by minimum |S1-S2| timestamp gap, paired through `sample_id` suffixes because the S1 and S2 zarr stores order seasons differently. Pairs capped at a 30-day S1-S2 gap (median ~1.7 days).
- Arrays: S1 float32 [2,264,264] dB (VV,VH); S2 float32 [12,264,264] reflectance (int16/10000); s2_valid bool = cloud_mask==0; cloud_fraction checked against the mask exactly. Normalization constants computed on train locations only and pinned to the split hash.
- Validation: LocalObservationDataset accepts all records; `train_jepa_distributed.py --dry-run --manifest ...` passes on the real data.

## Gate 1 objective screen

`slurm/07_jepa_gate1_4gpu.sbatch` runs four matched 50-step, seed-0 arms on the same manifest and batch quotas: baseline `clean_jepa`, `defensive_split_cfm` (64 shared + 32 S1-private + 32 S2-private dimensions, seven-day gap weighting, and an S1-private→S2-private conditional-flow bridge), `tessera_mixup`, and `vicreg`. Each arm writes the usual metrics/checkpoints plus `gate1_diagnostics.json` from `scripts/evaluate_jepa_gate1.py` (cross-view identifiability bound, SFA eigenvalues, subspace-Procrustes residual, and a strata-pseudolabel Matryoshka diagnostic). This is a bounded pipeline screen, not a downstream benchmark.
