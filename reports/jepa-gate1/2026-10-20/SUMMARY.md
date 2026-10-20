# Gate-1 four-arm GH200 objective screen — 2026-10-20

## Run record

- Slurm job: `7560286`
- State: `COMPLETED`, exit code `0:0`
- Start/end: `2026-10-20T01:59:08` → `2026-10-20T02:04:38` UTC
- Elapsed: `00:05:30`
- Resources: 1 node, 4× NVIDIA GH200 120GB, 30-minute limit
- Git SHA verified inside the job: `3fa7cf52b42c7b629898ef7676fed2a411007c00`
- Container: `/projects/u6xn/sif-images/pytorch_26.09-py3.sif` (SHA-256 `f5a41e030ff3d44936faa521480ea9164d095270d7d5bd999621579f3e400ec6`)
- Data: `/projects/u6xn/data/pilot_v1` approved SSL4EO-S12 pilot manifest
- Diagnostic evaluation: 5 steps × 32 samples = 160 unique pilot locations per arm
- Prior failed job: `7549777`; it trained the first arm but failed during evaluation because diagnostic batches reused pilot locations. The evaluator fix is included in `3fa7cf5`.

This is a bounded 50-step objective screen on training-pair cross-view diagnostics. It is **not** downstream benchmark evidence.

## Step-50 results

| Arm | Final loss | SIGReg | Tokens/s | Diagnostic accuracy | Procrustes residual |
|---|---:|---:|---:|---:|---:|
| `clean_jepa` | 0.1417599 | 2.4193058 | 42,824.9 | 0.4375 | 0.051385 |
| `defensive_split_cfm` | 0.2586757 | 2.7190237 | 27,706.6 | 0.48125 | 0.037230 |
| `tessera_mixup` | 2.0078821 | 1.8783541 | 23,423.4 | 0.5125 | 0.074840 |
| `vicreg` | 15.6196928 | 2.7857034 | 46,296.0 | 0.66875 | 0.088707 |

Losses are not directly comparable across objectives.

## Diagnostic readout

- `vicreg` produced the highest strata-pseudolabel diagnostic accuracy (`0.66875`) and the fastest steady-state throughput (`46,296 tokens/s`). Its Matryoshka prefix gaps versus random prefixes were mixed: `+0.0625` at 16 dims, `+0.04375` at 32 dims, `-0.03125` at 64 dims, `0` at 128 dims.
- `defensive_split_cfm` produced the best full Procrustes residual (`0.037230`) and the lowest identifiability alignment loss among the non-clean arms (`0.457579`), but no positive Matryoshka prefix advantage.
- `clean_jepa` had the lowest step-50 loss and stable optimization (`0.141760`), but only moderate diagnostic accuracy (`0.4375`).
- `tessera_mixup` was the weakest arm on this screen: loss increased from `1.563923` at step 10 to `2.007882` at step 50, and its whitening/identifiability diagnostics were poor.

## Artifacts

Raw copied evidence is under `artifacts/`:

- `artifacts/logs/jepa-gate1-4gpu-7560286.out` and `.err`
- `artifacts/logs/jepa-gate1-4gpu-7549777.out` and `.err`
- `artifacts/job-accounting.txt`
- Four arm directories containing:
  - `result.json`
  - `metrics.jsonl`
  - `manifest.json`
  - `checkpoint_index.json`
  - `gate1_diagnostics.json`

Checkpoint tensors remain on Isambard at the paths recorded in each `checkpoint_index.json`; only checkpoint paths and SHA-256 hashes are committed here.
