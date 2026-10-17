# GH200 GPU readiness run — 2026-10-17

Agent-run jobs on Isambard-AI Phase 2 after the human authenticated Clifton
(cert for project u6xn). Checkout: detached `30d6b3d` (= PR #3 head), clean tree.
SIF `/projects/u6xn/sif-images/pytorch_25.05-py3.sif`, SHA-256
`41bc95eb3ad44a361cfd795d03fa15a954a4afe0b924a2a69cc1c9d335c2a2a8` — identical
to the image used by the historical JEPA run (job 7365198).

## Results

| Job | Mode | GPUs | State | Outcome |
|---|---|---|---|---|
| 7506782 | single | 1 | COMPLETED (57s, exit 0) | `passed_synthetic_checks_only` |
| 7506803 | distributed | 2 | COMPLETED (50s, exit 0) | `passed_synthetic_checks_only` on both ranks |

Hardware: NVIDIA GH200 120GB, driver 580.173.02, CUDA runtime 12.9, cuDNN 9.10.1,
NCCL 2.26.5, torch `2.8.0a0+5228986c39.nv25.05` inside the NGC 25.05 container
(Python 3.12.3). Apptainer 1.4.1-1.brics on the host.

Checks passed (both ranks where applicable): `hardware_and_runtime`,
`bf16_matmul_vs_fp32`, `bf16_attention_forward_backward`, `nccl_all_reduce`,
`synthetic_image_training`, `checkpoint_write_read_continue`,
`ddp_parameter_consensus`.

NCCL on the same node initialized via P2P/CUMEM over the Slingshot NICs
(`NET/IB: No device found` on the login-facing interface is expected — intra-node
P2P was used and `nccl_all_reduce`/`ddp_parameter_consensus` passed). Multi-node
NCCL over Slingshot remains untested.

## Scope

`passed_synthetic_checks_only` means the generic CUDA/BF16/NCCL runtime is
healthy. It is **not** model readiness: the historical JEPA SyncBatchNorm
mismatch (job 7365198, tolerance exceeded 18.8% of elements) is model-specific
and must be re-tested separately.

## Evidence

- `gpu-readiness-7506782.out` / `.err` — single-GPU job log
- `gpu-readiness-7506803.out` / `.err` — 2-GPU distributed job log (NCCL init)
- `readiness-single-7506782.json`
- `readiness-distributed-rank0-7506803.json`, `readiness-distributed-rank1-7506803.json`
- Host inspection: `host-inspection-20261017T201816Z.json`, produced by
  `scripts/inspect_training_setup.py` on the login node
