# GH200 container smoke, Slurm job 7365120

One-GPU smoke on Isambard-AI Phase 2: architecture, CUDA and one 8192x8192 bfloat16 matmul. No data, no training.

- Commit: 4d315ccb5be8918ddc1a658bd821c10a261aa1df (clean checkout)
- Node: nid010010, aarch64, 1x NVIDIA GH200 120GB
- Container: nvcr.io/nvidia/pytorch:25.05-py3, SIF sha256 41bc95eb3ad44a361cfd795d03fa15a954a4afe0b924a2a69cc1c9d335c2a2a8
- torch 2.8.0a0+5228986c39.nv25.05, CUDA 12.9, cuDNN 91001
- Result: passed (smoke.json). Wall time 22 s.

Files: `airr-smoke-7365120.out/.err` (Slurm log), `smoke.json`, `manifest.json` (run provenance), `airr-pull-sif-7364955.out` (the job that pulled the container, 5 min 45 s).

This shows the container runs and bf16 works on one GH200. It says nothing about multi-GPU, SyncBatchNorm, or training throughput.
