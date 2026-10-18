# JEPA local-fit pilot — Isambard-AI GH200 (2026-10-18)

First real-data 4-GPU run for the first project (clean-JEPA P1).

- Job: 7527080, 1 node / 4x GH200 / completed 00:01:35, exit 0
- Commit: 34a675f1b756cb1463b91c65fce155037b40913e (branch devin/1791736800-jepa-p1-data-plan)
- Image: SIF /projects/u6xn/sif-images/pytorch_26.09-py3.sif (sha256 f5a41e03...); log label prints the 25.05 string but torch is 2.14.0a0+nv26.09
- Config: configs/jepa/gh200_local_pilot.yaml, seed 0, bf16, 50 steps, global batch 32
- Data: /projects/u6xn/data/pilot_v1 — 480 SSL4EO-S12 pairs (S1GRD+S2L2A shard_000001, rev 97a91733), manifest 439eb6cd...
- Result: status=completed; loss 0.1418, sigreg 2.4193, tokens/s ~30.2k, peak alloc ~471 MB/rank; checkpoints at steps 25/50 under outputs/jepa-gh200-local-pilot-slurm7527080/ on the cluster
- Caveat: bounded approved local-data pilot, 50 optimizer steps — a pipeline validation, not a benchmark result
