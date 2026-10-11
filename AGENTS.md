# Agent instructions

Read README.md, docs/PLAN.md, docs/WORKFLOW.md, and reports/SPEC.md first. Official source for anything about the cluster: https://docs.isambard.ac.uk

- Work locally. Never ssh to Isambard, run Clifton, or run sbatch, srun, salloc, scancel, squeue, or sacct. Write commands for the human to review and run.
- Never ask for, store, or use SSH keys, Clifton certificates, passwords, or tunnels.
- Isambard-AI Phase 2 is aarch64 GH200. Use the NGC PyTorch image with `apptainer exec --nv`. Never pip-install an x86 torch wheel.
- Every Slurm file sets `--nodes=1`, `--gpus=`, and `--time=` at or under 00:30:00. `make check-sbatch` enforces this.
- No downloads inside jobs. No fetch at all unless the source passes `make validate-sources` and `require_approved()`.
- Splits are geographic and temporal. Never random. Never change a test region or the cutoff because of a model result. reports/SPLIT_PROPOSAL.md is PROPOSED until a human accepts it.
- Do not claim a result, a smoke pass, or data readiness without a committed log.
- Before handing back: `make check`.
- Do not commit, push, or open a PR unless asked.
