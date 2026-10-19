# Agent instructions

Read README.md, docs/PLAN.md, docs/WORKFLOW.md, and reports/SPEC.md first. Official source for anything about the cluster: https://docs.isambard.ac.uk

- The agent may run `clifton auth` to start the device-code/SSO flow; the human still completes SSO approval interactively. Once a valid certificate exists on this machine, the agent may ssh to Isambard, run file/environment inspection on the login node, and run Slurm commands (sbatch, srun, salloc, scancel, squeue, sacct, sinfo, scontrol) within the limits below. Login nodes have no GPUs: no training, containers, or heavy jobs there — inspection and job submission only.
- Keep cluster actions bounded and reversible: no destructive filesystem commands, no work outside the project space, nothing beyond the Slurm limits below without an explicit ask. State what you are about to run before running it.
- Never ask for, store, or log SSH key contents, Clifton certificate contents, passwords, or tokens. Use the certificate files Clifton manages in place; do not copy credential material into the repo, reports, or chat.
- Isambard-AI Phase 2 is aarch64 GH200. Use the NGC PyTorch image with `apptainer exec --nv`. Never pip-install an x86 torch wheel.
- Every Slurm file sets `--nodes=1`, `--gpus=`, and `--time=` at or under 00:30:00. `make check-sbatch` enforces this.
- No downloads inside jobs. No fetch at all unless the source passes `make validate-sources` and `require_approved()`.
- Splits are geographic and temporal. Never random. Never change a test region or the cutoff because of a model result. reports/SPLIT_PROPOSAL.md is PROPOSED until a human accepts it.
- Do not claim a result, a smoke pass, or data readiness without a committed log.
- Before handing back: `make check`.
- Do not commit, push, or open a PR unless asked.
