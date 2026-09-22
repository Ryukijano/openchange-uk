# How this repo uses Isambard

Cursor stays on this PC. The cluster is reached only by a person, after Clifton auth. Official source: [Using AI agents with Isambard](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/) and [Slurm](https://docs.isambard.ac.uk/user-documentation/guides/slurm/).

```text
edit locally
  -> make test
  -> make check-sbatch
  -> commit
  -> you: clifton auth, ssh u6xn.aip2.isambard
  -> you: apptainer pull, then sbatch
  -> you: squeue --me
  -> paste the log back
```

## What the hooks do

`.cursor/hooks.json` runs on this project:

- `beforeShellExecution` denies `sbatch`, `srun`, `salloc`, `scancel`, `squeue`, `sinfo`, `sacct`, `scontrol`, `clifton`, and `ssh` to an Isambard host. The agent can still write the script.
- `sessionStart` reminds the session that Phase 2 is aarch64 GH200, one GPU is one superchip, and smoke comes before training.
- `postToolUse` runs `scripts/check_sbatch.py` after an edit under `slurm/` and feeds any failure back to the agent.

A crash in the deny hook blocks the shell command (`failClosed`). Other commands are allowed.

## What you run

On the login node, after the SIF exists:

```bash
sbatch slurm/00_smoke.sbatch
squeue --me
```

Do not wrap `squeue` in `watch`. The Isambard skill sets a 60-second floor on scheduler polling.

Interactive debug, when you want it, is one GH200 and is billed at 1.5×:

```bash
srun --gpus=1 --reservation=interactive --pty bash -i
```

Do not add `--partition` to that line. Batch scripts in this repo use `#SBATCH --nodes=1` and `#SBATCH --gpus=` instead of the interactive reservation.

## Checks before a longer job

`make check-sbatch` fails a script that:

- omits `--time`, `--nodes`, or `--gpus`
- asks for more than one node
- sets a walltime above 30 minutes
- calls `mpirun`, `mpiexec`, `curl`, or `wget`
- does not record `git rev-parse HEAD`

A longer or multi-node script is allowed only with the comment `OPENCHANGE_LONG_JOB` or `OPENCHANGE_MULTI_NODE` after you have asked for that in writing.

GitHub Actions runs the same checker and the unit tests on push. It does not log in to Isambard.
