# How this repo uses Isambard

Agents (Cursor on your machine, or a cloud agent) edit code and run local checks only. The cluster is reached only by a person, after Clifton auth. Official sources: [Using AI agents with Isambard](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/), [Login](https://docs.isambard.ac.uk/user-documentation/guides/login/), [Slurm](https://docs.isambard.ac.uk/user-documentation/guides/slurm/), [Containers](https://docs.isambard.ac.uk/user-documentation/guides/containers/), [ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/), [Storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/). If this page and the docs disagree, the docs win.

```text
agent or you, locally
  -> make check
  -> commit
you, on your machine
  -> clifton auth, ssh u6xn.aip2.isambard
you, on the login node
  -> git pull, apptainer pull (once), sbatch
  -> squeue --me, sacct -j <id>
  -> commit the log and manifest, paste the log back
```

## Local setup (no Isambard access)

Checked on Ubuntu 22.04.5 with uv 0.12.23, CPython 3.11.17 (uv-managed; Ubuntu's 3.10 is below `requires-python`), PyYAML 6.0.3, and ruff 0.16.10:

```bash
uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install -e ".[dev]"
make check
```

`make check` = `lint test check-sbatch validate-sources dry-run evaluate`. Torch is not installed locally; `scripts/smoke.py` runs only inside the container.

## Run records

Every script run through `openchange.provenance.start_run` writes `outputs/<job>-<run id>/manifest.json`:

- config path and SHA-256 of the config file
- seed (Python `random`, plus NumPy and torch when loaded)
- git SHA and a dirty flag
- host, architecture, Python, package versions
- container: `OPENCHANGE_SIF`, the image name, and `OPENCHANGE_SIF_SHA256` when a `<sif>.sha256` file sits next to the SIF
- a filtered environment: Slurm job, GPU, and `OPENCHANGE_` variables only; names containing TOKEN, SECRET, PASSWORD, or KEY are dropped

The run id is `slurm<SLURM_JOB_ID>` in a job and `local<UTC time>-<pid>` otherwise. An existing run directory is never overwritten. Slurm stdout and stderr go to `logs/%x-%j.out` and `logs/%x-%j.err`. The job header prints the same provenance and never runs a bare `env`.

`logs/` and `outputs/` are gitignored. To keep a smoke record: `git add -f logs/airr-smoke-<jobid>.out outputs/smoke-slurm<jobid>/`.

## Your Linux machine: Clifton and SSH

You run these. The agent never does. From the [login guide](https://docs.isambard.ac.uk/user-documentation/guides/login/):

```bash
# Use clifton-linux-musl-aarch64 on an Arm laptop.
curl -L https://github.com/isambard-sc/clifton/releases/latest/download/clifton-linux-musl-x86_64 -o clifton
chmod u+x clifton
mkdir -p ~/.local/bin && mv clifton ~/.local/bin/
clifton auth                                   # or: clifton auth --identity ~/.ssh/id_ed25519
clifton ssh-config write                       # once, and again after joining a project
ssh u6xn.aip2.isambard
```

The Windows equivalent is `winget install clifton` and `clifton auth --identity "$env:USERPROFILE\.ssh\id_ed25519"`. The certificate expires; re-run `clifton auth` when SSH is refused. Check the host key fingerprint against the login guide on first connect.

## What you run on the login node

Login nodes have no GPUs. Do not start agents, tmux, screen, watchers, or training there.

```bash
cd ~/openchange-uk            # your clone; git clone https://github.com/Ryukijano/openchange-uk.git first time
git pull --ff-only
make check-sbatch PYTHON=python3   # stdlib only, no install needed
mkdir -p "$PROJECTDIR/sif-images"
apptainer pull "$PROJECTDIR/sif-images/pytorch_25.05-py3.sif" docker://nvcr.io/nvidia/pytorch:25.05-py3
sbatch slurm/00_smoke.sbatch
squeue --me
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
cat logs/airr-smoke-<jobid>.out
```

The docs pull NGC images on the login node. If that pull is too slow or killed, `slurm/00_pull_sif.sbatch` does the same pull in a 30-minute, one-GPU job and also writes `<sif>.sha256`. It pulls the container, not data.

Do not wrap `squeue` in `watch`. Poll no more than once a minute.

Interactive debug is one GH200, billed at 1.5×. Do not add `--partition`:

```bash
srun --nodes=1 --gpus=1 --time=00:15:00 --reservation=interactive --pty bash -i
```

Batch scripts use `#SBATCH --nodes=1`, `--gpus=`, and `--time=`. They leave the partition to the default (`workq` in the docs).

## What the hooks do

`.cursor/hooks.json` calls `.cursor/hooks/hooks.py` (Python, so it runs on Linux, macOS, and Windows; the old `.ps1` files are kept for reference):

- `beforeShellExecution` denies `sbatch`, `srun`, `salloc`, `scancel`, `squeue`, `sinfo`, `sacct`, `scontrol`, `sacctmgr`, `clifton`, and `ssh`/`scp`/`sftp`/`rsync` to an Isambard host, including after `sudo`, `env`, `VAR=`, pipes, and `$(...)`. The agent can still write the script.
- `sessionStart` reminds the session that Phase 2 is aarch64 GH200, one GPU is one superchip, and smoke comes before training.
- `postToolUse` runs `scripts/check_sbatch.py` after an edit under `slurm/` and feeds any failure back to the agent.

A crash in the deny hook blocks the shell command (`failClosed`).

## Checks before a longer job

`make check-sbatch` fails a script that:

- omits `--time`, `--nodes`, or `--gpus`
- asks for more than one node or more than four GPUs
- sets a walltime above 30 minutes
- uses `#SBATCH --exclusive` or the interactive reservation
- calls `mpirun`, `mpiexec`, a downloader (`curl`, `wget`, `rsync`, `scp`, `git clone`, `hf download`, ...), or `pip install`
- pulls a container without the `OPENCHANGE_CONTAINER_PULL` mark
- runs `apptainer exec` without `--nv`
- dumps the whole environment
- does not use `set -euo pipefail` or record `git rev-parse HEAD`

A longer or multi-node script is allowed only with `OPENCHANGE_LONG_JOB` or `OPENCHANGE_MULTI_NODE` after you have asked for that in writing.

GitHub Actions runs `make check` on push. It does not log in to Isambard.
