"""Check Slurm templates without submitting them.

Isambard-AI Phase 2 rules baked in here (https://docs.isambard.ac.uk):
- explicit --nodes, --gpus, and --time in every file; one node unless marked multi-node
- at most four GPUs on one node (one node is four GH200s)
- agent-written jobs stay at or under 30 minutes unless marked long
- no mpirun/mpiexec, no #SBATCH --exclusive, no in-job downloads or pip installs
- GPU containers use apptainer --nv
- logs record the git SHA but never the whole environment
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"
MAX_MINUTES = 30
GPUS_PER_NODE = 4
TIME_RE = re.compile(r"^#SBATCH\s+--time=(\d+):(\d{2}):(\d{2})\s*$", re.M)
NODES_RE = re.compile(r"^#SBATCH\s+--nodes=(\S+)\s*$", re.M)
GPUS_RE = re.compile(r"^#SBATCH\s+--gpus=(\S+)\s*$", re.M)
LONG_MARK = "OPENCHANGE_LONG_JOB"
MULTI_MARK = "OPENCHANGE_MULTI_NODE"
PULL_MARK = "OPENCHANGE_CONTAINER_PULL"
FETCH_RE = re.compile(
    r"\b(curl|wget|rsync|scp|gsutil|rclone|aria2c)\b|\baws\s+s3\b|\bgit\s+clone\b"
    r"|\b(huggingface-cli|hf)\s+download\b"
)
PIP_RE = re.compile(r"\b(pip3?|uv\s+pip)\s+install\b")
ENV_DUMP_RE = re.compile(r"^\s*(env|printenv)\s*(\|\s*sort\s*)?$", re.M)


def minutes(match: re.Match[str]) -> int:
    hours, mins, secs = (int(match.group(i)) for i in range(1, 4))
    return hours * 60 + mins + (1 if secs else 0)


def code_lines(text: str) -> str:
    """Script body without comment lines, so notes can mention forbidden words."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def check_text(text: str) -> list[str]:
    errors: list[str] = []
    body = code_lines(text)
    if not text.startswith("#!/bin/bash"):
        errors.append("missing bash shebang")

    nodes = NODES_RE.findall(text)
    if not nodes:
        errors.append("missing #SBATCH --nodes=")
    elif nodes != ["1"] and MULTI_MARK not in text:
        errors.append("nodes is not 1; multi-node needs an OPENCHANGE_MULTI_NODE mark")

    found = TIME_RE.search(text)
    if found is None:
        errors.append("missing #SBATCH --time=HH:MM:SS")
    elif minutes(found) > MAX_MINUTES and LONG_MARK not in text:
        errors.append(
            f"walltime is over {MAX_MINUTES} minutes; add {LONG_MARK} only after an explicit decision"
        )

    gpus = GPUS_RE.findall(text)
    if not gpus:
        errors.append("missing #SBATCH --gpus= (Isambard-AI Phase 2 allocates GH200s)")
    elif not gpus[0].isdigit() or int(gpus[0]) < 1:
        errors.append("--gpus must be a positive integer")
    elif int(gpus[0]) > GPUS_PER_NODE and MULTI_MARK not in text:
        errors.append(f"more than {GPUS_PER_NODE} GPUs needs more than one node")

    if re.search(r"^#SBATCH\s+--exclusive", text, re.M):
        errors.append("#SBATCH --exclusive reserves a whole node; the docs say avoid it")
    if re.search(r"^#SBATCH\s+--(partition|reservation)=interactive", text, re.M):
        errors.append("the interactive reservation is for srun sessions, not batch templates")
    if re.search(r"\b(mpirun|mpiexec)\b", body):
        errors.append("mpirun/mpiexec is not used on Isambard; use srun")
    if FETCH_RE.search(body):
        errors.append("no downloads inside the job script")
    if re.search(r"\bapptainer\s+(pull|build)\b", body) and PULL_MARK not in text:
        errors.append(f"container pulls belong in a file marked {PULL_MARK}")
    if PIP_RE.search(body):
        errors.append("no pip installs inside a job; use the documented container")
    if re.search(r"\bapptainer\s+(exec|run)\b", body) and "--nv" not in body:
        errors.append("apptainer exec/run on a GPU node needs --nv")
    if ENV_DUMP_RE.search(body):
        errors.append("do not dump the whole environment into logs; filter it")
    if "set -euo pipefail" not in body:
        errors.append("missing set -euo pipefail")
    if "git rev-parse HEAD" not in body:
        errors.append("job log must record the git SHA")
    return errors


def check(path: Path) -> list[str]:
    return check_text(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    files = [Path(a) for a in args] if args else sorted(SLURM.glob("*.sbatch"))
    if not files:
        print(f"no sbatch files under {SLURM}", file=sys.stderr)
        return 1
    failed = False
    for path in files:
        errors = check(path)
        try:
            shown = path.resolve().relative_to(ROOT)
        except ValueError:
            shown = path
        if errors:
            failed = True
            print(f"{shown}:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"ok {shown}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
