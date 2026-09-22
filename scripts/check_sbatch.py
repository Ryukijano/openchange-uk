"""Check Slurm templates without submitting them.

Isambard-AI Phase 2 rules baked in here:
- one node unless the file is explicitly marked multi-node
- a walltime is always set
- agent-written jobs stay at or under 30 minutes unless marked long
- no mpirun/mpiexec, and no in-job downloads
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"
MAX_MINUTES = 30
TIME_RE = re.compile(r"^#SBATCH\s+--time=(\d+):(\d{2}):(\d{2})\s*$", re.M)
LONG_MARK = "OPENCHANGE_LONG_JOB"
MULTI_MARK = "OPENCHANGE_MULTI_NODE"


def minutes(match: re.Match[str]) -> int:
    hours, mins, secs = (int(match.group(i)) for i in range(1, 4))
    return hours * 60 + mins + (1 if secs else 0)


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not text.startswith("#!/bin/bash"):
        errors.append("missing bash shebang")
    if "#SBATCH --nodes=" not in text and MULTI_MARK not in text:
        errors.append("missing #SBATCH --nodes=")
    if "#SBATCH --nodes=" in text and "#SBATCH --nodes=1" not in text and MULTI_MARK not in text:
        errors.append("nodes is not 1; multi-node needs an OPENCHANGE_MULTI_NODE mark")
    found = TIME_RE.search(text)
    if found is None:
        errors.append("missing #SBATCH --time=HH:MM:SS")
    elif minutes(found) > MAX_MINUTES and LONG_MARK not in text:
        errors.append(f"walltime is over {MAX_MINUTES} minutes; add {LONG_MARK} only after an explicit decision")
    if "#SBATCH --gpus=" not in text:
        errors.append("missing #SBATCH --gpus= (Isambard-AI Phase 2 allocates GH200s)")
    if re.search(r"\b(mpirun|mpiexec)\b", text):
        errors.append("mpirun/mpiexec is not used on Isambard; use srun")
    if re.search(r"\b(curl|wget)\b", text):
        errors.append("no downloads inside the job script")
    if "set -euo pipefail" not in text:
        errors.append("missing set -euo pipefail")
    if "git rev-parse HEAD" not in text:
        errors.append("job log must record the git SHA")
    return errors


def main() -> int:
    files = sorted(SLURM.glob("*.sbatch"))
    if not files:
        print(f"no sbatch files under {SLURM}", file=sys.stderr)
        return 1
    failed = False
    for path in files:
        errors = check(path)
        if errors:
            failed = True
            print(f"{path.relative_to(ROOT)}:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"ok {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
