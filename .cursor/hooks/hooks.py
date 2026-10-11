#!/usr/bin/env python3
"""Cursor hooks for this repo, portable to Linux, macOS, and Windows.

  python3 .cursor/hooks/hooks.py deny          beforeShellExecution
  python3 .cursor/hooks/hooks.py context       sessionStart
  python3 .cursor/hooks/hooks.py check-sbatch  postToolUse

They never open a connection. The PowerShell versions next to this file are kept
for reference.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = "https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/"
SLURM_RE = re.compile(r"^(sbatch|srun|salloc|scancel|squeue|sinfo|sacct|scontrol|sacctmgr)(\.exe)?$")
REMOTE_RE = re.compile(r"^(ssh|scp|sftp|rsync|mosh)(\.exe)?$")
WRAPPERS = {"sudo", "env", "command", "exec", "nohup", "time", "nice", "builtin"}
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
CONTEXT = (
    "Isambard-AI Phase 2 (project u6xn) is aarch64 GH200. One GPU request is one GH200. "
    "Login nodes have no GPUs. Do not ssh, sbatch, srun, scancel, or run Clifton. "
    "Propose commands; the human runs them after https://docs.isambard.ac.uk . "
    "Order: make check, 10-minute 1-GPU smoke, then a dry-run. Cap suggested --time at "
    "00:30:00. Container: nvcr.io/nvidia/pytorch:25.05-py3 with apptainer exec --nv. "
    "Never pip-install an x86 torch wheel. Workflow: docs/WORKFLOW.md."
)


def first_command(segment: str) -> tuple[str, str]:
    words = segment.strip().split()
    while words and (words[0] in WRAPPERS or ASSIGN_RE.match(words[0]) or words[0].startswith("-")):
        words = words[1:]
    if not words:
        return "", ""
    return Path(words[0].replace("\\", "/")).name, " ".join(words)


def decide(command: str) -> str | None:
    """Return a reason to deny, or None to allow."""
    for segment in re.split(r"[;&|\n()`]+|\$\(", command):
        token, rest = first_command(segment)
        if not token:
            continue
        if SLURM_RE.match(token):
            return f"Slurm command '{token}' is for you to run on Isambard, not for the agent."
        if re.match(r"^clifton(\.exe)?$", token):
            return "Clifton auth stays in your terminal. The agent must not see the credentials."
        if REMOTE_RE.match(token) and re.search(r"isambard|aip2|bristol\.ac\.uk", rest, re.I):
            return "SSH to Isambard stays in your terminal. The agent proposes commands only."
    return None


def read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"command": raw}
    return data if isinstance(data, dict) else {}


def hook_deny() -> dict:
    payload = read_payload()
    command = str(payload.get("command") or payload.get("cmd") or "")
    why = decide(command)
    if why is None:
        return {"permission": "allow"}
    return {"permission": "deny", "user_message": why, "agent_message": f"{why} Docs: {DOCS}"}


def hook_context() -> dict:
    sys.stdin.read()
    return {"additional_context": CONTEXT}


def hook_check_sbatch() -> dict:
    blob = sys.stdin.read()
    if not re.search(r"\.sbatch|[/\\]slurm[/\\]", blob):
        return {}
    checker = ROOT / "scripts" / "check_sbatch.py"
    result = subprocess.run(
        [sys.executable, str(checker)], capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    if result.returncode == 0:
        note = "Slurm templates pass scripts/check_sbatch.py. Do not submit them. The human runs sbatch."
    else:
        note = "Slurm check failed. Fix the templates before proposing submission.\n" + result.stdout
    return {"additional_context": note.strip()}


HOOKS = {"deny": hook_deny, "context": hook_context, "check-sbatch": hook_check_sbatch}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in HOOKS:
        print(f"usage: hooks.py {{{'|'.join(HOOKS)}}}", file=sys.stderr)
        return 2
    print(json.dumps(HOOKS[argv[1]]()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
