"""Run provenance: config, seed, git SHA, environment, container identity, output paths.

Nothing here opens a network connection or imports torch. Values set by a Slurm
template (OPENCHANGE_GIT_SHA, OPENCHANGE_SIF, ...) win over values probed locally,
because git may be missing inside the container.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ENV_PREFIXES = ("SLURM_", "CUDA_", "NVIDIA_", "APPTAINER_", "SINGULARITY_", "OPENCHANGE_")
ENV_KEYS = ("PYTORCH_VERSION", "PYTORCH_BUILD_VERSION", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "KEY", "COOKIE")
PACKAGES = ("openchange-uk", "PyYAML", "torch", "numpy")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    sha = os.environ.get("OPENCHANGE_GIT_SHA")
    dirty_env = os.environ.get("OPENCHANGE_GIT_DIRTY")
    if sha and sha != "unavailable":
        dirty = {"0": False, "1": True}.get(dirty_env or "")
        return {"sha": sha, "dirty": dirty, "source": "env"}
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return {"sha": None, "dirty": None, "source": "unavailable"}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": None if status is None else bool(status), "source": "git"}


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def filtered_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Scheduler/GPU/container variables only. Never the full environment."""
    environ = dict(os.environ if environ is None else environ)
    kept = {
        key: value
        for key, value in environ.items()
        if (key.startswith(ENV_PREFIXES) or key in ENV_KEYS) and not is_secret_name(key)
    }
    return dict(sorted(kept.items()))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def container_identity() -> dict[str, Any]:
    env = os.environ
    return {
        "inside_container": bool(env.get("APPTAINER_CONTAINER") or env.get("SINGULARITY_CONTAINER")),
        "sif": env.get("OPENCHANGE_SIF") or env.get("APPTAINER_CONTAINER"),
        "sif_sha256": env.get("OPENCHANGE_SIF_SHA256"),
        "image": env.get("OPENCHANGE_IMAGE"),
        "nvidia_pytorch_version": env.get("NVIDIA_PYTORCH_VERSION"),
    }


def environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "system": platform.system(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": package_versions(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": sha256_file(resolved)}


def set_seed(seed: int) -> list[str]:
    """Seed what is already loaded. Does not import torch or numpy on its own."""
    random.seed(seed)
    seeded = ["random"]
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed)
        seeded.append("numpy")
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        seeded.append("torch")
    return seeded


def run_id(now: datetime | None = None) -> str:
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return f"slurm{job}"
    now = now or datetime.now(timezone.utc)
    return f"local{now:%Y%m%dT%H%M%SZ}-{os.getpid()}"


def make_run_dir(output_root: Path, job: str, rid: str | None = None) -> Path:
    path = Path(output_root) / f"{job}-{rid or run_id()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def start_run(
    job: str,
    config_path: Path,
    seed: int,
    output_root: Path,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create outputs/<job>-<run id>/manifest.json and return (run_dir, manifest)."""
    run_dir = make_run_dir(output_root, job)
    manifest: dict[str, Any] = {
        "job": job,
        "run_dir": str(run_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config_record(config_path),
        "seed": seed,
        "seeded": set_seed(seed),
        "git": git_state(),
        "environment": environment(),
        "container": container_identity(),
        "env": filtered_env(),
    }
    if extra:
        manifest.update(extra)
    write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summary_lines(manifest: dict[str, Any]) -> list[str]:
    git = manifest["git"]
    return [
        f"run_dir: {manifest['run_dir']}",
        f"config: {manifest['config']['path']} sha256={manifest['config']['sha256'][:12]}",
        f"seed: {manifest['seed']}",
        f"git_sha: {git['sha']} dirty={git['dirty']} source={git['source']}",
        f"machine: {manifest['environment']['machine']} python={manifest['environment']['python']}",
        f"container: {manifest['container']['sif'] or 'none'}",
    ]


def main() -> int:
    print(
        json.dumps(
            {"git": git_state(), "environment": environment(), "container": container_identity()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
