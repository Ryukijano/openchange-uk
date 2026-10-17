"""Human-run metadata inspection; stdlib only, no connection, install or GPU work."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

STORAGE_KEYS = ("PROJECTDIR", "SCRATCHDIR", "LOCALDIR")
PACKAGES = ("PyYAML", "torch", "numpy", "timm", "transformers", "rasterio", "xarray", "netCDF4")
TOOLS = ("git", "apptainer", "python3", "nvcc", "gcc", "lfs")
CHECKOUT_FILES = ("pyproject.toml", "configs/gpu_readiness.yaml", "configs/sources.yaml",
                  "slurm/06_gpu_readiness.sbatch", "scripts/check_gpu_env.py", "scripts/train.py")
SKIP_NAMES = {".git", ".venv", "__pycache__", ".ssh", ".aws", ".config", ".kube", ".cache"}
SECRET_MARKERS = ("secret", "token", "password", "passwd", "credential", "id_rsa", "id_ed25519")
CONFIG_KEYS = {
    "job", "seed", "repo_path", "expected_git_sha", "sif_path", "data_roots",
    "max_depth", "max_entries", "max_files", "deadline_seconds",
}
HEADER_LIMIT = 8192


def validate_config(config: object) -> dict:
    """Reject unknown options and unbounded scans."""
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("Config must contain exactly the documented keys")
    if config["job"] != "training-setup-inspection" or type(config["seed"]) is not int or config["seed"] != 0:
        raise ValueError("Expected training-setup-inspection and seed 0")
    for name, upper in (("max_depth", 4), ("max_entries", 2000), ("max_files", 500), ("deadline_seconds", 60)):
        if type(config[name]) is not int or not 1 <= config[name] <= upper:
            raise ValueError(f"Invalid bounded option: {name}")
    for name in ("repo_path", "sif_path"):
        if not isinstance(config[name], str) or not 1 <= len(config[name]) <= 1024:
            raise ValueError(f"Expected a path: {name}")
    roots = config["data_roots"]
    if not isinstance(roots, list) or len(roots) > 4:
        raise ValueError("Supply at most four explicit data directories")
    if any(not isinstance(root, str) or not 1 <= len(root) <= 1024 for root in roots):
        raise ValueError("Invalid data directory")
    sha = config["expected_git_sha"]
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha)):
        raise ValueError("Expected git SHA must be null or a full lowercase SHA")
    return config


def safe_name(name: str) -> bool:
    """Exclude credential locations and hidden files from data inspection."""
    return not name.startswith(".") and name not in SKIP_NAMES and not any(
        marker in name.lower() for marker in SECRET_MARKERS
    ) and not name.lower().endswith((".pem", ".key", ".p12", ".pfx"))


def expand_path(value: str) -> Path:
    """Expand only the documented storage variables, never arbitrary secrets."""
    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"

    def replace(match: re.Match) -> str:
        key = match.group(1) or match.group(2)
        if key not in (*STORAGE_KEYS, "HOME") or not os.environ.get(key):
            raise ValueError("Path uses an unset or non-storage environment variable")
        return os.environ[key]

    expanded = re.sub(pattern, replace, value)
    if "$" in expanded:
        raise ValueError("Unexpanded path variable")
    return Path(os.path.abspath(Path(expanded).expanduser()))


def run_git(repo: Path, *args: str) -> str | None:
    """Run a fixed read-only git query with a short timeout and no shell."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            timeout=5, check=True, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_info(repo: Path, expected_sha: str | None) -> dict:
    """Report commit and dirty state without exposing remotes or file contents."""
    top = run_git(repo, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != repo.resolve():
        return {"status": "not_a_checkout_root", "sha": None, "dirty": None}
    sha = run_git(repo, "rev-parse", "HEAD")
    dirty = run_git(repo, "status", "--porcelain", "--untracked-files=normal")
    return {
        "status": "inspected", "sha": sha, "dirty": None if dirty is None else bool(dirty),
        "expected_sha": expected_sha,
        "matches_expected": None if expected_sha is None or sha is None else sha == expected_sha,
    }


def checkout_files(repo: Path) -> dict:
    """Stat only known setup files, without reading their contents."""
    result: dict = {}
    for name in CHECKOUT_FILES:
        path = repo / name
        try:
            info = path.lstat()
            result[name] = {"regular_file": stat.S_ISREG(info.st_mode), "bytes": info.st_size}
        except OSError:
            result[name] = {"status": "missing_or_unreadable"}
    return result


def storage_sample(path: Path) -> dict:
    """List at most 50 immediate entries, without descending into storage roots."""
    result: dict = {"status": "sample_only", "entries_seen": 0, "entries": []}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if result["entries_seen"] >= 50:
                    result["status"] = "entry_limit"
                    break
                result["entries_seen"] += 1
                if safe_name(entry.name) and not entry.is_symlink():
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode):
                        result["entries"].append({
                            "name": entry.name, "directory": stat.S_ISDIR(info.st_mode),
                        })
    except OSError:
        result["status"] = "unavailable_or_partial"
    return result


def npy_header(path: Path) -> dict:
    """Read at most 8 KiB of NPY header; never load arrays or unpickle objects."""
    try:
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                return {"status": "invalid_magic"}
            version = stream.read(2)
            if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                return {"status": "unsupported_version"}
            size = 2 if version[0] == 1 else 4
            raw_length = stream.read(size)
            if len(raw_length) != size:
                return {"status": "truncated_header"}
            length = struct.unpack("<H" if size == 2 else "<I", raw_length)[0]
            if length > HEADER_LIMIT:
                return {"status": "header_limit"}
            raw = stream.read(length)
            if len(raw) != length:
                return {"status": "truncated_header"}
        header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1"))
        shape = header["shape"]
        if not isinstance(shape, tuple) or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid shape")
        dtype = header["descr"]
        # Structured dtype field names can contain private data; do not export them.
        dtype = dtype[:80] if isinstance(dtype, str) else "structured_dtype_not_exported"
        return {"status": "header_only", "shape": list(shape), "dtype": dtype}
    except (OSError, ValueError, SyntaxError, UnicodeError, KeyError, TypeError, RecursionError):
        return {"status": "unreadable_or_invalid_header"}


def inventory(root: Path, config: dict) -> dict:
    """Bound directory traversal; skip hidden paths, symlinks and special files."""
    resolved = root.resolve()
    forbidden = {Path("/"), Path.home().resolve(), Path("/home"), Path("/projects"), Path("/scratch")}
    protected = (Path("/etc"), Path("/proc"), Path("/sys"), Path("/dev"))
    if resolved in forbidden or any(resolved == p or p in resolved.parents for p in protected):
        return {"status": "refused_broad_or_system_root", "files": []}
    if any(not safe_name(part) for part in root.parts[1:]) or root.is_symlink() or root != resolved:
        return {"status": "refused_hidden_sensitive_or_symlink_path", "files": []}
    if not root.is_dir():
        return {"status": "missing_or_not_directory", "files": []}
    result: dict = {"status": "bounded_sample", "entries_seen": 0, "skipped": 0, "limits_hit": [], "files": []}
    queue = deque([(root, 0)])
    while queue:
        directory, depth = queue.popleft()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if result["entries_seen"] >= config["max_entries"]:
                        result["limits_hit"].append("max_entries")
                        return result
                    result["entries_seen"] += 1
                    if not safe_name(entry.name) or entry.is_symlink():
                        result["skipped"] += 1
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if depth < config["max_depth"]:
                            queue.append((Path(entry.path), depth + 1))
                        elif "max_depth" not in result["limits_hit"]:
                            result["limits_hit"].append("max_depth")
                    elif stat.S_ISREG(info.st_mode):
                        path = Path(entry.path)
                        row: dict = {"path": str(path.relative_to(root)), "bytes": info.st_size, "suffix": path.suffix}
                        if path.suffix.lower() == ".npy":
                            row["array_header"] = npy_header(path)
                        result["files"].append(row)
                        if len(result["files"]) >= config["max_files"]:
                            result["limits_hit"].append("max_files")
                            return result
                    else:
                        result["skipped"] += 1
        except OSError:
            result.setdefault("unreadable_directories", 0)
            result["unreadable_directories"] += 1
    return result


def path_info(value: str) -> dict:
    """Stat one staged path; do not hash or execute a container image."""
    try:
        path = expand_path(value)
        info = path.stat()
        return {"path": str(path), "status": "present", "regular_file": stat.S_ISREG(info.st_mode),
                "bytes": info.st_size, "readable_hint": os.access(path, os.R_OK), "sha256": "not_verified"}
    except (OSError, ValueError):
        return {"status": "missing_unreadable_or_unset_path", "sha256": "not_verified"}


def inspect(config: dict, report: dict) -> None:
    """Collect host-only facts; leave GPU, container runtime and model tests untested."""
    report["host"] = {"machine": platform.machine(), "system": platform.system(),
                      "python": platform.python_version(), "python_executable": sys.executable}
    report["tools"] = {name: shutil.which(name) for name in TOOLS}
    report["loaded_modules"] = [name for name in os.environ.get("LOADEDMODULES", "").split(":") if name][:100]
    report["host_packages"] = {}
    for name in PACKAGES:
        try:
            report["host_packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["host_packages"][name] = None
    repo = expand_path(config["repo_path"])
    report["repository"] = repository_info(repo, config["expected_git_sha"])
    report["checkout_files"] = checkout_files(repo)
    report["sif"] = path_info(config["sif_path"])
    report["storage"] = {}
    for key in STORAGE_KEYS:
        value = os.environ.get(key)
        row: dict = {"status": "unset"} if not value else {"path": value, "status": "unavailable"}
        if value:
            try:
                usage = shutil.disk_usage(value)
                row.update(status="present", filesystem_free_bytes=usage.free,
                           writable_hint=os.access(value, os.W_OK), project_quota="not_checked")
                row["top_level_sample"] = storage_sample(Path(value))
            except OSError:
                pass
        report["storage"][key] = row
    report["data"] = []
    for index, root_value in enumerate(config["data_roots"]):
        try:
            root = expand_path(root_value)
            row = {"root_index": index, "root": str(root), **inventory(root, config)}
        except ValueError:
            row = {"root_index": index, "status": "unset_or_invalid_path"}
        report["data"].append(row)
    report["next_actions"] = [
        "Review paths and filenames before sharing this report.",
        "Verify staged Arm64 SIF identity and SHA-256 separately on a compute allocation.",
        "Run the pinned one-GH200 readiness job, then review its logs before multi-GPU work.",
        "Select an explicit model/workload config before installing task-specific dependencies.",
    ]
    if report["host"]["machine"] != "aarch64":
        report["next_actions"].insert(0, "This host is not the target aarch64 environment.")
    if not config["data_roots"]:
        report["next_actions"].insert(0, "No data roots selected: data files were not inspected.")
    if report["repository"].get("dirty") is not False:
        report["next_actions"].insert(0, "Preserve local work; use a clean reviewed checkout for the GPU job.")
    if report["sif"]["status"] != "present":
        report["next_actions"].insert(0, "Locate the staged SIF or prepare a separately reviewed staging step.")
    if report["tools"]["apptainer"] is None:
        report["next_actions"].insert(0, "Apptainer is not on PATH; verify the documented site environment.")


def main(argv: list[str] | None = None) -> int:
    """Write a new report without modifying any inspected paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.config.open("rb") as stream:
        payload = stream.read(32769)
    if len(payload) > 32768:
        parser.error("Config exceeds 32 KiB")
    try:
        config = validate_config(json.loads(payload))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    # Exclusive creation prevents replacing an earlier inspection or following an output symlink.
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        started = time.monotonic()
        report = {
            "job": config["job"], "seed": config["seed"], "status": "inspection_incomplete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "config_path": str(args.config),
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu_readiness": "not_tested", "training_readiness": "not_tested",
            "container_execution": "not_tested", "quota_and_allocation": "not_checked",
        }

        def deadline(signum: int, frame: object) -> None:
            raise TimeoutError("Inspection deadline reached")

        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(config["deadline_seconds"])
        try:
            inspect(config, report)
            report["status"] = "inspection_complete_not_training_ready"
        except (OSError, ValueError, TimeoutError) as error:
            report["error_type"] = type(error).__name__
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            json.dump(report, output, indent=2)
            output.write("\n")
    return 0 if report["status"] == "inspection_complete_not_training_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
