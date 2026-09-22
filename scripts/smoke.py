"""GH200 smoke checks. Run inside the aarch64 PyTorch container, not on a login node."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    machine = platform.machine()
    expected = config["expect_machine"]
    cuda_ok = torch.cuda.is_available()
    record = {
        "architecture": machine,
        "expected_architecture": expected,
        "torch": torch.__version__,
        "cuda_available": cuda_ok,
        "cuda": torch.version.cuda,
        "seed": config["seed"],
    }
    print("architecture:", machine)
    print("torch:", record["torch"])
    print("cuda available:", cuda_ok)
    print("cuda:", record["cuda"])

    if machine != expected or not cuda_ok:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return 1

    torch.manual_seed(int(config["seed"]))
    name = torch.cuda.get_device_name(0)
    n = int(config["matmul"]["n"])
    x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    record.update(
        {
            "device": name,
            "matmul_shape": list(y.shape),
            "matmul_dtype": str(y.dtype),
        }
    )
    print("device:", name)
    print("matmul succeeded:", tuple(y.shape), y.dtype)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())