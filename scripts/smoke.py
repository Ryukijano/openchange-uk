"""GH200 smoke checks. Run inside the aarch64 PyTorch container, not on a login node."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

from openchange.provenance import start_run, summary_lines, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    run_dir, manifest = start_run(
        job=str(config.get("job", "smoke")),
        config_path=args.config,
        seed=seed,
        output_root=args.output_root,
        extra={"expected_image": config.get("container")},
    )
    for line in summary_lines(manifest):
        print(line)

    machine = manifest["environment"]["machine"]
    expected = config["expect_machine"]
    cuda_ok = torch.cuda.is_available()
    record = {
        "architecture": machine,
        "expected_architecture": expected,
        "torch": torch.__version__,
        "cuda_available": cuda_ok,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_ok else None,
        "seed": seed,
        "passed": False,
    }
    print("torch:", record["torch"])
    print("cuda available:", cuda_ok)
    print("cuda:", record["cuda"])

    if machine != expected or not cuda_ok:
        write_json(run_dir / "smoke.json", record)
        print(f"FAILED: machine={machine} expected={expected} cuda={cuda_ok}")
        print("report:", run_dir / "smoke.json")
        return 1

    torch.manual_seed(seed)
    name = torch.cuda.get_device_name(0)
    n = int(config["matmul"]["n"])
    dtype = getattr(torch, config["matmul"]["dtype"])
    x = torch.randn(n, n, device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = x @ x
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    record.update(
        {
            "device": name,
            "device_count": torch.cuda.device_count(),
            "matmul_shape": list(y.shape),
            "matmul_dtype": str(y.dtype),
            "matmul_seconds_first_call": seconds,
            "passed": True,
        }
    )
    print("device:", name)
    print("matmul succeeded:", tuple(y.shape), y.dtype)
    write_json(run_dir / "smoke.json", record)
    print("report:", run_dir / "smoke.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
