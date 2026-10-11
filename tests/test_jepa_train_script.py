from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("install .venv-jepa with CPU torch")

import yaml

from openchange.jepa.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "jepa" / "p1_synthetic.yaml"
SCRIPT_PATH = ROOT / "scripts" / "train_jepa.py"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _write_config(directory: Path, cfg: dict, name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run(config: Path, output: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config),
            *flags,
            "--output-root",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestJEPATrainCLI(unittest.TestCase):
    def test_dry_run_writes_provisional_manifest_and_model_check(self) -> None:
        cfg = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_config(root, cfg)
            completed = _run(config, root / "outputs", "--dry-run")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout.strip().splitlines()[-1])
            manifest = json.loads((Path(output["run_dir"]) / "manifest.json").read_text())
            self.assertEqual(manifest["config_hash"], output["config_hash"])
            self.assertEqual(manifest["split_status"], "provisional")
            self.assertIsNotNone(manifest["torch_version"])
            self.assertIsNotNone(manifest["numpy_version"])
            self.assertEqual(output["model_check"], "ok")
            self.assertEqual(output["token_table"]["per_view"]["224"], {"1": 20, "2": 39})
            self.assertEqual(
                output["parameter_count_actual"],
                output["parameter_count_formula"],
            )

    def test_cli_refusals_exit_two(self) -> None:
        cases = (
            ("data_root", lambda cfg: cfg.update(data_root=None)),
            ("status", lambda cfg: cfg.update(status="planned")),
            ("missing", lambda cfg: cfg.pop("job")),
            ("split", lambda cfg: cfg.update(split="train")),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in cases:
                with self.subTest(name=name):
                    cfg = _config()
                    mutate(cfg)
                    config = _write_config(root, cfg, f"{name}.yaml")
                    completed = _run(config, root / "outputs", "--synthetic-fit")
                    self.assertEqual(completed.returncode, 2)
            valid_config = _write_config(root, _config(), "valid.yaml")
            both = _run(valid_config, root / "outputs", "--dry-run", "--synthetic-fit")
            neither = _run(valid_config, root / "outputs")
            self.assertEqual(both.returncode, 2)
            self.assertEqual(neither.returncode, 2)

    def test_synthetic_fit_writes_finite_metrics_and_checkpoint(self) -> None:
        cfg = _config()
        cfg["global_batch"] = 8
        cfg["global_quotas"] = [2, 2, 2, 2]
        cfg["synthetic"]["pool_per_stratum"] = 8
        cfg["model"].update({"dim": 24, "depth": 2, "heads": 3, "mlp_ratio": 4})
        cfg["projector_hidden"] = [32, 32]
        cfg["projector_dim"] = 8
        cfg["sigreg_slices"] = 8
        cfg["schedule"].update({"warmup_steps": 1, "total_steps": 4})
        cfg["runtime"].update({"cpu_threads": 1, "max_wall_seconds": 600})
        cfg["checkpoint"]["every_steps"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_config(root, cfg)
            completed = _run(config, root / "outputs", "--synthetic-fit")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            run_dir = Path(result["run_dir"])
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(metrics), 4)
            self.assertTrue(all(math.isfinite(row["loss"]) for row in metrics))
            self.assertEqual(metrics[0]["step"], 1)
            self.assertEqual(result["status"], "complete")
            checkpoints = [
                json.loads(line)
                for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()
            ]
            checkpoint_steps = [entry["step"] for entry in checkpoints]
            self.assertEqual(checkpoint_steps, [2, 4])

    def test_zero_wall_cap_exits_three_and_saves_checkpoint(self) -> None:
        cfg = _config()
        cfg["global_batch"] = 8
        cfg["global_quotas"] = [2, 2, 2, 2]
        cfg["synthetic"]["pool_per_stratum"] = 8
        cfg["model"].update({"dim": 24, "depth": 2, "heads": 3, "mlp_ratio": 4})
        cfg["projector_hidden"] = [32, 32]
        cfg["projector_dim"] = 8
        cfg["sigreg_slices"] = 8
        cfg["schedule"].update({"warmup_steps": 1, "total_steps": 3})
        cfg["runtime"].update({"cpu_threads": 1, "max_wall_seconds": 0})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _write_config(root, cfg)
            completed = _run(config, root / "outputs", "--synthetic-fit")
            self.assertEqual(completed.returncode, 3, completed.stderr)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(result["status"], "runtime_cap")
            self.assertEqual(result["next_step"], 0)
            self.assertTrue((Path(result["run_dir"]) / "checkpoints" / "step_000000.pt").exists())

if __name__ == "__main__":
    unittest.main()
