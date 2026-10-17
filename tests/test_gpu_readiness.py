import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from openchange.gpu_readiness import IMAGE, execution_context, main, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "job": "gpu-readiness", "seed": 0, "container": IMAGE,
    "expect_machine": "aarch64", "expect_device": "GH200",
}
ENV = {
    "SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1",
    "APPTAINER_CONTAINER": "/project/pytorch.sif",
    "OPENCHANGE_GIT_SHA": "a" * 40, "OPENCHANGE_EXPECTED_GIT_SHA": "a" * 40,
    "OPENCHANGE_GIT_DIRTY": "0", "OPENCHANGE_IMAGE": IMAGE,
    "OPENCHANGE_SIF_SHA256": "b" * 64,
}


class GPUReadinessTest(unittest.TestCase):
    def test_fixed_config_and_repository_config(self):
        self.assertEqual(validate_config(CONFIG), CONFIG)
        actual = yaml.safe_load((ROOT / "configs/gpu_readiness.yaml").read_text())
        self.assertEqual(validate_config(actual), CONFIG)

    def test_unknown_missing_and_unsafe_config_refused(self):
        cases = [None, [], {}, {**CONFIG, "steps": 10000}]
        for key in CONFIG:
            missing = copy.deepcopy(CONFIG)
            del missing[key]
            cases.append(missing)
        for key in ("container", "expect_machine", "expect_device", "job"):
            cases.append({**CONFIG, key: "not-the-target"})
        for seed in (True, -1, 2**32, 1.0, "0", None):
            cases.append({**CONFIG, "seed": seed})
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_single_context(self):
        self.assertEqual(execution_context("single", ENV, "aarch64"), (0, 0, 1))

    def test_noncompute_and_unpinned_contexts_refused(self):
        for key in ENV:
            env = ENV.copy()
            del env[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                execution_context("single", env, "aarch64")
        for key, value in (
            ("SLURM_JOB_NUM_NODES", "2"), ("OPENCHANGE_GIT_DIRTY", "1"),
            ("OPENCHANGE_EXPECTED_GIT_SHA", "c" * 40), ("OPENCHANGE_GIT_SHA", "short"),
            ("OPENCHANGE_SIF_SHA256", "unverified"), ("OPENCHANGE_IMAGE", "newest"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                execution_context("single", {**ENV, key: value}, "aarch64")
        with self.assertRaises(ValueError):
            execution_context("single", ENV, "x86_64")

    def test_single_refuses_accidental_torchrun(self):
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                execution_context("single", {**ENV, key: "0"}, "aarch64")

    def test_distributed_contexts(self):
        for world in (2, 4):
            for rank in range(world):
                env = {**ENV, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world)}
                self.assertEqual(execution_context("distributed", env, "aarch64"), (rank, rank, world))

    def test_distributed_bad_ranks_world_and_mode_refused(self):
        for rank, local, world in ((0, 0, 1), (0, 0, 3), (0, 0, 8), (1, 0, 2), (2, 2, 2), (-1, -1, 2)):
            env = {**ENV, "RANK": str(rank), "LOCAL_RANK": str(local), "WORLD_SIZE": str(world)}
            with self.subTest(env=env), self.assertRaises(ValueError):
                execution_context("distributed", env, "aarch64")
        with self.assertRaises(ValueError):
            execution_context("cpu", ENV, "aarch64")

    def test_local_refusal_writes_report_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(CONFIG))
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("openchange.gpu_readiness.platform.machine", return_value="x86_64"),
                patch("openchange.gpu_readiness.run_cuda_checks") as cuda,
            ):
                code = main(["--config", str(config), "--mode", "single", "--output-root", str(root)])
            cuda.assert_not_called()
            self.assertEqual(code, 1)
            report_path, = root.glob("gpu-readiness-*/readiness.json")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["checks"], [])
            self.assertEqual(report["model_readiness"], "not_tested")
            self.assertEqual(report["active_check"], "execution_context")

    def test_missing_checks_cannot_report_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(CONFIG))
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch("openchange.gpu_readiness.platform.machine", return_value="aarch64"),
                patch("openchange.gpu_readiness.run_cuda_checks"),
            ):
                code = main(["--config", str(config), "--mode", "single", "--output-root", str(root)])
            self.assertEqual(code, 1)
            report_path, = root.glob("gpu-readiness-*/readiness.json")
            self.assertIn("not all required checks completed", json.loads(report_path.read_text())["error"])


if __name__ == "__main__":
    unittest.main()
