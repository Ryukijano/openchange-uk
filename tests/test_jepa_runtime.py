from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires the optional JEPA torch environment")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from openchange.jepa import runtime
from openchange.jepa.config import ConfigError, load_config, validate
from openchange.jepa.runtime import preserve_rng

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_jepa_distributed.py"


def tiny_config() -> dict:
    cfg = load_config(ROOT / "configs/jepa/p1_synthetic.yaml")
    cfg["global_batch"], cfg["global_quotas"] = 8, [2] * 4
    cfg["global_crop_pixels"], cfg["local_crop_pixels"] = 32, 16
    cfg["synthetic"]["image_pixels"] = 32
    cfg["model"].update(dim=32, depth=1, heads=2)
    cfg["projector_hidden"], cfg["projector_dim"] = [32, 32], 8
    cfg["projector_norm"] = "frozen"
    cfg["sigreg_slices"] = 8
    cfg["runtime"]["cpu_threads"] = 1
    cfg["schedule"].update(total_steps=3, warmup_steps=0)
    cfg["checkpoint"]["every_steps"] = 1
    return cfg


def launch(
    config: Path,
    output: Path,
    ranks: int = 1,
    flags: tuple = (),
    device: str = "cpu",
    mode: str = "synthetic-fit",
) -> dict:
    command = [sys.executable]
    if ranks > 1:
        command.extend(["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}"])
    command.extend(
        [
            str(SCRIPT),
            "--config",
            str(config),
            f"--{mode}",
            "--device",
            device,
            "--output-root",
            str(output),
            "--sigreg-chunk",
            "4",
            "--log-every",
            "1",
            *flags,
        ]
    )
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
    if result.returncode != 0:
        raise AssertionError(result.stdout + "\n" + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def checkpoint(result: dict, step: int = 3) -> dict:
    return torch.load(Path(result["run_dir"]) / "checkpoints" / f"step_{step:06d}.pt", weights_only=True)


def collective_worker(rank: int, directory: str) -> None:
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{root / 'rendezvous'}")
    try:
        ctx = runtime.get_context()
        decision = runtime.reduce(torch.tensor(int(rank == 1)), ctx, dist.ReduceOp.MAX)
        assert int(decision) == 1
        device = torch.device("cpu")
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
        state = torch.get_rng_state()
        path = root / "checkpoint.pt"
        runtime.save_distributed(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=2,
            local_tokens=10 + rank,
            digest="test",
            ctx=ctx,
            device=device,
        )
        torch.manual_seed(1234)
        step, tokens = runtime.load_distributed(
            path, model=model, optimizer=optimizer, scheduler=scheduler, digest="test", ctx=ctx, device=device
        )
        assert (step, tokens) == (2, 10 + rank)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        try:
            runtime.save_distributed(
                path / "impossible.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=3,
                local_tokens=10,
                digest="test",
                ctx=ctx,
                device=device,
            )
        except RuntimeError as exc:
            assert "checkpoint write failed" in str(exc)
        else:
            raise AssertionError("rank-zero write error was not propagated")
    finally:
        dist.destroy_process_group()


class TestDistributedRuntime(unittest.TestCase):
    def test_approved_local_array_adapter_runs_two_rank_fit(self):
        from test_jepa_data import PilotFixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PilotFixture(root)
            cfg = tiny_config()
            cfg.update(status="local_pilot", synthetic_only=False, split="pilot", normalization="manifest")
            del cfg["synthetic"]
            config = root / "local.yaml"
            config.write_text(yaml.safe_dump(cfg))
            result = launch(
                config,
                root / "fit",
                ranks=2,
                mode="local-fit",
                flags=(
                    "--manifest",
                    str(fixture.manifest_path),
                    "--sources-config",
                    str(fixture.sources_path),
                ),
            )
            self.assertEqual(result["next_step"], 3)
            record = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
            self.assertFalse(record["synthetic_only"])
            self.assertEqual(len(record["launch"]["manifest_sha256"]), 64)
            fixture.sources["approved_public_sources"][0]["use"] = "auxiliary"
            fixture.save_sources()
            with self.assertRaises(AssertionError):
                launch(
                    config,
                    root / "refused",
                    mode="local-fit",
                    flags=(
                        "--manifest",
                        str(fixture.manifest_path),
                        "--sources-config",
                        str(fixture.sources_path),
                    ),
                )

    def test_stop_consensus_rank_rng_and_checkpoint_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(collective_worker, args=(directory,), nprocs=2, join=True)

    def test_pilot_and_precision_boundaries_are_explicit(self):
        cfg = tiny_config()
        cfg.update(status="local_pilot", synthetic_only=False, split="pilot", normalization="manifest")
        del cfg["synthetic"]
        with self.assertRaises(ConfigError):
            validate(cfg)
        validate(cfg, local_pilot=True)
        cfg["precision"] = "bf16"
        with self.assertRaises(ConfigError):
            validate(cfg, local_pilot=True)
        validate(cfg, local_pilot=True, cuda=True)

    def test_provenance_rng_preservation(self):
        state = torch.get_rng_state()
        with preserve_rng(torch.device("cpu")):
            torch.manual_seed(912)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)

    def test_configure_precision_and_determinism_disables_tf32_in_fp32(self):
        cfg = tiny_config()
        cfg["precision"] = "fp32"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        runtime.configure_precision_and_determinism(cfg, torch.device("cpu"))
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_one_vs_two_rank_updates_and_exact_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "tiny.yaml"
            config.write_text(yaml.safe_dump(tiny_config()))
            single = launch(config, root / "single")
            multi = launch(config, root / "multi", ranks=2)
            one, two = checkpoint(single), checkpoint(multi)
            for key in one["model"]:
                torch.testing.assert_close(one["model"][key], two["model"][key], atol=2e-5, rtol=2e-5)
            resumed = launch(
                config,
                root / "resumed",
                ranks=2,
                flags=("--resume", str(Path(multi["run_dir"]) / "checkpoints/step_000001.pt")),
            )
            restored = checkpoint(resumed)
            self.assertEqual(restored["tokens_seen"], two["tokens_seen"])
            for key in two["model"]:
                torch.testing.assert_close(restored["model"][key], two["model"][key], rtol=0, atol=0)
            for left, right in zip(restored["rank_states"], two["rank_states"]):
                torch.testing.assert_close(left["torch"], right["torch"], rtol=0, atol=0)
            indices = json.loads((Path(multi["run_dir"]) / "checkpoint_index.json").read_text())
            self.assertEqual([row["step"] for row in indices], [1, 2, 3])
            with self.assertRaises(AssertionError):
                launch(
                    config,
                    root / "wrong_world",
                    flags=("--resume", str(Path(multi["run_dir"]) / "checkpoints/step_000001.pt")),
                )

    def test_workers_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "tiny.yaml"
            config.write_text(yaml.safe_dump(tiny_config()))
            zero = checkpoint(launch(config, root / "zero"))
            workers = checkpoint(launch(config, root / "workers", flags=("--workers", "2")))
            for key in zero["model"]:
                torch.testing.assert_close(zero["model"][key], workers["model"][key], rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2, "requires two real CUDA GPUs; CPU is not SyncBN evidence"
    )
    def test_cuda_syncbn_single_vs_two_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "cuda.yaml"
            cfg = tiny_config()
            cfg["projector_norm"] = "batch"
            config.write_text(yaml.safe_dump(cfg))
            single = checkpoint(launch(config, root / "cuda_single", device="cuda"))
            multi = checkpoint(launch(config, root / "cuda_multi", ranks=2, device="cuda"))
            for key in single["model"]:
                torch.testing.assert_close(
                    single["model"][key],
                    multi["model"][key],
                    atol=2e-4,
                    rtol=2e-4,
                    msg=f"mismatch in state key {key}",
                )

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2, "requires two real CUDA GPUs; bf16 restart is not CPU evidence"
    )
    def test_cuda_bf16_training_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "bf16.yaml"
            cfg = tiny_config()
            cfg.update(projector_norm="batch", precision="bf16")
            config.write_text(yaml.safe_dump(cfg))
            full = launch(config, root / "full", ranks=2, device="cuda")
            resumed = launch(
                config,
                root / "resume",
                ranks=2,
                device="cuda",
                flags=("--resume", str(Path(full["run_dir"]) / "checkpoints" / "step_000002.pt")),
            )
            original_state, resumed_state = checkpoint(full), checkpoint(resumed)
            for key in original_state["model"]:
                torch.testing.assert_close(
                    original_state["model"][key], resumed_state["model"][key], atol=2e-4, rtol=2e-4
                )
            self.assertEqual(original_state["tokens_seen"], resumed_state["tokens_seen"])
            for original_rank, resumed_rank in zip(
                original_state["rank_states"], resumed_state["rank_states"]
            ):
                self.assertTrue(torch.equal(original_rank["cuda"], resumed_rank["cuda"]))
            with (Path(full["run_dir"]) / "metrics.jsonl").open() as handle:
                metrics = [json.loads(row) for row in handle]
            self.assertTrue(all(torch.isfinite(torch.tensor(row["sigreg"])) for row in metrics))


if __name__ == "__main__":
    unittest.main()
