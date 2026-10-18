from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("install .venv-jepa with CPU torch")

import numpy as np
import torch

from openchange.jepa.checkpoints import load_checkpoint, save_checkpoint
from openchange.jepa.config import config_hash, load_config, lr_at
from openchange.jepa.model import JEPAEncoder
from openchange.jepa.views import make_views, synthetic_batch

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "jepa" / "p1_synthetic.yaml"


def _config() -> dict:
    cfg = load_config(CONFIG_PATH)
    cfg["global_batch"] = 8
    cfg["global_quotas"] = [2, 2, 2, 2]
    cfg["synthetic"]["pool_per_stratum"] = 8
    cfg["model"].update({"dim": 24, "depth": 2, "heads": 3, "mlp_ratio": 4})
    cfg["projector_hidden"] = [32, 32]
    cfg["projector_dim"] = 8
    cfg["projector_norm"] = "batch"
    cfg["sigreg_slices"] = 8
    cfg["schedule"].update({"warmup_steps": 2, "total_steps": 10})
    cfg["runtime"]["cpu_threads"] = 1
    return cfg


def _build(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    model = JEPAEncoder(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
        eps=cfg["optimizer"]["eps"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    peak = cfg["optimizer"]["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at(step, cfg) / peak,
    )
    return model, optimizer, scheduler


def _run(model, optimizer, scheduler, cfg: dict, start: int, stop: int):
    losses = []
    tokens_seen = 0
    metrics = []
    for step in range(start, stop):
        model.train()
        observations = synthetic_batch(cfg["seed"], step, cfg)
        batch = make_views(observations, cfg["seed"], step, cfg)
        output = model(batch)
        random_term = (
            torch.rand((), dtype=output.z.dtype)
            + float(np.random.random())
            + random.random()
        ) * 1e-4
        loss = output.z.square().mean() + output.z.mean() * random_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tokens_step = int(output.tokens.sum())
        tokens_seen += tokens_step
        losses.append(float(loss.detach()))
        metrics.append({"step": step + 1, "tokens_step": tokens_step})
    return losses, tokens_seen, metrics


def _equal_nested(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal_nested(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _equal_nested(a, b) for a, b in zip(left, right)
        )
    return left == right


class TestJEPACheckpoints(unittest.TestCase):
    def test_resume_matches_continuous_training_bitwise(self) -> None:
        cfg = _config()
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            continuous_model, continuous_optimizer, continuous_scheduler = _build(cfg)
            continuous_losses, continuous_tokens, continuous_metrics = _run(
                continuous_model, continuous_optimizer, continuous_scheduler, cfg, 0, 10
            )

            partial_model, partial_optimizer, partial_scheduler = _build(cfg)
            partial_losses, partial_tokens, partial_metrics = _run(
                partial_model, partial_optimizer, partial_scheduler, cfg, 0, 4
            )
            self.assertEqual(partial_losses, continuous_losses[:4])
            self.assertEqual(partial_metrics, continuous_metrics[:4])
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "step_000004.pt"
                checksum = save_checkpoint(
                    checkpoint,
                    model=partial_model,
                    optimizer=partial_optimizer,
                    scheduler=partial_scheduler,
                    step=4,
                    tokens_seen=partial_tokens,
                    config_hash=config_hash(cfg),
                    cfg=cfg,
                    metrics_tail=partial_metrics,
                )
                self.assertEqual(len(checksum), 64)
                loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(loaded["step"], 4)
                self.assertEqual(loaded["sampler_step"], 4)
                self.assertEqual(loaded["tokens_last_step"], partial_metrics[-1]["tokens_step"])

                resumed_model, resumed_optimizer, resumed_scheduler = _build(cfg)
                torch.manual_seed(888)
                np.random.seed(888)
                random.seed(888)
                restored = load_checkpoint(
                    checkpoint,
                    model=resumed_model,
                    optimizer=resumed_optimizer,
                    scheduler=resumed_scheduler,
                    config_hash=config_hash(cfg),
                    cfg=cfg,
                )
                self.assertEqual(restored["step"], 4)
                self.assertEqual(restored["tokens_seen"], partial_tokens)
                resumed_losses, resumed_tokens, resumed_metrics = _run(
                    resumed_model, resumed_optimizer, resumed_scheduler, cfg, 4, 10
                )
                self.assertEqual(resumed_losses, continuous_losses[4:])
                self.assertEqual(resumed_metrics, continuous_metrics[4:])
                self.assertEqual(partial_tokens + resumed_tokens, continuous_tokens)
                self.assertTrue(
                    _equal_nested(continuous_model.state_dict(), resumed_model.state_dict())
                )
                self.assertTrue(
                    _equal_nested(
                        continuous_optimizer.state_dict(),
                        resumed_optimizer.state_dict(),
                    )
                )
                self.assertEqual(
                    continuous_scheduler.get_last_lr(),
                    resumed_scheduler.get_last_lr(),
                )

                with self.assertRaisesRegex(ValueError, "config_hash"):
                    load_checkpoint(
                        checkpoint,
                        model=resumed_model,
                        optimizer=resumed_optimizer,
                        scheduler=resumed_scheduler,
                        config_hash="wrong",
                        cfg=cfg,
                    )
                torch.set_num_threads(2)
                with self.assertRaisesRegex(ValueError, "cpu_threads"):
                    load_checkpoint(
                        checkpoint,
                        model=resumed_model,
                        optimizer=resumed_optimizer,
                        scheduler=resumed_scheduler,
                        config_hash=config_hash(cfg),
                        cfg=cfg,
                    )
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic)


if __name__ == "__main__":
    unittest.main()
