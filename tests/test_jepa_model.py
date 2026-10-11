from __future__ import annotations

import dataclasses
import importlib.util
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("install .venv-jepa with CPU torch")

import torch
import torch.nn as nn

from openchange.jepa.config import load_config
from openchange.jepa.model import JEPAEncoder, sincos_pos_embed
from openchange.jepa.views import ViewBatch, ViewBucket, make_views, synthetic_observation

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "jepa" / "p1_synthetic.yaml"


def _config(projector_norm: str = "identity") -> dict:
    cfg = load_config(CONFIG_PATH)
    cfg["model"].update({"dim": 24, "depth": 2, "heads": 3, "mlp_ratio": 4})
    cfg["projector_hidden"] = [32, 32]
    cfg["projector_dim"] = 8
    cfg["projector_norm"] = projector_norm
    return cfg


def _batch(cfg: dict, observations=None) -> ViewBatch:
    if observations is None:
        observations = [
            synthetic_observation(0, 0, 13, cfg)[0],
            synthetic_observation(1, 2, 13, cfg)[0],
        ]
    return make_views(observations, 13, 2, cfg)


def _single_view(batch: ViewBatch, view: int) -> ViewBatch:
    buckets = []
    for bucket in batch.buckets:
        rows = torch.where(bucket.view_idx == view)[0]
        if rows.numel() == 0:
            continue
        buckets.append(
            ViewBucket(
                resolution=bucket.resolution,
                sensors=bucket.sensors,
                pixels={sensor: pixels.index_select(0, rows) for sensor, pixels in bucket.pixels.items()},
                keep_idx=bucket.keep_idx.index_select(0, rows),
                view_idx=torch.zeros_like(bucket.view_idx.index_select(0, rows)),
                sample_idx=bucket.sample_idx.index_select(0, rows),
            )
        )
    return ViewBatch(
        buckets=buckets,
        n_samples=batch.n_samples,
        n_views=1,
        n_global=1,
        strata=batch.strata,
        sample_ids=batch.sample_ids,
        synthetic_data=batch.synthetic_data,
        synthetic_condition=batch.synthetic_condition,
        view_sensors=[batch.view_sensors[view]],
        crop_boxes=batch.crop_boxes[view : view + 1],
        flip_rot=batch.flip_rot[view : view + 1],
        tokens=batch.tokens[view : view + 1],
        s2_valid_fraction=batch.s2_valid_fraction[view : view + 1],
    )


class TestJEPAModel(unittest.TestCase):
    def test_bucketed_forward_matches_one_view_at_a_time(self) -> None:
        cfg = _config()
        model = JEPAEncoder(cfg).double().eval()
        batch = _batch(cfg)
        with torch.no_grad():
            bucketed = model(batch)
            separate = [model(_single_view(batch, view)) for view in range(batch.n_views)]
        expected_b = torch.stack([output.b[0] for output in separate])
        expected_z = torch.stack([output.z[0] for output in separate])
        self.assertLessEqual(float((bucketed.b - expected_b).abs().max()), 1e-12)
        self.assertLessEqual(float((bucketed.z - expected_z).abs().max()), 1e-12)

    def test_embedding_layout_and_attention_set_equivariance(self) -> None:
        cfg = _config()
        model = JEPAEncoder(cfg).double().eval()
        image = torch.arange(2 * 32 * 32, dtype=torch.float64).reshape(1, 2, 32, 32)
        pixels = {"s1": image}
        embedded = model.embed(pixels)
        patches = (
            image.reshape(1, 2, 2, 16, 2, 16)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(1, 4, -1)
        )
        expected = (
            model.stems["s1"](patches)
            + sincos_pos_embed(2, cfg["model"]["dim"], dtype=torch.float64).unsqueeze(0)
            + model.modality_embed["s1"].view(1, 1, -1)
        )
        self.assertTrue(torch.equal(embedded, expected))

        sample = synthetic_observation(0, 0, 13, cfg)[0]
        batch = make_views([sample], 13, 2, cfg)
        bucket = next(bucket for bucket in batch.buckets if bucket.resolution == cfg["global_crop_pixels"])
        shuffled = bucket.keep_idx[:, torch.randperm(bucket.keep_idx.shape[1])]
        with torch.no_grad():
            reference = model.backbone(bucket.pixels, bucket.keep_idx)
            permuted = model.backbone(bucket.pixels, shuffled)
        self.assertLessEqual(float((reference - permuted).abs().max()), 1e-12)

    def test_position_table_projector_and_block_sequence(self) -> None:
        dim = 24
        actual = sincos_pos_embed(6, dim, dtype=torch.float64)
        position = (torch.arange(6, dtype=torch.float64) + 0.5) * 14 / 6
        rows, columns = torch.meshgrid(position, position, indexing="ij")
        frequency = 1 / (10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4)))

        def embed(values):
            phase = values.reshape(-1, 1) * frequency
            return torch.cat((phase.sin(), phase.cos()), dim=1)

        expected = torch.cat((embed(rows), embed(columns)), dim=1)
        self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1e-14))

        cfg = _config()
        model = JEPAEncoder(cfg).double().eval()
        self.assertIsInstance(model.projector[0], nn.Linear)
        self.assertIsInstance(model.projector[1], nn.Identity)
        self.assertIsInstance(model.projector[3], nn.Linear)
        self.assertIsInstance(model.projector[6], nn.Linear)
        self.assertEqual(model.projector[-1].out_features, cfg["projector_dim"])
        batch = _batch(cfg)
        seen_lengths = []
        hook = model.blocks[0].register_forward_pre_hook(
            lambda _module, arguments: seen_lengths.append(arguments[0].shape[1])
        )
        model(batch)
        hook.remove()
        for bucket, sequence_length in zip(batch.buckets, seen_lengths):
            self.assertEqual(sequence_length, bucket.keep_idx.shape[1] + 1)

    def test_projector_norm_modes_and_batch_state(self) -> None:
        cfg = _config("frozen")
        frozen = JEPAEncoder(cfg).double().train()
        batch = _batch(cfg)
        batch_norms = [module for module in frozen.projector.modules() if isinstance(module, nn.BatchNorm1d)]
        before = [(module.running_mean.clone(), module.running_var.clone()) for module in batch_norms]
        first = frozen(batch).z
        second = frozen(batch).z
        self.assertTrue(torch.equal(first, second))
        for module, (mean, variance) in zip(batch_norms, before):
            self.assertFalse(module.training)
            self.assertTrue(torch.equal(module.running_mean, mean))
            self.assertTrue(torch.equal(module.running_var, variance))

        cfg["projector_norm"] = "batch"
        batch_norm_model = JEPAEncoder(cfg).double()
        train_output = batch_norm_model.train()(batch).z
        eval_output = batch_norm_model.eval()(batch).z
        self.assertGreater(float((train_output - eval_output).detach().abs().max()), 1e-6)

    def test_invalid_bucket_shapes_missing_pairs_and_gradients(self) -> None:
        cfg = _config()
        model = JEPAEncoder(cfg).double()
        batch = _batch(cfg)
        bucket = batch.buckets[0]
        with self.assertRaisesRegex(ValueError, "keep_idx width"):
            model.backbone(bucket.pixels, bucket.keep_idx[:, :-1])

        target_view = int(bucket.view_idx[0])
        target_sample = int(bucket.sample_idx[0])
        new_buckets = []
        removed = False
        for current in batch.buckets:
            matches = (current.view_idx == target_view) & (current.sample_idx == target_sample)
            keep_rows = torch.where(~matches)[0]
            removed = removed or bool(matches.any())
            if keep_rows.numel():
                new_buckets.append(
                    dataclasses.replace(
                        current,
                        pixels={
                            sensor: pixels.index_select(0, keep_rows)
                            for sensor, pixels in current.pixels.items()
                        },
                        keep_idx=current.keep_idx.index_select(0, keep_rows),
                        view_idx=current.view_idx.index_select(0, keep_rows),
                        sample_idx=current.sample_idx.index_select(0, keep_rows),
                    )
                )
        self.assertTrue(removed)
        incomplete = dataclasses.replace(batch, buckets=new_buckets)
        with self.assertRaisesRegex(ValueError, "filled exactly once"):
            model(incomplete)

        model.zero_grad(set_to_none=True)
        output = model(batch)
        output.z.square().mean().backward()
        for parameter in [
            *model.stems["s1"].parameters(),
            *model.stems["s2"].parameters(),
            model.modality_embed["s1"],
            model.modality_embed["s2"],
        ]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
