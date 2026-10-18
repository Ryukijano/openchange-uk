from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("install .venv-jepa with CPU torch")

import numpy as np
import torch

from openchange.jepa.config import keep_count, load_config
from openchange.jepa.views import (
    Observation,
    _local_sensor_subset,
    audit_observation_batch_pathologies,
    make_views,
    synthetic_batch,
    synthetic_observation,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "jepa" / "p1_synthetic.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _batch_equal(left, right) -> bool:
    tensor_fields = (
        "strata",
        "sample_ids",
        "synthetic_data",
        "synthetic_condition",
        "crop_boxes",
        "flip_rot",
        "tokens",
        "s2_valid_fraction",
    )
    for key in tensor_fields:
        left_value, right_value = getattr(left, key), getattr(right, key)
        if left_value.is_floating_point():
            if not torch.allclose(left_value, right_value, equal_nan=True):
                return False
        elif not torch.equal(left_value, right_value):
            return False
    if left.view_sensors != right.view_sensors or len(left.buckets) != len(right.buckets):
        return False
    for left_bucket, right_bucket in zip(left.buckets, right.buckets):
        if (
            left_bucket.resolution != right_bucket.resolution
            or left_bucket.sensors != right_bucket.sensors
            or not torch.equal(left_bucket.keep_idx, right_bucket.keep_idx)
            or not torch.equal(left_bucket.view_idx, right_bucket.view_idx)
            or not torch.equal(left_bucket.sample_idx, right_bucket.sample_idx)
        ):
            return False
        if any(
            not torch.equal(left_bucket.pixels[sensor], right_bucket.pixels[sensor])
            for sensor in left_bucket.sensors
        ):
            return False
    return True


def _sample_signature(batch, sample_index: int) -> list[tuple]:
    signature = []
    for view in range(batch.n_views):
        content = []
        for bucket in batch.buckets:
            matches = torch.where(
                (bucket.view_idx == view) & (bucket.sample_idx == sample_index)
            )[0]
            if matches.numel():
                row = int(matches[0])
                content.append(
                    (
                        bucket.sensors,
                        bucket.keep_idx[row].clone(),
                        {
                            sensor: bucket.pixels[sensor][row].clone()
                            for sensor in bucket.sensors
                        },
                    )
                )
        signature.append(
            (
                batch.view_sensors[view][sample_index],
                batch.crop_boxes[view, sample_index].clone(),
                batch.flip_rot[view, sample_index].clone(),
                batch.tokens[view, sample_index].clone(),
                content,
            )
        )
    return signature


class TestJEPAViews(unittest.TestCase):
    def test_sampling_is_repeatable_and_step_dependent(self) -> None:
        cfg = _config()
        observations = [
            synthetic_observation(index, index % 4, 11, cfg)[0] for index in range(4)
        ]
        first = make_views(observations, 11, 3, cfg)
        repeated = make_views(observations, 11, 3, cfg)
        later = make_views(observations, 11, 4, cfg)
        self.assertTrue(_batch_equal(first, repeated))
        self.assertFalse(torch.equal(first.crop_boxes, later.crop_boxes))

    def test_sample_views_are_independent_of_batch_neighbors(self) -> None:
        cfg = _config()
        sample_17 = synthetic_observation(17, 0, 5, cfg)[0]
        sample_3 = synthetic_observation(3, 1, 5, cfg)[0]
        sample_5 = synthetic_observation(5, 2, 5, cfg)[0]
        left = make_views([sample_17, sample_3], 5, 7, cfg)
        right = make_views([sample_5, sample_17], 5, 7, cfg)
        for left_view, right_view in zip(
            _sample_signature(left, 0),
            _sample_signature(right, 1),
        ):
            self.assertEqual(left_view[0], right_view[0])
            for left_value, right_value in zip(left_view[1:4], right_view[1:4]):
                self.assertTrue(torch.equal(left_value, right_value))
            self.assertEqual(len(left_view[4]), len(right_view[4]))
            for left_bucket, right_bucket in zip(left_view[4], right_view[4]):
                self.assertEqual(left_bucket[0], right_bucket[0])
                self.assertTrue(torch.equal(left_bucket[1], right_bucket[1]))
                for sensor in left_bucket[0]:
                    self.assertTrue(torch.equal(left_bucket[2][sensor], right_bucket[2][sensor]))

    def test_sensor_layout_shapes_and_global_views(self) -> None:
        cfg = _config()
        observations = [
            synthetic_observation(index, index, 3, cfg)[0] for index in range(4)
        ]
        batch = make_views(observations, 3, 0, cfg)
        self.assertEqual(batch.tokens.shape, (8, 4))
        self.assertEqual(batch.crop_boxes.shape, (8, 4, 4))
        self.assertEqual(batch.strata.dtype, torch.int64)
        self.assertEqual(batch.sample_ids.dtype, torch.int64)
        self.assertEqual(batch.synthetic_data.dtype, torch.bool)
        self.assertEqual(batch.synthetic_condition.dtype, torch.bool)
        self.assertEqual(batch.view_sensors[0][0], ("s1", "s2"))
        self.assertEqual(batch.view_sensors[1][1], ("s1", "s2"))
        for view in range(batch.n_views):
            for sample_index in range(batch.n_samples):
                if batch.strata[sample_index] == 2:
                    self.assertNotIn("s2", batch.view_sensors[view][sample_index])
        for bucket in batch.buckets:
            self.assertTrue(torch.all(bucket.keep_idx[:, 1:] > bucket.keep_idx[:, :-1]))
            total = (bucket.resolution // cfg["patch_size"]) ** 2 * len(bucket.sensors)
            rho = (
                cfg["global_token_drop"]
                if bucket.resolution == cfg["global_crop_pixels"]
                else cfg["local_token_drop"]
            )
            self.assertEqual(bucket.keep_idx.shape[1], keep_count(total, rho))
            self.assertTrue(torch.all((bucket.keep_idx >= 0) & (bucket.keep_idx < total)))
            for row, (view, sample) in enumerate(zip(bucket.view_idx, bucket.sample_idx)):
                self.assertEqual(
                    int(batch.tokens[int(view), int(sample)]),
                    bucket.keep_idx.shape[1],
                )

    def test_local_sensor_frequencies_match_configured_probabilities(self) -> None:
        cfg = _config()
        counts = {"s1": 0, "s2": 0, "both": 0}
        for step in range(21):
            for sample_id in range(32):
                for view in range(cfg["global_views"], cfg["global_views"] + cfg["local_views"]):
                    subset = _local_sensor_subset(
                        17,
                        step,
                        sample_id,
                        view,
                        cfg["local_sensor_probs"],
                    )
                    key = "both" if len(subset) == 2 else subset[0]
                    counts[key] += 1
        total = sum(counts.values())
        for key, probability in cfg["local_sensor_probs"].items():
            observed = counts[key]
            standard_error = (probability * (1.0 - probability) / total) ** 0.5
            self.assertLessEqual(abs(observed / total - probability), 4 * standard_error)

    def test_shared_geometry_and_paired_sensor_pixels(self) -> None:
        cfg = _config()
        rng = np.random.default_rng(44)
        s1 = rng.standard_normal((2, 128, 128), dtype=np.float32)
        s2 = rng.standard_normal((12, 128, 128), dtype=np.float32)
        s2[0] = s1[0]
        observation = Observation(
            sample_id=12,
            stratum=0,
            s1=s1,
            s2=s2,
            s2_valid=np.ones((128, 128), dtype=np.bool_),
            synthetic_data=True,
            synthetic_condition=False,
            intervention="none",
            mask_fraction=0.0,
        )
        batch = make_views([observation], 8, 1, cfg)
        joint_bucket = next(bucket for bucket in batch.buckets if bucket.sensors == ("s1", "s2"))
        row = int(torch.where((joint_bucket.view_idx == 0) & (joint_bucket.sample_idx == 0))[0][0])
        self.assertTrue(torch.equal(joint_bucket.pixels["s1"][row, 0], joint_bucket.pixels["s2"][row, 0]))

    def test_observation_and_real_data_refusals(self) -> None:
        cfg = _config()
        with self.assertRaisesRegex(ValueError, "s1_only"):
            Observation(
                sample_id=1,
                stratum=2,
                s1=None,
                s2=None,
                s2_valid=None,
                synthetic_data=True,
                synthetic_condition=False,
                intervention="withhold_s2",
                mask_fraction=0.0,
            )
        observation = synthetic_observation(1, 0, 2, cfg)[0]
        real_observation = Observation(
            **{
                **observation.__dict__,
                "synthetic_data": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "real data"):
            make_views([real_observation], 2, 0, cfg)

    def test_generator_intervention_and_occlusion_metadata(self) -> None:
        cfg = _config()
        baseline, latent = synthetic_observation(9, 0, 2, cfg)
        changed, _ = synthetic_observation(
            9,
            0,
            2,
            cfg,
            private_override=np.ones(cfg["synthetic"]["private_s2_factors"], dtype=np.float32),
        )
        self.assertTrue(np.array_equal(baseline.s1, changed.s1))
        self.assertFalse(np.array_equal(baseline.s2, changed.s2))
        self.assertIn("shared", latent)
        self.assertIn("private", latent)
        occluded, _ = synthetic_observation(10, 1, 2, cfg)
        self.assertEqual(occluded.mask_fraction, 1 - occluded.s2_valid.mean())
        self.assertGreaterEqual(occluded.mask_fraction, 0.2)
        self.assertLessEqual(occluded.mask_fraction, 0.6)
        self.assertTrue(occluded.synthetic_condition)
        self.assertEqual(occluded.intervention, "synthetic_occlusion")

    def test_sampler_uses_stratum_blocks_that_shard_evenly(self) -> None:
        cfg = _config()
        batch = synthetic_batch(4, 2, cfg, layout_world_size=2)
        labels = torch.tensor([observation.stratum for observation in batch])
        self.assertEqual(labels.tolist(), [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8)
        for rank in range(2):
            self.assertEqual(torch.bincount(labels[rank::2], minlength=4).tolist(), [4, 4, 4, 4])
        with self.assertRaisesRegex(ValueError, "divisible"):
            synthetic_batch(4, 2, {**cfg, "global_quotas": [7, 8, 8, 8]}, layout_world_size=2)

    def test_pair_gap_and_batch_pathology_audit(self) -> None:
        cfg = _config()
        obs0, _ = synthetic_observation(0, 0, 2, cfg)
        obs1, _ = synthetic_observation(1, 1, 2, cfg)
        obs2, _ = synthetic_observation(2, 2, 2, cfg)
        obs_with_gap = Observation(
            **{**obs0.__dict__, "pair_gap_seconds": 14.0 * 86400.0}
        )
        with self.assertRaisesRegex(ValueError, "pair_gap_seconds"):
            Observation(**{**obs0.__dict__, "pair_gap_seconds": -1.0})

        batch = make_views([obs_with_gap, obs1, obs2], 2, 0, cfg)
        self.assertIsNotNone(batch.pair_gap_seconds)
        self.assertAlmostEqual(float(batch.pair_gap_seconds[0]), 14.0 * 86400.0, places=1)

        report = audit_observation_batch_pathologies(
            [obs_with_gap, obs1, obs2], speckle_z_threshold=2.5, gap_tau_days=7.0
        )
        self.assertEqual(report["n_observations"], 3)
        self.assertAlmostEqual(report["pair_gap_days_max"], 14.0, places=6)
        self.assertAlmostEqual(float(report["pair_gap_weights"][0]), math.exp(-2.0), places=5)
        self.assertAlmostEqual(float(report["pair_gap_weights"][1]), 1.0, places=6)
        self.assertGreater(report["cloud_halo_boundary_fraction"], 0.0)

if __name__ == "__main__":
    unittest.main()
