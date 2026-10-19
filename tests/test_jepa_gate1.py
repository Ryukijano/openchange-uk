import importlib.util
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Gate-1 tests require the optional torch dependency")

import torch
import yaml

from openchange.jepa.config import ConfigError, load_config, validate
from openchange.jepa.gate1 import (
    Gate1Model,
    gate1_loss,
    mix_global_view_batch,
    stratum_preserving_permutation,
)
from openchange.jepa.losses import sigreg_directions
from openchange.jepa.model import EncoderOutput
from openchange.jepa.views import ViewBatch, ViewBucket


def _cfg(name: str) -> dict:
    path = Path("configs/jepa/p1_synthetic.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["objective"]["name"] = name
    if name == "defensive_split_cfm":
        cfg["objective"].update(
            {
                "shared_dim": 64,
                "s1_private_dim": 32,
                "s2_private_dim": 32,
                "gap_tau_seconds": 604800,
                "cfm_weight": 0.1,
            }
        )
    validate(cfg)
    return cfg


def _batch(n_samples: int = 4, n_views: int = 2) -> ViewBatch:
    bucket = ViewBucket(
        resolution=16,
        sensors=("s1",),
        pixels={"s1": torch.arange(n_views * n_samples * 4).reshape(n_views * n_samples, 1, 2, 2).float()},
        keep_idx=torch.zeros((n_views * n_samples, 2), dtype=torch.int64),
        view_idx=torch.tensor([0] * n_samples + [1] * n_samples, dtype=torch.int64),
        sample_idx=torch.tensor(list(range(n_samples)) * n_views, dtype=torch.int64),
    )
    return ViewBatch(
        buckets=[bucket],
        n_samples=n_samples,
        n_views=n_views,
        n_global=2,
        strata=torch.tensor([0, 0, 1, 1][:n_samples], dtype=torch.int64),
        sample_ids=torch.arange(n_samples, dtype=torch.int64),
        synthetic_data=torch.ones(n_samples, dtype=torch.bool),
        synthetic_condition=torch.zeros(n_samples, dtype=torch.bool),
        view_sensors=[[("s1",)] * n_samples for _ in range(n_views)],
        crop_boxes=torch.zeros((n_views, n_samples, 4), dtype=torch.int64),
        flip_rot=torch.zeros((n_views, n_samples, 2), dtype=torch.int64),
        tokens=torch.ones((n_views, n_samples), dtype=torch.int64),
        s2_valid_fraction=torch.full((n_views, n_samples), float("nan")),
        pair_gap_seconds=torch.arange(n_samples, dtype=torch.float32) * 86400.0,
    )


class Gate1Tests(unittest.TestCase):
    def test_stratum_permutation_preserves_sensor_availability(self) -> None:
        strata = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64)
        perm = stratum_preserving_permutation(strata, seed=0, step=7)
        self.assertEqual(sorted(perm.tolist()), list(range(8)))
        for value in torch.unique(strata):
            self.assertTrue(torch.equal(strata[perm[strata == value]], strata[strata == value]))
        self.assertTrue(
            torch.equal(perm, stratum_preserving_permutation(strata, seed=0, step=7))
        )

    def test_mix_global_view_uses_view_b_same_sensor_only(self) -> None:
        batch = _batch()
        perm = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
        mixed = mix_global_view_batch(batch, perm, 0.25)
        values = mixed.buckets[0].pixels["s1"]
        self.assertEqual(values.shape, (8, 1, 2, 2))
        # View 0, sample 0 (row 0, value 0) mixes with view 1, sample 1 (row 5, value 20).
        self.assertEqual(float(values[0].mean()), 0.25 * 0.0 + 0.75 * 22.0)
        # Non-global view rows are unchanged by construction (there are none in this batch).
        self.assertTrue(torch.equal(mixed.buckets[0].keep_idx, batch.buckets[0].keep_idx))
        self.assertTrue(torch.equal(mixed.buckets[0].view_idx, batch.buckets[0].view_idx))

    def test_all_objectives_return_finite_logging_contract(self) -> None:
        batch = _batch(n_samples=8)
        strata = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
        z = torch.randn((2, 8, 128), dtype=torch.float64)
        output = EncoderOutput(b=torch.empty(0), z=z, tokens=torch.empty(0))
        directions = sigreg_directions(128, 8, seed=0, step=0, device="cpu", dtype=torch.float64)
        for name in ("clean_jepa", "defensive_split_cfm", "tessera_mixup", "vicreg"):
            cfg = _cfg(name)
            model = Gate1Model(cfg)
            mixed = EncoderOutput(b=torch.empty(0), z=torch.randn((2, 8, 128)), tokens=torch.empty(0))
            result = gate1_loss(
                cfg,
                model=model,
                output=output,
                batch=batch,
                strata=strata,
                step=0,
                directions=directions,
                directions_s1=sigreg_directions(32, 8, seed=17, step=0, device="cpu", dtype=torch.float64),
                directions_s2=sigreg_directions(32, 8, seed=31, step=0, device="cpu", dtype=torch.float64),
                mixed_output=mixed if name == "tessera_mixup" else None,
                mix_perm=torch.arange(8) if name == "tessera_mixup" else None,
                mix_alpha=0.5 if name == "tessera_mixup" else None,
                ctx=None,
                chunk=4,
            )
            for key in ("loss", "inv", "sigreg", "t_mu", "ep"):
                self.assertIn(key, result)
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertEqual(result["ep"].shape, (2, 4))

    def test_gate1_configs_are_reviewed_and_valid(self) -> None:
        expected = {
            "gh200_local_pilot": "clean_jepa",
            "gate1_defensive": "defensive_split_cfm",
            "gate1_tessera": "tessera_mixup",
            "gate1_vicreg": "vicreg",
        }
        for stem, name in expected.items():
            cfg = load_config(Path(f"configs/jepa/{stem}.yaml"), local_pilot=True, cuda=True)
            self.assertEqual(cfg["objective"]["name"], name)
        bad = _cfg("clean_jepa")
        bad["objective"]["name"] = "unreviewed"
        with self.assertRaises(ConfigError):
            validate(bad)


if __name__ == "__main__":
    unittest.main()
