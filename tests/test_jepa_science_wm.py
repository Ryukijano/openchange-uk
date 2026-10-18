from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("clean-JEPA tests require the optional torch dependency")

import torch

from openchange.jepa.losses import sigreg_directions
from openchange.jepa.science_wm import (
    FactorizedSpatiotemporalJEPAWorldModel,
    MultiSensorContinuumPatchStem,
    PhysicalForcingEncoder,
    gpu_counterfactual_observation_value,
)
from openchange.jepa.wam import HistoryStratifiedSIGReg

torch.set_num_threads(min(torch.get_num_threads(), 4))


class ScienceWorldModelTests(unittest.TestCase):
    def test_physical_forcing_encoder_accumulates_stress(self) -> None:
        torch.manual_seed(0)
        enc = PhysicalForcingEncoder(forcing_dim=3, dim=16, stress_decay=0.9)
        base = torch.zeros(2, 4, 3)
        # Sustained positive heat/drought anomaly grows cumulative stress monotonically
        forcing = torch.ones(2, 4, 3) * 0.5
        film, emb, stress = enc(forcing, base)
        self.assertEqual(film.shape, (2, 4, 32))
        self.assertEqual(emb.shape, (2, 4, 16))
        self.assertEqual(stress.shape, (2, 4, 3))
        for t in range(1, 4):
            self.assertTrue(bool((stress[:, t] > stress[:, t - 1]).all()))

    def test_multisensor_and_continuum_stem_omits_missing_sensors(self) -> None:
        torch.manual_seed(1)
        stem = MultiSensorContinuumPatchStem(
            image_size=32, patch_size=16, dim=12, s1_channels=2, s2_channels=12, continuum_channels=4
        )
        s1 = torch.randn(2, 3, 2, 32, 32)
        s2 = torch.randn(2, 3, 12, 32, 32)
        s2_valid = torch.ones(2, 3, 32, 32, dtype=torch.bool)
        s2_valid[:, :, :16, :16] = False

        z_s1 = stem(s1=s1)
        z_s2 = stem(s2=s2, s2_valid=s2_valid)
        z_joint = stem(s1=s1, s2=s2, s2_valid=s2_valid)
        self.assertEqual(z_s1.shape, (2, 3, 4, 12))
        self.assertEqual(z_s2.shape, (2, 3, 4, 12))
        self.assertEqual(z_joint.shape, (2, 3, 4, 12))
        self.assertFalse(torch.allclose(z_s1, z_joint))

        # Continuum PDE fields path (The Well / Walrus)
        fields = torch.randn(2, 3, 4, 32, 32)
        z_pde = stem(fields=fields)
        self.assertEqual(z_pde.shape, (2, 3, 4, 12))

    def test_factorized_spatiotemporal_world_model_loss_and_gradients(self) -> None:
        torch.manual_seed(2)
        stem = MultiSensorContinuumPatchStem(
            image_size=32, patch_size=16, dim=16, s1_channels=2, s2_channels=12, continuum_channels=3
        )
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem,
            forcing_dim=4,
            observable_dim=2,
            horizon=3,
            heads=2,
            depth=2,
            mlp_ratio=2,
            spatial_radius=1,
            max_timesteps=8,
        )
        s1_ctx = torch.randn(4, 3, 2, 32, 32)
        s2_ctx = torch.randn(4, 3, 12, 32, 32)
        s1_fut = torch.randn(4, 3, 2, 32, 32)
        s2_fut = torch.randn(4, 3, 12, 32, 32)
        f_ctx = torch.randn(4, 3, 4)
        f_fut = torch.randn(4, 3, 4)
        obs_tgt = torch.randn(4, 3, 2)
        hist = torch.tensor([0.0, 2.4, 0.0, 2.4])
        dirs = sigreg_directions(16, 12, seed=5, step=0, device="cpu", dtype=torch.float32)
        sigreg = HistoryStratifiedSIGReg((0.0, 2.4))

        z_ctx = stem(s1=s1_ctx, s2=s2_ctx)
        z_fut = stem(s1=s1_fut, s2=s2_fut)
        metrics = wm.science_wm_loss(
            z_context=z_ctx,
            z_future=z_fut,
            forcing_context=f_ctx,
            forcing_future=f_fut,
            history_seconds=hist,
            directions=dirs,
            sigreg=sigreg,
            target_observable=obs_tgt,
        )
        self.assertTrue(torch.isfinite(metrics["loss"]))
        self.assertGreater(float(metrics["mean_uncertainty_std"]), 0.0)
        self.assertGreater(float(metrics["forced_energy_ratio"]), 0.0)
        metrics["loss"].backward()
        self.assertGreater(float(wm.intrinsic_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(wm.forced_head[0].weight.grad.abs().sum()), 0.0)

    def test_gpu_counterfactual_observation_value_ranking(self) -> None:
        torch.manual_seed(3)
        stem = MultiSensorContinuumPatchStem(image_size=32, patch_size=16, dim=8)
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem, forcing_dim=2, observable_dim=1, horizon=2, heads=2, depth=1
        )
        z_ctx = torch.randn(2, 3, 4, 8)
        f_ctx = torch.randn(2, 3, 2)
        f_fut = torch.randn(2, 2, 2)

        candidates = [
            {
                "sensor": "S1_SAR_all_weather",
                "z_observation": torch.randn(2, 1, 4, 8) * 2.0,
                "forcing_observation": torch.randn(2, 1, 2),
                "cost_mb": 0.56,
            },
            {
                "sensor": "S2_Optical_cloudy_heavy",
                "z_observation": z_ctx[:, -1:].clone(),
                "forcing_observation": f_ctx[:, -1:].clone(),
                "cost_mb": 1.81,
            },
        ]
        result = gpu_counterfactual_observation_value(
            wm,
            z_context=z_ctx,
            candidate_observations=candidates,
            forcing_context=f_ctx,
            forcing_future=f_fut,
        )
        self.assertIn("recommended", result)
        self.assertEqual(len(result["rankings"]), 2)
        # Compact S1 SAR with non-trivial state update beats redundant 1.81MB optical pass
        self.assertEqual(result["recommended"]["sensor"], "S1_SAR_all_weather")


if __name__ == "__main__":
    unittest.main()
