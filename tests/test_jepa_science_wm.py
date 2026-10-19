from __future__ import annotations

import importlib.util
import math
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("clean-JEPA tests require the optional torch dependency")

import torch

from openchange.jepa.assimilation import (
    CandidateSensorPass,
    CounterfactualVoIEvaluator,
    SubspaceDataAssimilator,
    SubspacePartition,
)
from openchange.jepa.coastal_physics import (
    CoastalHydrodynamicConfig,
    CoastalHydrodynamicModel,
    ShoreForDisequilibriumResult,
    StockdonRunupResult,
)
from openchange.jepa.guarantees import (
    StratifiedConformalAlertCalibrator,
    ensemble_crps,
)
from openchange.jepa.losses import sigreg_directions
from openchange.jepa.science_wm import (
    EO_HISTORY_STRATA_SECONDS,
    FactorizedSpatiotemporalJEPAWorldModel,
    ForcingElasticityEvaluator,
    ForecastingStochasticInterpolantHead,
    MultiSensorContinuumPatchStem,
    PhysicalForcingEncoder,
    assimilate_subspace_observation,
    build_coastal_forcing_tensor,
    forcing_contrastive_loss,
    forcing_fd_elasticity_and_zero_probe,
    gpu_counterfactual_observation_value,
    shorefor_disequilibrium_stress,
    stockdon_wave_setup_and_runup,
)
from openchange.jepa.stochastic_interpolant import (
    EnsembleRolloutOutput,
    StochasticInterpolantConfig,
)
from openchange.jepa.wam import HistoryStratifiedSIGReg

torch.set_num_threads(min(torch.get_num_threads(), 4))


class ScienceWorldModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float32)

    def tearDown(self) -> None:
        torch.set_default_dtype(torch.float32)

    def test_physical_forcing_encoder_accumulates_stress(self) -> None:
        torch.manual_seed(0)
        enc = PhysicalForcingEncoder(forcing_dim=3, dim=16, stress_decay=0.9)
        base = torch.zeros(2, 4, 3)
        # Sustained positive heat/drought/storm anomaly grows cumulative stress monotonically
        forcing = torch.ones(2, 4, 3) * 0.5
        film, emb, stress = enc(forcing, base)
        self.assertEqual(film.shape, (2, 4, 32))
        self.assertEqual(emb.shape, (2, 4, 16))
        self.assertEqual(stress.shape, (2, 4, 3))
        for t in range(1, 4):
            self.assertTrue(bool((stress[:, t] > stress[:, t - 1]).all()))

    def test_coastal_stockdon_and_shorefor_physics_formulations(self) -> None:
        # Stockdon et al. (2006) wave setup, runup R_2, and horizontal waterline excursion
        hs = torch.tensor([[1.0, 2.5, 4.0, 1.2]], dtype=torch.float64)
        tp = torch.tensor([[8.0, 10.0, 12.0, 11.0]], dtype=torch.float64)
        tide = torch.tensor([[0.0, 0.5, 1.2, -0.4]], dtype=torch.float64)
        surge = torch.tensor([[0.0, 0.2, 0.6, 0.0]], dtype=torch.float64)
        slope = 0.1

        stock = stockdon_wave_setup_and_runup(hs, tp, slope, tide=tide, surge=surge)
        expected_l0 = (9.80665 / (2.0 * math.pi)) * tp.square()
        torch.testing.assert_close(stock["wavelength_l0"], expected_l0, atol=1e-12, rtol=1e-12)
        expected_setup = 0.35 * slope * (hs * expected_l0).sqrt()
        torch.testing.assert_close(stock["wave_setup"], expected_setup, atol=1e-12, rtol=1e-12)
        # Higher waves + storm surge + high tide produce larger landward waterline excursion
        self.assertGreater(
            float(stock["hydro_excursion"][0, 2]),
            float(stock["hydro_excursion"][0, 0]),
        )

        # ShoreFor (Yates et al. 2009): storm wave spike (t=2) produces positive erosion
        # disequilibrium, while post-storm low-energy swell (t=3) produces negative (accretion) disequilibrium
        shore = shorefor_disequilibrium_stress(hs, tp, fall_velocity=0.04, memory_decay=0.85)
        self.assertAlmostEqual(float(shore["disequilibrium"][0, 0]), 0.0, places=10)
        self.assertGreater(float(shore["disequilibrium"][0, 2]), 0.0)
        self.assertLess(float(shore["disequilibrium"][0, 3]), 0.0)
        self.assertGreater(
            float(shore["cumulative_erosion_stress"][0, 2]),
            float(shore["cumulative_erosion_stress"][0, 1]),
        )

        forcing_8, baseline_8 = build_coastal_forcing_tensor(
            hs.to(torch.float32),
            tp.to(torch.float32),
            tide=tide.to(torch.float32),
            surge=surge.to(torch.float32),
            beach_slope=slope,
        )
        self.assertEqual(forcing_8.shape, (1, 4, 8))
        self.assertEqual(baseline_8.shape, (1, 4, 8))

        # Object-oriented CoastalHydrodynamicModel & typed dataclass results
        cfg = CoastalHydrodynamicConfig(beach_slope=slope, fall_velocity=0.04, memory_decay=0.85)
        hydro_model = CoastalHydrodynamicModel(cfg)
        stock_res = hydro_model.compute_stockdon(hs, tp, tide=tide, surge=surge)
        shore_res = hydro_model.compute_shorefor(hs, tp)
        self.assertIsInstance(stock_res, StockdonRunupResult)
        self.assertIsInstance(shore_res, ShoreForDisequilibriumResult)
        torch.testing.assert_close(stock_res.wave_setup, stock["wave_setup"])
        torch.testing.assert_close(shore_res.disequilibrium, shore["disequilibrium"])
        f_oop, b_oop = hydro_model(
            hs.to(torch.float32),
            tp.to(torch.float32),
            tide=tide.to(torch.float32),
            surge=surge.to(torch.float32),
        )
        torch.testing.assert_close(f_oop, forcing_8)
        torch.testing.assert_close(b_oop, baseline_8)

    def test_multisensor_stem_unbiased_cloud_renormalization_and_zero_patch(self) -> None:
        torch.manual_seed(1)
        stem = MultiSensorContinuumPatchStem(
            image_size=32, patch_size=16, dim=12, s1_channels=2, s2_channels=12, continuum_channels=4
        ).to(dtype=torch.float64)

        # Uniform-valued patch: half-clouded (p = 0.5) unbiased conv must match full clear patch (p = 1.0)
        s2_const = torch.full((1, 2, 12, 32, 32), 0.75, dtype=torch.float64)
        mask_full = torch.ones((1, 2, 32, 32), dtype=torch.bool)
        mask_half = torch.ones((1, 2, 32, 32), dtype=torch.bool)
        mask_half[:, :, :8, :16] = False  # top-left patch is 50% clouded (8x16 out of 16x16)

        tok_full, w_full = stem._patchify_stream(s2_const, stem.s2_stem, 1, pixel_mask=mask_full)
        tok_half, w_half = stem._patchify_stream(s2_const, stem.s2_stem, 1, pixel_mask=mask_half)
        self.assertAlmostEqual(float(w_full[0, 0, 0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(w_half[0, 0, 0, 0]), 0.5, places=12)
        torch.testing.assert_close(tok_full[:, :, 0, :], tok_half[:, :, 0, :], atol=1e-10, rtol=1e-10)

        # 100% clouded patch (p = 0) must emit identically 0 on S2-only and not corrupt S1 on joint fusion
        mask_zero_p0 = torch.ones((2, 3, 32, 32), dtype=torch.bool)
        mask_zero_p0[:, :, :16, :16] = False
        s1 = torch.randn(2, 3, 2, 32, 32, dtype=torch.float64)
        s2 = torch.randn(2, 3, 12, 32, 32, dtype=torch.float64)

        z_s1 = stem(s1=s1)
        z_s2 = stem(s2=s2, s2_valid=mask_zero_p0)
        z_joint = stem(s1=s1, s2=s2, s2_valid=mask_zero_p0)
        self.assertEqual(z_s1.shape, (2, 3, 4, 12))
        self.assertEqual(z_s2.shape, (2, 3, 4, 12))
        self.assertEqual(z_joint.shape, (2, 3, 4, 12))
        # Patch 0 has p = 0 -> S2-only output for patch 0 is identically 0
        self.assertEqual(float(z_s2[:, :, 0, :].detach().abs().max()), 0.0)
        # On patch 0 (p = 0), joint S1+S2 equals pure S1 exactly
        torch.testing.assert_close(z_joint[:, :, 0, :], z_s1[:, :, 0, :], atol=1e-12, rtol=1e-12)
        # On clear patches (patches 1..3), joint differs from S1-only
        self.assertFalse(torch.allclose(z_s1[:, :, 1:, :], z_joint[:, :, 1:, :]))

        fields = torch.randn(2, 3, 4, 32, 32, dtype=torch.float64)
        z_pde = stem(fields=fields)
        self.assertEqual(z_pde.shape, (2, 3, 4, 12))

    def test_zero_forcing_invariant_and_shoreline_hydro_morph_decomposition(self) -> None:
        torch.manual_seed(4)
        stem = MultiSensorContinuumPatchStem(image_size=32, patch_size=16, dim=12)
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem,
            forcing_dim=8,
            observable_dim=1,
            horizon=3,
            heads=2,
            depth=2,
        )
        z_ctx = torch.randn(2, 3, 4, 12)
        zero_ctx = torch.zeros(2, 3, 8)
        zero_fut = torch.zeros(2, 3, 8)

        out_zero = wm.forward_latents(z_ctx, zero_ctx, zero_fut)
        # Zero-anchored invariant: Delta z_forced(h, 0) == 0 and hydro offset == 0 identically
        self.assertEqual(float(out_zero.z_forced_delta.detach().abs().max()), 0.0)
        assert out_zero.hydrodynamic_observable_offset is not None
        assert out_zero.morphological_observable is not None
        self.assertEqual(float(out_zero.hydrodynamic_observable_offset.detach().abs().max()), 0.0)
        torch.testing.assert_close(
            out_zero.observable_pred, out_zero.morphological_observable, atol=1e-7, rtol=1e-7
        )

        # Under coastal storm + tidal forcing, both forced latent delta and hydrodynamic offset activate
        hs = torch.tensor([[1.0, 1.2, 1.1, 3.5, 4.2, 1.5], [1.0, 1.0, 1.0, 2.8, 3.1, 1.2]])
        tp = torch.tensor([[8.0, 8.5, 8.0, 12.0, 13.0, 10.0], [9.0, 9.0, 9.0, 11.0, 12.0, 9.5]])
        tide = torch.tensor([[0.0, 0.2, -0.1, 1.1, 1.4, -0.5], [0.0, -0.2, 0.1, 0.9, 1.2, -0.3]])
        f_all, b_all = build_coastal_forcing_tensor(hs, tp, tide=tide)
        out_storm = wm.forward_latents(
            z_ctx,
            f_all[:, :3],
            f_all[:, 3:],
            baseline_context=b_all[:, :3],
            baseline_future=b_all[:, 3:],
        )
        self.assertGreater(float(out_storm.z_forced_delta.detach().abs().mean()), 0.0)
        assert out_storm.hydrodynamic_observable_offset is not None
        self.assertGreater(
            float(out_storm.hydrodynamic_observable_offset.detach().abs().mean()), 0.0
        )

        # Input validation guards
        with self.assertRaises(ValueError):
            wm.forward_latents(z_ctx, f_all[:, :3], torch.zeros(2, 10, 8))
        with self.assertRaises(ValueError):
            wm.forward_latents(z_ctx, torch.zeros(2, 3, 5), f_all[:, 3:])

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
        hist = torch.tensor(
            [
                EO_HISTORY_STRATA_SECONDS[0],
                EO_HISTORY_STRATA_SECONDS[1],
                EO_HISTORY_STRATA_SECONDS[2],
                EO_HISTORY_STRATA_SECONDS[3],
            ]
        )
        dirs = sigreg_directions(16, 12, seed=5, step=0, device="cpu", dtype=torch.float32)
        sigreg = HistoryStratifiedSIGReg(EO_HISTORY_STRATA_SECONDS[:4])

        for pool_mode in ("patch_flat", "frame_mean", "context_and_pred"):
            wm.zero_grad()
            stem.zero_grad()
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
                sigreg_pool=pool_mode,
            )
            self.assertTrue(torch.isfinite(metrics["loss"]))
            self.assertTrue(torch.isfinite(metrics["observable_crps"]))
            self.assertTrue(torch.isfinite(metrics["si_loss"]))
            self.assertGreater(float(metrics["effective_rank"]), 1.0)
            self.assertGreater(float(metrics["mean_uncertainty_std"]), 0.0)
            self.assertGreater(float(metrics["forced_energy_ratio"]), 0.0)
            for key in ("ep", "ep_over_n", "counts", "null"):
                self.assertIn(key, metrics)
            metrics["loss"].backward()
            self.assertGreater(float(wm.intrinsic_head.weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(wm.forced_head[0].weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(wm.si_head.drift_out.weight.grad.abs().sum()), 0.0)

    def test_forcing_contrastive_loss_and_fd_elasticity_probe(self) -> None:
        torch.manual_seed(6)
        stem = MultiSensorContinuumPatchStem(image_size=32, patch_size=16, dim=8)
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem, forcing_dim=3, observable_dim=1, horizon=2, heads=2, depth=1
        )
        z_ctx = torch.randn(3, 2, 4, 8)
        f_ctx = torch.randn(3, 2, 3)
        f_fut = torch.randn(3, 2, 3) * 1.5
        with torch.no_grad():
            z_target = wm.forward_latents(z_ctx, f_ctx, f_fut).z_hat

        contrast = forcing_contrastive_loss(wm, z_ctx, f_ctx, f_fut, z_target, temperature=0.05)
        self.assertTrue(torch.isfinite(contrast["loss"]))
        self.assertGreater(float(contrast["forcing_elasticity"].detach()), 0.0)
        self.assertAlmostEqual(float(contrast["forcing_identification_accuracy"]), 1.0, places=5)

        probe = forcing_fd_elasticity_and_zero_probe(wm, z_ctx, f_ctx, f_fut, delta=1e-2, n_probes=3)
        self.assertGreater(float(probe["fd_elasticity"]), 0.0)
        self.assertGreater(float(probe["zero_forcing_response_gap"]), 0.0)
        self.assertEqual(float(probe["zero_forced_delta_max"]), 0.0)

        # Direct OOP ForcingElasticityEvaluator check
        oop_eval = ForcingElasticityEvaluator(wm)
        oop_contrast = oop_eval.contrastive_loss(z_ctx, f_ctx, f_fut, z_target, temperature=0.05)
        torch.testing.assert_close(oop_contrast["loss"], contrast["loss"])

    def test_atlas_forecasting_stochastic_interpolant_and_multimodal_falsifier(self) -> None:
        torch.manual_seed(7)
        si_cfg = StochasticInterpolantConfig(
            dim=4, cond_dim=4, hidden_dim=64, time_dim=16, sigma_si=0.3, seed=7
        )
        si_head = ForecastingStochasticInterpolantHead(dim=4, cond_dim=4, config=si_cfg)
        # Exact endpoint boundary conditions gamma(0) == 0 and gamma(1) == 0
        self.assertAlmostEqual(float(si_head.gamma(torch.tensor(0.0))), 0.0, places=12)
        self.assertAlmostEqual(float(si_head.gamma(torch.tensor(1.0))), 0.0, places=12)

        # Train SI head on a bimodal coastal branching transition (storm erosion mode +z* vs swell recovery -z*)
        n_batch = 64
        z_prev = torch.zeros(n_batch, 4)
        cond = torch.ones(n_batch, 4)
        mode_a = torch.tensor([2.0, 2.0, -2.0, -2.0])
        mode_b = -mode_a
        signs = torch.where(torch.arange(n_batch) % 2 == 0, 1.0, -1.0).unsqueeze(-1)
        z_next = signs * mode_a.unsqueeze(0) + 0.05 * torch.randn(n_batch, 4)

        opt = torch.optim.Adam(si_head.parameters(), lr=1e-2)
        for step in range(120):
            opt.zero_grad()
            loss_dict = si_head.loss(z_prev, z_next, cond, seed=step)
            loss_dict["loss"].backward()
            opt.step()

        # Deterministic L2 mean predictor collapses to 0 (chord midpoint, high distance to both modes)
        l2_mean_pred = z_next.mean(dim=0, keepdim=True)
        dist_l2_to_modes = torch.minimum(
            (l2_mean_pred - mode_a).norm(dim=-1),
            (l2_mean_pred - mode_b).norm(dim=-1),
        ).mean()

        # SDE Stochastic Interpolant ensemble samples move off the zero midpoint and beat collapsed CRPS
        sde_samples = si_head.sample(
            torch.zeros(32, 4),
            torch.ones(32, 4),
            steps=12,
            solver="sde",
            sde_diffusion=0.6,
            seed=99,
        )
        dist_sde_to_modes = torch.minimum(
            (sde_samples - mode_a).norm(dim=-1),
            (sde_samples - mode_b).norm(dim=-1),
        ).mean()
        self.assertLess(float(dist_sde_to_modes.detach()), float(dist_l2_to_modes.detach()))

        # World-model ensemble rollout produces well-formed [M, B, K, G*G, D] and CRPS
        stem = MultiSensorContinuumPatchStem(image_size=32, patch_size=16, dim=8)
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem, forcing_dim=2, observable_dim=1, horizon=2, heads=2, depth=1
        )
        z_ctx = torch.randn(2, 3, 4, 8)
        f_ctx = torch.randn(2, 3, 2)
        f_fut = torch.randn(2, 2, 2)
        ens_out = wm.sample_ensemble_latents(
            z_ctx, f_ctx, f_fut, ensemble_size=6, si_steps=4, solver="sde", seed=11
        )
        self.assertIsInstance(ens_out, EnsembleRolloutOutput)
        self.assertEqual(ens_out.z_ensemble.shape, (6, 2, 2, 4, 8))
        self.assertEqual(ens_out["observable_ensemble"].shape, (6, 2, 2, 1))
        self.assertGreater(float(ens_out.z_std.detach().mean()), 0.0)
        crps_val = ensemble_crps(ens_out.observable_ensemble, torch.zeros(2, 2, 1))
        self.assertTrue(torch.isfinite(crps_val))

    def test_subspace_assimilation_and_batched_voi_policy_evaluation(self) -> None:
        torch.manual_seed(3)
        # 1. Verify OOP SubspacePartition & SubspaceDataAssimilator: S1 updates [z_sh, z_s1] and decays z_s2
        partition = SubspacePartition(total_dim=8, shared_dim=4, s1_private_dim=2, s2_private_dim=2)
        assimilator = SubspaceDataAssimilator(partition, default_gain=1.0, gap_tau_seconds=604800.0)
        z_prior = torch.ones(2, 1, 4, 8)
        z_obs_s1 = torch.full((2, 1, 4, 8), 5.0)
        assim_s1 = assimilate_subspace_observation(
            z_prior,
            z_obs_s1,
            modality="s1",
            shared_dim=4,
            s1_private_dim=2,
            s2_private_dim=2,
            assimilation_gain=1.0,
            gap_seconds=604800.0,
            gap_tau_seconds=604800.0,
        )
        torch.testing.assert_close(
            assim_s1,
            assimilator(z_prior, z_obs_s1, modality="s1", gap_seconds=604800.0),
        )
        # Shared [0:4] and S1-private [4:6] updated to 5.0; S2-private [6:8] preserved with 1/e decay
        torch.testing.assert_close(assim_s1[..., :6], torch.full((2, 1, 4, 6), 5.0))
        torch.testing.assert_close(
            assim_s1[..., 6:], torch.full((2, 1, 4, 2), math.exp(-1.0)), atol=1e-6, rtol=1e-6
        )

        # 100% cloudy S2 pass leaves shared & S2-private prior unchanged (clear_gain == 0)
        assim_cloudy_s2 = assimilate_subspace_observation(
            z_prior,
            z_obs_s1,
            modality="s2",
            shared_dim=4,
            s1_private_dim=2,
            s2_private_dim=2,
            assimilation_gain=1.0,
            cloud_fraction=1.0,
            gap_seconds=0.0,
        )
        torch.testing.assert_close(assim_cloudy_s2, z_prior)

        # 2. Verify batched VoI ranking, conformal calibrator wiring, and policy_evaluation
        stem = MultiSensorContinuumPatchStem(image_size=32, patch_size=16, dim=8)
        wm = FactorizedSpatiotemporalJEPAWorldModel(
            stem=stem, forcing_dim=2, observable_dim=1, horizon=2, heads=2, depth=1
        )
        z_ctx = torch.randn(2, 3, 4, 8)
        f_ctx = torch.randn(2, 3, 2)
        f_fut = torch.randn(2, 2, 2)

        cal = StratifiedConformalAlertCalibrator(alpha=0.1)
        cal.fit([0.5, 0.8, 1.1, 1.4, 1.9], groups=[("winter", "coast")] * 5)

        s1_obs = torch.randn(2, 1, 4, 8) * 2.0
        s1_f = torch.randn(2, 1, 2)
        candidates = [
            {
                "sensor": "S1_SAR_all_weather",
                "modality": "s1",
                "z_observation": s1_obs,
                "forcing_observation": s1_f,
                "cost_mb": 0.56,
            },
            {
                "sensor": "S2_Optical_cloudy_heavy",
                "modality": "s2",
                "cloud_fraction": 0.85,
                "z_observation": z_ctx[:, -1:].clone(),
                "forcing_observation": f_ctx[:, -1:].clone(),
                "cost_mb": 1.81,
            },
        ]

        # Target future matches the S1-assimilated trajectory so revealing S1 reduces realized error
        with torch.no_grad():
            z_s1_assim = assimilate_subspace_observation(
                z_ctx[:, -1:], s1_obs, modality="s1", shared_dim=4, s1_private_dim=2, s2_private_dim=2
            )
            z_s1_ctx = torch.cat((z_ctx[:, 1:], z_s1_assim), dim=1)
            f_s1_ctx = torch.cat((f_ctx[:, 1:], s1_f), dim=1)
            target_fut = wm.forward_latents(z_s1_ctx, f_s1_ctx, f_fut).z_hat

        result = gpu_counterfactual_observation_value(
            wm,
            z_context=z_ctx,
            candidate_observations=candidates,
            forcing_context=f_ctx,
            forcing_future=f_fut,
            calibrator=cal,
            calibrator_group=("winter", "coast"),
            shared_dim=4,
            s1_private_dim=2,
            s2_private_dim=2,
            target_future=target_fut,
        )
        self.assertIn("recommended", result)
        self.assertEqual(len(result["rankings"]), 2)
        self.assertEqual(result["recommended"]["sensor"], "S1_SAR_all_weather")
        self.assertIn("policy_evaluation", result)
        self.assertLess(
            result["policy_evaluation"]["always_s1"]["mean_loss"],
            result["policy_evaluation"]["stop"]["mean_loss"],
        )

        # Direct OOP CounterfactualVoIEvaluator + CandidateSensorPass check
        oop_voi = CounterfactualVoIEvaluator(
            wm,
            assimilator=assimilator,
            calibrator=cal,
            calibrator_group=("winter", "coast"),
        )
        typed_cands = [CandidateSensorPass.from_dict(c, i) for i, c in enumerate(candidates)]
        oop_res = oop_voi.evaluate(z_ctx, typed_cands, f_ctx, f_fut, target_future=target_fut)
        self.assertEqual(oop_res["recommended"]["sensor"], result["recommended"]["sensor"])


if __name__ == "__main__":
    unittest.main()
