from __future__ import annotations

import importlib.util
import math
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires the optional JEPA torch environment")

import torch

from openchange.jepa.guarantees import (
    StratifiedConformalAlertCalibrator,
    align_sfa_matryoshka_gauge,
    apply_orthogonal_procrustes,
    apply_subspace_procrustes,
    cross_unit_distance_ratio,
    evaluate_matryoshka_prefixes,
    evaluate_version_upgrade,
    fit_orthogonal_procrustes,
    fit_subspace_procrustes,
    latent_forecast_residuals,
    lejepa_identifiability_bound,
    lidar_score,
    rankme_score,
    score_observation_reveal_policies,
    selection_spearman,
)


class TestJEPAGuarantees(unittest.TestCase):
    def test_q1_rankme_and_lidar_scores(self) -> None:
        eye = torch.eye(8, dtype=torch.float64)
        self.assertAlmostEqual(rankme_score(eye), 8.0, places=5)

        rank_four = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        self.assertAlmostEqual(rankme_score(rank_four), 4.0, places=5)

        collapsed = torch.ones((16, 8), dtype=torch.float64)
        self.assertAlmostEqual(rankme_score(collapsed), 1.0, places=5)

        gen = torch.Generator().manual_seed(42)
        base_centers = torch.randn((32, 1, 6), generator=gen, dtype=torch.float64) * 3.0
        view_noise = torch.randn((32, 4, 6), generator=gen, dtype=torch.float64) * 0.1
        full_views = base_centers + view_noise

        rank2_centers = base_centers.clone()
        rank2_centers[:, :, 2:] = 0.0
        rank2_views = rank2_centers + view_noise
        collapsed_views = view_noise - view_noise.mean(dim=1, keepdim=True)

        full_lidar = lidar_score(full_views)
        rank2_lidar = lidar_score(rank2_views)
        collapsed_lidar = lidar_score(collapsed_views)

        self.assertGreater(full_lidar, 4.5)
        self.assertGreater(full_lidar, rank2_lidar + 1.5)
        self.assertGreater(rank2_lidar, 1.5)
        self.assertLess(rank2_lidar, 3.0)
        self.assertAlmostEqual(collapsed_lidar, 1.0, places=5)

        with self.assertRaises(ValueError):
            rankme_score(torch.ones(4))
        with self.assertRaises(ValueError):
            lidar_score(torch.ones((4, 4)))

    def test_q1_selection_spearman_and_regret(self) -> None:
        losses = [0.90, 0.75, 0.60, 0.45, 0.30]
        accuracies = [0.62, 0.68, 0.74, 0.81, 0.88]
        aligned = selection_spearman(losses, accuracies, higher_probe_is_better=True)
        self.assertAlmostEqual(aligned["spearman"], 1.0, places=12)
        self.assertAlmostEqual(aligned["raw_loss_spearman"], -1.0, places=12)
        self.assertAlmostEqual(aligned["top1_regret"], 0.0, places=12)
        self.assertEqual(aligned["selected_index"], 4)
        self.assertEqual(aligned["oracle_index"], 4)

        anti_accuracies = list(reversed(accuracies))
        anti = selection_spearman(losses, anti_accuracies, higher_probe_is_better=True)
        self.assertAlmostEqual(anti["spearman"], -1.0, places=12)
        self.assertAlmostEqual(anti["top1_regret"], 0.88 - 0.62, places=12)
        self.assertEqual(anti["selected_index"], 4)
        self.assertEqual(anti["oracle_index"], 0)

        rmse_probes = [0.50, 0.40, 0.30, 0.20, 0.10]
        lower_better = selection_spearman(losses, rmse_probes, higher_probe_is_better=False)
        self.assertAlmostEqual(lower_better["spearman"], 1.0, places=12)
        self.assertAlmostEqual(lower_better["top1_regret"], 0.0, places=12)

    def test_q2_orthogonal_procrustes_zero_reembedding_upgrade(self) -> None:
        gen = torch.Generator().manual_seed(7)
        dim = 12
        n_anchor = 64
        n_eval = 80

        q_true, _ = torch.linalg.qr(torch.randn((dim, dim), generator=gen, dtype=torch.float64))
        shift_old = torch.randn(dim, generator=gen, dtype=torch.float64) * 0.5
        shift_new = torch.randn(dim, generator=gen, dtype=torch.float64) * -0.7

        w_true = torch.randn((dim, 3), generator=gen, dtype=torch.float64)
        b_true = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)

        latent_anchor = torch.randn((n_anchor, dim), generator=gen, dtype=torch.float64)
        latent_eval = torch.randn((n_eval, dim), generator=gen, dtype=torch.float64)

        anchor_old = latent_anchor + shift_old
        anchor_new = latent_anchor @ q_true.T + shift_new

        eval_old = latent_eval + shift_old
        eval_new = latent_eval @ q_true.T + shift_new
        eval_labels = ((eval_old - shift_old) @ w_true + b_true).argmax(dim=-1)

        def frozen_old_head(features: torch.Tensor) -> torch.Tensor:
            return (features.to(torch.float64) - shift_old) @ w_true + b_true

        transform = fit_orthogonal_procrustes(anchor_source=anchor_new, anchor_target=anchor_old)
        self.assertLess(transform["orthogonality_error"], 1e-12)
        self.assertLess(transform["anchor_relative_residual"], 1e-12)

        recovered_eval = apply_orthogonal_procrustes(eval_new, transform)
        torch.testing.assert_close(recovered_eval, eval_old, atol=1e-10, rtol=1e-10)

        report = evaluate_version_upgrade(
            old_head_fn=frozen_old_head,
            anchor_old=anchor_old,
            anchor_new=anchor_new,
            eval_old=eval_old,
            eval_new=eval_new,
            eval_labels=eval_labels,
        )
        self.assertAlmostEqual(report["old_accuracy"], 1.0, places=12)
        self.assertAlmostEqual(report["aligned_accuracy"], 1.0, places=12)
        self.assertAlmostEqual(report["retained_accuracy_fraction"], 1.0, places=12)
        self.assertAlmostEqual(report["aligned_churn"], 0.0, places=12)
        self.assertLess(report["eval_relative_residual"], 1e-12)
        self.assertLess(report["unaligned_accuracy"], 0.65)
        self.assertGreater(report["unaligned_churn"], 0.35)

    def test_q3_q4_cross_unit_distance_ratio(self) -> None:
        gen = torch.Generator().manual_seed(19)
        base = torch.randn((48, 16), generator=gen, dtype=torch.float64)
        revisit_noise = torch.randn((48, 16), generator=gen, dtype=torch.float64) * 0.08
        swap_noise_calibrated = torch.randn((48, 16), generator=gen, dtype=torch.float64) * 0.08
        systematic_shift = torch.ones((48, 16), dtype=torch.float64) * 0.5

        unit_a = base
        same_revisit = base + revisit_noise
        unit_b_calibrated = base + swap_noise_calibrated
        unit_b_shifted = base + swap_noise_calibrated + systematic_shift

        calibrated = cross_unit_distance_ratio(unit_a, unit_b_calibrated, same_revisit)
        shifted = cross_unit_distance_ratio(unit_a, unit_b_shifted, same_revisit)

        self.assertGreater(calibrated["l2_ratio"], 0.8)
        self.assertLess(calibrated["l2_ratio"], 1.2)
        self.assertGreater(shifted["l2_ratio"], 5.0)
        self.assertGreater(shifted["cosine_ratio"], calibrated["cosine_ratio"] * 2.0)

        paired_revisit = torch.stack((unit_a, same_revisit), dim=1)
        paired_result = cross_unit_distance_ratio(unit_a, unit_b_calibrated, paired_revisit)
        self.assertAlmostEqual(paired_result["l2_ratio"], calibrated["l2_ratio"], places=12)

    def test_part2_stratified_conformal_alerts_and_s1_lead_time(self) -> None:
        gen = torch.Generator().manual_seed(101)
        dim = 8
        n_cal_per_group = 39
        alpha = 0.10

        cal_forecast = torch.zeros((2 * n_cal_per_group, dim), dtype=torch.float64)
        calm_noise = torch.randn((n_cal_per_group, dim), generator=gen, dtype=torch.float64) * 0.10
        storm_noise = torch.randn((n_cal_per_group, dim), generator=gen, dtype=torch.float64) * 0.50
        cal_actual = torch.cat((calm_noise, storm_noise), dim=0)
        cal_groups = [("summer", "inland")] * n_cal_per_group + [("winter", "coastal")] * n_cal_per_group

        calibrator = StratifiedConformalAlertCalibrator(alpha=alpha)
        thresholds = calibrator.fit(cal_actual, cal_forecast, cal_groups)
        self.assertGreater(thresholds[("winter", "coastal")], thresholds[("summer", "inland")] * 2.5)
        expected_level = math.ceil((n_cal_per_group + 1) * (1.0 - alpha)) / n_cal_per_group
        self.assertAlmostEqual(
            calibrator.group_effective_levels[("summer", "inland")],
            expected_level,
            places=12,
        )

        cal_residuals = latent_forecast_residuals(cal_actual, cal_forecast)
        calibrator_from_res = StratifiedConformalAlertCalibrator(alpha=alpha)
        thresholds_from_res = calibrator_from_res.fit(cal_residuals, None, cal_groups)
        self.assertAlmostEqual(
            thresholds_from_res[("winter", "coastal")],
            thresholds[("winter", "coastal")],
            places=12,
        )

        streams = 20
        steps = 6
        test_forecast = torch.zeros((streams, steps, dim), dtype=torch.float64)
        test_actual = torch.zeros((streams, steps, dim), dtype=torch.float64)
        test_groups: list[tuple[str, str]] = []
        onset_steps: list[int | None] = []
        reference_steps: list[int | None] = []
        s1_only_mask = torch.zeros((streams, steps), dtype=torch.bool)

        for s_idx in range(streams):
            is_winter = s_idx % 2 == 1
            group = ("winter", "coastal") if is_winter else ("summer", "inland")
            scale = 0.45 if is_winter else 0.09
            test_groups.append(group)
            test_actual[s_idx] = torch.randn((steps, dim), generator=gen, dtype=torch.float64) * scale
            s1_only_mask[s_idx, 2] = True
            s1_only_mask[s_idx, 3] = True
            if s_idx < 10:
                onset_steps.append(2)
                reference_steps.append(4)
                test_actual[s_idx, 2:] += 2.5
            else:
                onset_steps.append(None)
                reference_steps.append(None)

        evaluation = calibrator.evaluate_streams(
            test_actual,
            test_forecast,
            test_groups,
            onset_steps=onset_steps,
            reference_steps=reference_steps,
            s1_only_mask=s1_only_mask,
            action_cost=1.0,
            missed_loss=5.0,
        )
        self.assertLessEqual(evaluation["false_alarm_rate"], 0.15)
        self.assertAlmostEqual(evaluation["detection_rate"], 1.0, places=12)
        self.assertAlmostEqual(evaluation["mean_detection_lead_time"], 2.0, places=12)
        self.assertAlmostEqual(evaluation["mean_detection_delay"], 0.0, places=12)
        self.assertAlmostEqual(evaluation["s1_only_first_alert_fraction"], 1.0, places=12)
        self.assertGreater(evaluation["cost_loss_value"], 0.75)
        self.assertGreater(evaluation["net_cost_savings"], 0.0)

    def test_part2_observation_reveal_policies_under_budget(self) -> None:
        stop_losses = [0.05, 0.04, 0.90, 0.85, 0.95, 0.88]
        s1_losses = [0.04, 0.03, 0.20, 0.18, 0.35, 0.32]
        s2_losses = [0.03, 0.02, 0.82, 0.80, 0.08, 0.07]
        uncertainties = [0.02, 0.02, 0.85, 0.80, 0.90, 0.85]
        cloud_fractions = [0.10, 0.10, 0.95, 0.90, 0.05, 0.05]

        scored = score_observation_reveal_policies(
            stop_losses=stop_losses,
            s1_losses=s1_losses,
            s2_losses=s2_losses,
            uncertainties=uncertainties,
            cloud_fractions=cloud_fractions,
            cost_s1=0.2,
            cost_s2=1.0,
            budget_per_step=0.4,
            cost_weight=0.1,
            s1_gain_factor=0.7,
            s2_gain_factor=1.0,
        )

        for policy_name in ("stop", "always_s1", "always_s2", "uncertainty_voi"):
            self.assertIn(policy_name, scored)
            self.assertTrue(scored[policy_name]["within_budget"])

        voi = scored["uncertainty_voi"]
        self.assertEqual(voi["actions"][:2], ["stop", "stop"])
        self.assertEqual(voi["actions"][2:4], ["s1", "s1"])
        self.assertEqual(voi["actions"][4:6], ["s2", "s2"])
        self.assertLess(voi["mean_loss"], scored["stop"]["mean_loss"])
        self.assertLess(voi["mean_loss"], scored["always_s1"]["mean_loss"])
        self.assertLess(voi["mean_loss"], scored["always_s2"]["mean_loss"])
        self.assertGreater(voi["net_value"], scored["always_s1"]["net_value"])
        self.assertGreater(voi["net_value"], scored["always_s2"]["net_value"])

    def test_q2_subspace_procrustes_vs_full_procrustes_with_unaligned_private_blocks(self) -> None:
        gen = torch.Generator().manual_seed(301)
        n_samples = 96
        k_sh, k_s1, k_s2 = 8, 4, 4

        q_sh, _ = torch.linalg.qr(torch.randn((k_sh, k_sh), generator=gen, dtype=torch.float64))
        z_sh = torch.randn((n_samples, k_sh), generator=gen, dtype=torch.float64)
        # Private dimensions are independent nuisance/sensor-specific noise across versions
        priv_src = torch.randn((n_samples, k_s1 + k_s2), generator=gen, dtype=torch.float64)
        priv_tgt = torch.randn((n_samples, k_s1 + k_s2), generator=gen, dtype=torch.float64)

        anchor_src = torch.cat((z_sh @ q_sh.T, priv_src), dim=1)
        anchor_tgt = torch.cat((z_sh, priv_tgt), dim=1)

        sub_fit = fit_subspace_procrustes(
            anchor_src,
            anchor_tgt,
            shared_dim=k_sh,
            s1_private_dim=k_s1,
            s2_private_dim=k_s2,
        )
        self.assertLess(sub_fit["shared_relative_residual"], 1e-12)
        self.assertGreater(sub_fit["full_relative_residual"], 0.35)
        self.assertAlmostEqual(sub_fit["theoretical_full_floor"], 2.0 * 8.0 / 16.0, places=12)

        aligned_s1_only = apply_subspace_procrustes(
            anchor_src, sub_fit, zero_inactive_private="s2"
        )
        torch.testing.assert_close(aligned_s1_only[:, :k_sh], z_sh, atol=1e-10, rtol=1e-10)
        self.assertEqual(float(aligned_s1_only[:, k_sh + k_s1 :].abs().max()), 0.0)

    def test_matryoshka_prefix_evaluation(self) -> None:
        gen = torch.Generator().manual_seed(302)
        dim = 16
        prototypes = torch.randn((4, dim), generator=gen, dtype=torch.float64) * 3.0
        train_labels = torch.arange(4, dtype=torch.int64).repeat_interleave(20)
        eval_labels = torch.arange(4, dtype=torch.int64).repeat_interleave(15)
        train_x = prototypes[train_labels] + 0.2 * torch.randn((80, dim), generator=gen, dtype=torch.float64)
        eval_x = prototypes[eval_labels] + 0.2 * torch.randn((60, dim), generator=gen, dtype=torch.float64)

        report = evaluate_matryoshka_prefixes(
            train_x, train_labels, eval_x, eval_labels, prefixes=(4, 8, 16)
        )
        self.assertEqual(report["prefixes"], [4, 8, 16])
        self.assertGreaterEqual(report["full_accuracy"], 0.95)
        self.assertGreaterEqual(report["prefix_retention"][8], 0.85)
        self.assertIn("first_vs_random_gap", report)
        self.assertIn("group_std_staircase", report)

        # SFA gauge fixing orders coordinates by cross-view persistence without distorting unit variance
        sfa = align_sfa_matryoshka_gauge(train_x, train_x * 0.9 + 0.1 * torch.randn_like(train_x))
        self.assertEqual(sfa["rotation"].shape, (dim, dim))
        self.assertTrue(bool((sfa["eigenvalues"][:-1] >= sfa["eigenvalues"][1:] - 1e-9).all()))

    def test_klindt_lejepa_approximate_identifiability_bound(self) -> None:
        gen = torch.Generator().manual_seed(303)
        n_samples, dim = 256, 6
        rho = 0.9
        z_true = torch.randn((n_samples, dim), generator=gen, dtype=torch.float64)
        eta = torch.randn((n_samples, dim), generator=gen, dtype=torch.float64)
        z_next = rho * z_true + math.sqrt(1.0 - rho * rho) * eta

        q_orth, _ = torch.linalg.qr(torch.randn((dim, dim), generator=gen, dtype=torch.float64))
        h_a = z_true @ q_orth.T
        h_b = z_next @ q_orth.T

        bound_linear = lejepa_identifiability_bound(h_a, h_b, true_latents=z_true, rho=rho)
        self.assertTrue(bound_linear["bound_satisfied"])
        self.assertGreater(bound_linear["r2_h_to_z"], 0.999)
        self.assertLess(bound_linear["empirical_orthogonal_error"], 1e-10)

        # Nonlinearly distorted representation increases alignment gap and decreases R^2
        h_a_nonlin = h_a + 0.35 * (h_a.square() - 1.0)
        h_b_nonlin = h_b + 0.35 * (h_b.square() - 1.0)
        bound_nonlin = lejepa_identifiability_bound(h_a_nonlin, h_b_nonlin, true_latents=z_true, rho=rho)
        self.assertTrue(bound_nonlin["bound_satisfied"])
        self.assertLess(bound_nonlin["r2_h_to_z"], bound_linear["r2_h_to_z"])
        self.assertGreater(
            bound_nonlin["theoretical_recovery_bound"],
            bound_linear["theoretical_recovery_bound"],
        )


if __name__ == "__main__":
    unittest.main()
