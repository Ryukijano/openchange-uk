from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("clean-JEPA tests require the optional torch dependency")

import torch
import torch.nn.functional as functional

from openchange.jepa.losses import ep_per_view_stratum, sigreg_directions
from openchange.jepa.wam import (
    HISTORY_STRATA_SECONDS,
    CausalPatchJEPAWorldModel,
    EndToEndPixelWAM,
    FlowMatchingWAMHead,
    HistoryStratifiedSIGReg,
    LeRobotWindowDataset,
    LeWMPixelPatchEncoder,
    action_contrastive_loss,
    bidirectional_neighborhood_mask,
    causal_neighborhood_mask,
    compute_reachable_error_and_factorial,
    diffusion_steered_plan,
    robometer_preference_loss,
)

torch.set_num_threads(min(torch.get_num_threads(), 4))


def _model(**overrides) -> CausalPatchJEPAWorldModel:
    kwargs = dict(
        dim=8,
        grid_size=2,
        action_dim=3,
        proprio_dim=2,
        horizon=4,
        heads=2,
        depth=2,
        mlp_ratio=2,
        spatial_radius=1,
        max_timesteps=8,
        seed=0,
    )
    kwargs.update(overrides)
    return CausalPatchJEPAWorldModel(**kwargs)


class CausalMaskTests(unittest.TestCase):
    def test_spatiotemporal_allow_block_pattern(self) -> None:
        grid, steps, radius = 2, 3, 1
        mask = causal_neighborhood_mask(grid, steps, radius)
        n_tok = grid * grid + 1
        self.assertEqual(mask.shape, (steps * n_tok, steps * n_tok))

        def coords(u: int) -> tuple[int, int]:
            return u // grid, u % grid

        for t in range(steps):
            for tp in range(steps):
                for qi in range(n_tok):
                    for kj in range(n_tok):
                        allowed = mask[t * n_tok + qi, tp * n_tok + kj].item() == 0.0
                        if tp > t:
                            self.assertFalse(allowed, "future keys must be blocked")
                            continue
                        if qi == 0 or kj == 0:
                            self.assertTrue(allowed, "CLS visible to all same/past tokens")
                            continue
                        dr = abs(coords(qi - 1)[0] - coords(kj - 1)[0])
                        dc = abs(coords(qi - 1)[1] - coords(kj - 1)[1])
                        self.assertEqual(allowed, max(dr, dc) <= radius)

    def test_no_cls_and_validation(self) -> None:
        mask = causal_neighborhood_mask(2, 2, 0, include_cls=False)
        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(mask[0, 4].item(), float("-inf"))
        self.assertEqual(mask[4, 0].item(), 0.0)
        self.assertEqual(mask[4, 1].item(), float("-inf"))
        with self.assertRaises(ValueError):
            causal_neighborhood_mask(0, 2, 1)
        with self.assertRaises(ValueError):
            causal_neighborhood_mask(2, 0, 1)
        with self.assertRaises(ValueError):
            causal_neighborhood_mask(2, 2, -1)

    def test_zero_future_gradient_leakage(self) -> None:
        torch.manual_seed(0)
        grid, steps = 2, 3
        n_tok, dim = grid * grid + 1, 4
        length = steps * n_tok
        boundary = n_tok
        mask = causal_neighborhood_mask(grid, steps, 1)
        q = torch.randn(1, 2, length, dim, requires_grad=True)
        k = torch.randn(1, 2, length, dim, requires_grad=True)
        v = torch.randn(1, 2, length, dim, requires_grad=True)
        out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out[:, :, :boundary].sum().backward()
        self.assertTrue(torch.equal(k.grad[:, :, boundary:], torch.zeros_like(k.grad[:, :, boundary:])))
        self.assertTrue(torch.equal(v.grad[:, :, boundary:], torch.zeros_like(v.grad[:, :, boundary:])))
        self.assertGreater(float(q.grad[:, :, :boundary].abs().sum()), 0.0)

    def test_output_equals_truncated_past_attention(self) -> None:
        torch.manual_seed(1)
        grid, steps = 2, 3
        n_tok, dim = grid * grid + 1, 4
        length = steps * n_tok
        mask = causal_neighborhood_mask(grid, steps, 1)
        q, k, v = (torch.randn(1, 1, length, dim) for _ in range(3))
        full = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        cutoff = 2 * n_tok
        truncated = functional.scaled_dot_product_attention(
            q[:, :, :cutoff],
            k[:, :, :cutoff],
            v[:, :, :cutoff],
            attn_mask=mask[:cutoff, :cutoff],
        )
        torch.testing.assert_close(full[:, :, :cutoff], truncated, atol=1e-6, rtol=1e-6)


class HistoryStratifiedSIGRegTests(unittest.TestCase):
    def test_stratum_assignment(self) -> None:
        reg = HistoryStratifiedSIGReg()
        history = torch.tensor([0.0, 1.0, 2.4, 5.0, 9.6, 13.0, 19.2, 30.0, 38.4, 100.0])
        strata = reg.strata_index(history)
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4, 4, 4], dtype=torch.int64)
        self.assertTrue(torch.equal(strata, expected))
        with self.assertRaises(ValueError):
            reg.strata_index(torch.tensor([-1.0]))

    def test_matches_ep_per_view_stratum_fp32_and_fp64(self) -> None:
        torch.manual_seed(2)
        for dtype in (torch.float32, torch.float64):
            z = torch.randn(50, 6, dtype=dtype)
            history = torch.tensor([0.0, 2.4, 9.6, 19.2, 38.4] * 10)
            directions = sigreg_directions(6, 16, seed=7, step=0, device="cpu", dtype=dtype)
            reg = HistoryStratifiedSIGReg()
            out = reg(z, history, directions)
            strata = reg.strata_index(history)
            ep, counts = ep_per_view_stratum(z.unsqueeze(0), strata, 5, directions)
            torch.testing.assert_close(
                out["ep"][0], ep[0], atol=1e-6 if dtype == torch.float32 else 1e-12, rtol=0
            )
            torch.testing.assert_close(out["counts"], counts, atol=0, rtol=0)
            torch.testing.assert_close(out["sigreg"], ep[0].mean(), atol=1e-6, rtol=0)
            self.assertEqual(out["sigreg"].dtype, dtype)

    def test_fp32_exactness_under_autocast(self) -> None:
        torch.manual_seed(3)
        z = torch.randn(40, 8, dtype=torch.float32)
        history = torch.tensor([0.0, 2.4, 9.6, 19.2, 38.4] * 8)
        directions = sigreg_directions(8, 12, seed=7, step=0, device="cpu", dtype=torch.float32)
        reg = HistoryStratifiedSIGReg()
        expected = reg(z, history, directions)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = reg(z, history, directions)
        self.assertEqual(actual["sigreg"].dtype, torch.float32)
        torch.testing.assert_close(actual["sigreg"], expected["sigreg"], atol=0, rtol=0)

    def test_empty_stratum_raises(self) -> None:
        reg = HistoryStratifiedSIGReg()
        z = torch.randn(10, 4, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            reg(z, torch.zeros(10), sigreg_directions(4, 4, seed=1, step=0, device="cpu", dtype=torch.float64))

    def test_gaussian_inputs_near_null(self) -> None:
        torch.manual_seed(4)
        z = torch.randn(500, 16, dtype=torch.float64)
        history = torch.tensor(list(HISTORY_STRATA_SECONDS) * 100)
        directions = sigreg_directions(16, 64, seed=7, step=0, device="cpu", dtype=torch.float64)
        out = HistoryStratifiedSIGReg()(z, history, directions)
        self.assertLess(abs(float(out["ep"].mean()) - float(out["null"])), 0.5)


class WorldModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.model = _model().double()
        self.z_context = torch.randn(2, 3, 4, 8, dtype=torch.float64)
        self.proprio = torch.randn(2, 3, 2, dtype=torch.float64)
        self.actions = torch.randn(2, 4, 3, dtype=torch.float64)

    def test_forward_shape_and_causal_consistency(self) -> None:
        out = self.model(self.z_context, self.actions, self.proprio)
        self.assertEqual(out.z_hat.shape, (2, 4, 4, 8))
        short = self.model(self.z_context[:, :2], self.actions, self.proprio[:, :2])
        self.assertFalse(torch.allclose(short.z_hat, out.z_hat))

    def test_film_action_responsiveness(self) -> None:
        base = self.model(self.z_context, self.actions, self.proprio).z_hat
        other_actions = torch.randn(2, 4, 3, dtype=torch.float64)
        other = self.model(self.z_context, other_actions, self.proprio).z_hat
        self.assertGreater(float((base - other).detach().abs().max()), 1e-6)
        zeros = self.model(self.z_context, torch.zeros_like(self.actions), self.proprio).z_hat
        self.assertGreater(float((base - zeros).detach().abs().max()), 1e-6)
        repeated = self.model(self.z_context, self.actions, self.proprio).z_hat
        self.assertTrue(torch.equal(base, repeated))
        with self.assertRaises(ValueError):
            self.model(self.z_context, torch.randn(2, 99, 3, dtype=torch.float64), self.proprio)

    def test_stop_grad_vs_live_target_modes(self) -> None:
        history = torch.tensor([0.0, 38.4])
        directions = sigreg_directions(8, 8, seed=7, step=0, device="cpu", dtype=torch.float64)
        reg2 = HistoryStratifiedSIGReg((0.0, 38.4))
        kwargs = dict(
            history_seconds=history,
            directions=directions,
            sigreg=reg2,
            lam=0.1,
        )
        z_future_sg = torch.randn(2, 4, 4, 8, dtype=torch.float64, requires_grad=True)
        z_future_lv = z_future_sg.detach().clone().requires_grad_()
        loss_sg = self.model.wam_loss(
            self.z_context, self.actions, self.proprio, z_future_sg, target_mode="stop_grad", **kwargs
        )
        loss_lv = self.model.wam_loss(
            self.z_context, self.actions, self.proprio, z_future_lv, target_mode="live", **kwargs
        )
        torch.testing.assert_close(loss_sg["loss"], loss_lv["loss"], atol=1e-12, rtol=0)
        loss_sg["prediction"].backward(retain_graph=True)
        loss_lv["prediction"].backward()
        self.assertIsNone(z_future_sg.grad)
        self.assertGreater(float(z_future_lv.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            self.model.wam_loss(
                self.z_context,
                self.actions,
                self.proprio,
                z_future_sg.detach(),
                target_mode="bogus",
                **kwargs,
            )

    def test_reachable_error_and_oracle_factorial(self) -> None:
        goal = torch.tensor([[1.0], [1.0]])
        rollouts = {
            "open_pred": torch.zeros(2, 3, 1),
            "open_true": torch.zeros(2, 3, 1),
            "plan_pred": torch.ones(2, 3, 1),
            "plan_true": torch.full((2, 3, 1), -2.0),
            "oracle_pred": torch.zeros(2, 3, 1),
        }
        out = compute_reachable_error_and_factorial(rollouts, goal=goal)
        self.assertEqual(float(out["E_data"]), 0.0)
        self.assertGreater(float(out["E_reach"]), 0.0)
        self.assertGreater(float(out["E_rel"]), 0.0)
        self.assertGreater(float(out["separation"]), 0.0)
        factorial = out["factorial"]
        self.assertEqual(float(factorial["S(learned,latent)"]), 0.0)
        self.assertGreater(float(factorial["S(learned,true)"]), 0.0)
        self.assertEqual(float(factorial["S(true,latent)"]), 0.0)
        self.assertEqual(float(factorial["S(true,true)"]), 0.0)
        with self.assertRaises(ValueError):
            compute_reachable_error_and_factorial({"open_pred": torch.zeros(1)})

    def test_diffusion_steered_plan_stays_on_manifold(self) -> None:
        goal = torch.full((2,), 10.0)
        z0 = torch.zeros(2)

        def dynamics(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            return z + a

        def denoise(noise: torch.Tensor) -> torch.Tensor:
            return 0.5 * torch.tanh(noise)

        cem_actions, cem_info = diffusion_steered_plan(
            dynamics,
            z0,
            goal,
            horizon=3,
            action_dim=2,
            n_samples=32,
            n_iters=4,
            method="cem",
            action_bound=5.0,
            seed=0,
        )
        diff_actions, diff_info = diffusion_steered_plan(
            dynamics,
            z0,
            goal,
            horizon=3,
            action_dim=2,
            n_samples=32,
            n_iters=4,
            method="diffusion",
            denoise=denoise,
            action_bound=5.0,
            seed=0,
        )
        self.assertLessEqual(float(diff_actions.abs().max()), 0.5 + 1e-6)
        self.assertGreater(float(cem_actions.abs().max()), 0.5)
        self.assertGreater(float(diff_info["best_cost"]), float(cem_info["best_cost"]))
        again, _ = diffusion_steered_plan(
            dynamics,
            z0,
            goal,
            horizon=3,
            action_dim=2,
            n_samples=32,
            n_iters=4,
            method="diffusion",
            denoise=denoise,
            action_bound=5.0,
            seed=0,
        )
        self.assertTrue(torch.equal(diff_actions, again))
        with self.assertRaises(ValueError):
            diffusion_steered_plan(dynamics, z0, goal, horizon=1, action_dim=1, method="nope")

    def test_causal_vs_bidirectional_ablation_arm(self) -> None:
        bi_mask = bidirectional_neighborhood_mask(2, 3, 1)
        n_tok = 2 * 2 + 1
        self.assertEqual(bi_mask[1, 2 * n_tok + 1].item(), 0.0)
        causal_wm = _model(causal=True).double()
        bi_wm = _model(causal=False).double()
        out_c = causal_wm(self.z_context, self.actions, self.proprio)
        out_b = bi_wm(self.z_context, self.actions, self.proprio)
        self.assertEqual(out_c.z_hat.shape, out_b.z_hat.shape)
        self.assertFalse(torch.allclose(out_c.context, out_b.context))

    def test_pixel_encoder_end_to_end_wam_and_lerobot_dataset(self) -> None:
        torch.manual_seed(11)
        enc = LeWMPixelPatchEncoder(
            in_channels=3, image_size=32, patch_size=16, dim=8, heads=2, depth=1, mlp_ratio=2
        )
        wm = _model(dim=8, grid_size=2, action_dim=3, proprio_dim=2, horizon=2, max_timesteps=8)
        e2e = EndToEndPixelWAM(encoder=enc, world_model=wm, sigreg=HistoryStratifiedSIGReg((0.0, 2.4)))

        episodes = [
            {
                "frames": torch.randint(0, 255, (16, 3, 32, 32), dtype=torch.uint8),
                "actions": torch.randn(16, 3),
                "proprio": torch.randn(16, 2),
                "fps": 10.0,
            },
            {
                "frames": torch.randint(0, 255, (20, 3, 32, 32), dtype=torch.uint8),
                "actions": torch.randn(20, 3),
                "proprio": torch.randn(20, 2),
                "fps": 10.0,
            },
        ]
        ds = LeRobotWindowDataset(
            episodes, context_frames=3, horizon_frames=2, strata_seconds=(0.0, 2.4)
        )
        self.assertEqual(len(ds), 4)
        batch_items = [ds[i] for i in range(4)]
        f_ctx = torch.stack([b["frames_context"] for b in batch_items])
        f_fut = torch.stack([b["frames_future"] for b in batch_items])
        prop = torch.stack([b["proprio"] for b in batch_items])
        act = torch.stack([b["actions"] for b in batch_items])
        hist = torch.stack([b["history_seconds"] for b in batch_items])
        prog = torch.stack([b["target_progress"] for b in batch_items])

        dirs = sigreg_directions(8, 12, seed=3, step=0, device="cpu", dtype=torch.float32)
        losses = e2e.pixel_wam_loss(
            f_ctx, act, prop, f_fut, history_seconds=hist, directions=dirs, target_progress=prog
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        stem_grad = float(enc.stem.weight.grad.abs().sum())
        self.assertGreater(stem_grad, 0.0)

    def test_unconfounded_anchor_zero_history_and_action_alignment(self) -> None:
        # 20 frames at 20 fps -> max span is 17/20 = 0.85s
        ep = {
            "frames": torch.arange(20 * 3 * 16 * 16, dtype=torch.float32).reshape(20, 3, 16, 16) / 1000.0,
            "actions": torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2),
            "proprio": torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2),
            "fps": 20.0,
        }
        ds = LeRobotWindowDataset(
            [ep],
            context_frames=4,
            horizon_frames=3,
            strata_seconds=(0.0, 0.4, 38.4),
        )
        s0, s1, s2 = ds[0], ds[1], ds[2]
        # 1. Fixed prediction anchor across all history strata (unconfounded H intervention)
        self.assertEqual(int(s0["anchor_index"]), 17)
        self.assertEqual(int(s1["anchor_index"]), 17)
        self.assertEqual(int(s2["anchor_index"]), 17)
        self.assertTrue(torch.equal(s0["frames_future"], s1["frames_future"]))
        self.assertTrue(torch.equal(s1["frames_future"], s2["frames_future"]))

        # 2. H = 0.0 genuinely repeats only anchor - 1 (realized_hist_s == 0.0)
        self.assertAlmostEqual(float(s0["history_seconds"]), 0.0, places=6)
        self.assertFalse(bool(s0["history_truncated"]))
        self.assertTrue(torch.equal(s0["frames_context"][0], s0["frames_context"][-1]))

        # 3. Truncated history reports true realized_hist_s (0.80s), not 38.4s
        self.assertAlmostEqual(float(s2["history_seconds"]), 16.0 / 20.0, places=5)
        self.assertAlmostEqual(float(s2["requested_history_seconds"]), 38.4, places=4)
        self.assertTrue(bool(s2["history_truncated"]))

        # 4. Actions align with transitions from anchor - 1 (index 16, 17, 18)
        expected_actions = ep["actions"][16:19]
        torch.testing.assert_close(s0["actions"], expected_actions, atol=0, rtol=0)

    def test_robometer_preference_and_failure_margin_loss(self) -> None:
        target = torch.tensor([[0.25, 0.50, 0.75, 1.00], [0.25, 0.50, 0.75, 1.00]], dtype=torch.float64)
        monotone_pred = torch.tensor([[0.20, 0.45, 0.70, 0.95], [0.10, 0.15, 0.12, 0.10]], dtype=torch.float64)
        reversed_pred = torch.tensor([[0.95, 0.70, 0.45, 0.20], [0.90, 0.92, 0.95, 0.98]], dtype=torch.float64)
        failed = torch.tensor([False, True])

        good = robometer_preference_loss(monotone_pred, target, failed_mask=failed, margin=0.1)
        bad = robometer_preference_loss(reversed_pred, target, failed_mask=failed, margin=0.1)
        self.assertLess(float(good["ranking_loss"]), float(bad["ranking_loss"]))
        self.assertAlmostEqual(float(good["failure_margin_loss"]), 0.0, places=12)
        self.assertGreater(float(bad["failure_margin_loss"]), 0.5)

    def test_action_contrastive_loss_and_flow_matching_wam_head(self) -> None:
        torch.manual_seed(305)
        out_ac = action_contrastive_loss(
            self.model, self.z_context, self.actions, self.proprio,
            self.model(self.z_context, self.actions, self.proprio).z_hat,
            temperature=0.05,
        )
        self.assertTrue(torch.isfinite(out_ac["loss"]))
        self.assertGreater(float(out_ac["action_elasticity"].detach()), 0.0)
        self.assertAlmostEqual(float(out_ac["action_identification_accuracy"]), 1.0, places=6)

        # Frame-mean and context+pred SIGReg pooling modes
        directions = sigreg_directions(8, 8, seed=7, step=0, device="cpu", dtype=torch.float64)
        reg2 = HistoryStratifiedSIGReg((0.0, 38.4))
        for pool_mode in ("frame_mean", "context_and_pred"):
            res = self.model.wam_loss(
                self.z_context,
                self.actions,
                self.proprio,
                torch.randn(2, 4, 4, 8, dtype=torch.float64),
                history_seconds=torch.tensor([0.0, 38.4]),
                directions=directions,
                sigreg=reg2,
                sigreg_pool=pool_mode,
            )
            self.assertTrue(torch.isfinite(res["loss"]))
            self.assertGreaterEqual(float(res["effective_rank"]), 1.0)

        # FlowMatchingWAMHead dynamics + action generation
        cfm_head = FlowMatchingWAMHead(dim=8, action_dim=3, horizon=4, hidden_dim=32, time_dim=8, seed=11).double()
        ctx_summary = self.z_context.mean(dim=(1, 2))
        fut_summary = torch.randn(2, 4, 8, dtype=torch.float64)
        cfm_losses = cfm_head.cfm_loss(ctx_summary, self.actions, fut_summary)
        self.assertTrue(torch.isfinite(cfm_losses["loss"]))
        cfm_losses["loss"].backward()

        sampled_fut = cfm_head.sample_future(ctx_summary, self.actions, steps=8, solver="heun", seed=3)
        sampled_act = cfm_head.sample_actions(ctx_summary, fut_summary[:, -1, :], steps=8, solver="heun", seed=3)
        self.assertEqual(sampled_fut.shape, (2, 4, 8))
        self.assertEqual(sampled_act.shape, (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
