import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("clean-JEPA tests require the optional torch dependency")

import math

import numpy as np
import torch

from openchange.jepa.losses import (
    direction_seed,
    ep_null_expectation,
    ep_per_view_stratum,
    invariance_loss,
    jepa_loss,
    mean_penalty,
    quadrature,
    sigreg_directions,
)

torch.set_num_threads(min(torch.get_num_threads(), 4))


def _directions(features: int, count: int, step: int = 0) -> torch.Tensor:
    return sigreg_directions(
        features,
        count,
        seed=7,
        step=step,
        device="cpu",
        dtype=torch.float64,
    )


def _ep(values: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(values.shape[0], dtype=torch.int64)
    ep, _ = ep_per_view_stratum(values.unsqueeze(0), labels, 1, directions)
    return ep[0, 0]


class QuadratureTests(unittest.TestCase):
    def test_quadrature_constants_and_null(self) -> None:
        t, phi, q = quadrature(torch.float64, "cpu")
        self.assertEqual(t.shape, (17,))
        self.assertEqual(phi.shape, (17,))
        self.assertEqual(q.shape, (17,))
        self.assertAlmostEqual(q[0].item(), 3 / 16, places=14)
        self.assertAlmostEqual(q.sum().item(), 2.4996663000824, delta=1e-12)
        self.assertAlmostEqual(ep_null_expectation(), 1.052464154799, delta=1e-12)


class InvarianceTests(unittest.TestCase):
    def test_gradient_formula_gradcheck_and_translation_invariance(self) -> None:
        for n_global, n_local in ((2, 6), (1, 3), (3, 5), (2, 0)):
            with self.subTest(n_global=n_global, n_local=n_local):
                generator = torch.Generator().manual_seed(10 + n_global + n_local)
                z = torch.randn(
                    (n_global + n_local, 5, 7),
                    generator=generator,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                loss = invariance_loss(z, n_global)
                gradient = torch.autograd.grad(loss, z)[0]
                center = z[:n_global].mean(dim=0)
                expected = 2 * (z - center) / (5 * z.shape[0] * 7)
                if n_local:
                    local_center = z[n_global:].mean(dim=0)
                    expected[:n_global] += (
                        2
                        * (n_local / n_global)
                        * (center - local_center)
                        / (5 * z.shape[0] * 7)
                    )
                torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
                self.assertTrue(
                    torch.autograd.gradcheck(
                        lambda value: invariance_loss(value, n_global),
                        (z.detach().requires_grad_(),),
                        eps=1e-6,
                        atol=1e-7,
                        rtol=1e-5,
                    )
                )
                torch.testing.assert_close(
                    gradient.sum(dim=0),
                    torch.zeros_like(gradient.sum(dim=0)),
                    atol=1e-12,
                    rtol=0,
                )
                if n_local:
                    self.assertGreater(float(gradient[:n_global].sum().abs()), 1e-3)


class EppsPulleyTests(unittest.TestCase):
    def test_ep_matches_numpy_complex_characteristic_function(self) -> None:
        generator = torch.Generator().manual_seed(100)
        values = 1.3 * torch.randn((37, 16), generator=generator, dtype=torch.float64) + 0.2
        directions = _directions(16, 64)
        t, phi, q = (item.numpy() for item in quadrature(torch.float64, "cpu"))
        projected = values.numpy() @ directions.numpy()
        phase = projected[:, :, None] * t[None, None, :]
        characteristic = np.exp(1j * phase).mean(axis=0)
        discrepancy = (
            (characteristic.real - phi[None, :]) ** 2 + characteristic.imag**2
        ) @ q
        expected = values.shape[0] * discrepancy.sum() / directions.shape[1]
        self.assertAlmostEqual(_ep(values, directions).item(), float(expected), delta=1e-12)

    def test_ep_gradient_formula_and_gradcheck(self) -> None:
        generator = torch.Generator().manual_seed(101)
        values = torch.randn((6, 4), generator=generator, dtype=torch.float64)
        directions = _directions(4, 8)
        t, phi, q = quadrature(torch.float64, "cpu")
        labels = torch.zeros(values.shape[0], dtype=torch.int64)
        z = values.unsqueeze(0).requires_grad_()
        ep, _ = ep_per_view_stratum(z, labels, 1, directions)
        actual = torch.autograd.grad(ep.sum(), z)[0][0]
        expected = torch.zeros_like(values)
        for index in range(directions.shape[1]):
            direction = directions[:, index]
            phase = (values @ direction)[:, None] * t[None, :]
            cosine = torch.cos(phase)
            sine = torch.sin(phase)
            cosine_mean = cosine.mean(dim=0)
            sine_mean = sine.mean(dim=0)
            per_sample = (
                2
                * (
                    q[None, :]
                    * t[None, :]
                    * (
                        sine_mean[None, :] * cosine
                        - (cosine_mean - phi)[None, :] * sine
                    )
                ).sum(dim=1)
                / directions.shape[1]
            )
            expected += per_sample[:, None] * direction[None, :]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: _ep(value, directions),
                (values.detach().requires_grad_(),),
                eps=1e-6,
                atol=1e-7,
                rtol=1e-5,
            )
        )

    def test_view_averaging_and_stratified_weighting(self) -> None:
        generator = torch.Generator().manual_seed(102)
        z = torch.randn((4, 12, 8), generator=generator, dtype=torch.float64)
        labels = torch.tensor([0] * 4 + [1] * 3 + [2] * 5, dtype=torch.int64)
        directions = _directions(8, 20)
        pooled = jepa_loss(
            z,
            labels,
            n_global=2,
            n_strata=1,
            weights=[1.0],
            lam=0.2,
            lam_mu=0,
            directions=directions,
            mode="pooled",
        )
        single_views = torch.stack(
            [_ep(z[view], directions) for view in range(z.shape[0])]
        ).mean()
        torch.testing.assert_close(pooled["sigreg"], single_views, atol=1e-12, rtol=1e-12)

        weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        stratified = jepa_loss(
            z,
            labels,
            n_global=2,
            n_strata=3,
            weights=weights,
            lam=0.2,
            lam_mu=0,
            directions=directions,
            mode="stratified",
        )
        separate = []
        for stratum in range(3):
            selected = z[:, labels == stratum]
            separate.append(
                torch.stack(
                    [_ep(selected[view], directions) for view in range(z.shape[0])]
                ).mean()
            )
        expected = (weights * torch.stack(separate)).sum()
        torch.testing.assert_close(stratified["sigreg"], expected, atol=1e-12, rtol=1e-12)

    def test_ep_counts_weights_and_mean_penalty_are_applied_once(self) -> None:
        generator = torch.Generator().manual_seed(103)
        values = torch.randn((1, 8, 4), generator=generator, dtype=torch.float64)
        directions = _directions(4, 16)
        base, _ = ep_per_view_stratum(
            values,
            torch.zeros(8, dtype=torch.int64),
            1,
            directions,
        )
        repeated, _ = ep_per_view_stratum(
            torch.cat((values, values), dim=1),
            torch.zeros(16, dtype=torch.int64),
            1,
            directions,
        )
        torch.testing.assert_close(repeated / base, torch.full_like(base, 2), atol=1e-12, rtol=1e-12)

        strata = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int64)
        result = jepa_loss(
            values,
            strata,
            n_global=1,
            n_strata=2,
            weights=[1.0, 0.0],
            lam=1.0,
            lam_mu=0.5,
            directions=directions,
            mode="stratified",
        )
        first_stratum = _ep(values[0, :4], directions)
        torch.testing.assert_close(result["sigreg"], first_stratum, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            result["ep_over_n"] * result["counts"][None, :],
            result["ep"],
            atol=1e-12,
            rtol=1e-12,
        )
        expected_t_mu = 4 * values[0, :4].mean(dim=0).square().sum() / values.shape[-1]
        torch.testing.assert_close(result["t_mu"], expected_t_mu, atol=1e-12, rtol=1e-12)

    def test_standard_normal_sampling_null_with_four_se(self) -> None:
        repeats, samples, features, direction_count = 100, 32, 64, 128
        generator = torch.Generator().manual_seed(104)
        values = []
        for step in range(repeats):
            z = torch.randn((samples, features), generator=generator, dtype=torch.float64)
            values.append(_ep(z, _directions(features, direction_count, step)).item())
        measurements = torch.tensor(values, dtype=torch.float64)
        standard_error = measurements.std(unbiased=True) / math.sqrt(repeats)
        self.assertLess(
            abs(measurements.mean().item() - ep_null_expectation()),
            4 * standard_error.item(),
        )

    def test_collapsed_embedding_has_expected_value_and_zero_gradient(self) -> None:
        features, samples = 8, 256
        z = torch.zeros((samples, features), dtype=torch.float64, requires_grad=True)
        directions = _directions(features, 16)
        collapsed = _ep(z, directions)
        _, phi, q = quadrature(torch.float64, "cpu")
        expected = samples * (q * (1 - phi).square()).sum()
        torch.testing.assert_close(collapsed, expected, atol=1e-12, rtol=1e-12)
        gradient = torch.autograd.grad(collapsed, z)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(z), atol=0, rtol=0)
        generator = torch.Generator().manual_seed(105)
        nearby = 0.1 * torch.randn((samples, features), generator=generator, dtype=torch.float64)
        self.assertLess(_ep(nearby, directions).item(), collapsed.item())

    def test_isotropic_scale_matches_analytic_expectation_with_four_se(self) -> None:
        repeats, samples, features, direction_count = 10, 256, 64, 128
        generator = torch.Generator().manual_seed(106)
        t, phi, q = quadrature(torch.float64, "cpu")
        for sigma in (0.5, 0.9, 1.1, 2.0):
            measurements = []
            for step in range(repeats):
                z = sigma * torch.randn(
                    (samples, features),
                    generator=generator,
                    dtype=torch.float64,
                )
                measurements.append(
                    _ep(z, _directions(features, direction_count, step)).item()
                )
            observed = torch.tensor(measurements, dtype=torch.float64)
            standard_error = observed.std(unbiased=True) / math.sqrt(repeats)
            expected = samples * (
                q * (torch.exp(-sigma**2 * t.square() / 2) - phi).square()
            ).sum() + (q * (1 - torch.exp(-sigma**2 * t.square()))).sum()
            self.assertLess(
                abs(observed.mean().item() - expected.item()),
                4 * standard_error.item(),
                msg=f"sigma={sigma}",
            )

    def test_rotation_invariance(self) -> None:
        generator = torch.Generator().manual_seed(107)
        values = torch.randn((48, 9), generator=generator, dtype=torch.float64)
        raw = torch.randn((9, 9), generator=generator, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(raw)
        directions = _directions(9, 32)
        rotated_values = _ep(values @ rotation, directions)
        rotated_directions = _ep(values, rotation @ directions)
        torch.testing.assert_close(rotated_values, rotated_directions, atol=1e-12, rtol=1e-12)

    def test_one_coordinate_gap_is_detected_but_random_slices_are_near_null(self) -> None:
        repeats, samples, features, direction_count = 12, 1024, 64, 256
        generator = torch.Generator().manual_seed(108)
        measurements = []
        for step in range(repeats):
            values = torch.randn((samples, features), generator=generator, dtype=torch.float64)
            signs = torch.randint(0, 2, (samples,), generator=generator) * 2 - 1
            noise = torch.randn((samples,), generator=generator, dtype=torch.float64)
            values[:, 0] = (3 * signs + 0.3 * noise) / math.sqrt(9.09)
            measurements.append(
                _ep(values, _directions(features, direction_count, step)).item()
            )
        self.assertLess(abs(sum(measurements) / repeats - ep_null_expectation()), 0.15)
        aligned = torch.zeros((features, 1), dtype=torch.float64)
        aligned[0, 0] = 1
        self.assertGreater(_ep(values, aligned).item(), 50)

    def test_pooled_and_stratified_objectives_and_mean_penalty(self) -> None:
        samples, m = 1024, 4
        labels = torch.repeat_interleave(torch.arange(m), samples // m)
        weights = torch.full((m,), 1 / m, dtype=torch.float64)
        results: dict[int, list[float]] = {}
        for features in (64, 128):
            pooled_values = []
            for step in range(10):
                generator = torch.Generator().manual_seed(200 + step + features)
                values = torch.randn((samples, features), generator=generator, dtype=torch.float64)
                signs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)[labels]
                values[:, 0] = (
                    3 * signs
                    + torch.randn((samples,), generator=generator, dtype=torch.float64)
                ) / math.sqrt(10)
                directions = _directions(features, 256, step)
                pooled_values.append(
                    _ep(values, directions).item()
                )
                stratified, _ = ep_per_view_stratum(
                    values.unsqueeze(0),
                    labels,
                    m,
                    directions,
                )
                self.assertGreater((stratified[0] * weights).sum().item(), 1.5)
                mean_gap = mean_penalty(
                    values.unsqueeze(0),
                    labels,
                    m,
                    weights,
                    sum_reduce=lambda value: value,
                    count_reduce=lambda value: value,
                )
                self.assertGreater(mean_gap.item(), 2.0)

                clean_generator = torch.Generator().manual_seed(500 + step + features)
                clean = torch.randn(
                    (samples, features),
                    generator=clean_generator,
                    dtype=torch.float64,
                )
                clean_ep, _ = ep_per_view_stratum(
                    clean.unsqueeze(0),
                    labels,
                    m,
                    directions,
                )
                self.assertLess((clean_ep[0] * weights).sum().item(), 1.3)
                clean_mean = mean_penalty(
                    clean.unsqueeze(0),
                    labels,
                    m,
                    weights,
                    sum_reduce=lambda value: value,
                    count_reduce=lambda value: value,
                )
                self.assertLess(clean_mean.item(), 1.3)
            results[features] = pooled_values
            self.assertLess(abs(sum(pooled_values) / len(pooled_values) - ep_null_expectation()), 0.2)

    def test_mean_penalty_null_and_gradient(self) -> None:
        repeats, samples, features = 4000, 8, 64
        generator = torch.Generator().manual_seed(109)
        draws = torch.randn((repeats, samples, features), generator=generator, dtype=torch.float64)
        estimates = samples * draws.mean(dim=1).square().sum(dim=-1) / features
        standard_error = estimates.std(unbiased=True) / math.sqrt(repeats)
        self.assertLess(abs(estimates.mean().item() - 1), 4 * standard_error.item())

        z = torch.randn((4, 8, 5), generator=generator, dtype=torch.float64, requires_grad=True)
        strata = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int64)
        weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
        value = mean_penalty(
            z,
            strata,
            2,
            weights,
            sum_reduce=lambda tensor: tensor,
            count_reduce=lambda tensor: tensor,
        )
        actual = torch.autograd.grad(value, z)[0]
        means = torch.stack((z[:, :4].mean(dim=1), z[:, 4:].mean(dim=1)), dim=1)
        expected = torch.empty_like(z)
        for sample in range(z.shape[1]):
            stratum = int(strata[sample])
            expected[:, sample] = (
                2 * weights[stratum] * means[:, stratum] / (z.shape[0] * z.shape[-1])
            )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_total_loss_gradcheck(self) -> None:
        generator = torch.Generator().manual_seed(110)
        z = torch.randn((5, 8, 4), generator=generator, dtype=torch.float64)
        strata = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int64)
        directions = _directions(4, 8)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: jepa_loss(
                    value,
                    strata,
                    n_global=2,
                    n_strata=2,
                    weights=[0.4, 0.6],
                    lam=0.3,
                    lam_mu=0.2,
                    directions=directions,
                    mode="stratified",
                )["loss"],
                (z.requires_grad_(),),
                eps=1e-6,
                atol=1e-7,
                rtol=1e-5,
            )
        )

    def test_autocast_disables_reduced_precision_loss_math(self) -> None:
        generator = torch.Generator().manual_seed(111)
        z = torch.randn((8, 64, 128), generator=generator, dtype=torch.float32)
        strata = torch.tensor([0] * 32 + [1] * 32, dtype=torch.int64)
        directions = sigreg_directions(
            128,
            128,
            seed=7,
            step=0,
            device="cpu",
            dtype=torch.float32,
        )
        kwargs = {
            "n_global": 2,
            "n_strata": 2,
            "weights": [0.5, 0.5],
            "lam": 0.2,
            "lam_mu": 0.1,
            "directions": directions,
            "mode": "stratified",
        }
        expected = jepa_loss(z, strata, **kwargs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = jepa_loss(z, strata, **kwargs)
        for key in ("loss", "inv", "sigreg", "t_mu", "ep", "ep_over_n"):
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_chunking_and_checkpoint_preserve_values_and_gradients(self) -> None:
        generator = torch.Generator().manual_seed(112)
        z = torch.randn((3, 16, 8), generator=generator, dtype=torch.float64)
        strata = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4, dtype=torch.int64)
        directions = _directions(8, 100)
        kwargs = {
            "n_global": 2,
            "n_strata": 4,
            "weights": [0.25] * 4,
            "lam": 0.2,
            "lam_mu": 0.1,
            "directions": directions,
            "mode": "stratified",
        }
        plain_z = z.detach().requires_grad_()
        plain = jepa_loss(plain_z, strata, **kwargs)
        plain_gradient = torch.autograd.grad(plain["loss"], plain_z)[0]
        chunked_z = z.detach().requires_grad_()
        chunked = jepa_loss(chunked_z, strata, chunk=32, checkpoint=True, **kwargs)
        chunked_gradient = torch.autograd.grad(chunked["loss"], chunked_z)[0]
        torch.testing.assert_close(chunked["loss"], plain["loss"], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(chunked_gradient, plain_gradient, atol=1e-12, rtol=1e-12)

    def test_direction_seed_and_generator_determinism(self) -> None:
        self.assertEqual(direction_seed(0, 0), 7213634469391956062)
        first = _directions(12, 20, step=1)
        repeated = _directions(12, 20, step=1)
        later = _directions(12, 20, step=2)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, later))
        torch.testing.assert_close(
            torch.linalg.vector_norm(first, dim=0),
            torch.ones(20, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )

    def test_input_validation_errors(self) -> None:
        z = torch.randn((3, 8, 4), dtype=torch.float64)
        directions = _directions(4, 8)
        good = {
            "n_global": 2,
            "n_strata": 2,
            "weights": [0.5, 0.5],
            "lam": 0.2,
            "lam_mu": 0.1,
            "directions": directions,
            "mode": "stratified",
        }
        with self.assertRaises(ValueError):
            jepa_loss(z, torch.zeros(8, dtype=torch.int64), **good)
        with self.assertRaises(ValueError):
            jepa_loss(z, torch.tensor([0.0] * 4 + [1.0] * 4), **good)
        with self.assertRaises(ValueError):
            jepa_loss(z, torch.tensor([0] * 4 + [1] * 4), **{**good, "weights": [0.4, 0.4]})
        with self.assertRaises(ValueError):
            jepa_loss(z, torch.tensor([0] * 4 + [1] * 4), **{**good, "n_global": 0})


if __name__ == "__main__":
    unittest.main()
