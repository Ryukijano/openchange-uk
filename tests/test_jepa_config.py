from __future__ import annotations

import os
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from openchange.jepa.config import (
    REQUIRED_KEYS,
    ConfigError,
    config_hash,
    expected_tokens_per_item,
    keep_count,
    load_config,
    lr_at,
    validate,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "jepa" / "p1_synthetic.yaml"


def _config() -> dict:
    return load_config(CONFIG_PATH)


class TestJEPAConfig(unittest.TestCase):
    def test_example_config_is_torch_free_and_valid(self) -> None:
        cfg = _config()
        self.assertEqual(cfg["status"], "synthetic_debug")
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import openchange.jepa.config; assert 'torch' not in sys.modules",
            ],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_hash_is_stable_under_key_reordering(self) -> None:
        cfg = _config()
        reordered = dict(reversed(list(cfg.items())))
        self.assertEqual(config_hash(cfg), config_hash(reordered))

    def test_every_required_key_is_checked(self) -> None:
        for key in REQUIRED_KEYS:
            with self.subTest(key=key):
                cfg = _config()
                del cfg[key]
                with self.assertRaisesRegex(ConfigError, key):
                    validate(cfg)

    def test_unknown_and_forbidden_keys_are_rejected_even_when_null(self) -> None:
        cfg = _config()
        cfg["unexpected"] = 1
        with self.assertRaisesRegex(ConfigError, "unexpected"):
            validate(cfg)
        cfg = _config()
        cfg["data_root"] = None
        with self.assertRaisesRegex(ConfigError, "data_root"):
            validate(cfg)
        cfg = _config()
        cfg["optimizer"]["manifest"] = None
        with self.assertRaisesRegex(ConfigError, "manifest"):
            validate(cfg)

    def test_synthetic_contract_validation(self) -> None:
        cases = [
            ("synthetic_only", False, "synthetic_only"),
            ("status", "real_data", "status"),
            ("global_quotas", [7, 8, 8, 9], "equal"),
            ("global_quotas", [7, 7, 7, 7], "sum"),
            ("stratum_weights", [0.25, 0.25, 0.25, 0.24], "sum to 1"),
            ("gradient_accumulation", 2, "gradient_accumulation"),
        ]
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                cfg = _config()
                cfg[key] = value
                with self.assertRaisesRegex(ConfigError, message):
                    validate(cfg)

    def test_size_and_band_validation(self) -> None:
        cfg = _config()
        cfg["schedule"]["total_steps"] = 51
        with self.assertRaisesRegex(ConfigError, "50"):
            validate(cfg)
        cfg = _config()
        cfg["global_crop_pixels"] = 100
        with self.assertRaisesRegex(ConfigError, "divisible"):
            validate(cfg)
        cfg = _config()
        cfg["sensors"]["s2"][8] = "B10"
        with self.assertRaisesRegex(ConfigError, "B10"):
            validate(cfg)
        cfg = _config()
        cfg["sensors"]["s2"].remove("B8A")
        with self.assertRaisesRegex(ConfigError, "B8A"):
            validate(cfg)

    def test_noncanonical_quadrature_is_rejected(self) -> None:
        for key, value in (("sigreg_knots", 9), ("sigreg_knots", 17.0),
                           ("sigreg_t_max", 2.0), ("sigreg_t_max", float("nan"))):
            with self.subTest(key=key, value=value):
                cfg = _config()
                cfg[key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    validate(cfg)

    def test_keep_count_uses_exact_half_up_rounding(self) -> None:
        examples = (
            (196, 0.9, 20),
            (392, 0.9, 39),
            (36, 0.5, 18),
            (72, 0.5, 36),
            (36, 0.9, 4),
            (9, 0.5, 5),
            (25, 0.34, 17),
            (1, 0.99, 1),
        )
        for tokens, rho, expected in examples:
            with self.subTest(tokens=tokens, rho=rho):
                self.assertEqual(keep_count(tokens, rho), expected)
        for rho in (1.0, -0.1):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                keep_count(9, rho)

    def test_expected_tokens_match_the_view_mixture(self) -> None:
        cfg = _config()
        both = expected_tokens_per_item(cfg, 2)
        single = expected_tokens_per_item(cfg, 1)
        weighted = (both * sum(cfg["global_quotas"][:2]) + single * sum(cfg["global_quotas"][2:])) / cfg[
            "global_batch"
        ]
        self.assertEqual(both, Fraction(1038, 5))
        self.assertEqual(single, Fraction(148))
        self.assertEqual(weighted, Fraction(889, 5))

    def test_cosine_schedule_exact_points_and_monotonic_tail(self) -> None:
        cfg = _config()
        self.assertAlmostEqual(lr_at(0, cfg), 1.0e-4, places=15)
        self.assertAlmostEqual(lr_at(4, cfg), 5.0e-4, places=15)
        self.assertAlmostEqual(lr_at(5, cfg), 5.0e-4, places=15)
        self.assertAlmostEqual(lr_at(49, cfg), 5.0e-7, delta=1e-18)
        schedule = [lr_at(step, cfg) for step in range(4, 50)]
        self.assertTrue(all(left >= right for left, right in zip(schedule, schedule[1:])))


if __name__ == "__main__":
    unittest.main()
