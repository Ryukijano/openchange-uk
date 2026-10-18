"""Torch-free configuration and schedule helpers for the synthetic JEPA pilot."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a JEPA configuration violates the synthetic pilot contract."""


REQUIRED_KEYS = (
    "schema_version",
    "status",
    "synthetic_only",
    "split",
    "job",
    "seed",
    "loss_reduction",
    "strata",
    "global_batch",
    "global_quotas",
    "stratum_weights",
    "sensors",
    "normalization",
    "patch_size",
    "global_views",
    "local_views",
    "global_crop_pixels",
    "local_crop_pixels",
    "global_crop_scale",
    "local_crop_scale",
    "crop_log_aspect",
    "local_sensor_probs",
    "s2_gain_jitter",
    "global_token_drop",
    "local_token_drop",
    "token_keep_rounding",
    "model",
    "projector_hidden",
    "projector_dim",
    "projector_norm",
    "sigreg_mode",
    "sigreg_slices",
    "sigreg_knots",
    "sigreg_t_max",
    "lambda",
    "lambda_mu",
    "gaussian_target",
    "optimizer",
    "schedule",
    "gradient_accumulation",
    "precision",
    "runtime",
    "synthetic",
    "checkpoint",
)

_NESTED_KEYS = {
    "sensors": {"s1", "s2"},
    "local_sensor_probs": {"s1", "s2", "both"},
    "model": {"dim", "depth", "heads", "mlp_ratio", "pos_embed", "drop_path"},
    "optimizer": {
        "name",
        "lr",
        "lr_reference_batch",
        "lr_scaling",
        "betas",
        "eps",
        "weight_decay",
        "weight_decay_scope",
        "grad_clip_norm",
    },
    "schedule": {"name", "warmup_steps", "total_steps", "floor_ratio"},
    "runtime": {"cpu_threads", "max_wall_seconds", "deterministic"},
    "synthetic": {
        "image_pixels",
        "pool_per_stratum",
        "shared_factors",
        "private_s2_factors",
        "noise_std",
        "occlusion_fraction",
        "occlusion_fill",
    },
    "checkpoint": {"every_steps"},
}
_STRATA = ("joint_clear", "joint_occluded", "s1_only", "s2_only")
_S1_BANDS = ("VV", "VH")
_S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
_FORBIDDEN_KEYS = {"data_root", "sources", "manifest"}


def _contains_forbidden(value: Any, path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            shown = f"{path}.{key}" if path else str(key)
            if key in _FORBIDDEN_KEYS:
                return shown
            found = _contains_forbidden(nested, shown)
            if found:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _contains_forbidden(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def _check_keys(value: Any, required: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    missing = sorted(required - value.keys())
    if missing:
        raise ConfigError(f"missing key {path}.{missing[0]}")
    unknown = sorted(value.keys() - required)
    if unknown:
        raise ConfigError(f"unknown key {path}.{unknown[0]}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fraction(value: Any, name: str) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ConfigError(f"{name} must be a finite number") from exc


def load_config(path: Path, *, local_pilot: bool = False, cuda: bool = False) -> dict[str, Any]:
    """Load and validate a YAML configuration without importing torch."""
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not load config {path}: {exc}") from exc
    validate(loaded, local_pilot=local_pilot, cuda=cuda)
    return loaded


def validate(cfg: dict[str, Any], *, local_pilot: bool = False, cuda: bool = False) -> None:
    """Validate required fields and enforce the synthetic-only pilot boundary."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    forbidden = _contains_forbidden(cfg)
    if forbidden is not None:
        raise ConfigError(f"forbidden key {forbidden}")
    required = set(REQUIRED_KEYS) - ({"synthetic"} if local_pilot else set())
    _check_keys(cfg, required, "config")
    for key, keys in _NESTED_KEYS.items():
        if key == "synthetic" and local_pilot:
            continue
        _check_keys(cfg[key], keys, key)

    if cfg["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if local_pilot:
        if cfg["status"] != "local_pilot" or cfg["synthetic_only"] is not False or cfg["split"] != "pilot":
            raise ConfigError("local pilot requires status=local_pilot, synthetic_only=false, split=pilot")
        if cfg["normalization"] != "manifest":
            raise ConfigError("local pilot normalization must be manifest")
    elif cfg["status"] not in {"planned", "synthetic_debug"}:
        raise ConfigError("status must be planned or synthetic_debug")
    if not local_pilot and cfg["synthetic_only"] is not True:
        raise ConfigError("synthetic_only must be true")
    if not local_pilot and cfg["split"] != "none":
        raise ConfigError('split must equal "none"')
    if cfg["loss_reduction"] != "mean_n_v_k":
        raise ConfigError("loss_reduction must be mean_n_v_k")
    if tuple(cfg["strata"]) != _STRATA:
        raise ConfigError("strata must use the four formulation strata in order")

    global_batch = cfg["global_batch"]
    quotas = cfg["global_quotas"]
    if not _is_int(global_batch) or global_batch <= 0:
        raise ConfigError("global_batch must be a positive integer")
    if not isinstance(quotas, list) or len(quotas) != 4 or any(not _is_int(q) or q <= 0 for q in quotas):
        raise ConfigError("global_quotas must contain four positive integers")
    if len(set(quotas)) != 1:
        raise ConfigError("global_quotas must all be equal")
    if sum(quotas) != global_batch:
        raise ConfigError("global_quotas must sum to global_batch")

    weights = cfg["stratum_weights"]
    if not isinstance(weights, list) or len(weights) != 4:
        raise ConfigError("stratum_weights must contain four values")
    if any(not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0 for weight in weights):
        raise ConfigError("stratum_weights must be nonnegative numbers")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigError("stratum_weights must sum to 1")

    if cfg["sensors"]["s1"] != list(_S1_BANDS):
        raise ConfigError("sensors.s1 must be [VV, VH]")
    if cfg["sensors"]["s2"] != list(_S2_BANDS):
        raise ConfigError("sensors.s2 must include the required bands including B8A and exclude B10")

    for key in ("patch_size", "global_views", "local_views", "global_crop_pixels", "local_crop_pixels"):
        if not _is_int(cfg[key]) or cfg[key] <= 0:
            raise ConfigError(f"{key} must be a positive integer")
    for key in ("global_crop_pixels", "local_crop_pixels"):
        if cfg[key] % cfg["patch_size"]:
            raise ConfigError(f"{key} must be divisible by patch_size")

    probabilities = cfg["local_sensor_probs"]
    if any(not isinstance(probability, (int, float)) or probability < 0 for probability in probabilities.values()):
        raise ConfigError("local_sensor_probs must be nonnegative")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigError("local_sensor_probs must sum to 1")

    for key in ("global_token_drop", "local_token_drop"):
        rho = _fraction(cfg[key], key)
        if not 0 <= rho < 1:
            raise ConfigError(f"{key} must be in [0, 1)")
    if cfg["token_keep_rounding"] != "half_up_exact":
        raise ConfigError("token_keep_rounding must be half_up_exact")
    if not 0 <= cfg["lambda"] <= 1:
        raise ConfigError("lambda must be in [0, 1]")
    if cfg["lambda_mu"] < 0:
        raise ConfigError("lambda_mu must be nonnegative")
    if cfg["sigreg_mode"] not in {"pooled", "stratified"}:
        raise ConfigError("sigreg_mode must be pooled or stratified")
    if not _is_int(cfg["sigreg_knots"]) or cfg["sigreg_knots"] != 17:
        raise ConfigError("sigreg_knots must equal 17 for the implemented quadrature")
    if cfg["sigreg_t_max"] != 3.0:
        raise ConfigError("sigreg_t_max must equal 3.0 for the implemented quadrature")
    if cfg["projector_norm"] not in {"batch", "frozen", "identity"}:
        raise ConfigError("projector_norm must be batch, frozen, or identity")
    if cfg["gradient_accumulation"] != 1:
        raise ConfigError("gradient_accumulation must equal 1")
    if cfg["precision"] not in ({"fp32", "bf16"} if cuda else {"fp32"}):
        raise ConfigError("precision must be fp32")

    schedule = cfg["schedule"]
    if not _is_int(schedule["total_steps"]) or schedule["total_steps"] <= 0:
        raise ConfigError("schedule.total_steps must be a positive integer")
    if not _is_int(schedule["warmup_steps"]) or schedule["warmup_steps"] < 0:
        raise ConfigError("schedule.warmup_steps must be a nonnegative integer")
    if cfg["status"] == "synthetic_debug" or local_pilot:
        if schedule["total_steps"] > 50:
            raise ConfigError("synthetic_debug total_steps must not exceed 50")
        if schedule["warmup_steps"] >= schedule["total_steps"]:
            raise ConfigError("warmup_steps must be less than total_steps")


def config_hash(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def keep_count(tokens: int, rho: float) -> int:
    if not _is_int(tokens) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    fraction = Fraction(str(rho))
    if not 0 <= fraction < 1:
        raise ValueError("rho must be in [0, 1)")
    retained = tokens * (1 - fraction)
    return max(1, (retained.numerator * 2 + retained.denominator) // (2 * retained.denominator))


def lr_at(step: int, cfg: dict[str, Any]) -> float:
    if not _is_int(step) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    optimizer = cfg["optimizer"]
    schedule = cfg["schedule"]
    peak = float(optimizer["lr"])
    warmup = schedule["warmup_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, schedule["total_steps"] - 1 - warmup)
    floor = peak * schedule["floor_ratio"]
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def expected_tokens_per_item(cfg: dict[str, Any], sensors_available: int) -> Fraction:
    if sensors_available not in {1, 2}:
        raise ValueError("sensors_available must be 1 or 2")
    patch_size = cfg["patch_size"]
    global_tokens = (cfg["global_crop_pixels"] // patch_size) ** 2 * sensors_available
    local_tokens_one = (cfg["local_crop_pixels"] // patch_size) ** 2
    global_keep = keep_count(global_tokens, cfg["global_token_drop"])
    if sensors_available == 1:
        local_expectation = Fraction(keep_count(local_tokens_one, cfg["local_token_drop"]))
    else:
        probabilities = cfg["local_sensor_probs"]
        one_sensor = keep_count(local_tokens_one, cfg["local_token_drop"])
        two_sensors = keep_count(2 * local_tokens_one, cfg["local_token_drop"])
        local_expectation = sum(
            (Fraction(str(probabilities[key])) * value for key, value in (
                ("s1", one_sensor),
                ("s2", one_sensor),
                ("both", two_sensors),
            )),
            Fraction(0),
        )
    return Fraction(cfg["global_views"] * global_keep) + cfg["local_views"] * local_expectation
