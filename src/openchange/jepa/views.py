"""Sentinel-style observations and deterministic multi-crop views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from openchange.jepa.config import keep_count
from openchange.jepa.distributed import make_global_strata


@dataclass(frozen=True)
class Observation:
    sample_id: int
    stratum: int
    s1: np.ndarray | None
    s2: np.ndarray | None
    s2_valid: np.ndarray | None
    synthetic_data: bool
    synthetic_condition: bool
    intervention: str
    mask_fraction: float

    def __post_init__(self) -> None:
        if self.stratum not in range(4):
            raise ValueError("stratum must be in [0, 3]")
        if self.stratum in (0, 1) and (self.s1 is None or self.s2 is None):
            raise ValueError("joint strata require both sensors")
        if self.stratum == 2 and (self.s1 is None or self.s2 is not None):
            raise ValueError("s1_only requires S1 and no S2")
        if self.stratum == 3 and (self.s1 is not None or self.s2 is None):
            raise ValueError("s2_only requires S2 and no S1")
        if self.intervention not in {"none", "synthetic_occlusion", "withhold_s1", "withhold_s2"}:
            raise ValueError("unknown synthetic intervention")
        reference = self.s1 if self.s1 is not None else self.s2
        if reference is None or reference.ndim != 3:
            raise ValueError("sensor arrays must have shape [C,H,W]")
        height, width = reference.shape[-2:]
        if self.s1 is not None and (self.s1.shape != (2, height, width) or self.s1.dtype != np.float32):
            raise ValueError("S1 must be float32 [2,H,W]")
        if self.s2 is not None and (
            self.s2.shape != (12, height, width) or self.s2.dtype != np.float32
        ):
            raise ValueError("S2 must be float32 [12,H,W]")
        if self.s2 is None:
            if self.s2_valid is not None:
                raise ValueError("s2_valid must be None when S2 is absent")
        elif (
            self.s2_valid is None
            or self.s2_valid.shape != (height, width)
            or self.s2_valid.dtype != np.bool_
        ):
            raise ValueError("s2_valid must be bool [H,W] when S2 is present")
        if not 0.0 <= self.mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be in [0, 1]")


@dataclass
class ViewBucket:
    resolution: int
    sensors: tuple[str, ...]
    pixels: dict[str, Tensor]
    keep_idx: Tensor
    view_idx: Tensor
    sample_idx: Tensor


@dataclass
class ViewBatch:
    buckets: list[ViewBucket]
    n_samples: int
    n_views: int
    n_global: int
    strata: Tensor
    sample_ids: Tensor
    synthetic_data: Tensor
    synthetic_condition: Tensor
    view_sensors: list[list[tuple[str, ...]]]
    crop_boxes: Tensor
    flip_rot: Tensor
    tokens: Tensor
    s2_valid_fraction: Tensor

    def pin_memory(self) -> ViewBatch:
        self.strata = self.strata.pin_memory()
        for bucket in self.buckets:
            bucket.pixels = {key: value.pin_memory() for key, value in bucket.pixels.items()}
            for key in ("keep_idx", "view_idx", "sample_idx"):
                setattr(bucket, key, getattr(bucket, key).pin_memory())
        return self


@lru_cache(maxsize=16)
def _generator_constants(
    seed: int,
    height: int,
    shared_factors: int,
    private_factors: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng([seed, 11])
    w1 = rng.standard_normal((2, shared_factors)).astype(np.float32) / math.sqrt(shared_factors)
    w2_shared = (
        rng.standard_normal((12, shared_factors)).astype(np.float32) / math.sqrt(shared_factors)
    )
    w2_private = (
        rng.standard_normal((12, private_factors)).astype(np.float32) / math.sqrt(private_factors)
    )
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="ij",
    )

    def textures(count: int) -> np.ndarray:
        frequencies = rng.integers(1, max(2, height // 8), size=(count, 2))
        phases = rng.uniform(0.0, 2 * np.pi, size=count)
        return np.cos(
            2 * np.pi
            * (
                frequencies[:, 0, None, None] * xx[None]
                + frequencies[:, 1, None, None] * yy[None]
            )
            / height
            + phases[:, None, None]
        ).astype(np.float32)

    return w1, w2_shared, w2_private, textures(shared_factors), textures(private_factors)


def synthetic_observation(
    sample_id: int,
    stratum: int,
    seed: int,
    cfg: dict,
    private_override: np.ndarray | dict[str, np.ndarray] | None = None,
) -> tuple[Observation, dict[str, np.ndarray]]:
    spec = cfg["synthetic"]
    height = spec["image_pixels"]
    shared_factors = spec["shared_factors"]
    private_factors = spec["private_s2_factors"]
    w1, w2_shared, w2_private, t_shared, t_private = _generator_constants(
        seed, height, shared_factors, private_factors
    )
    shared = np.random.default_rng([seed, 7, sample_id, 0]).standard_normal(shared_factors).astype(
        np.float32
    )
    private = np.random.default_rng([seed, 7, sample_id, 1]).standard_normal(private_factors).astype(
        np.float32
    )
    if private_override is not None:
        private = np.asarray(
            private_override.get("private") if isinstance(private_override, dict) else private_override,
            dtype=np.float32,
        )
        if private.shape != (private_factors,):
            raise ValueError(f"private_override must have shape ({private_factors},)")
    shared_features = (1.0 + 0.5 * t_shared) * shared[:, None, None]
    private_features = (1.0 + 0.5 * t_private) * private[:, None, None]
    noise_scale = spec["noise_std"]
    s1_noise = np.random.default_rng([seed, 7, sample_id, 2]).standard_normal(
        (2, height, height)
    ).astype(np.float32)
    s2_noise = np.random.default_rng([seed, 7, sample_id, 3]).standard_normal(
        (12, height, height)
    ).astype(np.float32)
    s1_full = np.einsum("cf,fhw->chw", w1, shared_features).astype(np.float32)
    s1_full += noise_scale * s1_noise
    s2_full = np.einsum("cf,fhw->chw", w2_shared, shared_features).astype(np.float32)
    s2_full += np.einsum("cf,fhw->chw", w2_private, private_features).astype(np.float32)
    s2_full += noise_scale * s2_noise

    s1 = None if stratum == 3 else s1_full
    s2 = None if stratum == 2 else s2_full
    s2_valid = None if s2 is None else np.ones((height, height), dtype=np.bool_)
    intervention = "none"
    synthetic_condition = False
    mask_fraction = 0.0
    if stratum == 1:
        low, high = spec["occlusion_fraction"]
        requested = np.random.default_rng([seed, 7, sample_id, 4]).uniform(low, high)
        area = height * height
        minimum = math.ceil(low * area)
        maximum = math.floor(high * area)
        target = min(max(round(requested * area), minimum), maximum)
        rectangle_height = max(1, min(height, round(math.sqrt(target))))
        rectangle_width = max(1, min(height, round(target / rectangle_height)))
        while rectangle_height * rectangle_width < minimum and rectangle_width < height:
            rectangle_width += 1
        while rectangle_height * rectangle_width > maximum and rectangle_width > 1:
            rectangle_width -= 1
        if rectangle_height * rectangle_width > maximum:
            rectangle_height = max(1, maximum // rectangle_width)
        rng = np.random.default_rng([seed, 7, sample_id, 5])
        top = int(rng.integers(0, height - rectangle_height + 1))
        left = int(rng.integers(0, height - rectangle_width + 1))
        s2[:, top : top + rectangle_height, left : left + rectangle_width] = spec[
            "occlusion_fill"
        ]
        s2_valid[top : top + rectangle_height, left : left + rectangle_width] = False
        mask_fraction = float(1.0 - s2_valid.mean())
        intervention = "synthetic_occlusion"
        synthetic_condition = True
    elif stratum == 2:
        intervention = "withhold_s2"
    elif stratum == 3:
        intervention = "withhold_s1"

    observation = Observation(
        sample_id=sample_id,
        stratum=stratum,
        s1=s1,
        s2=s2,
        s2_valid=s2_valid,
        synthetic_data=True,
        synthetic_condition=synthetic_condition,
        intervention=intervention,
        mask_fraction=mask_fraction,
    )
    return observation, {"shared": shared, "private": private}


def synthetic_batch(seed: int, step: int, cfg: dict, layout_world_size: int = 1) -> list[Observation]:
    if layout_world_size < 1:
        raise ValueError("layout_world_size must be positive")
    quotas = cfg["global_quotas"]
    if any(quota % layout_world_size for quota in quotas):
        raise ValueError("each stratum quota must be divisible by layout_world_size")
    pool_size = cfg["synthetic"]["pool_per_stratum"]
    by_stratum: list[list[Observation]] = []
    for stratum, quota in enumerate(quotas):
        rng = np.random.default_rng([seed, step, 99, stratum])
        ids = rng.choice(pool_size, size=quota, replace=False)
        by_stratum.append(
            [
                synthetic_observation(stratum * pool_size + int(index), stratum, seed, cfg)[0]
                for index in ids
            ]
        )
    labels = make_global_strata(quotas).tolist()
    positions = [0] * len(quotas)
    batch: list[Observation] = []
    for stratum in labels:
        batch.append(by_stratum[stratum][positions[stratum]])
        positions[stratum] += 1
    return batch


def _crop_box(
    rng: np.random.Generator,
    height: int,
    width: int,
    scale: Sequence[float],
    log_aspect: Sequence[float],
) -> tuple[int, int, int, int]:
    for _ in range(10):
        area = rng.uniform(scale[0], scale[1]) * height * width
        aspect = math.exp(rng.uniform(log_aspect[0], log_aspect[1]))
        crop_height = max(1, round(math.sqrt(area / aspect)))
        crop_width = max(1, round(math.sqrt(area * aspect)))
        if crop_height <= height and crop_width <= width:
            top = int(rng.integers(0, height - crop_height + 1))
            left = int(rng.integers(0, width - crop_width + 1))
            return top, left, crop_height, crop_width
    side = min(height, width)
    return (height - side) // 2, (width - side) // 2, side, side


def _prepare_crop(
    array: np.ndarray,
    box: tuple[int, int, int, int],
    flip: int,
    rotation: int,
    resolution: int,
) -> Tensor:
    top, left, height, width = box
    cropped = torch.from_numpy(np.ascontiguousarray(array[:, top : top + height, left : left + width]))
    if flip:
        cropped = torch.flip(cropped, dims=(-1,))
    if rotation:
        cropped = torch.rot90(cropped, k=rotation, dims=(-2, -1))
    return functional.interpolate(
        cropped.unsqueeze(0),
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]


def _local_sensor_subset(
    seed: int,
    step: int,
    sample_id: int,
    view: int,
    probabilities: dict[str, float],
) -> tuple[str, ...]:
    subset = int(
        np.random.default_rng([seed, step, sample_id, view, 1]).choice(
            3,
            p=[probabilities["s1"], probabilities["s2"], probabilities["both"]],
        )
    )
    return (("s1",), ("s2",), ("s1", "s2"))[subset]


def make_views(
    observations: Sequence[Observation],
    seed: int,
    step: int,
    cfg: dict,
    *,
    allow_real: bool = False,
) -> ViewBatch:
    if not observations:
        raise ValueError("observations must not be empty")
    if not allow_real and any(not observation.synthetic_data for observation in observations):
        raise ValueError("real data needs data.py")
    n_samples = len(observations)
    n_global = cfg["global_views"]
    n_views = n_global + cfg["local_views"]
    tokens = torch.zeros((n_views, n_samples), dtype=torch.int64)
    crop_boxes = torch.zeros((n_views, n_samples, 4), dtype=torch.int64)
    flip_rot = torch.zeros((n_views, n_samples, 2), dtype=torch.int64)
    s2_valid_fraction = torch.full((n_views, n_samples), float("nan"), dtype=torch.float32)
    view_sensors: list[list[tuple[str, ...]]] = [[() for _ in observations] for _ in range(n_views)]
    grouped: dict[tuple[int, tuple[str, ...]], list[tuple[int, int, dict[str, Tensor], Tensor]]] = {}
    global_scale = cfg["global_crop_scale"]
    local_scale = cfg["local_crop_scale"]
    aspect = cfg["crop_log_aspect"]

    for view in range(n_views):
        is_global = view < n_global
        resolution = cfg["global_crop_pixels"] if is_global else cfg["local_crop_pixels"]
        scale = global_scale if is_global else local_scale
        rho = cfg["global_token_drop"] if is_global else cfg["local_token_drop"]
        for sample_index, observation in enumerate(observations):
            reference = observation.s1 if observation.s1 is not None else observation.s2
            height, width = reference.shape[-2:]
            geometry_rng = np.random.default_rng([seed, step, observation.sample_id, view, 0])
            box = _crop_box(geometry_rng, height, width, scale, aspect)
            flip = int(geometry_rng.random() < 0.5)
            rotation = int(geometry_rng.integers(0, 4))
            crop_boxes[view, sample_index] = torch.tensor(box)
            flip_rot[view, sample_index] = torch.tensor((flip, rotation))

            available = tuple(sensor for sensor, value in (("s1", observation.s1), ("s2", observation.s2)) if value is not None)
            if is_global or len(available) == 1:
                sensors = available
            else:
                sensors = _local_sensor_subset(
                    seed,
                    step,
                    observation.sample_id,
                    view,
                    cfg["local_sensor_probs"],
                )
            view_sensors[view][sample_index] = sensors
            inputs: dict[str, Tensor] = {}
            valid_fraction = None
            for sensor in sensors:
                pixels = observation.s1 if sensor == "s1" else observation.s2
                inputs[sensor] = _prepare_crop(pixels, box, flip, rotation, resolution)
                if sensor == "s2":
                    valid = observation.s2_valid.astype(np.float32, copy=False)[None]
                    valid_tensor = _prepare_crop(valid, box, flip, rotation, resolution)
                    valid_fraction = float(valid_tensor.mean())
            if valid_fraction is not None:
                s2_valid_fraction[view, sample_index] = valid_fraction
            total_tokens = (resolution // cfg["patch_size"]) ** 2 * len(sensors)
            keep = keep_count(total_tokens, rho)
            drop_rng = np.random.default_rng([seed, step, observation.sample_id, view, 2])
            keep_indices = np.sort(
                drop_rng.choice(total_tokens, size=keep, replace=False)
            ).astype(np.int64)
            index_tensor = torch.from_numpy(keep_indices)
            tokens[view, sample_index] = keep
            grouped.setdefault((resolution, sensors), []).append(
                (view, sample_index, inputs, index_tensor)
            )

    buckets: list[ViewBucket] = []
    bucket_keys = sorted(grouped, key=lambda key: (-key[0], key[1]))
    for resolution, sensors in bucket_keys:
        items = grouped[(resolution, sensors)]
        pixels = {
            sensor: torch.stack([item[2][sensor] for item in items]) for sensor in sensors
        }
        buckets.append(
            ViewBucket(
                resolution=resolution,
                sensors=sensors,
                pixels=pixels,
                keep_idx=torch.stack([item[3] for item in items]),
                view_idx=torch.tensor([item[0] for item in items], dtype=torch.int64),
                sample_idx=torch.tensor([item[1] for item in items], dtype=torch.int64),
            )
        )
    return ViewBatch(
        buckets=buckets,
        n_samples=n_samples,
        n_views=n_views,
        n_global=n_global,
        strata=torch.tensor([observation.stratum for observation in observations], dtype=torch.int64),
        sample_ids=torch.tensor([observation.sample_id for observation in observations], dtype=torch.int64),
        synthetic_data=torch.tensor([observation.synthetic_data for observation in observations]),
        synthetic_condition=torch.tensor(
            [observation.synthetic_condition for observation in observations]
        ),
        view_sensors=view_sensors,
        crop_boxes=crop_boxes,
        flip_rot=flip_rot,
        tokens=tokens,
        s2_valid_fraction=s2_valid_fraction,
    )
