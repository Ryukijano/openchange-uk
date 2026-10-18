"""Clean-JEPA invariance, SIGReg, and mean-penalty losses."""

import hashlib
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from openchange.jepa.distributed import (
    DistContext,
    all_reduce_sum_autograd,
    all_reduce_sum_nograd,
)

_KNOT_COUNT = 17
_T_MAX = 3.0


def _identity(value: Tensor) -> Tensor:
    return value


def quadrature(dtype: torch.dtype, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
    t = torch.arange(_KNOT_COUNT, dtype=torch.float64, device="cpu") * (_T_MAX / 16)
    phi = torch.exp(-(t**2) / 2)
    omega = torch.full_like(t, 2 * (_T_MAX / 16))
    omega[0] *= 0.5
    omega[-1] *= 0.5
    q = omega * phi
    return tuple(value.to(device=device, dtype=dtype) for value in (t, phi, q))


def ep_null_expectation() -> float:
    t, _, q = quadrature(torch.float64, "cpu")
    return float(torch.sum(q * (1 - torch.exp(-(t**2))))
                 .item())


def direction_seed(seed: int, step: int) -> int:
    digest = hashlib.sha256(f"sigreg:{seed}:{step}".encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)


def sigreg_directions(
    k: int,
    m: int,
    *,
    seed: int,
    step: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    if k <= 0 or m <= 0:
        raise ValueError("k and m must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(direction_seed(seed, step))
    directions = torch.randn((k, m), generator=generator, dtype=torch.float64, device="cpu")
    directions = directions / torch.linalg.vector_norm(directions, dim=0, keepdim=True)
    return directions.to(device=device, dtype=dtype)


def invariance_loss(z: Tensor, n_global: int) -> Tensor:
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if not 1 <= n_global <= views:
        raise ValueError("n_global must be between 1 and V")
    center = z[:n_global].mean(dim=0)
    return (z - center).square().sum() / (samples * views * features)


def cf_sums(z: Tensor, a: Tensor, onehot: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    projections = torch.einsum("vnk,km->vnm", z, a)
    phase = projections.unsqueeze(-1) * t
    cosine_sums = torch.einsum("vnml,ns->vsml", torch.cos(phase), onehot)
    sine_sums = torch.einsum("vnml,ns->vsml", torch.sin(phase), onehot)
    return cosine_sums, sine_sums


def ep_per_view_stratum(
    z: Tensor,
    strata: Tensor,
    n_strata: int,
    a: Tensor,
    *,
    sum_reduce: Callable[[Tensor], Tensor] = _identity,
    count_reduce: Callable[[Tensor], Tensor] = _identity,
    chunk: int | None = None,
    checkpoint: bool = False,
) -> tuple[Tensor, Tensor]:
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if strata.ndim != 1 or strata.shape[0] != samples or strata.dtype != torch.int64:
        raise ValueError("strata must be an int64 tensor of shape [N]")
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    if a.ndim != 2 or a.shape[0] != features or a.shape[1] <= 0:
        raise ValueError("directions must have shape [K,M] with M positive")
    if chunk is not None and chunk <= 0:
        raise ValueError("chunk must be positive")
    if strata.numel() and (int(strata.min()) < 0 or int(strata.max()) >= n_strata):
        raise ValueError("strata contains an out-of-range label")

    compute_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=z.device.type, enabled=False):
        z_compute = z.to(dtype=compute_dtype)
        directions = a.to(device=z.device, dtype=compute_dtype)
        onehot = functional.one_hot(strata.to(device=z.device), num_classes=n_strata).to(
            dtype=compute_dtype
        )
        with torch.no_grad():
            counts = count_reduce(onehot.sum(dim=0))
            if torch.any(counts == 0):
                raise ValueError("every stratum must contain at least one sample")
        t, phi, q = quadrature(compute_dtype, z.device)
        direction_count = directions.shape[1]
        chunk_size = direction_count if chunk is None else min(chunk, direction_count)
        accumulated = z_compute.new_zeros((views, n_strata))
        for start in range(0, direction_count, chunk_size):
            direction_chunk = directions[:, start : start + chunk_size]
            if checkpoint and z_compute.requires_grad:
                cosine_sums, sine_sums = checkpoint_fn(
                    cf_sums,
                    z_compute,
                    direction_chunk,
                    onehot,
                    t,
                    use_reentrant=False,
                )
            else:
                cosine_sums, sine_sums = cf_sums(z_compute, direction_chunk, onehot, t)
            cosine_sums, sine_sums = sum_reduce(torch.stack((cosine_sums, sine_sums))).unbind(0)
            denominator = counts[None, :, None, None]
            cosine_means = cosine_sums / denominator
            sine_means = sine_sums / denominator
            terms = (cosine_means - phi[None, None, None, :]).square() + sine_means.square()
            accumulated = accumulated + (terms * q[None, None, None, :]).sum(dim=(-1, -2))
        ep = counts[None, :] * accumulated / direction_count
    return ep, counts


def mean_penalty(
    z: Tensor,
    strata: Tensor,
    n_strata: int,
    weights: Tensor | list[float] | tuple[float, ...],
    *,
    sum_reduce: Callable[[Tensor], Tensor],
    count_reduce: Callable[[Tensor], Tensor],
) -> Tensor:
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if strata.ndim != 1 or strata.shape[0] != samples or strata.dtype != torch.int64:
        raise ValueError("strata must be an int64 tensor of shape [N]")
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    if strata.numel() and (int(strata.min()) < 0 or int(strata.max()) >= n_strata):
        raise ValueError("strata contains an out-of-range label")

    compute_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=z.device.type, enabled=False):
        z_compute = z.to(dtype=compute_dtype)
        onehot = functional.one_hot(strata.to(device=z.device), num_classes=n_strata).to(
            dtype=compute_dtype
        )
        with torch.no_grad():
            counts = count_reduce(onehot.sum(dim=0))
            if torch.any(counts == 0):
                raise ValueError("every stratum must contain at least one sample")
        weights_tensor = torch.as_tensor(weights, device=z.device, dtype=compute_dtype)
        if weights_tensor.shape != (n_strata,):
            raise ValueError("weights must have shape [n_strata]")
        sums = sum_reduce(torch.einsum("vnk,ns->vsk", z_compute, onehot))
        means = sums / counts[None, :, None]
        return (
            (weights_tensor[None, :] * counts[None, :])
            * means.square().sum(dim=-1)
            .div(features)
        ).sum(dim=1).mean()


def _validated_weights(weights: Tensor | list[float] | tuple[float, ...], n_strata: int) -> Tensor:
    values = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if values.shape != (n_strata,):
        raise ValueError("weights must have shape [n_strata]")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    if abs(float(values.sum()) - 1.0) > 1e-12:
        raise ValueError("weights must sum to 1")
    return values


def jepa_loss(
    z: Tensor,
    strata: Tensor,
    *,
    n_global: int,
    n_strata: int,
    weights: Tensor | list[float] | tuple[float, ...],
    lam: float,
    lam_mu: float,
    directions: Tensor,
    mode: Literal["pooled", "stratified"],
    dist: DistContext | None = None,
    chunk: int | None = None,
    checkpoint: bool = False,
) -> dict[str, Tensor]:
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if not 1 <= n_global <= views:
        raise ValueError("n_global must be between 1 and V")
    if strata.ndim != 1 or strata.shape[0] != samples or strata.dtype != torch.int64:
        raise ValueError("strata must be an int64 tensor of shape [N]")
    if mode not in ("pooled", "stratified"):
        raise ValueError("mode must be 'pooled' or 'stratified'")
    if directions.ndim != 2 or directions.shape[0] != features or directions.shape[1] <= 0:
        raise ValueError("directions must have shape [K,M]")

    if mode == "pooled":
        active_strata = torch.zeros(samples, dtype=torch.int64, device=z.device)
        active_n_strata = 1
        active_weights = torch.ones(1, dtype=torch.float64)
    else:
        if n_strata <= 0:
            raise ValueError("n_strata must be positive")
        if strata.numel() and (int(strata.min()) < 0 or int(strata.max()) >= n_strata):
            raise ValueError("strata contains an out-of-range label")
        active_strata = strata.to(device=z.device)
        active_n_strata = n_strata
        active_weights = _validated_weights(weights, active_n_strata)

    if chunk is not None and chunk <= 0:
        raise ValueError("chunk must be positive")
    compute_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    context = dist
    sum_reduce: Callable[[Tensor], Tensor] = (
        _identity if context is None else lambda value: all_reduce_sum_autograd(value, context)
    )
    count_reduce: Callable[[Tensor], Tensor] = (
        _identity if context is None else lambda value: all_reduce_sum_nograd(value, context)
    )
    with torch.autocast(device_type=z.device.type, enabled=False):
        z_compute = z.to(dtype=compute_dtype)
        inv = invariance_loss(z_compute, n_global)
        ep, counts = ep_per_view_stratum(
            z_compute,
            active_strata,
            active_n_strata,
            directions,
            sum_reduce=sum_reduce,
            count_reduce=count_reduce,
            chunk=chunk,
            checkpoint=checkpoint,
        )
        weights_compute = active_weights.to(device=z.device, dtype=compute_dtype)
        sigreg = (ep * weights_compute[None, :]).sum(dim=1).mean()
        if lam_mu == 0:
            with torch.no_grad():
                t_mu = mean_penalty(
                    z_compute,
                    active_strata,
                    active_n_strata,
                    weights_compute,
                    sum_reduce=sum_reduce,
                    count_reduce=count_reduce,
                )
        else:
            t_mu = mean_penalty(
                z_compute,
                active_strata,
                active_n_strata,
                weights_compute,
                sum_reduce=sum_reduce,
                count_reduce=count_reduce,
            )
        loss = (1 - lam) * inv + lam * sigreg + lam_mu * t_mu
        ep_detached = ep.detach()
        counts_detached = counts.detach()
        return {
            "loss": loss,
            "inv": inv,
            "sigreg": sigreg,
            "t_mu": t_mu,
            "ep": ep_detached,
            "ep_over_n": ep_detached / counts_detached[None, :],
            "counts": counts_detached,
            "null": torch.tensor(ep_null_expectation(), dtype=compute_dtype, device=z.device),
        }
