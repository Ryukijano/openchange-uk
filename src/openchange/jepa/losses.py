"""Clean-JEPA invariance, SIGReg, mean-penalty, split-subspace, Barlow/VICReg, and Flow Matching losses."""

import hashlib
import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
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


def weighted_invariance_loss(
    z: Tensor,
    n_global: int,
    *,
    pair_gap_seconds: Tensor | None = None,
    gap_tau_seconds: float | None = None,
    view_weights: Tensor | None = None,
) -> Tensor:
    """Invariance loss with optional S1-S2 acquisition time-gap and view-validity attenuation.

    When ``pair_gap_seconds`` [N] and ``gap_tau_seconds > 0`` are provided, each sample's
    cross-view alignment penalty is weighted by ``w_n = exp(-Delta_t_n / tau)`` so pairs
    separated by weeks of phenological/moisture change are not forced to zero residual.
    """
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if not 1 <= n_global <= views:
        raise ValueError("n_global must be between 1 and V")

    center = z[:n_global].mean(dim=0)
    sq_err = (z - center).square().mean(dim=-1)  # [V, N]

    weights = torch.ones((views, samples), dtype=z.dtype, device=z.device)
    if pair_gap_seconds is not None and gap_tau_seconds is not None:
        if gap_tau_seconds <= 0 or not math.isfinite(gap_tau_seconds):
            raise ValueError("gap_tau_seconds must be positive and finite")
        gaps = torch.as_tensor(pair_gap_seconds, dtype=z.dtype, device=z.device)
        if gaps.ndim != 1 or gaps.shape[0] != samples or bool((gaps < 0).any()):
            raise ValueError("pair_gap_seconds must be a nonnegative 1D tensor of shape [N]")
        sample_w = torch.exp(-gaps / float(gap_tau_seconds))
        weights = weights * sample_w.unsqueeze(0)

    if view_weights is not None:
        vw = torch.as_tensor(view_weights, dtype=z.dtype, device=z.device)
        if vw.shape != (views, samples) or bool((vw < 0).any()):
            raise ValueError("view_weights must be nonnegative with shape [V, N]")
        weights = weights * vw

    return (sq_err * weights).sum() / (views * samples)


def split_subspace_jepa_loss(
    z: Tensor,
    strata: Tensor,
    *,
    n_global: int,
    shared_dim: int,
    s1_private_dim: int,
    s2_private_dim: int,
    directions_shared: Tensor,
    directions_s1: Tensor | None = None,
    directions_s2: Tensor | None = None,
    lam: float = 0.05,
    lam_priv_inactive: float = 0.1,
    pair_gap_seconds: Tensor | None = None,
    gap_tau_seconds: float | None = None,
    dist: DistContext | None = None,
    chunk: int | None = None,
    checkpoint: bool = False,
) -> dict[str, Tensor]:
    """Factorized Shared + Sensor-Private Clean-JEPA loss resolving the Bayes-risk conflict.

    Decomposes ``z = [z_shared, z_s1_priv, z_s2_priv]`` of width ``K = K_sh + K_s1 + K_s2``:
      1. ``z_shared`` aligns across all views (with optional Delta-t attenuation) and is
         regularized to N(0, I_{K_sh}) across all active strata.
      2. ``z_s1_priv`` aligns across views for S1-observing samples (strata 0, 1, 2) and is
         regularized by Epps-Pulley SIGReg ONLY on S1-observing strata {0, 1, 2}. On S2-only
         samples (stratum 3), ``z_s1_priv`` is penalized toward zero energy instead of unit
         variance, avoiding hallucinated noise.
      3. ``z_s2_priv`` aligns across views for S2-observing samples (strata 0, 1, 3) and is
         regularized by Epps-Pulley SIGReg ONLY on S2-observing strata {0, 1, 3}. On S1-only
         samples (stratum 2), ``z_s2_priv`` is penalized toward zero energy, preserving
         optical-specific spectral features on S2 views without forcing S1 to guess them.
    """
    if z.ndim != 3:
        raise ValueError("z must have shape [V,N,K]")
    views, samples, features = z.shape
    if shared_dim <= 0 or s1_private_dim < 0 or s2_private_dim < 0:
        raise ValueError("shared_dim > 0 and private dims >= 0 required")
    if shared_dim + s1_private_dim + s2_private_dim != features:
        raise ValueError("shared_dim + s1_private_dim + s2_private_dim must equal K")
    if strata.ndim != 1 or strata.shape[0] != samples or strata.dtype != torch.int64:
        raise ValueError("strata must be an int64 tensor of shape [N]")
    if strata.numel() and (int(strata.min()) < 0 or int(strata.max()) >= 4):
        raise ValueError("strata must be in [0, 3]")
    if not 0.0 <= lam <= 1.0 or lam_priv_inactive < 0.0:
        raise ValueError("lam in [0, 1] and lam_priv_inactive >= 0 required")

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
        active_strata = strata.to(device=z.device)

        z_sh = z_compute[:, :, :shared_dim]
        z_s1 = z_compute[:, :, shared_dim : shared_dim + s1_private_dim]
        z_s2 = z_compute[:, :, shared_dim + s1_private_dim :]

        inv_sh = weighted_invariance_loss(
            z_sh,
            n_global,
            pair_gap_seconds=pair_gap_seconds,
            gap_tau_seconds=gap_tau_seconds,
        )

        # Shared SIGReg across present strata among {0, 1, 2, 3}
        present_sh = torch.unique(active_strata, sorted=True)
        remap_sh = torch.searchsorted(present_sh, active_strata)
        ep_sh, _ = ep_per_view_stratum(
            z_sh,
            remap_sh,
            int(present_sh.numel()),
            directions_shared,
            sum_reduce=sum_reduce,
            count_reduce=count_reduce,
            chunk=chunk,
            checkpoint=checkpoint,
        )
        sigreg_sh = ep_sh.mean()

        inv_priv_terms: list[Tensor] = []
        sigreg_priv_terms: list[Tensor] = []
        inactive_terms: list[Tensor] = []

        # S1 private block: active on strata {0, 1, 2}, inactive on stratum 3
        if s1_private_dim > 0:
            if directions_s1 is None or directions_s1.shape[0] != s1_private_dim:
                raise ValueError("directions_s1 must have shape [s1_private_dim, M]")
            s1_mask = active_strata != 3
            s1_inactive_mask = active_strata == 3
            if bool(s1_mask.any()):
                z_s1_active = z_s1[:, s1_mask, :]
                inv_priv_terms.append(invariance_loss(z_s1_active, n_global))
                s1_strata = active_strata[s1_mask]
                present_s1 = torch.unique(s1_strata, sorted=True)
                remap_s1 = torch.searchsorted(present_s1, s1_strata)
                ep_s1, _ = ep_per_view_stratum(
                    z_s1_active,
                    remap_s1,
                    int(present_s1.numel()),
                    directions_s1,
                    sum_reduce=sum_reduce,
                    count_reduce=count_reduce,
                    chunk=chunk,
                    checkpoint=checkpoint,
                )
                sigreg_priv_terms.append(ep_s1.mean())
            if bool(s1_inactive_mask.any()):
                inactive_terms.append(z_s1[:, s1_inactive_mask, :].square().mean())

        # S2 private block: active on strata {0, 1, 3}, inactive on stratum 2
        if s2_private_dim > 0:
            if directions_s2 is None or directions_s2.shape[0] != s2_private_dim:
                raise ValueError("directions_s2 must have shape [s2_private_dim, M]")
            s2_mask = active_strata != 2
            s2_inactive_mask = active_strata == 2
            if bool(s2_mask.any()):
                z_s2_active = z_s2[:, s2_mask, :]
                inv_priv_terms.append(invariance_loss(z_s2_active, n_global))
                s2_strata = active_strata[s2_mask]
                present_s2 = torch.unique(s2_strata, sorted=True)
                remap_s2 = torch.searchsorted(present_s2, s2_strata)
                ep_s2, _ = ep_per_view_stratum(
                    z_s2_active,
                    remap_s2,
                    int(present_s2.numel()),
                    directions_s2,
                    sum_reduce=sum_reduce,
                    count_reduce=count_reduce,
                    chunk=chunk,
                    checkpoint=checkpoint,
                )
                sigreg_priv_terms.append(ep_s2.mean())
            if bool(s2_inactive_mask.any()):
                inactive_terms.append(z_s2[:, s2_inactive_mask, :].square().mean())

        inv_priv = (
            torch.stack(inv_priv_terms).mean()
            if inv_priv_terms
            else torch.zeros((), dtype=compute_dtype, device=z.device)
        )
        sigreg_priv = (
            torch.stack(sigreg_priv_terms).mean()
            if sigreg_priv_terms
            else torch.zeros((), dtype=compute_dtype, device=z.device)
        )
        inactive_penalty = (
            torch.stack(inactive_terms).mean()
            if inactive_terms
            else torch.zeros((), dtype=compute_dtype, device=z.device)
        )

        inv_total = 0.5 * (inv_sh + inv_priv) if inv_priv_terms else inv_sh
        sigreg_total = 0.5 * (sigreg_sh + sigreg_priv) if sigreg_priv_terms else sigreg_sh
        loss = (
            (1.0 - lam) * inv_total
            + lam * sigreg_total
            + lam_priv_inactive * inactive_penalty
        )
        return {
            "loss": loss,
            "inv": inv_total,
            "inv_shared": inv_sh,
            "inv_private": inv_priv,
            "sigreg": sigreg_total,
            "sigreg_shared": sigreg_sh,
            "sigreg_private": sigreg_priv,
            "inactive_private_penalty": inactive_penalty,
            "null": torch.tensor(ep_null_expectation(), dtype=compute_dtype, device=z.device),
        }


def barlow_twins_loss(
    z: Tensor,
    *,
    lam_bt: float = 5e-3,
    eps: float = 1e-5,
    normalize_by_ceiling: bool = False,
    dist: DistContext | None = None,
) -> dict[str, Tensor]:
    """Distributed-exact Barlow Twins loss (TESSERA / TESSERA v2 comparator for Q1).

    Given ``z`` of shape ``[V, N, K]`` (with ``V >= 2``) or ``[2, N, K]``, normalizes the first
    two global views along the global batch dimension and computes the cross-correlation matrix
    ``C = (Z_A^T Z_B) / N_global`` in FP32/FP64 outside autocast.

    Per TESSERA v2 (arXiv:2607.03949, Appendix S.3 & T, Eq. 29), ``raw_bt_loss`` is the sum over K,
    ``loss`` is the per-coordinate mean (or ``raw_bt_loss / (K + lam_bt * K * (K - 1))`` when
    ``normalize_by_ceiling=True`` so runs with different projector widths K are comparable).
    """
    if z.ndim != 3 or z.shape[0] < 2:
        raise ValueError("z must have shape [V, N, K] with V >= 2")
    _, samples, features = z.shape
    if samples < 2 or features < 1:
        raise ValueError("barlow_twins_loss requires N >= 2 and K >= 1")
    if lam_bt < 0 or eps <= 0:
        raise ValueError("lam_bt >= 0 and eps > 0 required")

    compute_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    context = dist
    sum_reduce: Callable[[Tensor], Tensor] = (
        _identity if context is None else lambda value: all_reduce_sum_autograd(value, context)
    )
    count_reduce: Callable[[Tensor], Tensor] = (
        _identity if context is None else lambda value: all_reduce_sum_nograd(value, context)
    )

    with torch.autocast(device_type=z.device.type, enabled=False):
        za = z[0].to(dtype=compute_dtype)
        zb = z[1].to(dtype=compute_dtype)
        n_total = count_reduce(torch.tensor(float(samples), dtype=compute_dtype, device=z.device))

        sums = sum_reduce(torch.stack((za.sum(dim=0), zb.sum(dim=0))))
        mean_a, mean_b = sums[0] / n_total, sums[1] / n_total
        za_c = za - mean_a.unsqueeze(0)
        zb_c = zb - mean_b.unsqueeze(0)

        var_sums = sum_reduce(torch.stack((za_c.square().sum(dim=0), zb_c.square().sum(dim=0))))
        std_a = torch.sqrt(var_sums[0] / n_total + eps)
        std_b = torch.sqrt(var_sums[1] / n_total + eps)

        za_norm = za_c / std_a.unsqueeze(0)
        zb_norm = zb_c / std_b.unsqueeze(0)

        cross_cov = sum_reduce(za_norm.T @ zb_norm) / n_total
        diag = torch.diagonal(cross_cov)
        invariance_term = (1.0 - diag).square().sum()
        off_diag_sq = cross_cov.square().sum() - diag.square().sum()
        redundancy_term = lam_bt * off_diag_sq
        raw_bt = invariance_term + redundancy_term
        ceiling = float(features) + float(lam_bt) * float(features) * float(max(features - 1, 0))
        ceiling_norm_loss = raw_bt / max(ceiling, 1e-12)
        loss = ceiling_norm_loss if normalize_by_ceiling else raw_bt / features

        return {
            "loss": loss,
            "raw_bt_loss": raw_bt,
            "ceiling_normalized_loss": ceiling_norm_loss,
            "invariance_term": invariance_term / features,
            "redundancy_term": redundancy_term / features,
            "cross_correlation": cross_cov.detach(),
        }


def tessera_mixup_barlow_loss(
    z_a: Tensor,
    z_b: Tensor,
    z_mixed: Tensor,
    *,
    alpha_mix: float,
    perm: Tensor,
    lam_bt: float = 5e-3,
    lam_mix: float = 1.0,
    eps: float = 1e-5,
    normalize_by_ceiling: bool = False,
    dist: DistContext | None = None,
) -> dict[str, Tensor]:
    """TESSERA v1 / v2 Barlow Twins + Mix-Up linear consistency objective (arXiv:2607.03949, App. C & T).

    In TESSERA and TESSERA v2 (Eq. 29-31), Barlow Twins is paired with a mix-up regularizer that
    enforces linear consistency in cross-correlation space when input views are mixed as
    ``Y_M = alpha_mix * Y_A + (1 - alpha_mix) * Y_B[perm]``:
      C_target^{MA} = alpha_mix * (Z_A^T Z_A / B) + (1 - alpha_mix) * (Z_S^T Z_A / B)
      C_target^{MS} = alpha_mix * (Z_A^T Z_S / B) + (1 - alpha_mix) * (Z_S^T Z_S / B)
      L_MIX = 0.5 * ||C^{MA} - C_target^{MA}||_F^2 + 0.5 * ||C^{MS} - C_target^{MS}||_F^2
      L_total = L_BT + lam_mix * L_MIX
    """
    if z_a.ndim != 2 or z_b.ndim != 2 or z_mixed.ndim != 2:
        raise ValueError("z_a, z_b, and z_mixed must be 2D tensors [N, K]")
    if z_a.shape != z_b.shape or z_a.shape != z_mixed.shape:
        raise ValueError("z_a, z_b, and z_mixed must share shape [N, K]")
    samples, features = z_a.shape
    if samples < 2 or features < 1:
        raise ValueError("tessera_mixup_barlow_loss requires N >= 2 and K >= 1")
    if not 0.0 <= alpha_mix <= 1.0 or lam_mix < 0.0:
        raise ValueError("alpha_mix in [0, 1] and lam_mix >= 0 required")
    if perm.ndim != 1 or perm.shape[0] != samples:
        raise ValueError("perm must be a 1D index tensor of length N")

    bt = barlow_twins_loss(
        torch.stack((z_a, z_b), dim=0),
        lam_bt=lam_bt,
        eps=eps,
        normalize_by_ceiling=normalize_by_ceiling,
        dist=dist,
    )

    compute_dtype = torch.float64 if z_a.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=z_a.device.type, enabled=False):
        za = z_a.to(dtype=compute_dtype)
        zb = z_b.to(dtype=compute_dtype)
        zm = z_mixed.to(dtype=compute_dtype)

        def _bn(x: Tensor) -> Tensor:
            xc = x - x.mean(dim=0, keepdim=True)
            std = torch.sqrt(xc.square().mean(dim=0, keepdim=True) + eps)
            return xc / std

        za_n = _bn(za)
        zb_n = _bn(zb)
        zm_n = _bn(zm)
        zs_n = zb_n[perm.to(device=z_a.device, dtype=torch.long)]

        b_denom = float(samples)
        c_ma = (zm_n.T @ za_n) / b_denom
        c_ms = (zm_n.T @ zs_n) / b_denom
        c_aa = (za_n.T @ za_n) / b_denom
        c_sa = (zs_n.T @ za_n) / b_denom
        c_as = (za_n.T @ zs_n) / b_denom
        c_ss = (zs_n.T @ zs_n) / b_denom

        alpha = float(alpha_mix)
        target_ma = (alpha * c_aa + (1.0 - alpha) * c_sa).detach()
        target_ms = (alpha * c_as + (1.0 - alpha) * c_ss).detach()

        raw_mix = 0.5 * (
            (c_ma - target_ma).square().sum() + (c_ms - target_ms).square().sum()
        )
        scale = (
            float(features) + float(lam_bt) * float(features) * float(max(features - 1, 0))
            if normalize_by_ceiling
            else float(features)
        )
        mix_loss = raw_mix / max(scale, 1e-12)
        total = bt["loss"] + float(lam_mix) * mix_loss

        return {
            "loss": total,
            "bt_loss": bt["loss"],
            "mix_loss": mix_loss,
            "raw_bt_loss": bt["raw_bt_loss"],
            "raw_mix_loss": raw_mix,
            "ceiling_normalized_bt": bt["ceiling_normalized_loss"],
            "cross_correlation": bt["cross_correlation"],
        }


def vicreg_loss(
    z: Tensor,
    *,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
    dist: DistContext | None = None,
) -> dict[str, Tensor]:
    """Distributed-exact VICReg comparator loss (invariance + variance hinge + covariance)."""
    if z.ndim != 3 or z.shape[0] < 2:
        raise ValueError("z must have shape [V, N, K] with V >= 2")
    views, samples, features = z.shape
    if samples < 2 or features < 1:
        raise ValueError("vicreg_loss requires N >= 2 and K >= 1")

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
        sim_loss = invariance_loss(z_compute, n_global=2)
        n_total = count_reduce(torch.tensor(float(samples), dtype=compute_dtype, device=z.device))

        za, zb = z_compute[0], z_compute[1]
        sums = sum_reduce(torch.stack((za.sum(dim=0), zb.sum(dim=0))))
        za_c = za - (sums[0] / n_total).unsqueeze(0)
        zb_c = zb - (sums[1] / n_total).unsqueeze(0)

        var_sums = sum_reduce(torch.stack((za_c.square().sum(dim=0), zb_c.square().sum(dim=0))))
        std_a = torch.sqrt(var_sums[0] / (n_total - 1.0) + eps)
        std_b = torch.sqrt(var_sums[1] / (n_total - 1.0) + eps)
        std_loss = 0.5 * (functional.relu(gamma - std_a).mean() + functional.relu(gamma - std_b).mean())

        cov_ab = sum_reduce(torch.stack((za_c.T @ za_c, zb_c.T @ zb_c))) / (n_total - 1.0)
        cov_a, cov_b = cov_ab[0], cov_ab[1]
        off_a = cov_a.square().sum() - torch.diagonal(cov_a).square().sum()
        off_b = cov_b.square().sum() - torch.diagonal(cov_b).square().sum()
        cov_loss = (off_a + off_b) / features

        total = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
        return {
            "loss": total,
            "sim_loss": sim_loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss,
        }


def conditional_flow_matching_loss(
    velocity_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    x1: Tensor,
    cond: Tensor,
    *,
    sigma_min: float = 1e-4,
    t: Tensor | None = None,
    x0: Tensor | None = None,
) -> dict[str, Tensor]:
    """Optimal-Transport Conditional Flow Matching (OT-CFM / Rectified Flow) loss.

    Models a multimodal conditional distribution ``p(x_1 | cond)`` (such as optical S2 latents
    given radar S1 latents under 100% cloud cover, or multimodal future states/actions in WAM)
    without the mean-collapse of L2 JEPA regression:
      x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
      u_t = x_1 - (1 - sigma_min) * x_0
      L_CFM = E[ || v_theta(x_t, t, cond) - u_t ||_2^2 ]
    """
    if x1.ndim != 2 or cond.ndim != 2 or x1.shape[0] != cond.shape[0]:
        raise ValueError("x1 [B, D] and cond [B, C] must be 2D tensors with matching batch size")
    if sigma_min < 0.0 or sigma_min >= 1.0:
        raise ValueError("sigma_min must be in [0, 1)")

    batch, _ = x1.shape
    noise = torch.randn_like(x1) if x0 is None else x0.to(device=x1.device, dtype=x1.dtype)
    if noise.shape != x1.shape:
        raise ValueError("x0 must match x1 shape [B, D]")

    if t is None:
        tau = torch.rand((batch, 1), dtype=x1.dtype, device=x1.device)
    else:
        tau = t.to(device=x1.device, dtype=x1.dtype)
        if tau.ndim == 1:
            tau = tau.unsqueeze(1)
        if tau.shape != (batch, 1) or bool(((tau < 0.0) | (tau > 1.0)).any()):
            raise ValueError("t must have shape [B] or [B, 1] with values in [0, 1]")

    x_tau = (1.0 - (1.0 - sigma_min) * tau) * noise + tau * x1
    target_velocity = x1 - (1.0 - sigma_min) * noise
    pred_velocity = velocity_fn(x_tau, tau.squeeze(1), cond)
    if pred_velocity.shape != x1.shape:
        raise ValueError("velocity_fn must return a tensor matching x1 shape [B, D]")

    loss = (pred_velocity - target_velocity).square().mean()
    return {
        "loss": loss,
        "x_tau": x_tau.detach(),
        "target_velocity": target_velocity.detach(),
        "pred_velocity": pred_velocity.detach(),
    }


def sample_conditional_flow(
    velocity_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    cond: Tensor,
    *,
    dim: int,
    steps: int = 16,
    sigma_min: float = 1e-4,
    solver: Literal["euler", "heun"] = "heun",
    x0: Tensor | None = None,
    seed: int | None = None,
) -> Tensor:
    """Integrate the learned OT-CFM ODE from t=0 to t=1 using Euler or 2nd-order Heun."""
    if cond.ndim != 2 or dim <= 0 or steps <= 0:
        raise ValueError("cond must be 2D [B, C], dim > 0, and steps > 0")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'")
    batch = cond.shape[0]
    if x0 is not None:
        x = x0.to(device=cond.device, dtype=cond.dtype).clone()
        if x.shape != (batch, dim):
            raise ValueError("x0 must have shape [B, dim]")
    else:
        gen = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)
        x = torch.randn((batch, dim), generator=gen, dtype=cond.dtype, device="cpu").to(cond.device)

    dt = 1.0 / float(steps)
    for step in range(steps):
        t_cur = torch.full((batch,), step * dt, dtype=cond.dtype, device=cond.device)
        v1 = velocity_fn(x, t_cur, cond)
        if solver == "euler" or step == steps - 1:
            x = x + dt * v1
        else:
            x_euler = x + dt * v1
            t_next = torch.full((batch,), (step + 1) * dt, dtype=cond.dtype, device=cond.device)
            v2 = velocity_fn(x_euler, t_next, cond)
            x = x + 0.5 * dt * (v1 + v2)
    return x


class ConditionalFlowBridge(nn.Module):
    """Sinusoidal-time MLP velocity network for cross-modal (S1 -> S2) or WAM flow matching."""

    def __init__(
        self,
        *,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if target_dim <= 0 or cond_dim <= 0 or hidden_dim <= 0 or time_dim <= 0 or time_dim % 2 != 0:
            raise ValueError("positive dimensions and even time_dim required")
        torch.manual_seed(seed)
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(target_dim + cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def _time_embed(self, t: Tensor) -> Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=t.dtype, device=t.device) / max(half - 1, 1)
        )
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(self, x_tau: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self._time_embed(t.to(dtype=x_tau.dtype, device=x_tau.device))
        inp = torch.cat((x_tau, cond.to(dtype=x_tau.dtype, device=x_tau.device), t_emb), dim=-1)
        return self.net(inp)

    def loss(
        self,
        x1: Tensor,
        cond: Tensor,
        *,
        sigma_min: float = 1e-4,
        t: Tensor | None = None,
        x0: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return conditional_flow_matching_loss(
            self.forward, x1, cond, sigma_min=sigma_min, t=t, x0=x0
        )

    def sample(
        self,
        cond: Tensor,
        *,
        steps: int = 16,
        solver: Literal["euler", "heun"] = "heun",
        x0: Tensor | None = None,
        seed: int | None = None,
    ) -> Tensor:
        return sample_conditional_flow(
            self.forward,
            cond,
            dim=self.target_dim,
            steps=steps,
            solver=solver,
            x0=x0,
            seed=seed,
        )

