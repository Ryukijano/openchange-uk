"""Operational guarantees (Q1-Q4) and stratified split-conformal alert calibration for clean-JEPA."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor


def rankme_score(embeddings: Tensor, eps: float = 1e-7) -> float:
    """Compute the RankMe effective rank exp(-sum_i p_i log p_i) of singular values."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    samples, dim = embeddings.shape
    if samples < 1 or dim < 1:
        raise ValueError("embeddings must have positive dimensions")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be a positive finite number")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")

    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        z = embeddings.detach().to(dtype=torch.float64)
        singular_values = torch.linalg.svdvals(z)
        total = singular_values.sum()
        if float(total.item()) <= eps:
            return 1.0
        p = singular_values / total
        entropy = -(p * torch.log(p.clamp_min(eps))).sum()
        return float(torch.exp(entropy).item())


def lidar_score(embeddings: Tensor, eps: float = 1e-6) -> float:
    """Compute the LiDAR linear discriminant spectral surrogate across views [N, V, D]."""
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [N, V, D]")
    samples, views, dim = embeddings.shape
    if samples < 2 or views < 2 or dim < 1:
        raise ValueError("lidar_score requires N >= 2, V >= 2, and D >= 1")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be a positive finite number")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")

    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        z = embeddings.detach().to(dtype=torch.float64)
        class_means = z.mean(dim=1)
        grand_mean = class_means.mean(dim=0, keepdim=True)
        centered_between = class_means - grand_mean
        sigma_b = (centered_between.T @ centered_between) / (samples - 1)

        centered_within = z - class_means.unsqueeze(1)
        flat_within = centered_within.reshape(samples * views, dim)
        sigma_w = (flat_within.T @ flat_within) / (samples * (views - 1))

        evals_w, evecs_w = torch.linalg.eigh(sigma_w)
        inv_sqrt_w = (evals_w.clamp_min(0.0) + eps).rsqrt()
        whitening = (evecs_w * inv_sqrt_w.unsqueeze(0)) @ evecs_w.T
        discriminant = whitening @ sigma_b @ whitening
        discriminant = 0.5 * (discriminant + discriminant.T)
        eigvals = torch.linalg.eigvalsh(discriminant).clamp_min(0.0)
        total = eigvals.sum()
        if float(total.item()) <= eps:
            return 1.0
        p = eigvals / total
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
        return float(torch.exp(entropy).item())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    idx = 0
    n = len(values)
    while idx < n:
        end = idx + 1
        while end < n and values[order[end]] == values[order[idx]]:
            end += 1
        avg_rank = 0.5 * (idx + 1 + end)
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def selection_spearman(
    validation_losses: Sequence[float],
    probe_scores: Sequence[float],
    higher_probe_is_better: bool = True,
) -> dict[str, Any]:
    """Compute Spearman rank correlation between negative validation loss and probe utility."""
    losses = np.asarray(list(validation_losses), dtype=np.float64)
    probes = np.asarray(list(probe_scores), dtype=np.float64)
    if losses.ndim != 1 or probes.ndim != 1 or len(losses) != len(probes):
        raise ValueError("validation_losses and probe_scores must be 1D sequences of equal length")
    if len(losses) < 2:
        raise ValueError("at least two checkpoints are required to score selection")
    if not (np.isfinite(losses).all() and np.isfinite(probes).all()):
        raise ValueError("validation_losses and probe_scores must be finite")

    selector_scores = -losses
    probe_utility = probes if higher_probe_is_better else -probes

    rank_x = _average_ranks(selector_scores)
    rank_y = _average_ranks(probe_utility)
    dx = rank_x - rank_x.mean()
    dy = rank_y - rank_y.mean()
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    spearman = float(np.sum(dx * dy) / denom) if denom > 0.0 else 0.0

    selected_index = int(np.argmin(losses))
    oracle_index = int(np.argmax(probe_utility))
    selected_probe = float(probes[selected_index])
    oracle_probe = float(probes[oracle_index])
    top1_regret = float(probe_utility[oracle_index] - probe_utility[selected_index])

    return {
        "spearman": spearman,
        "spearman_rho": spearman,
        "raw_loss_spearman": -spearman,
        "top1_regret": top1_regret,
        "regret": top1_regret,
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "selected_probe": selected_probe,
        "oracle_probe": oracle_probe,
        "higher_probe_is_better": bool(higher_probe_is_better),
        "sign_convention": "negative_validation_loss",
    }


def fit_orthogonal_procrustes(anchor_source: Tensor, anchor_target: Tensor) -> dict[str, Any]:
    """Fit centered Orthogonal Procrustes alignment mapping anchor_source to anchor_target."""
    if anchor_source.ndim != 2 or anchor_target.ndim != 2:
        raise ValueError("anchor_source and anchor_target must be 2D tensors [N, D]")
    if anchor_source.shape != anchor_target.shape:
        raise ValueError("anchor_source and anchor_target must have identical shape [N, D]")
    samples, dim = anchor_source.shape
    if samples < 2 or dim < 1:
        raise ValueError("fit_orthogonal_procrustes requires N >= 2 and D >= 1")
    if not (torch.isfinite(anchor_source).all() and torch.isfinite(anchor_target).all()):
        raise ValueError("anchor matrices must contain only finite values")

    with torch.autocast(device_type=anchor_source.device.type, enabled=False):
        x = anchor_source.detach().to(dtype=torch.float64)
        y = anchor_target.detach().to(dtype=torch.float64, device=x.device)
        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)
        x_centered = x - mu_x
        y_centered = y - mu_y
        cross_cov = x_centered.T @ y_centered
        u, singular_values, vh = torch.linalg.svd(cross_cov, full_matrices=False)
        rotation = u @ vh
        eye = torch.eye(dim, dtype=torch.float64, device=x.device)
        orth_error = float(torch.linalg.matrix_norm(rotation.T @ rotation - eye, ord="fro").item())
        denom = float(torch.linalg.matrix_norm(y_centered, ord="fro").square().item())
        numer = float(torch.linalg.matrix_norm(x_centered @ rotation - y_centered, ord="fro").square().item())
        relative_residual = numer / max(denom, 1e-12)

    out_dtype = anchor_source.dtype if anchor_source.is_floating_point() else torch.float32
    return {
        "rotation": rotation.to(dtype=out_dtype),
        "source_mean": mu_x.to(dtype=out_dtype),
        "target_mean": mu_y.to(dtype=out_dtype),
        "singular_values": singular_values.to(dtype=out_dtype),
        "orthogonality_error": orth_error,
        "anchor_relative_residual": relative_residual,
    }


def apply_orthogonal_procrustes(features: Tensor, transform: dict[str, Any]) -> Tensor:
    """Apply a fitted Orthogonal Procrustes transform (X - mu_X) Q + mu_Y."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor [N, D]")
    rotation = transform["rotation"]
    source_mean = transform["source_mean"]
    target_mean = transform["target_mean"]
    if features.shape[1] != rotation.shape[0]:
        raise ValueError("feature dimension does not match Procrustes rotation")

    with torch.autocast(device_type=features.device.type, enabled=False):
        x = features.to(dtype=torch.float64)
        q = rotation.to(device=features.device, dtype=torch.float64)
        mu_x = source_mean.to(device=features.device, dtype=torch.float64)
        mu_y = target_mean.to(device=features.device, dtype=torch.float64)
        aligned = (x - mu_x) @ q + mu_y
    out_dtype = features.dtype if features.is_floating_point() else torch.float32
    return aligned.to(dtype=out_dtype)


def _predict_labels(head_fn: Callable[[Tensor], Tensor], features: Tensor) -> Tensor:
    with torch.no_grad():
        output = head_fn(features)
    if output.ndim == 2:
        return output.argmax(dim=-1).to(dtype=torch.int64)
    if output.ndim == 1:
        if output.is_floating_point():
            return (output >= 0.0).to(dtype=torch.int64)
        return output.to(dtype=torch.int64)
    raise ValueError("old_head_fn must return 1D predictions or 2D logits")


def evaluate_version_upgrade(
    old_head_fn: Callable[[Tensor], Tensor],
    anchor_old: Tensor,
    anchor_new: Tensor,
    eval_old: Tensor,
    eval_new: Tensor,
    eval_labels: Tensor,
) -> dict[str, Any]:
    """Evaluate zero-re-embedding model upgrade via Orthogonal Procrustes on held-out locations."""
    if eval_old.ndim != 2 or eval_new.ndim != 2 or eval_old.shape != eval_new.shape:
        raise ValueError("eval_old and eval_new must have identical 2D shape [N_eval, D]")
    labels = eval_labels.to(device=eval_old.device, dtype=torch.int64)
    if labels.ndim != 1 or labels.shape[0] != eval_old.shape[0]:
        raise ValueError("eval_labels must be a 1D tensor of length N_eval")

    transform = fit_orthogonal_procrustes(anchor_source=anchor_new, anchor_target=anchor_old)
    eval_aligned = apply_orthogonal_procrustes(eval_new, transform)

    pred_old = _predict_labels(old_head_fn, eval_old)
    pred_unaligned = _predict_labels(old_head_fn, eval_new)
    pred_aligned = _predict_labels(old_head_fn, eval_aligned)

    old_acc = float((pred_old == labels).to(dtype=torch.float64).mean().item())
    unaligned_acc = float((pred_unaligned == labels).to(dtype=torch.float64).mean().item())
    aligned_acc = float((pred_aligned == labels).to(dtype=torch.float64).mean().item())

    unaligned_churn = float((pred_unaligned != pred_old).to(dtype=torch.float64).mean().item())
    aligned_churn = float((pred_aligned != pred_old).to(dtype=torch.float64).mean().item())

    retained_fraction = aligned_acc / old_acc if old_acc > 0.0 else 0.0
    unaligned_retained_fraction = unaligned_acc / old_acc if old_acc > 0.0 else 0.0

    with torch.autocast(device_type=eval_old.device.type, enabled=False):
        diff = eval_aligned.to(torch.float64) - eval_old.to(torch.float64)
        centered_old = eval_old.to(torch.float64) - transform["target_mean"].to(
            device=eval_old.device, dtype=torch.float64
        )
        eval_rel_residual = float(
            (
                torch.linalg.matrix_norm(diff, ord="fro").square()
                / torch.linalg.matrix_norm(centered_old, ord="fro").square().clamp_min(1e-12)
            ).item()
        )

    return {
        "old_accuracy": old_acc,
        "unaligned_accuracy": unaligned_acc,
        "aligned_accuracy": aligned_acc,
        "unaligned_churn": unaligned_churn,
        "aligned_churn": aligned_churn,
        "retained_accuracy_fraction": retained_fraction,
        "unaligned_retained_accuracy_fraction": unaligned_retained_fraction,
        "eval_relative_residual": eval_rel_residual,
        "transform": transform,
    }


def fit_subspace_procrustes(
    anchor_source: Tensor,
    anchor_target: Tensor,
    *,
    shared_dim: int,
    s1_private_dim: int = 0,
    s2_private_dim: int = 0,
) -> dict[str, Any]:
    """Fit block-wise Orthogonal Procrustes alignment on shared vs. sensor-private subspaces.

    Under partial multi-sensor identifiability (``z = [z_shared, z_s1_priv, z_s2_priv]``),
    only ``z_shared`` is identifiable across sensors; fitting a single full-K orthogonal
    matrix mixes aligned shared directions with unalignable private subspaces (which have
    theoretical orthogonal residual floor ``2 * (K - K_sh) / K``).
    """
    if anchor_source.ndim != 2 or anchor_target.ndim != 2 or anchor_source.shape != anchor_target.shape:
        raise ValueError("anchor_source and anchor_target must share 2D shape [N, K]")
    samples, total_dim = anchor_source.shape
    if shared_dim <= 0 or s1_private_dim < 0 or s2_private_dim < 0:
        raise ValueError("shared_dim > 0 and private dims >= 0 required")
    if shared_dim + s1_private_dim + s2_private_dim != total_dim:
        raise ValueError("shared_dim + s1_private_dim + s2_private_dim must equal K")

    full_transform = fit_orthogonal_procrustes(anchor_source, anchor_target)
    shared_transform = fit_orthogonal_procrustes(
        anchor_source[:, :shared_dim], anchor_target[:, :shared_dim]
    )

    s1_transform = None
    if s1_private_dim > 0:
        s1_slice = slice(shared_dim, shared_dim + s1_private_dim)
        s1_transform = fit_orthogonal_procrustes(anchor_source[:, s1_slice], anchor_target[:, s1_slice])

    s2_transform = None
    if s2_private_dim > 0:
        s2_slice = slice(shared_dim + s1_private_dim, total_dim)
        s2_transform = fit_orthogonal_procrustes(anchor_source[:, s2_slice], anchor_target[:, s2_slice])

    theoretical_private_noise_floor = 2.0 * float(total_dim - shared_dim) / float(total_dim)
    return {
        "shared_dim": shared_dim,
        "s1_private_dim": s1_private_dim,
        "s2_private_dim": s2_private_dim,
        "shared_transform": shared_transform,
        "s1_transform": s1_transform,
        "s2_transform": s2_transform,
        "full_transform": full_transform,
        "shared_relative_residual": shared_transform["anchor_relative_residual"],
        "full_relative_residual": full_transform["anchor_relative_residual"],
        "theoretical_full_floor": theoretical_private_noise_floor,
    }


def apply_subspace_procrustes(
    features: Tensor,
    subspace_transform: dict[str, Any],
    *,
    zero_inactive_private: Literal["none", "s1", "s2", "both"] = "none",
) -> Tensor:
    """Apply block-wise Orthogonal Procrustes alignment to ``[z_shared, z_s1_priv, z_s2_priv]``."""
    if features.ndim != 2:
        raise ValueError("features must be 2D [N, K]")
    sh_dim = int(subspace_transform["shared_dim"])
    s1_dim = int(subspace_transform["s1_private_dim"])
    s2_dim = int(subspace_transform["s2_private_dim"])
    if features.shape[1] != sh_dim + s1_dim + s2_dim:
        raise ValueError("features width does not match subspace_transform")
    if zero_inactive_private not in ("none", "s1", "s2", "both"):
        raise ValueError("zero_inactive_private must be 'none', 's1', 's2', or 'both'")

    parts: list[Tensor] = [
        apply_orthogonal_procrustes(features[:, :sh_dim], subspace_transform["shared_transform"])
    ]
    if s1_dim > 0:
        s1_feat = features[:, sh_dim : sh_dim + s1_dim]
        if zero_inactive_private in ("s1", "both"):
            parts.append(torch.zeros_like(s1_feat))
        else:
            parts.append(apply_orthogonal_procrustes(s1_feat, subspace_transform["s1_transform"]))
    if s2_dim > 0:
        s2_feat = features[:, sh_dim + s1_dim :]
        if zero_inactive_private in ("s2", "both"):
            parts.append(torch.zeros_like(s2_feat))
        else:
            parts.append(apply_orthogonal_procrustes(s2_feat, subspace_transform["s2_transform"]))
    return torch.cat(parts, dim=1)


def evaluate_matryoshka_prefixes(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
    *,
    prefixes: Sequence[int] = (16, 32, 64, 128),
) -> dict[str, Any]:
    """Evaluate nested Matryoshka prefix representations (TESSERA v2 / Clean-JEPA prefix test).

    Fits a parameter-free nearest-class-centroid classifier on ``train_features[:, :k]``
    for each prefix ``k`` in ``prefixes`` and reports held-out accuracy and retention
    fraction relative to the full embedding width ``D``.
    """
    if train_features.ndim != 2 or eval_features.ndim != 2:
        raise ValueError("train_features and eval_features must be 2D tensors")
    if train_features.shape[1] != eval_features.shape[1]:
        raise ValueError("train and eval feature dimensions must match")
    dim = train_features.shape[1]
    valid_prefixes = sorted({int(k) for k in prefixes if 1 <= int(k) <= dim})
    if not valid_prefixes:
        raise ValueError("at least one prefix in [1, D] is required")
    if valid_prefixes[-1] != dim:
        valid_prefixes.append(dim)

    y_train = train_labels.to(device=train_features.device, dtype=torch.int64)
    y_eval = eval_labels.to(device=eval_features.device, dtype=torch.int64)
    classes = torch.unique(y_train, sorted=True)
    if classes.numel() < 2:
        raise ValueError("at least two classes are required in train_labels")

    prefix_accuracies: dict[int, float] = {}
    with torch.autocast(device_type=train_features.device.type, enabled=False):
        x_tr = train_features.detach().to(dtype=torch.float64)
        x_ev = eval_features.detach().to(device=x_tr.device, dtype=torch.float64)
        for k in valid_prefixes:
            tr_k = functional.normalize(x_tr[:, :k], dim=-1, eps=1e-12)
            ev_k = functional.normalize(x_ev[:, :k], dim=-1, eps=1e-12)
            centroids = torch.stack(
                [tr_k[y_train == c].mean(dim=0) for c in classes], dim=0
            )
            centroids = functional.normalize(centroids, dim=-1, eps=1e-12)
            sims = ev_k @ centroids.T
            preds = classes[sims.argmax(dim=-1)]
            acc = float((preds == y_eval).to(dtype=torch.float64).mean().item())
            prefix_accuracies[k] = acc

    full_acc = prefix_accuracies[dim]
    retention = {
        k: (acc / full_acc if full_acc > 0.0 else 0.0) for k, acc in prefix_accuracies.items()
    }
    return {
        "prefixes": valid_prefixes,
        "prefix_accuracies": prefix_accuracies,
        "prefix_retention": retention,
        "full_dim": dim,
        "full_accuracy": full_acc,
    }


def cross_unit_distance_ratio(
    unit_a: Tensor,
    unit_b: Tensor,
    same_unit_revisit: Tensor,
) -> dict[str, float]:
    """Compute ratio of cross-unit distance (e.g. S2A vs S2C) to same-unit revisit distance."""
    if unit_a.ndim != 2 or unit_b.ndim != 2 or unit_a.shape != unit_b.shape:
        raise ValueError("unit_a and unit_b must have identical 2D shape [N, D]")
    if unit_a.shape[0] < 1 or unit_a.shape[1] < 1:
        raise ValueError("unit_a and unit_b must be non-empty")
    if not (
        torch.isfinite(unit_a).all()
        and torch.isfinite(unit_b).all()
        and torch.isfinite(same_unit_revisit).all()
    ):
        raise ValueError("all input tensors must contain finite values")

    with torch.autocast(device_type=unit_a.device.type, enabled=False):
        a = unit_a.detach().to(dtype=torch.float64)
        b = unit_b.detach().to(device=a.device, dtype=torch.float64)
        revisit = same_unit_revisit.detach().to(device=a.device, dtype=torch.float64)

        if revisit.ndim == 2:
            if revisit.shape != a.shape:
                raise ValueError("2D same_unit_revisit must match unit_a shape [N, D]")
            rev_first, rev_second = a, revisit
        elif revisit.ndim == 3 and revisit.shape[1] == 2 and revisit.shape[2] == a.shape[1]:
            rev_first, rev_second = revisit[:, 0, :], revisit[:, 1, :]
        else:
            raise ValueError("same_unit_revisit must have shape [N, D] or [M, 2, D]")

        cross_l2 = torch.linalg.vector_norm(a - b, dim=-1)
        revisit_l2 = torch.linalg.vector_norm(rev_first - rev_second, dim=-1)
        mean_cross_l2 = float(cross_l2.mean().item())
        mean_revisit_l2 = float(revisit_l2.mean().item())
        if mean_revisit_l2 <= 1e-12:
            raise ValueError("same_unit_revisit distance is zero; ratio is undefined")

        cross_cos = 1.0 - functional.cosine_similarity(a, b, dim=-1, eps=1e-12)
        revisit_cos = 1.0 - functional.cosine_similarity(rev_first, rev_second, dim=-1, eps=1e-12)
        mean_cross_cos = float(cross_cos.mean().item())
        mean_revisit_cos = float(revisit_cos.mean().item())

        return {
            "l2_ratio": mean_cross_l2 / mean_revisit_l2,
            "cosine_ratio": mean_cross_cos / max(mean_revisit_cos, 1e-12),
            "mean_cross_l2": mean_cross_l2,
            "mean_revisit_l2": mean_revisit_l2,
            "mean_cross_cosine": mean_cross_cos,
            "mean_revisit_cosine": mean_revisit_cos,
        }


def latent_forecast_residuals(z_actual: Tensor, z_forecast: Tensor) -> Tensor:
    """Compute L2 forecast residuals ||z_actual - z_forecast||_2 along the feature axis."""
    if z_actual.shape != z_forecast.shape or z_actual.ndim < 2:
        raise ValueError("z_actual and z_forecast must share shape [..., D] with ndim >= 2")
    if not (torch.isfinite(z_actual).all() and torch.isfinite(z_forecast).all()):
        raise ValueError("z_actual and z_forecast must be finite")
    with torch.autocast(device_type=z_actual.device.type, enabled=False):
        diff = z_actual.to(dtype=torch.float64) - z_forecast.to(
            device=z_actual.device, dtype=torch.float64
        )
        norms = torch.linalg.vector_norm(diff, dim=-1)
    out_dtype = z_actual.dtype if z_actual.is_floating_point() else torch.float32
    return norms.to(dtype=out_dtype)


def _conformal_quantile(values: np.ndarray, alpha: float) -> tuple[float, float]:
    n_g = len(values)
    if n_g < 1:
        raise ValueError("each conformal group must contain at least one calibration residual")
    rank_1based = math.ceil((n_g + 1) * (1.0 - alpha))
    effective_level = rank_1based / n_g
    clamped_idx = min(n_g, max(1, rank_1based)) - 1
    ordered = np.sort(values, kind="mergesort")
    return float(ordered[clamped_idx]), float(effective_level)


class StratifiedConformalAlertCalibrator:
    """Split-conformal latent forecast residual alert calibrator stratified by (season, land_cover)."""

    def __init__(self, alpha: float = 0.1, allow_global_fallback: bool = False) -> None:
        if not (0.0 < alpha < 1.0) or not math.isfinite(alpha):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.allow_global_fallback = bool(allow_global_fallback)
        self.group_thresholds: dict[tuple[str, str], float] = {}
        self.group_counts: dict[tuple[str, str], int] = {}
        self.group_effective_levels: dict[tuple[str, str], float] = {}
        self._group_sorted_residuals: dict[tuple[str, str], np.ndarray] = {}
        self.global_threshold: float | None = None
        self._global_sorted_residuals: np.ndarray | None = None

    def fit(
        self,
        z_actual: Tensor | Sequence[float],
        z_forecast: Tensor | None = None,
        groups: Sequence[tuple[str, str]] = (),
    ) -> dict[tuple[str, str], float]:
        """Calibrate finite-sample (1 - alpha) residual quantiles per (season, land_cover) group."""
        if z_forecast is None:
            residuals_tensor = torch.as_tensor(z_actual, dtype=torch.float64)
            if residuals_tensor.ndim != 1:
                raise ValueError("when z_forecast is None, z_actual must be 1D precomputed residuals")
        else:
            actual_tensor = torch.as_tensor(z_actual)
            forecast_tensor = torch.as_tensor(z_forecast)
            if actual_tensor.ndim != 2 or forecast_tensor.ndim != 2:
                raise ValueError("calibration z_actual and z_forecast must have shape [N, D]")
            residuals_tensor = latent_forecast_residuals(actual_tensor, forecast_tensor).to(
                dtype=torch.float64
            )

        residuals = residuals_tensor.detach().cpu().numpy().astype(np.float64)
        if len(residuals) < 1 or len(residuals) != len(groups):
            raise ValueError("groups must have the same non-zero length as calibration samples")
        if not np.isfinite(residuals).all() or np.any(residuals < 0.0):
            raise ValueError("calibration residuals must be finite and non-negative")

        bucketed: dict[tuple[str, str], list[float]] = {}
        for residual, raw_group in zip(residuals, groups):
            if not isinstance(raw_group, tuple) or len(raw_group) != 2:
                raise ValueError("each group must be a (season, land_cover) 2-tuple")
            group = (str(raw_group[0]), str(raw_group[1]))
            bucketed.setdefault(group, []).append(float(residual))

        self.group_thresholds.clear()
        self.group_counts.clear()
        self.group_effective_levels.clear()
        self._group_sorted_residuals.clear()

        for group, group_values in sorted(bucketed.items()):
            arr = np.sort(np.asarray(group_values, dtype=np.float64), kind="mergesort")
            threshold, effective_level = _conformal_quantile(arr, self.alpha)
            self.group_thresholds[group] = threshold
            self.group_counts[group] = len(arr)
            self.group_effective_levels[group] = effective_level
            self._group_sorted_residuals[group] = arr

        all_sorted = np.sort(residuals, kind="mergesort")
        self.global_threshold, _ = _conformal_quantile(all_sorted, self.alpha)
        self._global_sorted_residuals = all_sorted
        return dict(self.group_thresholds)

    calibrate = fit

    def threshold_for(self, group: tuple[str, str]) -> float:
        if not self.group_thresholds:
            raise RuntimeError("calibrator has not been fitted")
        key = (str(group[0]), str(group[1]))
        if key in self.group_thresholds:
            return self.group_thresholds[key]
        if self.allow_global_fallback and self.global_threshold is not None:
            return self.global_threshold
        raise KeyError(f"uncalibrated (season, land_cover) group: {key}")

    def _resolve_step_groups(
        self,
        groups: Sequence[tuple[str, str]] | Sequence[Sequence[tuple[str, str]]],
        streams: int,
        steps: int,
    ) -> list[list[tuple[str, str]]]:
        if len(groups) != streams:
            raise ValueError("groups length must equal number of streams S")
        resolved: list[list[tuple[str, str]]] = []
        for entry in groups:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and all(isinstance(item, str) for item in entry)
            ):
                resolved.append([(entry[0], entry[1])] * steps)
            else:
                step_list = list(entry)  # type: ignore[arg-type]
                if len(step_list) != steps:
                    raise ValueError("per-step group sequence must have length T")
                resolved.append([(str(g[0]), str(g[1])) for g in step_list])
        return resolved

    def predict_streams(
        self,
        z_actual: Tensor,
        z_forecast: Tensor | None,
        groups: Sequence[tuple[str, str]] | Sequence[Sequence[tuple[str, str]]],
    ) -> dict[str, Tensor]:
        """Compute residuals, group thresholds, conformal p-values, and binary alerts on [S, T, D]."""
        if not self.group_thresholds:
            raise RuntimeError("calibrator has not been fitted")
        if z_forecast is None:
            if z_actual.ndim != 2:
                raise ValueError("precomputed stream residuals must have shape [S, T]")
            residuals = z_actual.detach().to(dtype=torch.float64)
        else:
            if z_actual.ndim != 3 or z_forecast.ndim != 3:
                raise ValueError("stream z_actual and z_forecast must have shape [S, T, D]")
            residuals = latent_forecast_residuals(z_actual, z_forecast).to(dtype=torch.float64)

        streams, steps = residuals.shape
        step_groups = self._resolve_step_groups(groups, streams, steps)
        thresholds = torch.empty((streams, steps), dtype=torch.float64, device=residuals.device)
        p_values = torch.empty((streams, steps), dtype=torch.float64, device=residuals.device)

        residuals_cpu = residuals.detach().cpu().numpy()
        for s_idx in range(streams):
            for t_idx in range(steps):
                group = step_groups[s_idx][t_idx]
                thresh = self.threshold_for(group)
                thresholds[s_idx, t_idx] = thresh
                cal_arr = self._group_sorted_residuals.get(group, self._global_sorted_residuals)
                assert cal_arr is not None
                exceed_or_equal = int(np.sum(cal_arr >= residuals_cpu[s_idx, t_idx]))
                p_values[s_idx, t_idx] = (1.0 + exceed_or_equal) / (len(cal_arr) + 1.0)

        alerts = residuals > thresholds
        return {
            "residuals": residuals,
            "thresholds": thresholds,
            "p_values": p_values,
            "alerts": alerts,
        }

    def evaluate_streams(
        self,
        z_actual: Tensor,
        z_forecast: Tensor | None,
        groups: Sequence[tuple[str, str]] | Sequence[Sequence[tuple[str, str]]],
        onset_steps: Sequence[int | None],
        *,
        reference_steps: Sequence[int | None] | None = None,
        s1_only_mask: Tensor | None = None,
        action_cost: float = 1.0,
        missed_loss: float = 5.0,
    ) -> dict[str, Any]:
        """Evaluate false-alarm rate, detection lead time, and cost-loss economic value on test streams."""
        if action_cost <= 0.0 or missed_loss <= action_cost:
            raise ValueError("require 0 < action_cost < missed_loss for cost-loss evaluation")

        pred = self.predict_streams(z_actual, z_forecast, groups)
        alerts = pred["alerts"]
        streams, steps = alerts.shape
        if len(onset_steps) != streams:
            raise ValueError("onset_steps must have length S")
        if reference_steps is not None and len(reference_steps) != streams:
            raise ValueError("reference_steps must have length S")

        if s1_only_mask is not None:
            if s1_only_mask.shape != (streams, steps):
                raise ValueError("s1_only_mask must have shape [S, T]")
            s1_mask_bool = s1_only_mask.to(device=alerts.device, dtype=torch.bool)
        else:
            s1_mask_bool = torch.zeros_like(alerts, dtype=torch.bool)

        stable_mask = torch.zeros((streams, steps), dtype=torch.bool, device=alerts.device)
        lead_times: list[float] = []
        detection_delays: list[float] = []
        s1_first_alerts: list[float] = []
        positive_streams = 0
        detected_streams = 0
        negative_streams = 0
        false_alarm_streams = 0

        for s_idx, raw_onset in enumerate(onset_steps):
            has_onset = raw_onset is not None and int(raw_onset) >= 0
            if not has_onset:
                stable_mask[s_idx, :] = True
                negative_streams += 1
                if bool(alerts[s_idx].any().item()):
                    false_alarm_streams += 1
                continue

            onset = int(raw_onset)  # type: ignore[arg-type]
            if onset >= steps:
                raise ValueError("onset_step must be in [0, T)")
            if onset > 0:
                stable_mask[s_idx, :onset] = True
            positive_streams += 1

            ref_step = (
                steps - 1
                if reference_steps is None or reference_steps[s_idx] is None
                else int(reference_steps[s_idx])  # type: ignore[arg-type]
            )
            if ref_step < onset or ref_step >= steps:
                raise ValueError("reference_step must satisfy onset_step <= reference_step < T")

            post_onset_alerts = torch.nonzero(alerts[s_idx, onset:], as_tuple=False).flatten()
            if post_onset_alerts.numel() > 0:
                first_alert_step = onset + int(post_onset_alerts[0].item())
                detected_streams += 1
                lead_times.append(float(max(0, ref_step - first_alert_step)))
                detection_delays.append(float(first_alert_step - onset))
                s1_first_alerts.append(float(bool(s1_mask_bool[s_idx, first_alert_step].item())))

        stable_count = int(stable_mask.sum().item())
        step_far = (
            float(alerts[stable_mask].to(dtype=torch.float64).mean().item())
            if stable_count > 0
            else 0.0
        )
        stable_s1_mask = stable_mask & s1_mask_bool
        stable_clear_mask = stable_mask & (~s1_mask_bool)
        s1_far = (
            float(alerts[stable_s1_mask].to(dtype=torch.float64).mean().item())
            if int(stable_s1_mask.sum().item()) > 0
            else 0.0
        )
        clear_far = (
            float(alerts[stable_clear_mask].to(dtype=torch.float64).mean().item())
            if int(stable_clear_mask.sum().item()) > 0
            else 0.0
        )

        stream_tpr = detected_streams / positive_streams if positive_streams > 0 else 0.0
        stream_fpr = false_alarm_streams / negative_streams if negative_streams > 0 else step_far
        prevalence = positive_streams / streams if streams > 0 else 0.0

        expected_expense = (
            prevalence * (stream_tpr * action_cost + (1.0 - stream_tpr) * missed_loss)
            + (1.0 - prevalence) * stream_fpr * action_cost
        )
        baseline_expense = min(action_cost, prevalence * missed_loss)
        oracle_expense = prevalence * action_cost
        denom = baseline_expense - oracle_expense
        cost_loss_value = (
            (baseline_expense - expected_expense) / denom if denom > 1e-12 else 0.0
        )

        return {
            "alpha": self.alpha,
            "false_alarm_rate": step_far,
            "false_alarm_rate_s1_only": s1_far,
            "false_alarm_rate_clear": clear_far,
            "stream_false_alarm_rate": stream_fpr,
            "detection_rate": stream_tpr,
            "mean_detection_lead_time": float(np.mean(lead_times)) if lead_times else 0.0,
            "mean_detection_delay": float(np.mean(detection_delays)) if detection_delays else 0.0,
            "s1_only_first_alert_fraction": (
                float(np.mean(s1_first_alerts)) if s1_first_alerts else 0.0
            ),
            "cost_loss_value": float(cost_loss_value),
            "net_cost_savings": float(baseline_expense - expected_expense),
            "expected_expense": float(expected_expense),
            "baseline_expense": float(baseline_expense),
            "oracle_expense": float(oracle_expense),
            "alerts": alerts,
            "thresholds": pred["thresholds"],
            "residuals": pred["residuals"],
            "p_values": pred["p_values"],
        }


def score_observation_reveal_policies(
    stop_losses: Tensor | Sequence[float],
    s1_losses: Tensor | Sequence[float],
    s2_losses: Tensor | Sequence[float],
    uncertainties: Tensor | Sequence[float],
    *,
    cloud_fractions: Tensor | Sequence[float] | None = None,
    cost_s1: float = 0.2,
    cost_s2: float = 1.0,
    budget_per_step: float | None = None,
    cost_weight: float = 0.1,
    s1_gain_factor: float = 0.7,
    s2_gain_factor: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Evaluate active observation reveal policies (stop, always_s1, always_s2, uncertainty_voi)."""
    loss_stop = np.asarray(
        stop_losses.detach().cpu().numpy() if isinstance(stop_losses, Tensor) else list(stop_losses),
        dtype=np.float64,
    )
    loss_s1 = np.asarray(
        s1_losses.detach().cpu().numpy() if isinstance(s1_losses, Tensor) else list(s1_losses),
        dtype=np.float64,
    )
    loss_s2 = np.asarray(
        s2_losses.detach().cpu().numpy() if isinstance(s2_losses, Tensor) else list(s2_losses),
        dtype=np.float64,
    )
    unc = np.asarray(
        uncertainties.detach().cpu().numpy()
        if isinstance(uncertainties, Tensor)
        else list(uncertainties),
        dtype=np.float64,
    )
    n = len(loss_stop)
    if n < 1 or any(len(arr) != n for arr in (loss_s1, loss_s2, unc)):
        raise ValueError("stop_losses, s1_losses, s2_losses, and uncertainties must share length N >= 1")
    if cloud_fractions is None:
        clouds = np.zeros(n, dtype=np.float64)
    else:
        clouds = np.asarray(
            cloud_fractions.detach().cpu().numpy()
            if isinstance(cloud_fractions, Tensor)
            else list(cloud_fractions),
            dtype=np.float64,
        )
        if len(clouds) != n:
            raise ValueError("cloud_fractions must have length N")
    if not (
        np.isfinite(loss_stop).all()
        and np.isfinite(loss_s1).all()
        and np.isfinite(loss_s2).all()
        and np.isfinite(unc).all()
        and np.isfinite(clouds).all()
    ):
        raise ValueError("all policy inputs must be finite")
    if np.any(unc < 0.0) or np.any((clouds < 0.0) | (clouds > 1.0)):
        raise ValueError("uncertainties must be >= 0 and cloud_fractions must be in [0, 1]")
    if cost_s1 <= 0.0 or cost_s2 < cost_s1:
        raise ValueError("require 0 < cost_s1 <= cost_s2")
    if budget_per_step is not None and budget_per_step < 0.0:
        raise ValueError("budget_per_step must be non-negative")

    total_budget = float("inf") if budget_per_step is None else float(budget_per_step * n)
    costs = {"stop": 0.0, "s1": float(cost_s1), "s2": float(cost_s2)}
    loss_table = {"stop": loss_stop, "s1": loss_s1, "s2": loss_s2}

    actions_stop = ["stop"] * n

    actions_s1: list[str] = []
    spent_s1 = 0.0
    for _ in range(n):
        if spent_s1 + cost_s1 <= total_budget + 1e-12:
            actions_s1.append("s1")
            spent_s1 += cost_s1
        else:
            actions_s1.append("stop")

    actions_s2: list[str] = []
    spent_s2 = 0.0
    for _ in range(n):
        if spent_s2 + cost_s2 <= total_budget + 1e-12:
            actions_s2.append("s2")
            spent_s2 += cost_s2
        else:
            actions_s2.append("stop")

    expected_gain = {
        "stop": np.zeros(n, dtype=np.float64),
        "s1": unc * float(s1_gain_factor),
        "s2": unc * (1.0 - clouds) * float(s2_gain_factor),
    }
    actions_voi = ["stop"] * n
    spent_voi = 0.0
    while True:
        best_ratio = 0.0
        best_step = -1
        best_target = "stop"
        for idx in range(n):
            current = actions_voi[idx]
            cur_cost = costs[current]
            cur_gain = float(expected_gain[current][idx])
            for candidate in ("s1", "s2"):
                if candidate == current:
                    continue
                extra_cost = costs[candidate] - cur_cost
                if extra_cost <= 0.0 or spent_voi + extra_cost > total_budget + 1e-12:
                    continue
                extra_gain = float(expected_gain[candidate][idx]) - cur_gain
                net_improvement = extra_gain - cost_weight * extra_cost
                if net_improvement <= 1e-12:
                    continue
                ratio = extra_gain / extra_cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_step = idx
                    best_target = candidate
        if best_step < 0:
            break
        spent_voi += costs[best_target] - costs[actions_voi[best_step]]
        actions_voi[best_step] = best_target

    mean_stop_loss = float(np.mean(loss_stop))
    results: dict[str, dict[str, Any]] = {}
    for name, chosen in (
        ("stop", actions_stop),
        ("always_s1", actions_s1),
        ("always_s2", actions_s2),
        ("uncertainty_voi", actions_voi),
    ):
        realized_losses = np.array(
            [loss_table[action][idx] for idx, action in enumerate(chosen)],
            dtype=np.float64,
        )
        realized_costs = np.array([costs[action] for action in chosen], dtype=np.float64)
        mean_loss = float(np.mean(realized_losses))
        mean_cost = float(np.mean(realized_costs))
        total_cost = float(np.sum(realized_costs))
        loss_reduction = mean_stop_loss - mean_loss
        net_value = loss_reduction - cost_weight * mean_cost
        results[name] = {
            "mean_loss": mean_loss,
            "mean_cost": mean_cost,
            "total_cost": total_cost,
            "within_budget": bool(total_cost <= total_budget + 1e-9),
            "loss_reduction": float(loss_reduction),
            "net_value": float(net_value),
            "action_counts": {
                "stop": int(sum(a == "stop" for a in chosen)),
                "s1": int(sum(a == "s1" for a in chosen)),
                "s2": int(sum(a == "s2" for a in chosen)),
            },
            "actions": list(chosen),
        }
    return results
