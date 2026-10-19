"""Object-oriented subspace-aware data assimilation and Value-of-Information (VoI) controller.

Implements SDA-style guided subspace assimilation across the factorized
Clean-JEPA representation ``z = [z_sh, z_s1, z_s2]`` and batched GPU
counterfactual rollout evaluation for the archive-reveal active observation game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from openchange.jepa.guarantees import (
    StratifiedConformalAlertCalibrator,
    gaussian_crps,
    score_observation_reveal_policies,
)

if TYPE_CHECKING:
    from openchange.jepa.science_wm import FactorizedSpatiotemporalJEPAWorldModel


@dataclass(frozen=True)
class SubspacePartition:
    """Factorized Clean-JEPA subspace layout ``z = [z_sh, z_s1, z_s2]``."""

    total_dim: int
    shared_dim: int
    s1_private_dim: int = 0
    s2_private_dim: int = 0

    def __post_init__(self) -> None:
        if self.total_dim <= 0 or self.shared_dim <= 0:
            raise ValueError("total_dim and shared_dim must be positive")
        if self.s1_private_dim < 0 or self.s2_private_dim < 0:
            raise ValueError("private dimensions must be non-negative")
        if self.shared_dim + self.s1_private_dim + self.s2_private_dim != self.total_dim:
            raise ValueError(
                "shared_dim + s1_private_dim + s2_private_dim must equal total_dim"
            )

    @classmethod
    def resolve(
        cls,
        total_dim: int,
        *,
        shared_dim: int | None = None,
        s1_private_dim: int = 0,
        s2_private_dim: int = 0,
    ) -> SubspacePartition:
        resolved_shared = (
            (total_dim - s1_private_dim - s2_private_dim)
            if shared_dim is None
            else int(shared_dim)
        )
        return cls(
            total_dim=total_dim,
            shared_dim=resolved_shared,
            s1_private_dim=s1_private_dim,
            s2_private_dim=s2_private_dim,
        )

    @property
    def shared_slice(self) -> slice:
        return slice(0, self.shared_dim)

    @property
    def s1_slice(self) -> slice:
        return slice(self.shared_dim, self.shared_dim + self.s1_private_dim)

    @property
    def s2_slice(self) -> slice:
        return slice(self.shared_dim + self.s1_private_dim, self.total_dim)


@dataclass(frozen=True)
class CandidateSensorPass:
    """Typed representation of a candidate archived/taskable satellite pass."""

    candidate_index: int
    sensor: str
    modality: Literal["s1", "s2", "joint"]
    z_observation: Tensor
    forcing_observation: Tensor
    cost_mb: float = 1.0
    cloud_fraction: float = 0.0
    gap_seconds: float = 0.0
    assimilation_gain: float | None = None

    @staticmethod
    def infer_modality(raw_modality: Any, sensor_name: str) -> Literal["s1", "s2", "joint"]:
        if isinstance(raw_modality, str):
            low = raw_modality.lower()
            if low in ("s1", "s2", "joint"):
                return low  # type: ignore[return-value]
        low_sensor = sensor_name.lower()
        if "s1" in low_sensor or "sar" in low_sensor:
            return "s1"
        if "s2" in low_sensor or "optical" in low_sensor:
            return "s2"
        return "joint"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int) -> CandidateSensorPass:
        z_obs = raw.get("z_observation")
        f_obs = raw.get("forcing_observation")
        if not isinstance(z_obs, Tensor):
            raise ValueError(f"candidate {index} z_observation must be a Tensor")
        if not isinstance(f_obs, Tensor):
            raise ValueError(f"candidate {index} forcing_observation must be a Tensor")
        cost_mb = float(raw.get("cost_mb", 1.0))
        if cost_mb <= 0.0:
            raise ValueError("cost_mb must be positive")
        cloud_fraction = float(raw.get("cloud_fraction", 0.0))
        if not (0.0 <= cloud_fraction <= 1.0):
            raise ValueError("cloud_fraction must be in [0, 1]")
        gap_seconds = float(raw.get("gap_seconds", 0.0))
        if gap_seconds < 0.0:
            raise ValueError("gap_seconds must be non-negative")
        sensor = str(raw.get("sensor", f"candidate_{index}"))
        modality = cls.infer_modality(raw.get("modality"), sensor)
        gain_raw = raw.get("assimilation_gain")
        gain = None if gain_raw is None else float(gain_raw)
        return cls(
            candidate_index=index,
            sensor=sensor,
            modality=modality,
            z_observation=z_obs,
            forcing_observation=f_obs,
            cost_mb=cost_mb,
            cloud_fraction=cloud_fraction,
            gap_seconds=gap_seconds,
            assimilation_gain=gain,
        )


class SubspaceDataAssimilator(nn.Module):
    """Object-oriented SDA-style subspace assimilator over ``z = [z_sh, z_s1, z_s2]``.

    Ensures an all-weather Sentinel-1 SAR pass updates ``[z_sh, z_s1]`` without
    overwriting optical-private memory ``z_s2`` (which is carried forward from
    ``z_prior`` with exponential staleness decay ``exp(-Delta t / tau)``), while a
    cloudy Sentinel-2 pass attenuates its optical update by ``(1 - cloud_fraction)``.
    """

    def __init__(
        self,
        partition: SubspacePartition,
        *,
        default_gain: float = 1.0,
        gap_tau_seconds: float = 604800.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= default_gain <= 1.0):
            raise ValueError("default_gain must be in [0, 1]")
        if gap_tau_seconds <= 0.0:
            raise ValueError("gap_tau_seconds must be positive")
        self.partition = partition
        self.default_gain = float(default_gain)
        self.gap_tau_seconds = float(gap_tau_seconds)

    def assimilate(
        self,
        z_prior: Tensor,
        z_obs: Tensor,
        *,
        modality: Literal["s1", "s2", "joint"] = "joint",
        assimilation_gain: float | None = None,
        cloud_fraction: float = 0.0,
        gap_seconds: float = 0.0,
    ) -> Tensor:
        if z_prior.shape != z_obs.shape or z_prior.ndim < 2:
            raise ValueError("z_prior and z_obs must share shape [..., D]")
        if z_prior.shape[-1] != self.partition.total_dim:
            raise ValueError("last dimension of z_prior must match partition.total_dim")
        if modality not in ("s1", "s2", "joint"):
            raise ValueError("modality must be 's1', 's2', or 'joint'")
        gain = self.default_gain if assimilation_gain is None else float(assimilation_gain)
        if not (0.0 <= gain <= 1.0):
            raise ValueError("assimilation_gain must be in [0, 1]")
        if not (0.0 <= cloud_fraction <= 1.0):
            raise ValueError("cloud_fraction must be in [0, 1]")
        if gap_seconds < 0.0:
            raise ValueError("gap_seconds must be non-negative")

        obs_cast = z_obs.to(device=z_prior.device, dtype=z_prior.dtype)
        decay = math.exp(-float(gap_seconds) / self.gap_tau_seconds)

        sh = self.partition.shared_slice
        s1 = self.partition.s1_slice
        s2 = self.partition.s2_slice

        out = z_prior.clone()
        if modality == "s1":
            out[..., sh] = (1.0 - gain) * z_prior[..., sh] + gain * obs_cast[..., sh]
            if self.partition.s1_private_dim > 0:
                out[..., s1] = (1.0 - gain) * z_prior[..., s1] + gain * obs_cast[..., s1]
            if self.partition.s2_private_dim > 0:
                out[..., s2] = decay * z_prior[..., s2]
        elif modality == "s2":
            clear_gain = gain * (1.0 - float(cloud_fraction))
            out[..., sh] = (1.0 - clear_gain) * z_prior[..., sh] + clear_gain * obs_cast[..., sh]
            if self.partition.s1_private_dim > 0:
                out[..., s1] = decay * z_prior[..., s1]
            if self.partition.s2_private_dim > 0:
                out[..., s2] = (1.0 - clear_gain) * z_prior[..., s2] + clear_gain * obs_cast[..., s2]
        else:
            clear_gain = gain * (1.0 - float(cloud_fraction))
            out[..., sh] = (1.0 - gain) * z_prior[..., sh] + gain * obs_cast[..., sh]
            if self.partition.s1_private_dim > 0:
                out[..., s1] = (1.0 - gain) * z_prior[..., s1] + gain * obs_cast[..., s1]
            if self.partition.s2_private_dim > 0:
                out[..., s2] = (1.0 - clear_gain) * z_prior[..., s2] + clear_gain * obs_cast[..., s2]

        return out

    def forward(
        self,
        z_prior: Tensor,
        z_obs: Tensor,
        *,
        modality: Literal["s1", "s2", "joint"] = "joint",
        assimilation_gain: float | None = None,
        cloud_fraction: float = 0.0,
        gap_seconds: float = 0.0,
    ) -> Tensor:
        return self.assimilate(
            z_prior,
            z_obs,
            modality=modality,
            assimilation_gain=assimilation_gain,
            cloud_fraction=cloud_fraction,
            gap_seconds=gap_seconds,
        )


class CounterfactualVoIEvaluator:
    """Object-oriented Value-of-Information (VoI) evaluator for active sensor reveal.

    Executes the baseline trajectory and all ``C`` candidate passes in a single
    batched ``[(C + 1) * B]`` GPU rollout, assimilates observations via
    ``SubspaceDataAssimilator``, and optionally scores policies against withheld
    future truth via ``score_observation_reveal_policies``.
    """

    def __init__(
        self,
        model: FactorizedSpatiotemporalJEPAWorldModel,
        *,
        assimilator: SubspaceDataAssimilator | None = None,
        conformal_quantile: float = 1.645,
        calibrator: StratifiedConformalAlertCalibrator | None = None,
        calibrator_group: tuple[str, str] | None = None,
    ) -> None:
        if conformal_quantile <= 0.0:
            raise ValueError("conformal_quantile must be positive")
        self.model = model
        self.assimilator = (
            assimilator
            if assimilator is not None
            else SubspaceDataAssimilator(SubspacePartition.resolve(model.dim))
        )
        self.conformal_quantile = float(conformal_quantile)
        self.calibrator = calibrator
        self.calibrator_group = calibrator_group

    def resolve_conformal_quantile(self) -> float:
        if self.calibrator is None:
            return self.conformal_quantile
        if self.calibrator_group is not None:
            return float(self.calibrator.threshold_for(self.calibrator_group))
        if self.calibrator.global_threshold is not None:
            return float(self.calibrator.global_threshold)
        raise ValueError("calibrator_group required when calibrator has no global fallback")

    def evaluate(
        self,
        z_context: Tensor,
        candidate_observations: list[dict[str, Any] | CandidateSensorPass],
        forcing_context: Tensor,
        forcing_future: Tensor,
        *,
        target_future: Tensor | None = None,
        target_observable: Tensor | None = None,
        cost_s1: float = 0.2,
        cost_s2: float = 1.0,
        budget_per_step: float | None = None,
        cost_weight: float = 0.1,
    ) -> dict[str, Any]:
        if not candidate_observations:
            raise ValueError("candidate_observations must be non-empty")

        effective_quantile = self.resolve_conformal_quantile()
        batch = z_context.shape[0]
        n_cands = len(candidate_observations)
        z_prior = z_context[:, -1:, :, :]

        z_batch_list: list[Tensor] = [z_context]
        f_batch_list: list[Tensor] = [forcing_context]
        parsed_candidates: list[CandidateSensorPass] = []
        assimilated_states: list[Tensor] = []

        for idx, raw_cand in enumerate(candidate_observations):
            cand = (
                raw_cand
                if isinstance(raw_cand, CandidateSensorPass)
                else CandidateSensorPass.from_dict(raw_cand, idx)
            )
            if cand.z_observation.shape != (batch, 1, self.model.n_patches, self.model.dim):
                raise ValueError(
                    f"candidate {idx} z_observation must have shape [B,1,G*G,D]"
                )
            if cand.forcing_observation.shape != (
                forcing_context.shape[0],
                1,
                forcing_context.shape[2],
            ):
                raise ValueError(
                    f"candidate {idx} forcing_observation must have shape [B,1,F]"
                )

            z_assimilated = self.assimilator.assimilate(
                z_prior,
                cand.z_observation,
                modality=cand.modality,
                assimilation_gain=cand.assimilation_gain,
                cloud_fraction=cand.cloud_fraction,
                gap_seconds=cand.gap_seconds,
            )
            z_aug = torch.cat((z_context[:, 1:], z_assimilated), dim=1)
            f_aug = torch.cat(
                (
                    forcing_context[:, 1:],
                    cand.forcing_observation.to(
                        dtype=forcing_context.dtype, device=forcing_context.device
                    ),
                ),
                dim=1,
            )
            z_batch_list.append(z_aug)
            f_batch_list.append(f_aug)
            parsed_candidates.append(cand)
            assimilated_states.append(z_assimilated)

        with torch.no_grad():
            z_stacked = torch.cat(z_batch_list, dim=0)
            f_ctx_stacked = torch.cat(f_batch_list, dim=0)
            f_fut_stacked = forcing_future.repeat(n_cands + 1, 1, 1)

            all_out = self.model.forward_latents(z_stacked, f_ctx_stacked, f_fut_stacked)
            z_hat_chunks = all_out.z_hat.reshape(
                n_cands + 1,
                batch,
                all_out.z_hat.shape[1],
                self.model.n_patches,
                self.model.dim,
            )
            unc_chunks = all_out.uncertainty_std.reshape(
                n_cands + 1,
                batch,
                all_out.uncertainty_std.shape[1],
                self.model.observable_dim,
            )
            obs_chunks = all_out.observable_pred.reshape(
                n_cands + 1,
                batch,
                all_out.observable_pred.shape[1],
                self.model.observable_dim,
            )

            base_z_hat = z_hat_chunks[0]
            base_unc = unc_chunks[0]
            base_obs = obs_chunks[0]
            base_width = float((2.0 * effective_quantile * base_unc).mean().item())

            rankings: list[dict[str, Any]] = []
            for c_idx, cand in enumerate(parsed_candidates):
                aug_z_hat = z_hat_chunks[c_idx + 1]
                aug_unc = unc_chunks[c_idx + 1]
                aug_width = float((2.0 * effective_quantile * aug_unc).mean().item())
                width_reduction = max(0.0, base_width - aug_width)
                latent_shift = float((aug_z_hat - base_z_hat).norm(dim=-1).mean().item())
                usable_weight = (
                    1.0 - float(cand.cloud_fraction) if cand.modality == "s2" else 1.0
                )
                voi_per_mb = (
                    usable_weight * (width_reduction + 0.1 * latent_shift) / float(cand.cost_mb)
                )
                rankings.append(
                    {
                        "candidate_index": cand.candidate_index,
                        "sensor": cand.sensor,
                        "modality": cand.modality,
                        "cost_mb": cand.cost_mb,
                        "cloud_fraction": cand.cloud_fraction,
                        "base_conformal_width": base_width,
                        "post_conformal_width": aug_width,
                        "width_reduction": width_reduction,
                        "latent_shift": latent_shift,
                        "voi_per_mb": voi_per_mb,
                        "z_assimilated": assimilated_states[c_idx],
                    }
                )

            rankings.sort(key=lambda r: float(r["voi_per_mb"]), reverse=True)
            result: dict[str, Any] = {
                "recommended": rankings[0],
                "rankings": rankings,
                "base_conformal_width": base_width,
                "effective_conformal_quantile": effective_quantile,
            }

            if target_future is not None or target_observable is not None:
                if target_future is not None:
                    if target_future.shape != base_z_hat.shape:
                        raise ValueError("target_future must match z_hat shape [B,K,G*G,D]")
                    tgt_z = target_future.to(
                        device=base_z_hat.device, dtype=base_z_hat.dtype
                    )
                    stop_losses = (base_z_hat - tgt_z).square().flatten(1).mean(dim=1)
                    cand_losses = [
                        (z_hat_chunks[c_idx + 1] - tgt_z).square().flatten(1).mean(dim=1)
                        for c_idx in range(n_cands)
                    ]
                else:
                    assert target_observable is not None
                    if target_observable.shape != base_obs.shape:
                        raise ValueError(
                            "target_observable must match observable_pred shape [B,K,O]"
                        )
                    tgt_o = target_observable.to(
                        device=base_obs.device, dtype=base_obs.dtype
                    )
                    stop_losses = (
                        gaussian_crps(base_obs, base_unc, tgt_o, reduction="none")
                        .flatten(1)
                        .mean(dim=1)
                    )
                    cand_losses = [
                        gaussian_crps(
                            obs_chunks[c_idx + 1],
                            unc_chunks[c_idx + 1],
                            tgt_o,
                            reduction="none",
                        )
                        .flatten(1)
                        .mean(dim=1)
                        for c_idx in range(n_cands)
                    ]

                s1_indices = [
                    i for i, c in enumerate(parsed_candidates) if c.modality == "s1"
                ]
                s2_indices = [
                    i for i, c in enumerate(parsed_candidates) if c.modality == "s2"
                ]
                s1_losses = cand_losses[s1_indices[0]] if s1_indices else cand_losses[0]
                s2_losses = cand_losses[s2_indices[0]] if s2_indices else cand_losses[-1]
                s2_cloud = (
                    float(parsed_candidates[s2_indices[0]].cloud_fraction)
                    if s2_indices
                    else 0.0
                )
                step_unc = (2.0 * effective_quantile * base_unc).flatten(1).mean(dim=1)
                cloud_tensor = torch.full_like(step_unc, s2_cloud)

                result["stop_losses"] = stop_losses
                result["s1_losses"] = s1_losses
                result["s2_losses"] = s2_losses
                result["uncertainties"] = step_unc
                result["policy_evaluation"] = score_observation_reveal_policies(
                    stop_losses,
                    s1_losses,
                    s2_losses,
                    step_unc,
                    cloud_fractions=cloud_tensor,
                    cost_s1=cost_s1,
                    cost_s2=cost_s2,
                    budget_per_step=budget_per_step,
                    cost_weight=cost_weight,
                )

            return result


def assimilate_subspace_observation(
    z_prior: Tensor,
    z_obs: Tensor,
    *,
    modality: Literal["s1", "s2", "joint"] = "joint",
    shared_dim: int | None = None,
    s1_private_dim: int = 0,
    s2_private_dim: int = 0,
    assimilation_gain: float = 1.0,
    cloud_fraction: float = 0.0,
    gap_seconds: float = 0.0,
    gap_tau_seconds: float = 604800.0,
) -> Tensor:
    """Functional wrapper around ``SubspaceDataAssimilator.assimilate``."""
    if z_prior.shape != z_obs.shape or z_prior.ndim < 2:
        raise ValueError("z_prior and z_obs must share shape [..., D]")
    partition = SubspacePartition.resolve(
        int(z_prior.shape[-1]),
        shared_dim=shared_dim,
        s1_private_dim=s1_private_dim,
        s2_private_dim=s2_private_dim,
    )
    assimilator = SubspaceDataAssimilator(
        partition,
        default_gain=assimilation_gain,
        gap_tau_seconds=gap_tau_seconds,
    )
    return assimilator.assimilate(
        z_prior,
        z_obs,
        modality=modality,
        assimilation_gain=assimilation_gain,
        cloud_fraction=cloud_fraction,
        gap_seconds=gap_seconds,
    )


def gpu_counterfactual_observation_value(
    model: FactorizedSpatiotemporalJEPAWorldModel,
    *,
    z_context: Tensor,
    candidate_observations: list[dict[str, Tensor | float | str]],
    forcing_context: Tensor,
    forcing_future: Tensor,
    conformal_quantile: float = 1.645,
    calibrator: StratifiedConformalAlertCalibrator | None = None,
    calibrator_group: tuple[str, str] | None = None,
    shared_dim: int | None = None,
    s1_private_dim: int = 0,
    s2_private_dim: int = 0,
    assimilation_gain: float = 1.0,
    gap_tau_seconds: float = 604800.0,
    target_future: Tensor | None = None,
    target_observable: Tensor | None = None,
    cost_s1: float = 0.2,
    cost_s2: float = 1.0,
    budget_per_step: float | None = None,
    cost_weight: float = 0.1,
) -> dict[str, Any]:
    """Functional wrapper around ``CounterfactualVoIEvaluator.evaluate``."""
    partition = SubspacePartition.resolve(
        model.dim,
        shared_dim=shared_dim,
        s1_private_dim=s1_private_dim,
        s2_private_dim=s2_private_dim,
    )
    assimilator = SubspaceDataAssimilator(
        partition,
        default_gain=assimilation_gain,
        gap_tau_seconds=gap_tau_seconds,
    )
    evaluator = CounterfactualVoIEvaluator(
        model,
        assimilator=assimilator,
        conformal_quantile=conformal_quantile,
        calibrator=calibrator,
        calibrator_group=calibrator_group,
    )
    return evaluator.evaluate(
        z_context,
        candidate_observations,
        forcing_context,
        forcing_future,
        target_future=target_future,
        target_observable=target_observable,
        cost_s1=cost_s1,
        cost_s2=cost_s2,
        budget_per_step=budget_per_step,
        cost_weight=cost_weight,
    )


__all__ = [
    "CandidateSensorPass",
    "CounterfactualVoIEvaluator",
    "SubspaceDataAssimilator",
    "SubspacePartition",
    "assimilate_subspace_observation",
    "gpu_counterfactual_observation_value",
]
