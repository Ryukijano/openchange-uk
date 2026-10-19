"""Gate-1 matched objective arms and TESSERA-style input mixing."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from openchange.jepa.distributed import DistContext
from openchange.jepa.losses import (
    ConditionalFlowBridge,
    jepa_loss,
    split_subspace_jepa_loss,
    tessera_mixup_barlow_loss,
    vicreg_loss,
)
from openchange.jepa.model import EncoderOutput, JEPAEncoder
from openchange.jepa.views import ViewBatch, ViewBucket


class Gate1Model(nn.Module):
    """JEPA encoder plus the optional S1-to-S2-private conditional-flow bridge."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.encoder = JEPAEncoder(cfg)
        objective = cfg["objective"]
        self.bridge = None
        if objective["name"] == "defensive_split_cfm":
            self.bridge = ConditionalFlowBridge(
                target_dim=objective["s2_private_dim"],
                cond_dim=objective["s1_private_dim"],
                hidden_dim=objective["cfm_hidden_dim"],
                time_dim=objective["cfm_time_dim"],
                seed=cfg["seed"],
            )

    def forward(self, batch: ViewBatch, *, validate: bool = True) -> EncoderOutput:
        return self.encoder(batch, validate=validate)

    def train(self, mode: bool = True) -> Gate1Model:
        super().train(mode)
        self.encoder.train(mode)
        return self


def stratum_preserving_permutation(strata: Tensor, *, seed: int, step: int) -> Tensor:
    """Permute samples within each stratum so mixed views have the same sensors."""
    if strata.ndim != 1 or strata.dtype != torch.int64:
        raise ValueError("strata must be an int64 tensor of shape [N]")
    perm = torch.arange(strata.shape[0], dtype=torch.int64)
    generator = torch.Generator().manual_seed((seed + 1) * 1_000_003 + step)
    for stratum in torch.unique(strata.cpu(), sorted=True):
        indices = torch.nonzero(strata.cpu() == stratum, as_tuple=False).flatten()
        order = torch.randperm(indices.numel(), generator=generator)
        perm[indices] = indices[order]
    return perm.to(device=strata.device)


def mix_global_view_batch(batch: ViewBatch, perm: Tensor, alpha_mix: float) -> ViewBatch:
    """Mix each global view-A input with the same sensor from a shuffled view-B sample."""
    if not 0.0 <= alpha_mix <= 1.0:
        raise ValueError("alpha_mix must be in [0, 1]")
    if perm.ndim != 1 or perm.shape[0] != batch.n_samples or perm.dtype != torch.int64:
        raise ValueError("perm must be an int64 tensor of shape [N]")

    buckets: list[ViewBucket] = []
    for bucket in batch.buckets:
        row_lookup = {
            (int(view), int(sample)): row
            for row, (view, sample) in enumerate(
                zip(bucket.view_idx.tolist(), bucket.sample_idx.tolist(), strict=True)
            )
        }
        mixed_pixels = {}
        for sensor, values in bucket.pixels.items():
            mixed = values.clone()
            global_rows = torch.nonzero(bucket.view_idx == 0, as_tuple=False).flatten()
            source_rows = []
            for row_tensor in global_rows:
                row = int(row_tensor)
                source_sample = int(perm[int(bucket.sample_idx[row])])
                try:
                    source_rows.append(row_lookup[(1, source_sample)])
                except KeyError as exc:
                    raise ValueError("mixed source view is missing required sensors") from exc
            if source_rows:
                source_index = torch.tensor(source_rows, dtype=torch.int64)
                mixed[global_rows] = alpha_mix * values[global_rows] + (
                    1.0 - alpha_mix
                ) * values[source_index]
            mixed_pixels[sensor] = mixed
        buckets.append(replace(bucket, pixels=mixed_pixels))
    return replace(batch, buckets=buckets)


def gate1_loss(
    cfg: dict,
    *,
    model: Gate1Model,
    output: EncoderOutput,
    batch: ViewBatch,
    strata: Tensor,
    step: int,
    directions: Tensor,
    directions_s1: Tensor | None,
    directions_s2: Tensor | None,
    mixed_output: EncoderOutput | None,
    mix_perm: Tensor | None,
    mix_alpha: float | None,
    ctx: DistContext,
    chunk: int,
) -> dict[str, Tensor]:
    """Evaluate one of the four matched Gate-1 objective arms."""
    objective = cfg["objective"]
    name = objective["name"]
    z = output.z.float()
    zero = torch.zeros((), dtype=z.dtype, device=z.device)
    ep_zero = torch.zeros((z.shape[0], 4), dtype=z.dtype, device=z.device)

    if name == "clean_jepa":
        return jepa_loss(
            z,
            strata,
            n_global=cfg["global_views"],
            n_strata=4,
            weights=cfg["stratum_weights"],
            lam=cfg["lambda"],
            lam_mu=cfg["lambda_mu"],
            directions=directions,
            mode=cfg["sigreg_mode"],
            dist=ctx,
            chunk=chunk,
            checkpoint=True,
        )
    if name == "defensive_split_cfm":
        if model.bridge is None or directions_s1 is None or directions_s2 is None:
            raise ValueError("defensive objective requires the CFM bridge and private directions")
        result = split_subspace_jepa_loss(
            z,
            strata,
            n_global=cfg["global_views"],
            shared_dim=objective["shared_dim"],
            s1_private_dim=objective["s1_private_dim"],
            s2_private_dim=objective["s2_private_dim"],
            directions_shared=directions[: objective["shared_dim"]],
            directions_s1=directions_s1,
            directions_s2=directions_s2,
            lam=cfg["lambda"],
            lam_priv_inactive=objective["lam_priv_inactive"],
            pair_gap_seconds=(
                batch.pair_gap_seconds.to(device=z.device)
                if batch.pair_gap_seconds is not None
                else None
            ),
            gap_tau_seconds=objective["gap_tau_seconds"],
            dist=ctx,
            chunk=chunk,
            checkpoint=True,
        )
        joint = (strata == 0) | (strata == 1)
        shared_end = objective["shared_dim"]
        s1_end = shared_end + objective["s1_private_dim"]
        bridge_dtype = next(model.bridge.parameters()).dtype
        s1_latent = z[: cfg["global_views"], joint, shared_end:s1_end].mean(dim=0).to(bridge_dtype)
        s2_latent = z[: cfg["global_views"], joint, s1_end:].mean(dim=0).to(bridge_dtype)
        cfm = model.bridge.loss(s2_latent, s1_latent)
        result["cfm_loss"] = cfm["loss"]
        result["loss"] = result["loss"] + objective["cfm_weight"] * cfm["loss"]
        result["ep"] = ep_zero
        result["t_mu"] = zero
        return result
    if name == "tessera_mixup":
        if mixed_output is None or mix_perm is None or mix_alpha is None:
            raise ValueError("tessera_mixup requires mixed_output, mix_perm, and mix_alpha")
        result = tessera_mixup_barlow_loss(
            z[0],
            z[1],
            mixed_output.z.float()[0],
            alpha_mix=mix_alpha,
            perm=mix_perm,
            lam_bt=objective["barlow_lambda"],
            lam_mix=objective["mixup_lambda"],
            normalize_by_ceiling=objective["normalize_by_ceiling"],
            dist=ctx,
        )
        result["inv"] = result["bt_loss"]
        result["sigreg"] = result["mix_loss"]
        result["t_mu"] = zero
        result["ep"] = ep_zero
        return result
    if name == "vicreg":
        result = vicreg_loss(
            z,
            sim_coeff=objective["vicreg_sim_coeff"],
            std_coeff=objective["vicreg_std_coeff"],
            cov_coeff=objective["vicreg_cov_coeff"],
            gamma=objective["vicreg_gamma"],
            dist=ctx,
        )
        result["inv"] = result["sim_loss"]
        result["sigreg"] = result["std_loss"] + result["cov_loss"]
        result["t_mu"] = zero
        result["ep"] = ep_zero
        return result
    raise ValueError(f"unknown Gate-1 objective {name}")
