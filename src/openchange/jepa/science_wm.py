"""GPU-native Open-Science Spatiotemporal World Model (Sci-JEPA-WM).

Unifies multi-sensor Earth Observation (Sentinel-1 SAR + Sentinel-2 Optical)
and continuum PDE physical fields (The Well / Walrus / PDE-JEPA) into a
decoder-free, forcing-conditioned latent world model designed for 4x GH200 GPUs:

  1. PhysicalForcingEncoder (EO-WM arXiv:2606.27277 + PDE-JEPA arXiv:2609.34715):
     Decomposes external forcing f_t into three physical pathways:
       - Climatological / equilibrium baseline f_t^{base}
       - Instantaneous anomaly delta_f_t = f_t - f_t^{base}
       - Cumulative physical stress integrator S_t = sum_{tau<=t} gamma^{t-tau} ReLU(delta_f_tau)
  2. MultiSensorContinuumPatchStem:
     Encodes co-registered S1 [B,T,2,H,W] + S2 [B,T,12,H,W] (with true missing-sensor
     omission and cloud validity masking) OR continuum physical fields [B,T,C,H,W]
     into spatial patch tokens [B,T,G*G,D].
  3. FactorizedSpatiotemporalJEPAWorldModel:
     Alternates Walrus-style spatial attention with Theorem-1 causal temporal
     attention, decomposing latent transitions into parameter-free intrinsic
     evolution + forcing-modulated response, regularized by HistoryStratifiedSIGReg
     and scored by a Goal-Agnostic Physical Observable Probe (arXiv:2607.21644).
  4. gpu_counterfactual_observation_value:
     Batched parallel GPU counterfactual rollouts ranking candidate sensor passes
     by expected conformal uncertainty reduction + forecast gain per megabyte.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from openchange.jepa.model import sincos_pos_embed
from openchange.jepa.wam import (
    HistoryStratifiedSIGReg,
    _CausalBlock,
    causal_neighborhood_mask,
)


@dataclass
class ScienceWMOutput:
    z_hat: Tensor
    z_intrinsic: Tensor
    z_forced_delta: Tensor
    observable_pred: Tensor
    uncertainty_std: Tensor
    cumulative_stress: Tensor


class PhysicalForcingEncoder(nn.Module):
    """Three-pathway physical forcing encoder (baseline, anomaly, cumulative stress)."""

    def __init__(self, forcing_dim: int, dim: int, stress_decay: float = 0.85) -> None:
        super().__init__()
        if forcing_dim <= 0 or dim <= 0:
            raise ValueError("forcing_dim and dim must be positive")
        if not 0.0 < stress_decay <= 1.0:
            raise ValueError("stress_decay must be in (0, 1]")
        self.forcing_dim = forcing_dim
        self.dim = dim
        self.stress_decay = float(stress_decay)

        self.baseline_proj = nn.Linear(forcing_dim, dim)
        self.anomaly_proj = nn.Linear(forcing_dim, dim)
        self.stress_proj = nn.Linear(forcing_dim, dim)
        self.film_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in (self.baseline_proj, self.anomaly_proj, self.stress_proj):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            nn.init.zeros_(mod.bias)
        nn.init.trunc_normal_(self.film_head[1].weight, std=0.02)
        nn.init.zeros_(self.film_head[1].bias)

    def compute_cumulative_stress(self, anomaly: Tensor) -> Tensor:
        """Causal exponential accumulator S_t = gamma * S_{t-1} + ReLU(anomaly_t)."""
        if anomaly.ndim != 3 or anomaly.shape[2] != self.forcing_dim:
            raise ValueError("anomaly must have shape [B,T,forcing_dim]")
        pos_excess = functional.relu(anomaly)
        stress_steps: list[Tensor] = []
        acc = torch.zeros_like(pos_excess[:, 0])
        for t in range(anomaly.shape[1]):
            acc = self.stress_decay * acc + pos_excess[:, t]
            stress_steps.append(acc)
        return torch.stack(stress_steps, dim=1)

    def forward(
        self, forcing: Tensor, baseline: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (film_gamma_beta [B,T,2*D], forcing_embed [B,T,D], cumulative_stress [B,T,F])."""
        if forcing.ndim != 3 or forcing.shape[2] != self.forcing_dim:
            raise ValueError("forcing must have shape [B,T,forcing_dim]")
        base = torch.zeros_like(forcing) if baseline is None else baseline.to(dtype=forcing.dtype)
        if base.shape != forcing.shape:
            raise ValueError("baseline shape must match forcing shape [B,T,forcing_dim]")
        anomaly = forcing - base
        stress = self.compute_cumulative_stress(anomaly)
        embed = (
            self.baseline_proj(base)
            + self.anomaly_proj(anomaly)
            + self.stress_proj(stress)
        )
        return self.film_head(embed), embed, stress


class MultiSensorContinuumPatchStem(nn.Module):
    """GPU patch stem supporting S1 SAR + S2 Optical (with missing-sensor omission)
    and general continuum PDE fields [B,T,C,H,W].
    """

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 16,
        dim: int = 128,
        s1_channels: int = 2,
        s2_channels: int = 12,
        continuum_channels: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be a positive multiple of patch_size")
        torch.manual_seed(seed)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.dim = dim
        self.s1_channels = s1_channels
        self.s2_channels = s2_channels
        self.continuum_channels = continuum_channels

        self.s1_stem = nn.Conv2d(s1_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.s2_stem = nn.Conv2d(s2_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.continuum_stem = (
            nn.Conv2d(continuum_channels, dim, kernel_size=patch_size, stride=patch_size)
            if continuum_channels is not None
            else None
        )
        self.modality_embed = nn.Embedding(3, dim)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(self.grid_size, dim, dtype=torch.float32), persistent=False
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _patchify_stream(self, x: Tensor, stem: nn.Conv2d, mod_idx: int) -> Tensor:
        batch, timesteps, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected spatial size {self.image_size}x{self.image_size}")
        flat = x.reshape(batch * timesteps, channels, height, width).to(dtype=stem.weight.dtype)
        patches = stem(flat).flatten(2).transpose(1, 2)
        patches = (
            patches
            + self.pos_embed.to(dtype=patches.dtype).unsqueeze(0)
            + self.modality_embed.weight[mod_idx].to(dtype=patches.dtype).view(1, 1, -1)
        )
        return patches.reshape(batch, timesteps, self.n_patches, self.dim)

    def forward(
        self,
        *,
        s1: Tensor | None = None,
        s2: Tensor | None = None,
        s2_valid: Tensor | None = None,
        fields: Tensor | None = None,
    ) -> Tensor:
        """Encode available sensor streams or continuum fields into [B,T,G*G,D] tokens."""
        if fields is not None:
            if self.continuum_stem is None:
                raise ValueError("continuum_channels was not configured on this stem")
            return self.norm(self._patchify_stream(fields, self.continuum_stem, 2))

        if s1 is None and s2 is None:
            raise ValueError("at least one of s1, s2, or fields must be provided")

        streams: list[Tensor] = []
        weights: list[Tensor] = []
        if s1 is not None:
            p1 = self._patchify_stream(s1, self.s1_stem, 0)
            streams.append(p1)
            weights.append(torch.ones_like(p1[..., :1]))

        if s2 is not None:
            s2_in = s2
            if s2_valid is not None:
                if s2_valid.shape != (s2.shape[0], s2.shape[1], s2.shape[3], s2.shape[4]):
                    raise ValueError("s2_valid must have shape [B,T,H,W]")
                mask_bool = s2_valid.to(dtype=s2.dtype).unsqueeze(2)
                s2_in = s2 * mask_bool
                # Compute per-patch clear fraction to weight optical fusion honestly
                patch_valid = functional.avg_pool2d(
                    mask_bool.reshape(-1, 1, s2.shape[3], s2.shape[4]),
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ).reshape(s2.shape[0], s2.shape[1], self.n_patches, 1)
                w2 = patch_valid.clamp(min=0.05 if s1 is None else 0.0)
            else:
                w2 = torch.ones((s2.shape[0], s2.shape[1], self.n_patches, 1), dtype=s2.dtype, device=s2.device)
            p2 = self._patchify_stream(s2_in, self.s2_stem, 1)
            streams.append(p2)
            weights.append(w2)

        if len(streams) == 1:
            fused = streams[0]
        else:
            w_sum = (weights[0] + weights[1]).clamp(min=1e-6)
            fused = (streams[0] * weights[0] + streams[1] * weights[1]) / w_sum
        return self.norm(fused)


class FactorizedSpatiotemporalJEPAWorldModel(nn.Module):
    """GPU-native Spatiotemporal Science World Model (Sci-JEPA-WM).

    Decomposes latent dynamics into:
      * Intrinsic parameter-free evolution (autonomous PDE / land-surface evolution)
      * Forcing-conditioned response (weather anomalies, cumulative heat/drought stress,
        or boundary/actuator forcing)
    and predicts both future patch latents and goal-agnostic physical observables
    with calibrated epistemic uncertainty.
    """

    def __init__(
        self,
        *,
        stem: MultiSensorContinuumPatchStem,
        forcing_dim: int,
        observable_dim: int = 1,
        horizon: int = 4,
        heads: int = 4,
        depth: int = 2,
        mlp_ratio: int = 2,
        spatial_radius: int = 1,
        max_timesteps: int = 16,
        stress_decay: float = 0.85,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.stem = stem
        self.dim = stem.dim
        self.grid_size = stem.grid_size
        self.n_patches = stem.n_patches
        self.n_tok = self.n_patches + 1
        self.horizon = horizon
        self.spatial_radius = spatial_radius
        self.max_timesteps = max_timesteps
        self.observable_dim = observable_dim

        self.cls_token = nn.Parameter(torch.empty(1, 1, self.dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.time_embed = nn.Embedding(max_timesteps + horizon + 1, self.dim)
        nn.init.trunc_normal_(self.time_embed.weight, std=0.02)

        self.forcing_encoder = PhysicalForcingEncoder(
            forcing_dim=forcing_dim, dim=self.dim, stress_decay=stress_decay
        )
        self.spatial_blocks = nn.ModuleList(
            [_CausalBlock(self.dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [_CausalBlock(self.dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)

        # PDE-JEPA decomposed transition heads: intrinsic vs. forcing-modulated response
        self.intrinsic_head = nn.Linear(self.dim, self.dim)
        self.forced_head = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        # Goal-agnostic physical observable probe + epistemic log-variance head
        self.observable_probe = nn.Linear(self.dim, observable_dim)
        self.uncertainty_head = nn.Linear(self.dim, observable_dim)
        self.register_buffer(
            "spatial_mask",
            torch.zeros((self.n_tok, self.n_tok), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _apply_film(tokens: Tensor, film_gb: Tensor, n_tok: int) -> Tensor:
        gamma, beta = film_gb.chunk(2, dim=-1)
        gamma = gamma.repeat_interleave(n_tok, dim=1)
        beta = beta.repeat_interleave(n_tok, dim=1)
        return tokens * (1.0 + gamma) + beta

    def _prepend_cls_and_time(self, z_patches: Tensor, t_offset: int = 0) -> Tensor:
        batch, timesteps, _, _ = z_patches.shape
        out_steps: list[Tensor] = []
        for t in range(timesteps):
            cls = self.cls_token.to(dtype=z_patches.dtype).expand(batch, -1, -1)
            step_tok = torch.cat((cls, z_patches[:, t]), dim=1)
            step_tok = step_tok + self.time_embed.weight[t_offset + t].to(dtype=z_patches.dtype).view(1, 1, -1)
            out_steps.append(step_tok)
        return torch.cat(out_steps, dim=1)

    def forward_latents(
        self,
        z_context: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        *,
        baseline_context: Tensor | None = None,
        baseline_future: Tensor | None = None,
    ) -> ScienceWMOutput:
        """Roll out factorized spatiotemporal latent dynamics over future forcing steps."""
        if z_context.ndim != 4 or z_context.shape[2] != self.n_patches or z_context.shape[3] != self.dim:
            raise ValueError("z_context must have shape [B,T,G*G,D]")
        batch, timesteps, _, _ = z_context.shape
        horizon = forcing_future.shape[1]
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"forcing_future horizon must be in 1..{self.horizon}")

        full_forcing = torch.cat((forcing_context, forcing_future), dim=1)
        full_base = (
            None
            if (baseline_context is None or baseline_future is None)
            else torch.cat((baseline_context, baseline_future), dim=1)
        )
        film_all, force_emb_all, stress_all = self.forcing_encoder(full_forcing, full_base)
        film_ctx = film_all[:, :timesteps]
        force_fut = force_emb_all[:, timesteps:]
        stress_fut = stress_all[:, timesteps:]

        tokens = self._prepend_cls_and_time(z_context, t_offset=0)
        tokens = self._apply_film(tokens, film_ctx.to(dtype=tokens.dtype), self.n_tok)

        s_mask = self.spatial_mask.to(dtype=tokens.dtype, device=tokens.device)
        t_mask = causal_neighborhood_mask(
            self.grid_size, timesteps, self.spatial_radius, include_cls=True, device=tokens.device
        ).to(dtype=tokens.dtype)

        past_temporal: list[Tensor] = []
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            per_step = tokens.reshape(batch * timesteps, self.n_tok, self.dim)
            per_step = s_block(per_step, s_mask)
            tokens = per_step.reshape(batch, timesteps * self.n_tok, self.dim)
            past_temporal.append(tokens)
            tokens = t_block(tokens, t_mask)

        h_last = tokens[:, -self.n_tok :]
        z_hats: list[Tensor] = []
        z_intrinsics: list[Tensor] = []
        z_forced_deltas: list[Tensor] = []
        obs_preds: list[Tensor] = []
        unc_stds: list[Tensor] = []

        for step in range(horizon):
            f_step = force_fut[:, step].to(dtype=h_last.dtype)
            f_film = film_all[:, timesteps + step : timesteps + step + 1].to(dtype=h_last.dtype)
            conditioned = self._apply_film(h_last, f_film, self.n_tok)
            for idx, (s_block, t_block) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
                conditioned = s_block(conditioned, s_mask)
                updated = t_block.step(conditioned, past_temporal[idx])
                past_temporal[idx] = torch.cat((past_temporal[idx], conditioned), dim=1)
                conditioned = updated

            normed = self.norm(conditioned)
            patch_hidden = normed[:, 1:, :]
            cls_hidden = normed[:, 0, :]

            z_int = self.intrinsic_head(patch_hidden)
            f_expanded = f_step.unsqueeze(1).expand(-1, self.n_patches, -1)
            z_delta = self.forced_head(torch.cat((patch_hidden, f_expanded), dim=-1))
            z_next = z_int + z_delta

            pooled_state = 0.5 * (cls_hidden + z_next.mean(dim=1))
            obs_step = self.observable_probe(pooled_state)
            std_step = functional.softplus(self.uncertainty_head(pooled_state)) + 1e-4

            z_hats.append(z_next)
            z_intrinsics.append(z_int)
            z_forced_deltas.append(z_delta)
            obs_preds.append(obs_step)
            unc_stds.append(std_step)

            cls_next = self.cls_token.to(dtype=z_next.dtype).expand(batch, -1, -1)
            h_last = torch.cat((cls_next, z_next), dim=1) + self.time_embed.weight[timesteps + step].to(
                dtype=z_next.dtype
            ).view(1, 1, -1)

        return ScienceWMOutput(
            z_hat=torch.stack(z_hats, dim=1),
            z_intrinsic=torch.stack(z_intrinsics, dim=1),
            z_forced_delta=torch.stack(z_forced_deltas, dim=1),
            observable_pred=torch.stack(obs_preds, dim=1),
            uncertainty_std=torch.stack(unc_stds, dim=1),
            cumulative_stress=stress_fut,
        )

    def science_wm_loss(
        self,
        *,
        z_context: Tensor,
        z_future: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        history_seconds: Tensor,
        directions: Tensor,
        sigreg: HistoryStratifiedSIGReg,
        target_observable: Tensor | None = None,
        baseline_context: Tensor | None = None,
        baseline_future: Tensor | None = None,
        lam_sigreg: float = 0.05,
        observable_weight: float = 0.5,
    ) -> dict[str, Tensor]:
        """Compute joint latent prediction + HistoryStratifiedSIGReg + Gaussian NLL observable loss."""
        out = self.forward_latents(
            z_context,
            forcing_context,
            forcing_future,
            baseline_context=baseline_context,
            baseline_future=baseline_future,
        )
        if z_future.shape != out.z_hat.shape:
            raise ValueError("z_future must match out.z_hat shape [B,K,G*G,D]")
        latent_mse = (out.z_hat - z_future.detach()).square().mean()

        batch, horizon, n_patches, dim = out.z_hat.shape
        flat = out.z_hat.reshape(batch * horizon * n_patches, dim)
        repeated_hist = history_seconds.to(device=flat.device).repeat_interleave(horizon * n_patches)
        reg = sigreg(flat, repeated_hist, directions)

        if target_observable is not None:
            if target_observable.shape != out.observable_pred.shape:
                raise ValueError("target_observable must match observable_pred shape [B,K,O]")
            tgt = target_observable.to(dtype=out.observable_pred.dtype)
            var = out.uncertainty_std.square()
            obs_nll = 0.5 * ((tgt - out.observable_pred).square() / var + torch.log(var)).mean()
            obs_mse = (tgt - out.observable_pred).square().mean()
        else:
            obs_nll = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)
            obs_mse = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)

        total = latent_mse + lam_sigreg * reg["sigreg"] + observable_weight * obs_nll
        return {
            "loss": total,
            "latent_mse": latent_mse,
            "sigreg": reg["sigreg"],
            "observable_nll": obs_nll,
            "observable_mse": obs_mse,
            "mean_uncertainty_std": out.uncertainty_std.detach().mean(),
            "forced_energy_ratio": (
                out.z_forced_delta.detach().norm()
                / (out.z_intrinsic.detach().norm() + out.z_forced_delta.detach().norm() + 1e-8)
            ),
        }


def gpu_counterfactual_observation_value(
    model: FactorizedSpatiotemporalJEPAWorldModel,
    *,
    z_context: Tensor,
    candidate_observations: list[dict[str, Tensor | float | str]],
    forcing_context: Tensor,
    forcing_future: Tensor,
    conformal_quantile: float = 1.645,
) -> dict[str, object]:
    """GPU batched Value-of-Information (VoI) evaluator for active sensor tasking/reveal.

    Evaluates a baseline trajectory (no extra observation revealed) against each
    candidate archived/taskable observation (e.g., an all-weather Sentinel-1 SAR
    pass or clear optical pass appended to the context) in parallel on GPU, and
    ranks candidates by expected conformal interval width reduction per megabyte.
    """
    if not candidate_observations:
        raise ValueError("candidate_observations must be non-empty")
    if conformal_quantile <= 0:
        raise ValueError("conformal_quantile must be positive")

    with torch.no_grad():
        base_out = model.forward_latents(z_context, forcing_context, forcing_future)
        base_width = float((2.0 * conformal_quantile * base_out.uncertainty_std).mean().item())

        rankings: list[dict[str, object]] = []
        for idx, cand in enumerate(candidate_observations):
            z_cand = cand["z_observation"]
            if not isinstance(z_cand, Tensor) or z_cand.shape != (z_context.shape[0], 1, model.n_patches, model.dim):
                raise ValueError(f"candidate {idx} z_observation must have shape [B,1,G*G,D]")
            f_cand = cand["forcing_observation"]
            if not isinstance(f_cand, Tensor) or f_cand.shape != (forcing_context.shape[0], 1, forcing_context.shape[2]):
                raise ValueError(f"candidate {idx} forcing_observation must have shape [B,1,F]")
            cost_mb = float(cand.get("cost_mb", 1.0))
            if cost_mb <= 0:
                raise ValueError("cost_mb must be positive")

            z_aug = torch.cat((z_context[:, 1:], z_cand.to(dtype=z_context.dtype, device=z_context.device)), dim=1)
            f_aug = torch.cat((forcing_context[:, 1:], f_cand.to(dtype=forcing_context.dtype, device=forcing_context.device)), dim=1)
            aug_out = model.forward_latents(z_aug, f_aug, forcing_future)
            aug_width = float((2.0 * conformal_quantile * aug_out.uncertainty_std).mean().item())
            width_reduction = max(0.0, base_width - aug_width)
            latent_shift = float((aug_out.z_hat - base_out.z_hat).norm(dim=-1).mean().item())
            voi_per_mb = (width_reduction + 0.1 * latent_shift) / cost_mb
            rankings.append({
                "candidate_index": idx,
                "sensor": str(cand.get("sensor", f"candidate_{idx}")),
                "cost_mb": cost_mb,
                "base_conformal_width": base_width,
                "post_conformal_width": aug_width,
                "width_reduction": width_reduction,
                "latent_shift": latent_shift,
                "voi_per_mb": voi_per_mb,
            })

        rankings.sort(key=lambda r: float(r["voi_per_mb"]), reverse=True)
        return {
            "recommended": rankings[0],
            "rankings": rankings,
            "base_conformal_width": base_width,
        }
