"""GPU-native Open-Science & Coastal Hydrodynamic-Morphodynamic World Model (Sci-JEPA-WM).

Object-oriented world model composing:
  1. ``openchange.jepa.coastal_physics``:
     - ``CoastalHydrodynamicConfig``, ``CoastalHydrodynamicModel``,
       ``StockdonRunupResult``, ``ShoreForDisequilibriumResult``
  2. ``openchange.jepa.stochastic_interpolant``:
     - ``StochasticInterpolantConfig``, ``ForecastingStochasticInterpolantHead``,
       ``EnsembleRolloutOutput``
  3. ``openchange.jepa.assimilation``:
     - ``SubspacePartition``, ``CandidateSensorPass``, ``SubspaceDataAssimilator``,
       ``CounterfactualVoIEvaluator``
  4. Core Spatiotemporal JEPA Modules in this file:
     - ``PhysicalForcingEncoder``
     - ``MultiSensorContinuumPatchStem``
     - ``FactorizedSpatiotemporalJEPAWorldModel``
     - ``ForcingElasticityEvaluator``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from openchange.jepa.assimilation import (
    CandidateSensorPass,
    CounterfactualVoIEvaluator,
    SubspaceDataAssimilator,
    SubspacePartition,
    assimilate_subspace_observation,
    gpu_counterfactual_observation_value,
)
from openchange.jepa.coastal_physics import (
    EO_HISTORY_STRATA_SECONDS,
    CoastalHydrodynamicConfig,
    CoastalHydrodynamicModel,
    ShoreForDisequilibriumResult,
    StockdonRunupResult,
    build_coastal_forcing_tensor,
    shorefor_disequilibrium_stress,
    stockdon_wave_setup_and_runup,
)
from openchange.jepa.guarantees import (
    StratifiedConformalAlertCalibrator,
    ensemble_crps,
    gaussian_crps,
    score_observation_reveal_policies,
)
from openchange.jepa.model import sincos_pos_embed
from openchange.jepa.stochastic_interpolant import (
    EnsembleRolloutOutput,
    ForecastingStochasticInterpolantHead,
    StochasticInterpolantConfig,
)
from openchange.jepa.wam import (
    HistoryStratifiedSIGReg,
    _CausalBlock,
    causal_neighborhood_mask,
)


@dataclass
class ScienceWMOutput:
    """Typed output of ``FactorizedSpatiotemporalJEPAWorldModel.forward_latents``."""

    z_hat: Tensor
    z_intrinsic: Tensor
    z_forced_delta: Tensor
    observable_pred: Tensor
    uncertainty_std: Tensor
    cumulative_stress: Tensor
    morphological_observable: Tensor | None = None
    hydrodynamic_observable_offset: Tensor | None = None


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
        base = (
            torch.zeros_like(forcing)
            if baseline is None
            else baseline.to(device=forcing.device, dtype=forcing.dtype)
        )
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
    """GPU patch stem supporting S1 SAR + S2 Optical (with missing-sensor omission
    and unbiased masked-convolution cloud renormalization) and continuum PDE fields [B,T,C,H,W].
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
        if dim <= 0 or s1_channels <= 0 or s2_channels <= 0:
            raise ValueError("dim, s1_channels, and s2_channels must be positive")
        if continuum_channels is not None and continuum_channels <= 0:
            raise ValueError("continuum_channels must be positive when provided")
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
        self._init_weights()

    def _init_weights(self) -> None:
        for conv in (self.s1_stem, self.s2_stem, self.continuum_stem):
            if conv is not None:
                nn.init.trunc_normal_(conv.weight, std=0.02)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)

    def _patchify_stream(
        self,
        x: Tensor,
        stem: nn.Conv2d,
        mod_idx: int,
        *,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(patches [B,T,G*G,D], patch_clear_fraction [B,T,G*G,1])``."""
        if x.ndim != 5:
            raise ValueError("input stream must have 5D shape [B,T,C,H,W]")
        batch, timesteps, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected spatial size {self.image_size}x{self.image_size}")
        if channels != stem.in_channels:
            raise ValueError(f"expected {stem.in_channels} channels, got {channels}")

        flat = x.reshape(batch * timesteps, channels, height, width).to(dtype=stem.weight.dtype)

        if pixel_mask is not None:
            if pixel_mask.shape != (batch, timesteps, height, width):
                raise ValueError("pixel_mask must have shape [B,T,H,W]")
            mask_flat = pixel_mask.reshape(batch * timesteps, 1, height, width).to(
                device=flat.device, dtype=flat.dtype
            )
            patch_valid = functional.avg_pool2d(
                mask_flat, kernel_size=self.patch_size, stride=self.patch_size
            )  # [B*T, 1, G, G]
            patch_mean = functional.avg_pool2d(
                flat * mask_flat, kernel_size=self.patch_size, stride=self.patch_size
            ) / patch_valid.clamp_min(1e-6)
            patch_mean_up = patch_mean.repeat_interleave(self.patch_size, dim=2).repeat_interleave(
                self.patch_size, dim=3
            )
            unbiased_input = flat * mask_flat + patch_mean_up * (1.0 - mask_flat)
            raw_conv = stem(unbiased_input)
            valid_gate = (patch_valid > 0.0).to(dtype=raw_conv.dtype)
            conv_tokens = (raw_conv * valid_gate).flatten(2).transpose(1, 2)
            p_weights = patch_valid.flatten(2).transpose(1, 2)  # [B*T, G*G, 1]
            pos_and_mod = (
                self.pos_embed.to(dtype=conv_tokens.dtype).unsqueeze(0)
                + self.modality_embed.weight[mod_idx].to(dtype=conv_tokens.dtype).view(1, 1, -1)
            )
            gate_tokens = (p_weights > 0.0).to(dtype=conv_tokens.dtype)
            patches = (conv_tokens + pos_and_mod) * gate_tokens
        else:
            conv_tokens = stem(flat).flatten(2).transpose(1, 2)
            patches = (
                conv_tokens
                + self.pos_embed.to(dtype=conv_tokens.dtype).unsqueeze(0)
                + self.modality_embed.weight[mod_idx].to(dtype=conv_tokens.dtype).view(1, 1, -1)
            )
            p_weights = torch.ones(
                (batch * timesteps, self.n_patches, 1),
                dtype=patches.dtype,
                device=patches.device,
            )

        return (
            patches.reshape(batch, timesteps, self.n_patches, self.dim),
            p_weights.reshape(batch, timesteps, self.n_patches, 1),
        )

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
            patches, _ = self._patchify_stream(fields, self.continuum_stem, 2)
            return self.norm(patches)

        if s1 is None and s2 is None:
            raise ValueError("at least one of s1, s2, or fields must be provided")

        p1: Tensor | None = None
        w1: Tensor | None = None
        if s1 is not None:
            p1, w1 = self._patchify_stream(s1, self.s1_stem, 0)

        p2: Tensor | None = None
        w2: Tensor | None = None
        if s2 is not None:
            p2, w2 = self._patchify_stream(s2, self.s2_stem, 1, pixel_mask=s2_valid)

        if p1 is not None and p2 is None:
            return self.norm(p1)

        if p2 is not None and p1 is None:
            assert w2 is not None
            active_gate = (w2 > 0.0).to(dtype=p2.dtype)
            return self.norm(p2) * w2.to(dtype=p2.dtype) * active_gate

        assert p1 is not None and p2 is not None and w1 is not None and w2 is not None
        if p1.shape[:3] != p2.shape[:3]:
            raise ValueError("s1 and s2 must share batch, timesteps, and patch grid dimensions")
        w2_cast = w2.to(dtype=p1.dtype, device=p1.device)
        w1_cast = w1.to(dtype=p1.dtype, device=p1.device)
        w_sum = (w1_cast + w2_cast).clamp_min(1e-6)
        fused = (p1 * w1_cast + p2.to(dtype=p1.dtype) * w2_cast) / w_sum
        return self.norm(fused)


class FactorizedSpatiotemporalJEPAWorldModel(nn.Module):
    """GPU-native Spatiotemporal Science & Coastal World Model (Sci-JEPA-WM).

    Decomposes latent dynamics into:
      * Intrinsic parameter-free evolution (autonomous PDE / equilibrium morphology)
      * Zero-anchored forcing-conditioned response ``Delta z_forced(h, f) = g(h, f) - g(h, 0)``
        (wave power disequilibrium, atmospheric storm surge, drought/heat stress)
      * Atlas Forecasting Stochastic Interpolant (``si_head``) for multi-step ensemble rollouts
    and decomposes physical observables into persistent ``morphological_observable`` plus
    reversible ``hydrodynamic_observable_offset``.
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
        si_hidden_dim: int = 128,
        si_sigma: float = 0.25,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if forcing_dim <= 0 or observable_dim <= 0 or horizon <= 0:
            raise ValueError("forcing_dim, observable_dim, and horizon must be positive")
        if heads <= 0 or depth <= 0 or mlp_ratio <= 0 or max_timesteps <= 0:
            raise ValueError("heads, depth, mlp_ratio, and max_timesteps must be positive")
        torch.manual_seed(seed)
        self.stem = stem
        self.dim = stem.dim
        self.grid_size = stem.grid_size
        self.n_patches = stem.n_patches
        self.n_tok = self.n_patches + 1
        self.forcing_dim = forcing_dim
        self.horizon = horizon
        self.spatial_radius = spatial_radius
        self.max_timesteps = max_timesteps
        self.observable_dim = observable_dim

        self.cls_token = nn.Parameter(torch.empty(1, 1, self.dim))
        self.time_embed = nn.Embedding(max_timesteps + horizon + 1, self.dim)

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

        self.intrinsic_head = nn.Linear(self.dim, self.dim)
        self.forced_head = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.si_head = ForecastingStochasticInterpolantHead(
            dim=self.dim,
            cond_dim=2 * self.dim,
            hidden_dim=si_hidden_dim,
            sigma_si=si_sigma,
            seed=seed,
        )

        self.observable_probe = nn.Linear(self.dim, observable_dim)
        self.hydro_offset_head = nn.Sequential(
            nn.Linear(forcing_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, observable_dim),
        )
        self.uncertainty_head = nn.Linear(self.dim, observable_dim)
        self.register_buffer(
            "spatial_mask",
            torch.zeros((self.n_tok, self.n_tok), dtype=torch.float32),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.time_embed.weight, std=0.02)
        for mod in (
            self.intrinsic_head,
            self.forced_head[0],
            self.forced_head[2],
            self.observable_probe,
            self.hydro_offset_head[0],
            self.hydro_offset_head[2],
            self.uncertainty_head,
        ):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    @staticmethod
    def _apply_film(tokens: Tensor, film_gb: Tensor, n_tok: int) -> Tensor:
        gamma, beta = film_gb.chunk(2, dim=-1)
        gamma = gamma.repeat_interleave(n_tok, dim=1)
        beta = beta.repeat_interleave(n_tok, dim=1)
        return tokens * (1.0 + gamma) + beta

    def _embed_step_tokens(self, z_step: Tensor, t_index: int, *, add_pos: bool = False) -> Tensor:
        batch = z_step.shape[0]
        patches = (
            z_step + self.stem.pos_embed.to(device=z_step.device, dtype=z_step.dtype).unsqueeze(0)
            if add_pos
            else z_step
        )
        cls = self.cls_token.to(dtype=z_step.dtype).expand(batch, -1, -1)
        step_tok = torch.cat((cls, patches), dim=1)
        return step_tok + self.time_embed.weight[t_index].to(dtype=z_step.dtype).view(1, 1, -1)

    def _prepend_cls_and_time(self, z_patches: Tensor, t_offset: int = 0) -> Tensor:
        timesteps = z_patches.shape[1]
        out_steps = [
            self._embed_step_tokens(z_patches[:, t], t_offset + t, add_pos=False)
            for t in range(timesteps)
        ]
        return torch.cat(out_steps, dim=1)

    def _compute_forced_delta(self, patch_hidden: Tensor, f_step: Tensor) -> Tensor:
        f_expanded = f_step.unsqueeze(1).expand(-1, self.n_patches, -1)
        zero_f = torch.zeros_like(f_expanded)
        active_out = self.forced_head(torch.cat((patch_hidden, f_expanded), dim=-1))
        null_out = self.forced_head(torch.cat((patch_hidden, zero_f), dim=-1))
        return active_out - null_out

    def _compute_hydro_offset(self, anomaly_step: Tensor) -> Tensor:
        zero_anom = torch.zeros_like(anomaly_step)
        return self.hydro_offset_head(anomaly_step) - self.hydro_offset_head(zero_anom)

    def _validate_rollout_inputs(
        self,
        z_context: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        baseline_context: Tensor | None,
        baseline_future: Tensor | None,
    ) -> tuple[int, int, int]:
        if (
            z_context.ndim != 4
            or z_context.shape[2] != self.n_patches
            or z_context.shape[3] != self.dim
        ):
            raise ValueError("z_context must have shape [B,T,G*G,D]")
        batch, timesteps, _, _ = z_context.shape
        if (
            forcing_context.ndim != 3
            or forcing_context.shape[0] != batch
            or forcing_context.shape[1] != timesteps
            or forcing_context.shape[2] != self.forcing_dim
        ):
            raise ValueError("forcing_context must have shape [B,T,forcing_dim]")
        if (
            forcing_future.ndim != 3
            or forcing_future.shape[0] != batch
            or forcing_future.shape[2] != self.forcing_dim
        ):
            raise ValueError("forcing_future must have shape [B,K,forcing_dim]")
        horizon = int(forcing_future.shape[1])
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"forcing_future horizon must be in 1..{self.horizon}")
        if timesteps + horizon > self.max_timesteps + self.horizon:
            raise ValueError("timesteps + horizon exceeds max_timesteps + horizon")
        if (baseline_context is None) != (baseline_future is None):
            raise ValueError("baseline_context and baseline_future must both be provided or both be None")
        if baseline_context is not None and baseline_context.shape != forcing_context.shape:
            raise ValueError("baseline_context must match forcing_context shape [B,T,forcing_dim]")
        if baseline_future is not None and baseline_future.shape != forcing_future.shape:
            raise ValueError("baseline_future must match forcing_future shape [B,K,forcing_dim]")
        return batch, timesteps, horizon

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
        batch, timesteps, horizon = self._validate_rollout_inputs(
            z_context, forcing_context, forcing_future, baseline_context, baseline_future
        )

        full_forcing = torch.cat((forcing_context, forcing_future), dim=1)
        full_base = (
            torch.zeros_like(full_forcing)
            if (baseline_context is None or baseline_future is None)
            else torch.cat((baseline_context, baseline_future), dim=1).to(
                device=full_forcing.device, dtype=full_forcing.dtype
            )
        )
        film_all, force_emb_all, stress_all = self.forcing_encoder(full_forcing, full_base)
        anomaly_fut = (full_forcing - full_base)[:, timesteps:]
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
        morph_preds: list[Tensor] = []
        hydro_offsets: list[Tensor] = []
        unc_stds: list[Tensor] = []

        for step in range(horizon):
            f_step = force_fut[:, step].to(dtype=h_last.dtype)
            anom_step = anomaly_fut[:, step].to(dtype=h_last.dtype)
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
            z_delta = self._compute_forced_delta(patch_hidden, f_step)
            z_next = z_int + z_delta

            pooled_state = 0.5 * (cls_hidden + z_next.mean(dim=1))
            morph_step = self.observable_probe(pooled_state)
            hydro_step = self._compute_hydro_offset(anom_step)
            obs_step = morph_step + hydro_step
            std_step = functional.softplus(self.uncertainty_head(pooled_state)) + 1e-4

            z_hats.append(z_next)
            z_intrinsics.append(z_int)
            z_forced_deltas.append(z_delta)
            morph_preds.append(morph_step)
            hydro_offsets.append(hydro_step)
            obs_preds.append(obs_step)
            unc_stds.append(std_step)

            h_last = self._embed_step_tokens(z_next, timesteps + step, add_pos=True)

        return ScienceWMOutput(
            z_hat=torch.stack(z_hats, dim=1),
            z_intrinsic=torch.stack(z_intrinsics, dim=1),
            z_forced_delta=torch.stack(z_forced_deltas, dim=1),
            observable_pred=torch.stack(obs_preds, dim=1),
            uncertainty_std=torch.stack(unc_stds, dim=1),
            cumulative_stress=stress_fut,
            morphological_observable=torch.stack(morph_preds, dim=1),
            hydrodynamic_observable_offset=torch.stack(hydro_offsets, dim=1),
        )

    def sample_ensemble_latents(
        self,
        z_context: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        *,
        baseline_context: Tensor | None = None,
        baseline_future: Tensor | None = None,
        ensemble_size: int = 8,
        si_steps: int = 8,
        solver: Literal["euler", "heun", "sde"] = "sde",
        sde_diffusion: float = 0.15,
        seed: int = 0,
    ) -> EnsembleRolloutOutput:
        """Roll out an M-member Atlas Stochastic-Interpolant ensemble over future forcing steps."""
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        batch, timesteps, horizon = self._validate_rollout_inputs(
            z_context, forcing_context, forcing_future, baseline_context, baseline_future
        )

        m = int(ensemble_size)
        z_ctx_exp = (
            z_context.unsqueeze(0)
            .expand(m, batch, timesteps, self.n_patches, self.dim)
            .reshape(m * batch, timesteps, self.n_patches, self.dim)
        )
        f_ctx_exp = (
            forcing_context.unsqueeze(0)
            .expand(m, batch, timesteps, self.forcing_dim)
            .reshape(m * batch, timesteps, self.forcing_dim)
        )
        f_fut_exp = (
            forcing_future.unsqueeze(0)
            .expand(m, batch, horizon, self.forcing_dim)
            .reshape(m * batch, horizon, self.forcing_dim)
        )
        full_forcing = torch.cat((f_ctx_exp, f_fut_exp), dim=1)
        if baseline_context is not None and baseline_future is not None:
            b_ctx_exp = (
                baseline_context.unsqueeze(0)
                .expand(m, batch, timesteps, self.forcing_dim)
                .reshape(m * batch, timesteps, self.forcing_dim)
            )
            b_fut_exp = (
                baseline_future.unsqueeze(0)
                .expand(m, batch, horizon, self.forcing_dim)
                .reshape(m * batch, horizon, self.forcing_dim)
            )
            full_base = torch.cat((b_ctx_exp, b_fut_exp), dim=1).to(
                device=full_forcing.device, dtype=full_forcing.dtype
            )
        else:
            full_base = torch.zeros_like(full_forcing)

        film_all, force_emb_all, _ = self.forcing_encoder(full_forcing, full_base)
        anomaly_fut = (full_forcing - full_base)[:, timesteps:]
        film_ctx = film_all[:, :timesteps]
        force_fut = force_emb_all[:, timesteps:]

        tokens = self._prepend_cls_and_time(z_ctx_exp, t_offset=0)
        tokens = self._apply_film(tokens, film_ctx.to(dtype=tokens.dtype), self.n_tok)

        s_mask = self.spatial_mask.to(dtype=tokens.dtype, device=tokens.device)
        t_mask = causal_neighborhood_mask(
            self.grid_size, timesteps, self.spatial_radius, include_cls=True, device=tokens.device
        ).to(dtype=tokens.dtype)

        past_temporal: list[Tensor] = []
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            per_step = tokens.reshape(m * batch * timesteps, self.n_tok, self.dim)
            per_step = s_block(per_step, s_mask)
            tokens = per_step.reshape(m * batch, timesteps * self.n_tok, self.dim)
            past_temporal.append(tokens)
            tokens = t_block(tokens, t_mask)

        h_last = tokens[:, -self.n_tok :]
        z_prev = z_ctx_exp[:, -1]

        z_steps: list[Tensor] = []
        obs_steps: list[Tensor] = []
        morph_steps: list[Tensor] = []
        hydro_steps: list[Tensor] = []

        for step in range(horizon):
            f_step = force_fut[:, step].to(dtype=h_last.dtype)
            anom_step = anomaly_fut[:, step].to(dtype=h_last.dtype)
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

            f_expanded = f_step.unsqueeze(1).expand(-1, self.n_patches, -1)
            si_cond = torch.cat((patch_hidden, f_expanded), dim=-1)
            z_next = self.si_head.sample(
                z_prev,
                si_cond,
                steps=si_steps,
                solver=solver,
                sde_diffusion=sde_diffusion,
                seed=seed + 1009 * step,
            )

            pooled_state = 0.5 * (cls_hidden + z_next.mean(dim=1))
            morph_step = self.observable_probe(pooled_state)
            hydro_step = self._compute_hydro_offset(anom_step)
            obs_step = morph_step + hydro_step

            z_steps.append(z_next.reshape(m, batch, self.n_patches, self.dim))
            morph_steps.append(morph_step.reshape(m, batch, self.observable_dim))
            hydro_steps.append(hydro_step.reshape(m, batch, self.observable_dim))
            obs_steps.append(obs_step.reshape(m, batch, self.observable_dim))

            z_prev = z_next
            h_last = self._embed_step_tokens(z_next, timesteps + step, add_pos=True)

        z_ens = torch.stack(z_steps, dim=2)
        obs_ens = torch.stack(obs_steps, dim=2)
        morph_ens = torch.stack(morph_steps, dim=2)
        hydro_ens = torch.stack(hydro_steps, dim=2)

        return EnsembleRolloutOutput(
            z_ensemble=z_ens,
            z_mean=z_ens.mean(dim=0),
            z_std=z_ens.std(dim=0, unbiased=False),
            observable_ensemble=obs_ens,
            observable_mean=obs_ens.mean(dim=0),
            observable_std=obs_ens.std(dim=0, unbiased=False),
            morphological_ensemble=morph_ens,
            hydrodynamic_offset=hydro_ens[0],
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
        si_weight: float = 0.25,
        target_mode: Literal["live", "stop_grad"] = "stop_grad",
        sigreg_pool: Literal["patch_flat", "frame_mean", "context_and_pred"] = "patch_flat",
        observable_loss_mode: Literal["crps", "nll"] = "crps",
        weights: Tensor | list[float] | tuple[float, ...] | None = None,
        chunk: int | None = None,
        checkpoint: bool = False,
    ) -> dict[str, Tensor]:
        """Compute joint latent prediction + Atlas Forecasting-SI + HistoryStratifiedSIGReg + CRPS loss."""
        if target_mode not in ("live", "stop_grad"):
            raise ValueError("target_mode must be 'live' or 'stop_grad'")
        if sigreg_pool not in ("patch_flat", "frame_mean", "context_and_pred"):
            raise ValueError("sigreg_pool must be 'patch_flat', 'frame_mean', or 'context_and_pred'")
        if observable_loss_mode not in ("crps", "nll"):
            raise ValueError("observable_loss_mode must be 'crps' or 'nll'")
        if lam_sigreg < 0.0 or observable_weight < 0.0 or si_weight < 0.0:
            raise ValueError("lam_sigreg, observable_weight, and si_weight must be non-negative")

        out = self.forward_latents(
            z_context,
            forcing_context,
            forcing_future,
            baseline_context=baseline_context,
            baseline_future=baseline_future,
        )
        if z_future.shape != out.z_hat.shape:
            raise ValueError("z_future must match out.z_hat shape [B,K,G*G,D]")

        batch, horizon, n_patches, dim = out.z_hat.shape
        history = torch.as_tensor(history_seconds, device=z_future.device)
        if history.ndim != 1 or history.shape[0] != batch:
            raise ValueError("history_seconds must have shape [B]")

        target_z = z_future if target_mode == "live" else z_future.detach()
        latent_mse = (out.z_hat - target_z).square().mean()

        if si_weight > 0.0:
            full_forcing = torch.cat((forcing_context, forcing_future), dim=1)
            full_base = (
                torch.zeros_like(full_forcing)
                if (baseline_context is None or baseline_future is None)
                else torch.cat((baseline_context, baseline_future), dim=1).to(
                    device=full_forcing.device, dtype=full_forcing.dtype
                )
            )
            _, force_emb_all, _ = self.forcing_encoder(full_forcing, full_base)
            force_fut = force_emb_all[:, z_context.shape[1] :].to(dtype=out.z_hat.dtype)
            z_prev_seq = torch.cat((z_context[:, -1:].detach(), target_z[:, :-1]), dim=1)
            f_seq = force_fut.unsqueeze(2).expand(-1, -1, n_patches, -1)
            si_cond = torch.cat((out.z_intrinsic.detach(), f_seq), dim=-1)
            si_out = self.si_head.loss(z_prev_seq, target_z, si_cond)
            si_loss = si_out["loss"]
        else:
            si_loss = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)

        if sigreg_pool == "frame_mean":
            pooled_frames = out.z_hat.mean(dim=2).reshape(batch * horizon, dim)
            repeated_hist = history.repeat_interleave(horizon)
            reg = sigreg(
                pooled_frames,
                repeated_hist,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )
        elif sigreg_pool == "context_and_pred":
            ctx_frames = z_context.mean(dim=2)
            pred_frames = out.z_hat.mean(dim=2)
            all_frames = torch.cat((ctx_frames, pred_frames), dim=1)
            n_total = all_frames.shape[1]
            repeated_hist = history.repeat_interleave(n_total)
            reg = sigreg(
                all_frames.reshape(batch * n_total, dim),
                repeated_hist,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )
        else:
            flat = out.z_hat.reshape(batch * horizon * n_patches, dim)
            repeated_hist = history.repeat_interleave(horizon * n_patches)
            reg = sigreg(
                flat,
                repeated_hist,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )

        with torch.no_grad():
            frame_latents = (
                out.z_hat.detach().mean(dim=2).reshape(batch * horizon, dim).to(torch.float64)
            )
            svals = torch.linalg.svdvals(frame_latents)
            sval_sum = svals.sum()
            if float(sval_sum.item()) > 1e-12:
                probs = (svals / sval_sum).clamp_min(1e-12)
                eff_rank = torch.exp(-(probs * torch.log(probs)).sum()).to(dtype=latent_mse.dtype)
            else:
                eff_rank = torch.ones((), dtype=latent_mse.dtype, device=latent_mse.device)

        if target_observable is not None:
            if target_observable.shape != out.observable_pred.shape:
                raise ValueError("target_observable must match observable_pred shape [B,K,O]")
            tgt = target_observable.to(
                device=out.observable_pred.device, dtype=out.observable_pred.dtype
            )
            var = out.uncertainty_std.square()
            obs_nll = 0.5 * ((tgt - out.observable_pred).square() / var + torch.log(var)).mean()
            obs_crps = gaussian_crps(out.observable_pred, out.uncertainty_std, tgt)
            obs_mse = (tgt - out.observable_pred).square().mean()
            obs_term = obs_crps if observable_loss_mode == "crps" else obs_nll
        else:
            obs_nll = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)
            obs_crps = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)
            obs_mse = torch.zeros((), dtype=latent_mse.dtype, device=latent_mse.device)
            obs_term = obs_crps

        total = (
            latent_mse
            + float(si_weight) * si_loss
            + float(lam_sigreg) * reg["sigreg"]
            + float(observable_weight) * obs_term
        )
        return {
            "loss": total,
            "latent_mse": latent_mse,
            "si_loss": si_loss,
            "sigreg": reg["sigreg"],
            "ep": reg["ep"],
            "ep_over_n": reg["ep_over_n"],
            "counts": reg["counts"],
            "null": reg["null"],
            "effective_rank": eff_rank,
            "observable_crps": obs_crps,
            "observable_nll": obs_nll,
            "observable_mse": obs_mse,
            "mean_uncertainty_std": out.uncertainty_std.detach().mean(),
            "forced_energy_ratio": (
                out.z_forced_delta.detach().norm()
                / (out.z_intrinsic.detach().norm() + out.z_forced_delta.detach().norm() + 1e-8)
            ),
        }


class ForcingElasticityEvaluator:
    """Object-oriented evaluator for counterfactual forcing responsiveness and elasticity."""

    def __init__(self, model: FactorizedSpatiotemporalJEPAWorldModel) -> None:
        self.model = model

    def contrastive_loss(
        self,
        z_context: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        z_future: Tensor,
        *,
        baseline_context: Tensor | None = None,
        baseline_future: Tensor | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Tensor]:
        """Compute B x B counterfactual Forcing-InfoNCE and forcing elasticity."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        batch = z_context.shape[0]
        if batch < 2:
            raise ValueError("forcing_contrastive_loss requires batch size B >= 2")

        target_flat = z_future.detach().reshape(batch, -1)
        ctx_rep = (
            z_context.unsqueeze(1)
            .expand(batch, batch, *z_context.shape[1:])
            .reshape(batch * batch, *z_context.shape[1:])
        )
        f_ctx_rep = (
            forcing_context.unsqueeze(1)
            .expand(batch, batch, *forcing_context.shape[1:])
            .reshape(batch * batch, *forcing_context.shape[1:])
        )
        f_fut_rep = (
            forcing_future.unsqueeze(0)
            .expand(batch, batch, *forcing_future.shape[1:])
            .reshape(batch * batch, *forcing_future.shape[1:])
        )
        b_ctx_rep = (
            None
            if baseline_context is None
            else baseline_context.unsqueeze(1)
            .expand(batch, batch, *baseline_context.shape[1:])
            .reshape(batch * batch, *baseline_context.shape[1:])
        )
        b_fut_rep = (
            None
            if baseline_future is None
            else baseline_future.unsqueeze(0)
            .expand(batch, batch, *baseline_future.shape[1:])
            .reshape(batch * batch, *baseline_future.shape[1:])
        )

        pred_all = self.model.forward_latents(
            ctx_rep,
            f_ctx_rep,
            f_fut_rep,
            baseline_context=b_ctx_rep,
            baseline_future=b_fut_rep,
        ).z_hat.reshape(batch, batch, -1)

        diffs = pred_all - target_flat.unsqueeze(1)
        dist_matrix = diffs.square().mean(dim=-1)
        logits = -dist_matrix / float(temperature)
        labels = torch.arange(batch, dtype=torch.long, device=z_context.device)
        nce_loss = functional.cross_entropy(logits, labels)

        true_preds = pred_all[labels, labels].unsqueeze(1)
        off_diag_mask = ~torch.eye(batch, dtype=torch.bool, device=z_context.device)
        cf_shifts = (pred_all - true_preds).square().mean(dim=-1)[off_diag_mask].sqrt()
        elasticity = cf_shifts.mean()
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).to(dtype=z_context.dtype).mean()

        return {
            "loss": nce_loss,
            "forcing_elasticity": elasticity,
            "forcing_identification_accuracy": acc,
            "distance_matrix": dist_matrix.detach(),
        }

    def fd_elasticity_and_zero_probe(
        self,
        z_context: Tensor,
        forcing_context: Tensor,
        forcing_future: Tensor,
        *,
        baseline_context: Tensor | None = None,
        baseline_future: Tensor | None = None,
        delta: float = 1e-2,
        n_probes: int = 4,
        seed: int = 0,
    ) -> dict[str, Tensor]:
        """Compute Hutchinson finite-difference forcing elasticity and zero-forcing probe."""
        if delta <= 0.0 or n_probes < 1:
            raise ValueError("delta > 0 and n_probes >= 1 required")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            base_out = self.model.forward_latents(
                z_context,
                forcing_context,
                forcing_future,
                baseline_context=baseline_context,
                baseline_future=baseline_future,
            )
            batch = z_context.shape[0]
            base_flat = base_out.z_hat.reshape(batch, -1)

            dir_norms: list[Tensor] = []
            for _ in range(n_probes):
                eps_f = torch.randn(
                    forcing_future.shape,
                    generator=gen,
                    dtype=forcing_future.dtype,
                    device="cpu",
                ).to(forcing_future.device)
                eps_f = eps_f / torch.linalg.vector_norm(
                    eps_f.reshape(batch, -1), dim=-1, keepdim=True
                ).clamp_min(1e-12).view(batch, 1, 1)
                pert_out = self.model.forward_latents(
                    z_context,
                    forcing_context,
                    forcing_future + float(delta) * eps_f,
                    baseline_context=baseline_context,
                    baseline_future=baseline_future,
                )
                pert_flat = pert_out.z_hat.reshape(batch, -1)
                dir_norms.append(
                    torch.linalg.vector_norm(pert_flat - base_flat, dim=-1) / float(delta)
                )
            fd_elasticity = torch.stack(dir_norms, dim=0).mean()

            zero_ctx = torch.zeros_like(forcing_context)
            zero_fut = torch.zeros_like(forcing_future)
            zero_out = self.model.forward_latents(
                z_context,
                zero_ctx,
                zero_fut,
                baseline_context=torch.zeros_like(forcing_context)
                if baseline_context is not None
                else None,
                baseline_future=torch.zeros_like(forcing_future)
                if baseline_future is not None
                else None,
            )
            zero_response_gap = (base_out.z_hat - zero_out.z_hat).square().mean().sqrt()
            zero_forced_delta_norm = zero_out.z_forced_delta.abs().max()

        return {
            "fd_elasticity": fd_elasticity,
            "zero_forcing_response_gap": zero_response_gap,
            "zero_forced_delta_max": zero_forced_delta_norm,
        }


def forcing_contrastive_loss(
    model: FactorizedSpatiotemporalJEPAWorldModel,
    z_context: Tensor,
    forcing_context: Tensor,
    forcing_future: Tensor,
    z_future: Tensor,
    *,
    baseline_context: Tensor | None = None,
    baseline_future: Tensor | None = None,
    temperature: float = 0.1,
) -> dict[str, Tensor]:
    """Functional wrapper around ``ForcingElasticityEvaluator.contrastive_loss``."""
    return ForcingElasticityEvaluator(model).contrastive_loss(
        z_context,
        forcing_context,
        forcing_future,
        z_future,
        baseline_context=baseline_context,
        baseline_future=baseline_future,
        temperature=temperature,
    )


def forcing_fd_elasticity_and_zero_probe(
    model: FactorizedSpatiotemporalJEPAWorldModel,
    z_context: Tensor,
    forcing_context: Tensor,
    forcing_future: Tensor,
    *,
    baseline_context: Tensor | None = None,
    baseline_future: Tensor | None = None,
    delta: float = 1e-2,
    n_probes: int = 4,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Functional wrapper around ``ForcingElasticityEvaluator.fd_elasticity_and_zero_probe``."""
    return ForcingElasticityEvaluator(model).fd_elasticity_and_zero_probe(
        z_context,
        forcing_context,
        forcing_future,
        baseline_context=baseline_context,
        baseline_future=baseline_future,
        delta=delta,
        n_probes=n_probes,
        seed=seed,
    )


__all__ = [
    "CandidateSensorPass",
    "CoastalHydrodynamicConfig",
    "CoastalHydrodynamicModel",
    "CounterfactualVoIEvaluator",
    "EO_HISTORY_STRATA_SECONDS",
    "EnsembleRolloutOutput",
    "FactorizedSpatiotemporalJEPAWorldModel",
    "ForcingElasticityEvaluator",
    "ForecastingStochasticInterpolantHead",
    "MultiSensorContinuumPatchStem",
    "PhysicalForcingEncoder",
    "ScienceWMOutput",
    "ShoreForDisequilibriumResult",
    "StochasticInterpolantConfig",
    "StockdonRunupResult",
    "StratifiedConformalAlertCalibrator",
    "SubspaceDataAssimilator",
    "SubspacePartition",
    "assimilate_subspace_observation",
    "build_coastal_forcing_tensor",
    "ensemble_crps",
    "forcing_contrastive_loss",
    "forcing_fd_elasticity_and_zero_probe",
    "gaussian_crps",
    "gpu_counterfactual_observation_value",
    "score_observation_reveal_policies",
    "shorefor_disequilibrium_stress",
    "stockdon_wave_setup_and_runup",
]
