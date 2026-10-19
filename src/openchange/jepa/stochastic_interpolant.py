"""Object-oriented Forecasting Stochastic Interpolant (NVIDIA Earth-2 Atlas).

Implements the state-to-state Forecasting Stochastic Interpolant formulation
from NVIDIA Earth-2 Atlas (Kossaifi et al. 2026, arXiv:2601.18111) and Chen
et al. (ICML 2024, arXiv:2403.13724) for multi-step latent world-model
ensemble rollouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class StochasticInterpolantConfig:
    """Configuration for ``ForecastingStochasticInterpolantHead``."""

    dim: int
    cond_dim: int
    hidden_dim: int = 128
    time_dim: int = 32
    sigma_si: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.dim <= 0
            or self.cond_dim <= 0
            or self.hidden_dim <= 0
            or self.time_dim <= 0
            or self.time_dim % 2 != 0
        ):
            raise ValueError("positive dimensions and even time_dim required")
        if self.sigma_si < 0.0 or not math.isfinite(self.sigma_si):
            raise ValueError("sigma_si must be finite and non-negative")


@dataclass(frozen=True)
class EnsembleRolloutOutput:
    """Typed output of an M-member stochastic-interpolant world-model rollout."""

    z_ensemble: Tensor
    z_mean: Tensor
    z_std: Tensor
    observable_ensemble: Tensor
    observable_mean: Tensor
    observable_std: Tensor
    morphological_ensemble: Tensor
    hydrodynamic_offset: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "z_ensemble": self.z_ensemble,
            "z_mean": self.z_mean,
            "z_std": self.z_std,
            "observable_ensemble": self.observable_ensemble,
            "observable_mean": self.observable_mean,
            "observable_std": self.observable_std,
            "morphological_ensemble": self.morphological_ensemble,
            "hydrodynamic_offset": self.hydrodynamic_offset,
        }

    def __getitem__(self, key: str) -> Tensor:
        return self.as_dict()[key]


class ForecastingStochasticInterpolantHead(nn.Module):
    """State-to-state Forecasting Stochastic Interpolant (NVIDIA Earth-2 Atlas arXiv:2601.18111;
    Chen et al. ICML 2024 arXiv:2403.13724).

    Bridges point-mass current latent ``x_0 = z_t`` and next latent ``x_1 = z_{t+1}``:
      ``I_tau = (1 - tau) * z_t + tau * z_{t+1} + gamma(tau) * eps``,
      ``gamma(tau) = sigma_si * tau * (1 - tau)``,  ``d gamma / d tau = sigma_si * (1 - 2 * tau)``.
    Learns both conditional drift ``b_theta`` and denoiser ``eta_theta`` so sampling
    supports deterministic ODE (Euler/Heun) and stochastic SDE rollouts with
    post-training tunable diffusion ``sde_diffusion``.
    """

    def __init__(
        self,
        *,
        dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        sigma_si: float = 0.25,
        seed: int = 0,
        config: StochasticInterpolantConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = (
            config
            if config is not None
            else StochasticInterpolantConfig(
                dim=dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                sigma_si=sigma_si,
                seed=seed,
            )
        )
        self.config = cfg
        torch.manual_seed(cfg.seed)
        self.dim = cfg.dim
        self.cond_dim = cfg.cond_dim
        self.hidden_dim = cfg.hidden_dim
        self.time_dim = cfg.time_dim
        self.sigma_si = float(cfg.sigma_si)

        in_features = 2 * self.dim + self.cond_dim + self.time_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.drift_out = nn.Linear(self.hidden_dim, self.dim)
        self.noise_out = nn.Linear(self.hidden_dim, self.dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def _time_embed(self, tau: Tensor) -> Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=tau.dtype, device=tau.device)
            / max(half - 1, 1)
        )
        angles = tau.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def gamma(self, tau: Tensor) -> Tensor:
        return self.sigma_si * tau * (1.0 - tau)

    def d_gamma(self, tau: Tensor) -> Tensor:
        return self.sigma_si * (1.0 - 2.0 * tau)

    def forward(
        self, x_tau: Tensor, tau: Tensor, z_prev: Tensor, cond: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(drift_pred [..., D], noise_pred [..., D])``."""
        lead_shape = x_tau.shape[:-1]
        flat_x = x_tau.reshape(-1, self.dim)
        flat_prev = z_prev.to(device=flat_x.device, dtype=flat_x.dtype).reshape(-1, self.dim)
        flat_cond = cond.to(device=flat_x.device, dtype=flat_x.dtype).reshape(-1, self.cond_dim)
        flat_tau = tau.to(device=flat_x.device, dtype=flat_x.dtype).reshape(-1)
        if flat_tau.numel() == 1 and flat_x.shape[0] > 1:
            flat_tau = flat_tau.expand(flat_x.shape[0])
        elif flat_tau.numel() != flat_x.shape[0]:
            raise ValueError("tau must broadcast to the flattened batch/token count")

        t_emb = self._time_embed(flat_tau)
        h = self.backbone(torch.cat((flat_x, flat_prev, flat_cond, t_emb), dim=-1))
        drift = self.drift_out(h).reshape(*lead_shape, self.dim)
        noise = self.noise_out(h).reshape(*lead_shape, self.dim)
        return drift, noise

    def loss(
        self,
        z_prev: Tensor,
        z_next: Tensor,
        cond: Tensor,
        *,
        tau: Tensor | None = None,
        eps: Tensor | None = None,
        seed: int = 0,
    ) -> dict[str, Tensor]:
        """Compute Forecasting Stochastic Interpolant drift + denoiser regression loss."""
        if z_prev.shape != z_next.shape or z_prev.shape[-1] != self.dim:
            raise ValueError("z_prev and z_next must share shape [..., dim]")
        if cond.shape[:-1] != z_prev.shape[:-1] or cond.shape[-1] != self.cond_dim:
            raise ValueError("cond must have shape [*z_prev.shape[:-1], cond_dim]")

        flat_prev = z_prev.reshape(-1, self.dim)
        flat_next = z_next.reshape(-1, self.dim)
        flat_cond = cond.reshape(-1, self.cond_dim)
        n_items = flat_prev.shape[0]

        gen = torch.Generator(device="cpu").manual_seed(seed)
        if tau is None:
            tau_flat = torch.rand(
                (n_items,), generator=gen, dtype=flat_prev.dtype, device="cpu"
            ).to(flat_prev.device)
            tau_flat = tau_flat.clamp(1e-3, 1.0 - 1e-3)
        else:
            tau_flat = tau.to(device=flat_prev.device, dtype=flat_prev.dtype).reshape(-1)
            if tau_flat.numel() == 1:
                tau_flat = tau_flat.expand(n_items)
            elif tau_flat.numel() != n_items:
                raise ValueError("tau must match flattened sample count")

        if eps is None:
            eps_flat = torch.randn(
                flat_prev.shape, generator=gen, dtype=flat_prev.dtype, device="cpu"
            ).to(flat_prev.device)
        else:
            eps_flat = eps.to(device=flat_prev.device, dtype=flat_prev.dtype).reshape(
                n_items, self.dim
            )

        gamma_t = self.gamma(tau_flat).unsqueeze(-1)
        d_gamma_t = self.d_gamma(tau_flat).unsqueeze(-1)
        tau_col = tau_flat.unsqueeze(-1)

        i_tau = (1.0 - tau_col) * flat_prev + tau_col * flat_next + gamma_t * eps_flat
        target_drift = (flat_next - flat_prev) + d_gamma_t * eps_flat

        drift_pred, noise_pred = self.forward(i_tau, tau_flat, flat_prev, flat_cond)
        drift_loss = (drift_pred - target_drift).square().mean()
        noise_loss = (noise_pred - eps_flat).square().mean()
        total = drift_loss + 0.5 * noise_loss
        return {
            "loss": total,
            "si_drift_loss": drift_loss,
            "si_noise_loss": noise_loss,
        }

    def sample(
        self,
        z_prev: Tensor,
        cond: Tensor,
        *,
        steps: int = 8,
        solver: Literal["euler", "heun", "sde"] = "heun",
        sde_diffusion: float = 0.15,
        seed: int | None = None,
    ) -> Tensor:
        """Integrate the Forecasting Stochastic Interpolant from ``tau=0 (z_prev)`` to ``tau=1``."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if solver not in ("euler", "heun", "sde"):
            raise ValueError("solver must be 'euler', 'heun', or 'sde'")
        if sde_diffusion < 0.0:
            raise ValueError("sde_diffusion must be non-negative")
        if (
            z_prev.shape[-1] != self.dim
            or cond.shape[:-1] != z_prev.shape[:-1]
            or cond.shape[-1] != self.cond_dim
        ):
            raise ValueError("z_prev and cond shapes must align on leading dimensions")

        lead_shape = z_prev.shape[:-1]
        flat_prev = z_prev.reshape(-1, self.dim)
        flat_cond = cond.to(device=flat_prev.device, dtype=flat_prev.dtype).reshape(
            -1, self.cond_dim
        )
        n_items = flat_prev.shape[0]
        dt = 1.0 / float(steps)

        gen = torch.Generator(device="cpu").manual_seed(0 if seed is None else int(seed))
        x = flat_prev.clone()

        if solver == "sde" and self.sigma_si > 0.0:
            latent_bridge_xi = torch.randn(
                flat_prev.shape, generator=gen, dtype=flat_prev.dtype, device="cpu"
            ).to(flat_prev.device)
        else:
            latent_bridge_xi = None

        for step in range(steps):
            t0 = (step + 0.5 / steps) if step == 0 else (step * dt)
            tau_cur = torch.full(
                (n_items,), t0, dtype=flat_prev.dtype, device=flat_prev.device
            )
            v1, n1 = self.forward(x, tau_cur, flat_prev, flat_cond)

            if solver == "sde":
                dw = (
                    torch.randn(
                        flat_prev.shape, generator=gen, dtype=flat_prev.dtype, device="cpu"
                    ).to(flat_prev.device)
                    * math.sqrt(dt)
                )
                d_gamma_cur = self.d_gamma(tau_cur).unsqueeze(-1)
                mean_vel = v1 - d_gamma_cur * n1
                bridge_term = (
                    d_gamma_cur * latent_bridge_xi * dt
                    if latent_bridge_xi is not None
                    else torch.zeros_like(x)
                )
                x = x + dt * mean_vel + bridge_term + float(sde_diffusion) * dw
            elif solver == "euler" or step == steps - 1:
                x = x + dt * v1
            else:
                x_euler = x + dt * v1
                tau_next = torch.full(
                    (n_items,),
                    min(1.0 - 1e-4, (step + 1) * dt),
                    dtype=flat_prev.dtype,
                    device=flat_prev.device,
                )
                v2, _ = self.forward(x_euler, tau_next, flat_prev, flat_cond)
                x = x + 0.5 * dt * (v1 + v2)

        return x.reshape(*lead_shape, self.dim)


__all__ = [
    "EnsembleRolloutOutput",
    "ForecastingStochasticInterpolantHead",
    "StochasticInterpolantConfig",
]
