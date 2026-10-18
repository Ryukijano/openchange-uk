"""Track A: causal clean-JEPA world-action model (WAM).

Implements:
  * Theorem-1 spatiotemporal causal attention masks (patch (u,t) attends to
    (u',t') with t' <= t and |u-u'|_inf <= r, plus same-or-earlier CLS tokens).
  * History-stratified Epps-Pulley SIGReg over H in {0.0, 2.4, 9.6, 19.2, 38.4}s.
  * A FiLM-conditioned causal patch-token world model g_theta with multi-step
    latent rollout and a live/stop-grad target mode.
  * Reachable-error / 2x2 Oracle Factorial evaluation.
  * Policy-noise-space (LeWAM/MBDPO-style) planning vs. plain CEM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from openchange.jepa.losses import ep_null_expectation, ep_per_view_stratum
from openchange.jepa.model import sincos_pos_embed

HISTORY_STRATA_SECONDS: tuple[float, ...] = (0.0, 2.4, 9.6, 19.2, 38.4)
_NEG_INF = float("-inf")


def causal_neighborhood_mask(
    grid_size: int,
    timesteps: int,
    spatial_radius: int,
    include_cls: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """Additive attention mask implementing the Theorem-1 causal neighborhood.

    Token layout per timestep t is [cls_t?, patch_(u,t) for u in row-major
    grid order]. Query (u,t) may attend key (u',t') iff t' <= t and
    max(|dr|,|dc|) <= spatial_radius; a CLS key at time t' <= t is always
    allowed. Any key with t' > t is strictly blocked (-inf), so no future
    information or gradient can flow.

    Returns a float32 [L, L] tensor with 0 for allowed and -inf for blocked
    pairs, suitable as ``attn_mask`` for scaled_dot_product_attention.
    """
    for name, value in (
        ("grid_size", grid_size),
        ("timesteps", timesteps),
        ("spatial_radius", spatial_radius),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if spatial_radius < 0:
        raise ValueError("spatial_radius must be nonnegative")

    n_patches = grid_size * grid_size
    n_tok = n_patches + (1 if include_cls else 0)

    coordinates = torch.arange(grid_size, dtype=torch.float64)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    positions = torch.stack((rows.reshape(-1), columns.reshape(-1)), dim=1)
    chebyshev = (positions[:, None, :] - positions[None, :, :]).abs().amax(dim=-1)
    spatial_allowed = chebyshev <= spatial_radius

    if include_cls:
        block = torch.ones((n_tok, n_tok), dtype=torch.bool)
        block[1:, 1:] = spatial_allowed
    else:
        block = spatial_allowed

    time_allowed = torch.tril(torch.ones((timesteps, timesteps), dtype=torch.bool))
    allowed = time_allowed.repeat_interleave(n_tok, dim=0).repeat_interleave(n_tok, dim=1)
    allowed &= block.repeat(timesteps, timesteps)

    mask = torch.zeros((timesteps * n_tok, timesteps * n_tok), dtype=torch.float32)
    mask.masked_fill_(~allowed, _NEG_INF)
    return mask.to(device=device) if device is not None else mask


def bidirectional_neighborhood_mask(
    grid_size: int,
    timesteps: int,
    spatial_radius: int,
    include_cls: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """Matched bidirectional context mask (control arm for the Long-WAM vs. Causal-JEPA MVE).

    Allows tokens at any context timestep t to attend to tokens at any context
    timestep t' within Chebyshev spatial radius ``spatial_radius`` (plus CLS).
    """
    for name, value in (
        ("grid_size", grid_size),
        ("timesteps", timesteps),
        ("spatial_radius", spatial_radius),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if grid_size <= 0 or timesteps <= 0 or spatial_radius < 0:
        raise ValueError("grid_size > 0, timesteps > 0, spatial_radius >= 0 required")

    n_patches = grid_size * grid_size
    n_tok = n_patches + (1 if include_cls else 0)
    coordinates = torch.arange(grid_size, dtype=torch.float64)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    positions = torch.stack((rows.reshape(-1), columns.reshape(-1)), dim=1)
    chebyshev = (positions[:, None, :] - positions[None, :, :]).abs().amax(dim=-1)
    spatial_allowed = chebyshev <= spatial_radius

    if include_cls:
        block = torch.ones((n_tok, n_tok), dtype=torch.bool)
        block[1:, 1:] = spatial_allowed
    else:
        block = spatial_allowed

    allowed = block.repeat(timesteps, timesteps)
    mask = torch.zeros((timesteps * n_tok, timesteps * n_tok), dtype=torch.float32)
    mask.masked_fill_(~allowed, _NEG_INF)
    return mask.to(device=device) if device is not None else mask


class HistoryStratifiedSIGReg(nn.Module):
    """FP32 Epps-Pulley Gaussian regularizer stratified by history length.

    Samples are assigned to the nearest stratum in HISTORY_STRATA_SECONDS;
    ties go to the lower index. All characteristic-function math runs in
    float32 (or float64 for float64 inputs) with autocast disabled via
    :func:`ep_per_view_stratum`.
    """

    n_strata: int = len(HISTORY_STRATA_SECONDS)

    def __init__(self, strata_seconds: tuple[float, ...] = HISTORY_STRATA_SECONDS):
        super().__init__()
        if len(strata_seconds) == 0 or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            for value in strata_seconds
        ):
            raise ValueError("strata_seconds must be nonnegative numbers")
        boundaries = torch.as_tensor(strata_seconds, dtype=torch.float64)
        if boundaries.ndim != 1 or torch.any(boundaries[1:] <= boundaries[:-1]):
            raise ValueError("strata_seconds must be strictly increasing")
        self.n_strata = int(boundaries.numel())
        self.register_buffer("boundaries", boundaries, persistent=False)

    def strata_index(self, history_seconds: Tensor) -> Tensor:
        """Map history lengths (seconds) to stratum indices [0, n_strata)."""
        history = torch.as_tensor(history_seconds)
        if history.ndim != 1:
            raise ValueError("history_seconds must have shape [N]")
        if not torch.isfinite(history).all() or bool((history < 0).any()):
            raise ValueError("history_seconds must be finite and nonnegative")
        distances = (history.to(torch.float64)[:, None] - self.boundaries.cpu()[None, :]).abs()
        return distances.argmin(dim=1).to(dtype=torch.int64, device=history.device)

    def forward(
        self,
        z: Tensor,
        history_seconds: Tensor,
        directions: Tensor,
        *,
        weights: Tensor | list[float] | tuple[float, ...] | None = None,
        chunk: int | None = None,
        checkpoint: bool = False,
    ) -> dict[str, Tensor]:
        """Stratified EP loss for embeddings z of shape [N, K] or [V, N, K]."""
        if z.ndim == 2:
            z = z.unsqueeze(0)
        if z.ndim != 3:
            raise ValueError("z must have shape [N,K] or [V,N,K]")
        _, samples, features = z.shape
        strata = self.strata_index(history_seconds)
        if strata.shape[0] != samples:
            raise ValueError("history_seconds length must equal N")
        if directions.ndim != 2 or directions.shape[0] != features or directions.shape[1] <= 0:
            raise ValueError("directions must have shape [K,M] with M positive")
        if weights is None:
            weights = torch.full((self.n_strata,), 1.0 / self.n_strata, dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64)
        if weights.shape != (self.n_strata,):
            raise ValueError("weights must have shape [n_strata]")
        if bool((weights < 0).any()) or abs(float(weights.sum()) - 1.0) > 1e-12:
            raise ValueError("weights must be nonnegative and sum to 1")

        ep, counts = ep_per_view_stratum(
            z, strata, self.n_strata, directions, chunk=chunk, checkpoint=checkpoint
        )
        weights_compute = weights.to(device=z.device, dtype=ep.dtype)
        sigreg = (ep * weights_compute[None, :]).sum(dim=1).mean()
        return {
            "sigreg": sigreg,
            "ep": ep.detach(),
            "ep_over_n": ep.detach() / counts[None, :],
            "counts": counts.detach(),
            "strata": strata,
            "null": torch.tensor(ep_null_expectation(), dtype=ep.dtype, device=z.device),
        }


class _CausalBlock(nn.Module):
    """Pre-norm attention block; ``forward`` uses an additive mask, ``step``
    appends one timestep's tokens attending to cached previous block inputs."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int):
        super().__init__()
        if dim % heads:
            raise ValueError("model dimension must be divisible by the number of heads")
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def _qkv(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = value.shape
        qkv = self.qkv(self.norm1(value)).reshape(batch, length, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, value: Tensor, attn_mask: Tensor) -> Tensor:
        query, key, content = self._qkv(value)
        attended = functional.scaled_dot_product_attention(query, key, content, attn_mask=attn_mask)
        attended = attended.transpose(1, 2).reshape(value.shape[0], value.shape[1], -1)
        value = value + self.proj(attended)
        return value + self.mlp(self.norm2(value))

    def step(self, cur: Tensor, past: Tensor) -> Tensor:
        """One-timestep update: cur [B,n,D] attends to cat(past, cur)."""
        joined = torch.cat((past, cur), dim=1)
        query, key, content = self._qkv(joined)
        n_cur = cur.shape[1]
        attended = functional.scaled_dot_product_attention(
            query[:, :, -n_cur:], key, content
        )
        attended = attended.transpose(1, 2).reshape(cur.shape[0], n_cur, -1)
        cur = cur + self.proj(attended)
        return cur + self.mlp(self.norm2(cur))


@dataclass
class WAMOutput:
    z_hat: Tensor
    context: Tensor


class CausalPatchJEPAWorldModel(nn.Module):
    """Causal patch-token world model g_theta(z_{<=t}, a_{t:t+k-1}, p_{<=t})."""

    def __init__(
        self,
        *,
        dim: int,
        grid_size: int,
        action_dim: int,
        proprio_dim: int,
        horizon: int,
        heads: int,
        depth: int,
        mlp_ratio: int,
        spatial_radius: int,
        max_timesteps: int,
        causal: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        for name, value in (
            ("dim", dim),
            ("grid_size", grid_size),
            ("action_dim", action_dim),
            ("proprio_dim", proprio_dim),
            ("horizon", horizon),
            ("depth", depth),
            ("spatial_radius", spatial_radius),
            ("max_timesteps", max_timesteps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if horizon > 64 or max_timesteps + horizon > 4096:
            raise ValueError("horizon/max_timesteps exceed supported bounds")
        torch.manual_seed(seed)
        self.dim = dim
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.n_tok = self.n_patches + 1
        self.horizon = horizon
        self.spatial_radius = spatial_radius
        self.max_timesteps = max_timesteps
        self.causal = bool(causal)

        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.register_buffer(
            "pos_embed", sincos_pos_embed(grid_size, dim, dtype=torch.float32), persistent=False
        )
        self.time_embed = nn.Embedding(max_timesteps + horizon + 1, dim)
        self.proprio_film = nn.Linear(proprio_dim, 2 * dim)
        self.action_film = nn.Linear(action_dim, 2 * dim)
        self.blocks = nn.ModuleList(
            [_CausalBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.predict = nn.Linear(dim, dim)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proprio_film.weight, std=0.02)
        nn.init.zeros_(self.proprio_film.bias)
        nn.init.trunc_normal_(self.action_film.weight, std=0.02)
        nn.init.zeros_(self.action_film.bias)

    @staticmethod
    def _film(tokens: Tensor, gamma_beta: Tensor, n_tok: int) -> Tensor:
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.repeat_interleave(n_tok, dim=1)
        beta = beta.repeat_interleave(n_tok, dim=1)
        return tokens * (1 + gamma) + beta

    def _embed_tokens(self, patches: Tensor, t_index: int) -> Tensor:
        batch = patches.shape[0]
        cls = self.cls_token.to(dtype=patches.dtype).expand(batch, -1, -1)
        tokens = torch.cat(
            (cls, patches + self.pos_embed.to(dtype=patches.dtype).unsqueeze(0)), dim=1
        )
        return tokens + self.time_embed.weight[t_index].to(dtype=patches.dtype).view(1, 1, -1)

    def _encode(self, z_context: Tensor, proprio: Tensor) -> tuple[Tensor, list[Tensor]]:
        _, timesteps, _, _ = z_context.shape
        steps = [self._embed_tokens(z_context[:, t], t) for t in range(timesteps)]
        value = torch.cat(steps, dim=1)
        film_params = self.proprio_film(proprio.to(dtype=value.dtype))
        value = self._film(value, film_params, self.n_tok)
        mask_fn = causal_neighborhood_mask if self.causal else bidirectional_neighborhood_mask
        mask = mask_fn(
            self.grid_size, timesteps, self.spatial_radius, include_cls=True, device=value.device
        ).to(dtype=value.dtype)
        block_inputs: list[Tensor] = []
        for block in self.blocks:
            block_inputs.append(value)
            value = block(value, mask)
        return value, block_inputs

    def forward(self, z_context: Tensor, actions: Tensor, proprio: Tensor) -> WAMOutput:
        """z_context [B,T,N,D], actions [B,K,A], proprio [B,T,P] -> z_hat [B,K,N,D]."""
        if (
            z_context.ndim != 4
            or z_context.shape[2] != self.n_patches
            or z_context.shape[3] != self.dim
        ):
            raise ValueError("z_context must have shape [B,T,G*G,dim]")
        batch, timesteps, _, _ = z_context.shape
        if (
            actions.ndim != 3
            or actions.shape[0] != batch
            or actions.shape[2] != self.action_film.in_features
        ):
            raise ValueError("actions must have shape [B,K,action_dim]")
        horizon = actions.shape[1]
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"actions must provide 1..{self.horizon} future steps")
        if (
            proprio.ndim != 3
            or proprio.shape[:2] != (batch, timesteps)
            or proprio.shape[2] != self.proprio_film.in_features
        ):
            raise ValueError("proprio must have shape [B,T,proprio_dim]")
        if timesteps + horizon > self.max_timesteps + self.horizon:
            raise ValueError("timesteps + horizon exceeds max_timesteps")

        hidden, block_inputs = self._encode(z_context, proprio)
        h_last = hidden[:, -self.n_tok :]
        past = list(block_inputs)
        latents: list[Tensor] = []
        for step in range(horizon):
            film_params = self.action_film(
                actions[:, step].to(dtype=h_last.dtype)
            ).unsqueeze(1)
            conditioned = self._film(h_last, film_params, self.n_tok)
            for index, block in enumerate(self.blocks):
                updated = block.step(conditioned, past[index])
                past[index] = torch.cat((past[index], conditioned), dim=1)
                conditioned = updated
            nxt = self.predict(self.norm(conditioned))
            latents.append(nxt[:, 1:, :])
            h_last = self._embed_tokens(nxt[:, 1:, :], timesteps + step)
        return WAMOutput(z_hat=torch.stack(latents, dim=1), context=hidden)

    def wam_loss(
        self,
        z_context: Tensor,
        actions: Tensor,
        proprio: Tensor,
        z_future: Tensor,
        *,
        history_seconds: Tensor,
        directions: Tensor,
        sigreg: HistoryStratifiedSIGReg,
        target_mode: Literal["live", "stop_grad"] = "stop_grad",
        lam: float = 1.0,
        weights: Tensor | list[float] | tuple[float, ...] | None = None,
        chunk: int | None = None,
        checkpoint: bool = False,
    ) -> dict[str, Tensor]:
        """Multi-step latent prediction loss + history-stratified SIGReg."""
        if target_mode not in ("live", "stop_grad"):
            raise ValueError("target_mode must be 'live' or 'stop_grad'")
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        output = self.forward(z_context, actions, proprio)
        if z_future.shape != output.z_hat.shape:
            raise ValueError("z_future must have shape [B,K,G*G,dim] matching predictions")
        target = z_future if target_mode == "live" else z_future.detach()
        prediction = (output.z_hat - target).square().sum() / output.z_hat.numel()

        batch, horizon, n_patches, dim = output.z_hat.shape
        flat = output.z_hat.reshape(batch * horizon * n_patches, dim)
        history = torch.as_tensor(history_seconds, device=z_future.device)
        if history.ndim != 1 or history.shape[0] != batch:
            raise ValueError("history_seconds must have shape [B]")
        repeated_history = history.repeat_interleave(horizon * n_patches)
        reg = sigreg(
            flat,
            repeated_history,
            directions,
            weights=weights,
            chunk=chunk,
            checkpoint=checkpoint,
        )
        loss = prediction + lam * reg["sigreg"]
        return {
            "loss": loss,
            "prediction": prediction,
            "sigreg": reg["sigreg"],
            "ep": reg["ep"],
            "ep_over_n": reg["ep_over_n"],
            "counts": reg["counts"],
            "null": reg["null"],
        }


def _mse(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).square().sum() / a.numel()


def compute_reachable_error_and_factorial(
    rollouts: dict[str, Tensor],
    *,
    goal: Tensor | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Open-loop, reachable, cost-relevant errors and the 2x2 Oracle Factorial."""
    required = ("open_pred", "open_true", "plan_pred", "plan_true")
    missing = [key for key in required if key not in rollouts]
    if missing:
        raise ValueError(f"rollouts missing key {missing[0]}")
    for key in required:
        if rollouts[key].shape != rollouts[key.replace("pred", "true")].shape:
            raise ValueError(f"rollouts[{key!r}] shape must match its true counterpart")

    e_data = _mse(rollouts["open_pred"], rollouts["open_true"])
    e_reach = _mse(rollouts["plan_pred"], rollouts["plan_true"])
    if goal is not None:
        goal_t = torch.as_tensor(
            goal, dtype=rollouts["plan_true"].dtype, device=rollouts["plan_true"].device
        )
        plan_pred = rollouts["plan_pred"]
        plan_true = rollouts["plan_true"]
        if goal_t.ndim == 1:
            goal_t = goal_t.view(*([1] * (plan_pred.ndim - 1)), -1).expand_as(plan_pred)
        elif goal_t.ndim == 2 and plan_pred.ndim == 3:
            goal_t = goal_t.unsqueeze(1).expand_as(plan_pred)
        elif goal_t.shape != plan_pred.shape:
            goal_t = goal_t.expand_as(plan_pred)
        imagined_cost = (plan_pred - goal_t).square().flatten(1).sum(dim=1)
        realized_cost = (plan_true - goal_t).square().flatten(1).sum(dim=1)
        e_rel = (imagined_cost - realized_cost).abs().mean()
    else:
        e_rel = e_reach

    if "oracle_pred" in rollouts:
        if rollouts["oracle_pred"].shape != rollouts["open_true"].shape:
            raise ValueError("oracle_pred must match open_true shape")
        s_true_latent = _mse(rollouts["oracle_pred"], rollouts["open_true"])
    else:
        s_true_latent = torch.zeros((), dtype=e_data.dtype, device=e_data.device)
    s_true_true = torch.zeros((), dtype=e_data.dtype, device=e_data.device)

    factorial = {
        "S(learned,latent)": e_data,
        "S(learned,true)": e_reach,
        "S(true,latent)": s_true_latent,
        "S(true,true)": s_true_true,
    }
    return {
        "E_data": e_data,
        "E_reach": e_reach,
        "E_rel": e_rel,
        "separation": e_reach - e_data,
        "factorial": factorial,
    }


def _rollout_cost(
    dynamics: Callable[[Tensor, Tensor], Tensor],
    z0: Tensor,
    goal: Tensor,
    actions: Tensor,
) -> Tensor:
    z = z0.to(dtype=actions.dtype).reshape(1, -1).expand(actions.shape[0], -1).clone()
    for step in range(actions.shape[1]):
        z = dynamics(z, actions[:, step])
    return (z - goal.reshape(1, -1).to(dtype=z.dtype)).square().sum(dim=1)


def diffusion_steered_plan(
    dynamics: Callable[[Tensor, Tensor], Tensor],
    z0: Tensor,
    goal: Tensor,
    *,
    horizon: int,
    action_dim: int,
    n_samples: int = 64,
    n_iters: int = 8,
    elite_frac: float = 0.1,
    sigma: float = 1.0,
    alpha: float = 0.5,
    method: Literal["diffusion", "cem"] = "diffusion",
    denoise: Callable[[Tensor], Tensor] | None = None,
    action_bound: float | None = None,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | str]]:
    """Trajectory optimization in policy-noise space vs. standard CEM."""
    if method not in ("diffusion", "cem"):
        raise ValueError("method must be 'diffusion' or 'cem'")
    for name, value in (
        ("horizon", horizon),
        ("action_dim", action_dim),
        ("n_samples", n_samples),
        ("n_iters", n_iters),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < elite_frac <= 1 or sigma <= 0 or alpha < 0:
        raise ValueError("elite_frac in (0,1], sigma > 0, alpha >= 0 required")
    elite_count = max(1, int(round(n_samples * elite_frac)))
    device = z0.device
    generator = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=z0.dtype).to(device)

    def decode(noise: Tensor) -> Tensor:
        actions = denoise(noise) if denoise is not None else noise
        if action_bound is not None:
            actions = actions.clamp(-action_bound, action_bound)
        return actions

    best_actions: Tensor | None = None
    best_cost = torch.tensor(float("inf"), dtype=z0.dtype, device=device)

    if method == "cem":
        mean = torch.zeros((horizon, action_dim), dtype=z0.dtype, device=device)
        std = torch.full_like(mean, sigma)
        for _ in range(n_iters):
            actions = mean + std * randn(n_samples, horizon, action_dim)
            if action_bound is not None:
                actions = actions.clamp(-action_bound, action_bound)
            costs = _rollout_cost(dynamics, z0, goal, actions)
            order = costs.argsort()
            elites = actions[order[:elite_count]]
            if costs[order[0]] < best_cost:
                best_cost = costs[order[0]]
                best_actions = actions[order[0]].clone()
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False) + 1e-6
    else:
        noises = sigma * randn(n_samples, horizon, action_dim)
        for _ in range(n_iters):
            actions = decode(noises)
            costs = _rollout_cost(dynamics, z0, goal, actions)
            order = costs.argsort()
            if costs[order[0]] < best_cost:
                best_cost = costs[order[0]]
                best_actions = actions[order[0]].clone()
            elite_noises = noises[order[:elite_count]]
            picks = torch.randint(elite_count, (n_samples,), generator=generator).to(device)
            noises = elite_noises[picks] + alpha * randn(n_samples, horizon, action_dim)

    assert best_actions is not None
    return best_actions, {
        "best_cost": best_cost,
        "method": method,
        "elite_frac": torch.tensor(elite_frac),
    }


class LeWMPixelPatchEncoder(nn.Module):
    """GPU-native pixel-to-patch encoder (LeWorldModel recipe, arXiv:2603.19312).

    Maps video tensors ``frames [B, T, C, H, W]`` (RGB or RGB+Depth from LeRobot v0.6)
    into dense spatial patch latents ``z [B, T, G*G, D]`` without an EMA teacher or
    pixel decoder.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 64,
        patch_size: int = 16,
        dim: int = 128,
        heads: int = 4,
        depth: int = 2,
        mlp_ratio: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be a positive multiple of patch_size")
        torch.manual_seed(seed)
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.dim = dim

        self.stem = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(self.grid_size, dim, dtype=torch.float32), persistent=False
        )
        self.blocks = nn.ModuleList([_CausalBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.register_buffer(
            "spatial_mask",
            torch.zeros((self.n_patches, self.n_patches), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, frames: Tensor) -> Tensor:
        """Encode ``frames [B, T, C, H, W]`` -> ``z [B, T, G*G, D]``."""
        if (
            frames.ndim != 5
            or frames.shape[2] != self.in_channels
            or frames.shape[3] != self.image_size
            or frames.shape[4] != self.image_size
        ):
            raise ValueError(
                f"frames must have shape [B,T,{self.in_channels},{self.image_size},{self.image_size}]"
            )
        batch, timesteps, channels, height, width = frames.shape
        flat = frames.reshape(batch * timesteps, channels, height, width).to(dtype=self.stem.weight.dtype)
        tokens = self.stem(flat).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype).unsqueeze(0)
        mask = self.spatial_mask.to(dtype=tokens.dtype, device=tokens.device)
        for block in self.blocks:
            tokens = block(tokens, mask)
        tokens = self.norm(tokens)
        return tokens.reshape(batch, timesteps, self.n_patches, self.dim)


class EndToEndPixelWAM(nn.Module):
    """End-to-end GPU pixel World-Action Model combining LeWMPixelPatchEncoder,
    CausalPatchJEPAWorldModel, HistoryStratifiedSIGReg, and a Robometer-style
    progress prediction head (arXiv:2603.02115).
    """

    def __init__(
        self,
        *,
        encoder: LeWMPixelPatchEncoder,
        world_model: CausalPatchJEPAWorldModel,
        sigreg: HistoryStratifiedSIGReg | None = None,
    ) -> None:
        super().__init__()
        if encoder.grid_size != world_model.grid_size or encoder.dim != world_model.dim:
            raise ValueError("encoder and world_model must share grid_size and dim")
        self.encoder = encoder
        self.world_model = world_model
        self.sigreg = sigreg if sigreg is not None else HistoryStratifiedSIGReg()
        self.progress_head = nn.Linear(world_model.dim, 1)
        nn.init.trunc_normal_(self.progress_head.weight, std=0.02)
        nn.init.zeros_(self.progress_head.bias)

    def forward_pixels(
        self, frames_context: Tensor, actions: Tensor, proprio: Tensor
    ) -> dict[str, Tensor]:
        z_context = self.encoder(frames_context)
        out = self.world_model(z_context, actions, proprio)
        pooled = out.z_hat.mean(dim=2)
        progress_pred = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
        return {
            "z_context": z_context,
            "z_hat": out.z_hat,
            "context_hidden": out.context,
            "progress_pred": progress_pred,
        }

    def pixel_wam_loss(
        self,
        frames_context: Tensor,
        actions: Tensor,
        proprio: Tensor,
        frames_future: Tensor,
        *,
        history_seconds: Tensor,
        directions: Tensor,
        target_progress: Tensor | None = None,
        target_mode: Literal["live", "stop_grad"] = "live",
        lam: float = 0.05,
        progress_weight: float = 0.2,
    ) -> dict[str, Tensor]:
        z_context = self.encoder(frames_context)
        z_future = self.encoder(frames_future)
        base = self.world_model.wam_loss(
            z_context,
            actions,
            proprio,
            z_future,
            history_seconds=history_seconds,
            directions=directions,
            sigreg=self.sigreg,
            target_mode=target_mode,
            lam=lam,
        )
        out = self.world_model(z_context, actions, proprio)
        pooled = out.z_hat.mean(dim=2)
        progress_pred = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
        if target_progress is not None:
            if target_progress.shape != progress_pred.shape:
                raise ValueError("target_progress must have shape [B,K]")
            prog_loss = functional.mse_loss(
                progress_pred, target_progress.to(dtype=progress_pred.dtype)
            )
        else:
            prog_loss = torch.zeros((), dtype=base["loss"].dtype, device=base["loss"].device)
        total = base["loss"] + progress_weight * prog_loss
        return {
            **base,
            "loss": total,
            "progress_loss": prog_loss,
            "progress_pred": progress_pred.detach(),
        }


class LeRobotWindowDataset(torch.utils.data.Dataset):
    """History-stratified window sampler compatible with LeRobot v3.0 / RoboCasa365
    episodes (``ember-lab-berkeley/robocasa365-*``).

    Each episode dict contains:
      * ``frames``: float32/uint8 tensor ``[T_ep, C, H, W]``
      * ``actions``: float32 tensor ``[T_ep, A]``
      * ``proprio``: float32 tensor ``[T_ep, P]``
      * ``fps``: float (default 20.0 for RoboCasa365)
      * optional ``subtask_id``: int tensor ``[T_ep]``
    """

    def __init__(
        self,
        episodes: list[dict[str, Tensor | float]],
        *,
        context_frames: int = 4,
        horizon_frames: int = 4,
        strata_seconds: tuple[float, ...] = HISTORY_STRATA_SECONDS,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if context_frames < 1 or horizon_frames < 1:
            raise ValueError("context_frames and horizon_frames must be positive")
        min_len = context_frames + horizon_frames
        for idx, ep in enumerate(episodes):
            if "frames" not in ep or "actions" not in ep or "proprio" not in ep:
                raise ValueError(f"episode {idx} missing frames/actions/proprio")
            if int(ep["frames"].shape[0]) < min_len:
                raise ValueError(f"episode {idx} shorter than context_frames + horizon_frames")
        self.episodes = episodes
        self.context_frames = context_frames
        self.horizon_frames = horizon_frames
        self.strata_seconds = strata_seconds
        self.seed = seed

    def __len__(self) -> int:
        return len(self.episodes) * len(self.strata_seconds)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ep_idx = index // len(self.strata_seconds)
        stratum_idx = index % len(self.strata_seconds)
        ep = self.episodes[ep_idx]
        frames: Tensor = ep["frames"]
        actions: Tensor = ep["actions"]
        proprio: Tensor = ep["proprio"]
        total_t = int(frames.shape[0])
        fps = float(ep.get("fps", 20.0))
        target_hist_s = float(self.strata_seconds[stratum_idx])

        # Span of context in raw frames determined by target_hist_s
        span_frames = max(self.context_frames, int(round(target_hist_s * fps)) + 1)
        max_anchor = total_t - self.horizon_frames
        anchor = min(max_anchor, max(self.context_frames, span_frames))
        start = max(0, anchor - span_frames)
        ctx_indices = torch.linspace(
            start, anchor - 1, steps=self.context_frames, dtype=torch.float64
        ).round().to(torch.int64)
        fut_indices = torch.arange(anchor, anchor + self.horizon_frames, dtype=torch.int64)

        f_ctx = frames[ctx_indices].to(dtype=torch.float32)
        f_fut = frames[fut_indices].to(dtype=torch.float32)
        if f_ctx.max() > 1.5:
            f_ctx = f_ctx / 255.0
            f_fut = f_fut / 255.0
        p_ctx = proprio[ctx_indices].to(dtype=torch.float32)
        a_fut = actions[fut_indices].to(dtype=torch.float32)
        progress = (fut_indices.to(dtype=torch.float32) + 1.0) / float(total_t)
        realized_hist_s = float(ctx_indices[-1].item() - ctx_indices[0].item()) / max(fps, 1e-6)
        return {
            "frames_context": f_ctx,
            "frames_future": f_fut,
            "proprio": p_ctx,
            "actions": a_fut,
            "history_seconds": torch.tensor(
                max(target_hist_s, realized_hist_s), dtype=torch.float32
            ),
            "target_progress": progress.clamp(0.0, 1.0),
        }
