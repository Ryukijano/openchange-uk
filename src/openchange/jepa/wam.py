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

from openchange.jepa.losses import (
    conditional_flow_matching_loss,
    ep_null_expectation,
    ep_per_view_stratum,
    sample_conditional_flow,
)
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


def lewam_mot_attention_mask(
    n_history: int,
    n_clean_actions: int,
    n_noisy_actions: int,
    *,
    include_state_query: bool = True,
    include_goal: bool = False,
    device: torch.device | str | None = None,
) -> Tensor:
    """Token-level Mixture-of-Transformers (MoT) attention mask (LeWAM, arXiv:2610.12818, Eq. 10).

    Token ordering:
      ``[z_hist (n_history), q_state? (1), A_clean (n_clean_actions), c_goal? (1), A_noisy (n_noisy_actions)]``

    Enforces LeWAM's strict information firewall across layers (Appendix A, Eq. 10 & Fig. 10):
      * ``V(z_i) = {z_hist}``: history tokens attend ONLY to history tokens (never clean actions,
        blocking indirect multi-layer leakage of ground-truth actions into the policy branch).
      * ``V(q_state) = {z_hist, q_state, A_clean}``: state query reads history and clean actions.
      * ``V(a_j^clean) = {z_hist, a_{<=j}^clean}``: clean actions read history and causal prefix.
      * ``V(c_goal) = {z_hist, c_goal}``: goal token (if present) reads history and itself.
      * ``V(a_j^{(s)}) = {z_hist, c_goal?, a_{<=j}^{(s)}}``: noisy action-flow tokens read history,
        optional goal context, and their own causal noisy-action prefix—NEVER ``q_state`` or ``A_clean``.
    """
    for name, value in (
        ("n_history", n_history),
        ("n_clean_actions", n_clean_actions),
        ("n_noisy_actions", n_noisy_actions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    q_count = 1 if include_state_query else 0
    g_count = 1 if include_goal else 0
    total = n_history + q_count + n_clean_actions + g_count + n_noisy_actions
    allowed = torch.zeros((total, total), dtype=torch.bool)

    hist_slice = slice(0, n_history)
    q_idx = n_history if include_state_query else None
    clean_start = n_history + q_count
    clean_slice = slice(clean_start, clean_start + n_clean_actions)
    goal_idx = clean_start + n_clean_actions if include_goal else None
    noisy_start = clean_start + n_clean_actions + g_count
    noisy_slice = slice(noisy_start, total)

    # 1. History tokens attend only to history tokens
    allowed[hist_slice, hist_slice] = True

    # 2. State query attends to history, itself, and clean actions
    if q_idx is not None:
        allowed[q_idx, hist_slice] = True
        allowed[q_idx, q_idx] = True
        allowed[q_idx, clean_slice] = True

    # 3. Clean actions attend to history + causal prefix of clean actions
    allowed[clean_slice, hist_slice] = True
    allowed[clean_slice, clean_slice] = torch.tril(
        torch.ones((n_clean_actions, n_clean_actions), dtype=torch.bool)
    )

    # 4. Optional goal token attends to history and itself
    if goal_idx is not None:
        allowed[goal_idx, hist_slice] = True
        allowed[goal_idx, goal_idx] = True

    # 5. Noisy action-flow tokens attend to history (+ goal) + causal prefix of noisy actions
    allowed[noisy_slice, hist_slice] = True
    if goal_idx is not None:
        allowed[noisy_slice, goal_idx] = True
    allowed[noisy_slice, noisy_slice] = torch.tril(
        torch.ones((n_noisy_actions, n_noisy_actions), dtype=torch.bool)
    )

    mask = torch.zeros((total, total), dtype=torch.float32)
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
        boundaries = self.boundaries.to(device=history.device, dtype=torch.float64)
        distances = (history.to(torch.float64)[:, None] - boundaries[None, :]).abs()
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
        sigreg_pool: Literal["patch_flat", "frame_mean", "context_and_pred"] = "patch_flat",
    ) -> dict[str, Tensor]:
        """Multi-step latent prediction loss + history-stratified SIGReg + rank/elasticity diagnostics."""
        if target_mode not in ("live", "stop_grad"):
            raise ValueError("target_mode must be 'live' or 'stop_grad'")
        if sigreg_pool not in ("patch_flat", "frame_mean", "context_and_pred"):
            raise ValueError("sigreg_pool must be 'patch_flat', 'frame_mean', or 'context_and_pred'")
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        output = self.forward(z_context, actions, proprio)
        if z_future.shape != output.z_hat.shape:
            raise ValueError("z_future must have shape [B,K,G*G,dim] matching predictions")
        target = z_future if target_mode == "live" else z_future.detach()
        prediction = (output.z_hat - target).square().sum() / output.z_hat.numel()

        batch, horizon, n_patches, dim = output.z_hat.shape
        history = torch.as_tensor(history_seconds, device=z_future.device)
        if history.ndim != 1 or history.shape[0] != batch:
            raise ValueError("history_seconds must have shape [B]")

        if sigreg_pool == "frame_mean":
            pooled_frames = output.z_hat.mean(dim=2).reshape(batch * horizon, dim)
            repeated_history = history.repeat_interleave(horizon)
            reg = sigreg(
                pooled_frames,
                repeated_history,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )
        elif sigreg_pool == "context_and_pred":
            ctx_frames = z_context.mean(dim=2)
            pred_frames = output.z_hat.mean(dim=2)
            all_frames = torch.cat((ctx_frames, pred_frames), dim=1)
            n_total_frames = all_frames.shape[1]
            repeated_history = history.repeat_interleave(n_total_frames)
            reg = sigreg(
                all_frames.reshape(batch * n_total_frames, dim),
                repeated_history,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )
        else:
            flat = output.z_hat.reshape(batch * horizon * n_patches, dim)
            repeated_history = history.repeat_interleave(horizon * n_patches)
            reg = sigreg(
                flat,
                repeated_history,
                directions,
                weights=weights,
                chunk=chunk,
                checkpoint=checkpoint,
            )

        with torch.no_grad():
            frame_latents = output.z_hat.detach().mean(dim=2).reshape(batch * horizon, dim).to(torch.float64)
            svals = torch.linalg.svdvals(frame_latents)
            sval_sum = svals.sum()
            if float(sval_sum.item()) > 1e-12:
                probs = (svals / sval_sum).clamp_min(1e-12)
                eff_rank = torch.exp(-(probs * torch.log(probs)).sum()).to(dtype=prediction.dtype)
            else:
                eff_rank = torch.ones((), dtype=prediction.dtype, device=prediction.device)

        loss = prediction + lam * reg["sigreg"]
        return {
            "loss": loss,
            "prediction": prediction,
            "sigreg": reg["sigreg"],
            "ep": reg["ep"],
            "ep_over_n": reg["ep_over_n"],
            "counts": reg["counts"],
            "null": reg["null"],
            "effective_rank": eff_rank,
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
    method: Literal["diffusion", "cem", "lewam_grad"] = "diffusion",
    denoise: Callable[[Tensor], Tensor] | None = None,
    action_bound: float | None = None,
    lr_grad: float = 0.05,
    lam_prox: float = 0.01,
    grad_clip: float = 10.0,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | str]]:
    """Trajectory optimization in policy-noise space, CEM, or LeWAM gradient-refined planning.

    When ``method == 'lewam_grad'`` (LeWAM Algorithm 1, arXiv:2610.12818, Appendix A.1, Eq. 7-8 & 11):
      1. Samples ``n_samples`` initial action sequences ``A_sample`` from the learned action flow
         policy ``denoise`` (or Gaussian prior when ``denoise`` is None).
      2. Refines all ``n_samples`` candidates in parallel for ``n_iters`` Adam steps through the
         differentiable ``dynamics`` minimizing the proximity-regularized objective:
         ``J_prox(A; A_sample) = C(A) + lam_prox * ||A - A_sample||_F^2``
         while clipping action norms to ``action_bound`` (default 3.0 in LeWAM) and retaining the
         best iterate per candidate using the unpenalized goal cost ``C(A)``.
    """
    if method not in ("diffusion", "cem", "lewam_grad"):
        raise ValueError("method must be 'diffusion', 'cem', or 'lewam_grad'")
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
    if lr_grad <= 0 or lam_prox < 0 or grad_clip <= 0:
        raise ValueError("lr_grad > 0, lam_prox >= 0, and grad_clip > 0 required")
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
    elif method == "lewam_grad":
        with torch.no_grad():
            a_sample = decode(sigma * randn(n_samples, horizon, action_dim)).detach()
            c_init = _rollout_cost(dynamics, z0, goal, a_sample)
            a_star = a_sample.clone()
            c_star = c_init.clone()

        a_var = a_sample.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([a_var], lr=float(lr_grad))
        for _ in range(n_iters):
            opt.zero_grad()
            c_unpenalized = _rollout_cost(dynamics, z0, goal, a_var)
            prox_term = (a_var - a_sample).square().flatten(1).sum(dim=1)
            j_prox = (c_unpenalized + float(lam_prox) * prox_term).sum()
            j_prox.backward()
            torch.nn.utils.clip_grad_norm_([a_var], max_norm=float(grad_clip))
            opt.step()
            with torch.no_grad():
                if action_bound is not None:
                    a_var.clamp_(-action_bound, action_bound)
                c_cur = _rollout_cost(dynamics, z0, goal, a_var)
                improved = c_cur < c_star
                if bool(improved.any()):
                    c_star[improved] = c_cur[improved]
                    a_star[improved] = a_var.detach()[improved]

        best_idx = int(c_star.argmin().item())
        best_cost = c_star[best_idx].detach()
        best_actions = a_star[best_idx].detach().clone()
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


def robometer_preference_loss(
    progress_pred: Tensor,
    target_progress: Tensor,
    *,
    failed_mask: Tensor | None = None,
    margin: float = 0.05,
    temperature: float = 0.1,
    rank_weight: float = 0.5,
) -> dict[str, Tensor]:
    """Robometer-style hybrid progress supervision (arXiv:2603.02115).

    Combines:
      1. Scalar calibration MSE on valid trajectories.
      2. Bradley-Terry pairwise temporal ordering preference ``-log sigmoid((s_{t2} - s_{t1}) / tau)``
         for ordered step pairs ``t2 > t1`` where ``target_progress[:, t2] > target_progress[:, t1]``.
      3. Failure/stalled trajectory margin penalty ensuring failed/reversed rollouts score below
         successful completions.
    """
    if progress_pred.ndim != 2 or target_progress.shape != progress_pred.shape:
        raise ValueError("progress_pred and target_progress must share 2D shape [B, K]")
    if temperature <= 0.0 or margin < 0.0 or rank_weight < 0.0:
        raise ValueError("temperature > 0, margin >= 0, and rank_weight >= 0 required")

    batch, horizon = progress_pred.shape
    target = target_progress.to(dtype=progress_pred.dtype, device=progress_pred.device)

    if failed_mask is not None:
        fail_bool = failed_mask.to(device=progress_pred.device, dtype=torch.bool)
        if fail_bool.shape != (batch,):
            raise ValueError("failed_mask must have shape [B]")
    else:
        fail_bool = torch.zeros((batch,), dtype=torch.bool, device=progress_pred.device)

    valid_mask = ~fail_bool
    if bool(valid_mask.any()):
        mse_term = functional.mse_loss(progress_pred[valid_mask], target[valid_mask])
    else:
        mse_term = functional.mse_loss(progress_pred, target)

    if horizon >= 2 and bool(valid_mask.any()):
        valid_pred = progress_pred[valid_mask]
        valid_targ = target[valid_mask]
        diff_pred = valid_pred.unsqueeze(2) - valid_pred.unsqueeze(1)  # [B_v, t1, t2] -> s_{t1} - s_{t2}
        diff_targ = valid_targ.unsqueeze(2) - valid_targ.unsqueeze(1)
        # Pairs where step j > step i has strictly higher target progress
        ordered_pairs = diff_targ < -1e-6
        if bool(ordered_pairs.any()):
            # We want s_j - s_i > 0, i.e. -diff_pred[ordered_pairs] > 0
            bt_loss = functional.softplus(diff_pred[ordered_pairs] / float(temperature)).mean()
        else:
            bt_loss = torch.zeros((), dtype=progress_pred.dtype, device=progress_pred.device)
    else:
        bt_loss = torch.zeros((), dtype=progress_pred.dtype, device=progress_pred.device)

    if bool(fail_bool.any()) and bool(valid_mask.any()):
        valid_final = progress_pred[valid_mask, -1].mean()
        failed_final = progress_pred[fail_bool, -1]
        fail_penalty = functional.relu(failed_final - valid_final + float(margin)).mean()
    else:
        fail_penalty = torch.zeros((), dtype=progress_pred.dtype, device=progress_pred.device)

    total = mse_term + float(rank_weight) * (bt_loss + fail_penalty)
    return {
        "loss": total,
        "mse_loss": mse_term,
        "ranking_loss": bt_loss,
        "failure_margin_loss": fail_penalty,
    }


def action_contrastive_loss(
    world_model: CausalPatchJEPAWorldModel,
    z_context: Tensor,
    actions: Tensor,
    proprio: Tensor,
    z_future: Tensor,
    *,
    temperature: float = 0.1,
) -> dict[str, Tensor]:
    """Counterfactual Action-InfoNCE & Elasticity loss (FLEX-WAM / OpenWAM action responsiveness).

    Prevents the 'action blindness' shortcut (where ``d z_hat / d a -> 0`` because ``z_context``
    alone predicts most of ``z_future``) by evaluating every batch context ``b`` under all
    candidate action sequences ``b'`` in the batch and requiring the true action ``b' = b`` to
    produce the rollout closest to ``z_future[b]``.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    batch = z_context.shape[0]
    if batch < 2:
        raise ValueError("action_contrastive_loss requires batch size B >= 2")

    target_flat = z_future.detach().reshape(batch, -1)
    # Compute predicted rollouts for all (context_b, action_b') pairs
    ctx_rep = z_context.unsqueeze(1).expand(batch, batch, *z_context.shape[1:]).reshape(
        batch * batch, *z_context.shape[1:]
    )
    prop_rep = proprio.unsqueeze(1).expand(batch, batch, *proprio.shape[1:]).reshape(
        batch * batch, *proprio.shape[1:]
    )
    act_rep = actions.unsqueeze(0).expand(batch, batch, *actions.shape[1:]).reshape(
        batch * batch, *actions.shape[1:]
    )
    pred_all = world_model(ctx_rep, act_rep, prop_rep).z_hat.reshape(batch, batch, -1)

    # Squared L2 distance matrix D[b, b'] = mean((z_hat(ctx_b, act_b') - z_future_b)^2)
    diffs = pred_all - target_flat.unsqueeze(1)
    dist_matrix = diffs.square().mean(dim=-1)
    logits = -dist_matrix / float(temperature)
    labels = torch.arange(batch, dtype=torch.long, device=z_context.device)
    nce_loss = functional.cross_entropy(logits, labels)

    # Action elasticity: mean L2 shift between true action (diag) and counterfactual actions (off-diag)
    true_preds = pred_all[labels, labels].unsqueeze(1)
    off_diag_mask = ~torch.eye(batch, dtype=torch.bool, device=z_context.device)
    cf_shifts = (pred_all - true_preds).square().mean(dim=-1)[off_diag_mask].sqrt()
    elasticity = cf_shifts.mean()
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels).to(dtype=z_context.dtype).mean()

    return {
        "loss": nce_loss,
        "action_elasticity": elasticity,
        "action_identification_accuracy": acc,
        "distance_matrix": dist_matrix.detach(),
    }


def flex_wam_fd_elasticity_and_stop_probe(
    world_model: CausalPatchJEPAWorldModel,
    z_context: Tensor,
    actions: Tensor,
    proprio: Tensor,
    *,
    delta: float = 1e-2,
    n_probes: int = 4,
    tau_state: float = 0.5,
    tau_action: float = 0.5,
    wm_setpoint: float | None = None,
    seed: int = 0,
) -> dict[str, Tensor]:
    """FLEX-WAM Hutchinson Forward-Dynamics (FD) Elasticity, Stop Probe, and Gradient Cross-Weights.

    Implements the three training-time and diagnostic mechanisms from FLEX-WAM (arXiv:2610.05483,
    Sec. II-2 & III-B1, Eq. 3-4 & Fig. 2):
      1. ``fd_elasticity``: Hutchinson directional-derivative estimator of the action Jacobian
         ``E_{eps ~ N(0, I)}[ ||z_hat(z_{<=t}, a + delta * eps) - z_hat(z_{<=t}, a)||_2 / delta ]``.
      2. ``stop_command_drift`` vs ``active_command_displacement``: Counterfactual stop-command probe
         (setting ``a_stop = 0`` for relative actions and measuring distance from the last context
         state ``z_context[:, -1]`` vs. active trajectory displacement).
      3. ``phi_act`` and ``phi_obs`` (Eq. 4): Noise-cleanness cross-weights
         ``Phi_act = (tau_s + eps) / (tau_s + tau_a + 2*eps)`` and
         ``Phi_obs = (tau_a + eps) / (tau_s + tau_a + 2*eps)``, plus closed-loop ``wm_sampling_bias``
         regulating sensitivity around ``0.9 * wm_setpoint``.
    """
    if delta <= 0.0 or n_probes < 1:
        raise ValueError("delta > 0 and n_probes >= 1 required")
    if not (0.0 <= tau_state <= 1.0 and 0.0 <= tau_action <= 1.0):
        raise ValueError("tau_state and tau_action must be in [0, 1]")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        base_pred = world_model(z_context, actions, proprio).z_hat
        batch = z_context.shape[0]
        base_flat = base_pred.reshape(batch, -1)

        dir_norms: list[Tensor] = []
        for _ in range(n_probes):
            eps_a = torch.randn(actions.shape, generator=gen, dtype=actions.dtype, device="cpu").to(
                actions.device
            )
            eps_a = eps_a / torch.linalg.vector_norm(
                eps_a.reshape(batch, -1), dim=-1, keepdim=True
            ).clamp_min(1e-12).view(batch, 1, 1)
            pert_pred = world_model(z_context, actions + float(delta) * eps_a, proprio).z_hat.reshape(
                batch, -1
            )
            dir_norms.append(torch.linalg.vector_norm(pert_pred - base_flat, dim=-1) / float(delta))
        fd_elasticity = torch.stack(dir_norms, dim=0).mean()

        # Counterfactual stop-command probe (a_stop = 0) vs active trajectory command
        zero_actions = torch.zeros_like(actions)
        stop_pred = world_model(z_context, zero_actions, proprio).z_hat
        last_state = z_context[:, -1:, :, :].expand_as(stop_pred)
        stop_drift = (stop_pred - last_state).square().mean().sqrt()
        active_disp = (base_pred - last_state).square().mean().sqrt()
        stop_response_gap = (base_pred - stop_pred).square().mean().sqrt()

        eps_w = 1e-3
        denom = float(tau_state) + float(tau_action) + 2.0 * eps_w
        phi_act = torch.tensor((float(tau_state) + eps_w) / denom, dtype=z_context.dtype, device=z_context.device)
        phi_obs = torch.tensor((float(tau_action) + eps_w) / denom, dtype=z_context.dtype, device=z_context.device)

        if wm_setpoint is not None and wm_setpoint > 0.0:
            target_setpoint = 0.9 * float(wm_setpoint)
            deficit = max(0.0, (target_setpoint - float(fd_elasticity.item())) / target_setpoint)
            wm_bias = min(1.0, max(0.2, 0.2 + 0.8 * deficit))
        else:
            wm_bias = 1.0

    return {
        "fd_elasticity": fd_elasticity,
        "stop_command_drift": stop_drift,
        "active_command_displacement": active_disp,
        "stop_response_gap": stop_response_gap,
        "phi_act": phi_act,
        "phi_obs": phi_obs,
        "wm_sampling_bias": torch.tensor(wm_bias, dtype=z_context.dtype, device=z_context.device),
    }


class FlowMatchingWAMHead(nn.Module):
    """Conditional Flow Matching (CFM / Diffusion) World-Action Model head (OpenWAM / FLEX-WAM / Long-WAM).

    Addresses the core limitations of pure L2 JEPA world models and supports Long-WAM's IDM mode:
      1. Multimodal dynamics ``v_dyn(z_tau, tau | h_ctx, a)``: learns the full conditional
         distribution ``p(z_{t+1:t+K} | z_{<=t}, a_{t:t+K-1})`` so bifurcating trajectories
         do not collapse to an off-manifold mean.
      2. Multimodal action policy ``v_act(a_tau, tau | h_ctx, z_fut_pred_detached?, g)``: generates
         horizon action chunks conditioned on causal context, optional detached predicted future
         latents (Long-WAM's Predict-Then-Act / IDM mode, arXiv:2610.10528 Sec. 3.2), and goal ``g``.
    """

    def __init__(
        self,
        *,
        dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        predict_then_act: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if dim <= 0 or action_dim <= 0 or horizon <= 0 or hidden_dim <= 0 or time_dim <= 0 or time_dim % 2 != 0:
            raise ValueError("positive dimensions and even time_dim required")
        torch.manual_seed(seed)
        self.dim = dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.time_dim = time_dim
        self.predict_then_act = bool(predict_then_act)
        self.latent_chunk_dim = horizon * dim
        self.action_chunk_dim = horizon * action_dim

        # Velocity field for future pooled latents [B, K*D] given (context [B, D], actions [B, K*A])
        self.dyn_net = nn.Sequential(
            nn.Linear(self.latent_chunk_dim + dim + self.action_chunk_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_chunk_dim),
        )
        act_cond_dim = (2 * dim + self.latent_chunk_dim) if self.predict_then_act else (2 * dim)
        self.act_net = nn.Sequential(
            nn.Linear(self.action_chunk_dim + act_cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_chunk_dim),
        )

    def _time_embed(self, t: Tensor) -> Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=t.dtype, device=t.device))
            * torch.arange(half, dtype=t.dtype, device=t.device)
            / max(half - 1, 1)
        )
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def velocity_dynamics(self, z_tau: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self._time_embed(t.to(dtype=z_tau.dtype, device=z_tau.device))
        return self.dyn_net(torch.cat((z_tau, cond, t_emb), dim=-1))

    def velocity_actions(self, a_tau: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self._time_embed(t.to(dtype=a_tau.dtype, device=a_tau.device))
        return self.act_net(torch.cat((a_tau, cond, t_emb), dim=-1))

    def _build_action_cond(
        self,
        context_summary: Tensor,
        goal_summary: Tensor,
        predicted_future_summary: Tensor | None = None,
    ) -> Tensor:
        batch = context_summary.shape[0]
        if not self.predict_then_act:
            return torch.cat((context_summary, goal_summary), dim=-1)
        if predicted_future_summary is None:
            raise ValueError("predicted_future_summary [B, K, dim] required when predict_then_act=True")
        fut_detached = predicted_future_summary.detach().reshape(batch, self.latent_chunk_dim)
        return torch.cat((context_summary, goal_summary, fut_detached), dim=-1)

    def cfm_loss(
        self,
        context_summary: Tensor,
        actions: Tensor,
        target_future_summary: Tensor,
        *,
        goal_summary: Tensor | None = None,
        predicted_future_summary: Tensor | None = None,
        sigma_min: float = 1e-4,
    ) -> dict[str, Tensor]:
        """Compute joint dynamics CFM + action-policy CFM losses."""
        if context_summary.ndim != 2 or context_summary.shape[1] != self.dim:
            raise ValueError("context_summary must have shape [B, dim]")
        batch = context_summary.shape[0]
        if actions.shape != (batch, self.horizon, self.action_dim):
            raise ValueError("actions must have shape [B, K, action_dim]")
        if target_future_summary.shape != (batch, self.horizon, self.dim):
            raise ValueError("target_future_summary must have shape [B, K, dim]")

        act_flat = actions.reshape(batch, self.action_chunk_dim)
        fut_flat = target_future_summary.reshape(batch, self.latent_chunk_dim)
        dyn_cond = torch.cat((context_summary, act_flat), dim=-1)
        dyn_out = conditional_flow_matching_loss(
            self.velocity_dynamics, fut_flat, dyn_cond, sigma_min=sigma_min
        )

        goal = target_future_summary[:, -1, :] if goal_summary is None else goal_summary
        if goal.shape != (batch, self.dim):
            raise ValueError("goal_summary must have shape [B, dim]")
        idm_future = target_future_summary if (self.predict_then_act and predicted_future_summary is None) else predicted_future_summary
        act_cond = self._build_action_cond(context_summary, goal, idm_future)
        act_out = conditional_flow_matching_loss(
            self.velocity_actions, act_flat, act_cond, sigma_min=sigma_min
        )

        total = dyn_out["loss"] + act_out["loss"]
        return {
            "loss": total,
            "dynamics_cfm_loss": dyn_out["loss"],
            "action_cfm_loss": act_out["loss"],
        }

    def sample_future(
        self,
        context_summary: Tensor,
        actions: Tensor,
        *,
        steps: int = 16,
        solver: Literal["euler", "heun"] = "heun",
        x0: Tensor | None = None,
        seed: int | None = None,
    ) -> Tensor:
        batch = context_summary.shape[0]
        dyn_cond = torch.cat((context_summary, actions.reshape(batch, self.action_chunk_dim)), dim=-1)
        flat = sample_conditional_flow(
            self.velocity_dynamics,
            dyn_cond,
            dim=self.latent_chunk_dim,
            steps=steps,
            solver=solver,
            x0=x0,
            seed=seed,
        )
        return flat.reshape(batch, self.horizon, self.dim)

    def sample_actions(
        self,
        context_summary: Tensor,
        goal_summary: Tensor,
        *,
        predicted_future_summary: Tensor | None = None,
        steps: int = 16,
        solver: Literal["euler", "heun"] = "heun",
        x0: Tensor | None = None,
        seed: int | None = None,
    ) -> Tensor:
        batch = context_summary.shape[0]
        act_cond = self._build_action_cond(context_summary, goal_summary, predicted_future_summary)
        flat = sample_conditional_flow(
            self.velocity_actions,
            act_cond,
            dim=self.action_chunk_dim,
            steps=steps,
            solver=solver,
            x0=x0,
            seed=seed,
        )
        return flat.reshape(batch, self.horizon, self.action_dim)


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
        failed_mask: Tensor | None = None,
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
            prog_details = robometer_preference_loss(
                progress_pred, target_progress, failed_mask=failed_mask
            )
            prog_loss = prog_details["loss"]
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

    Fixes the three experimental confounds identified in the scientific audit:
      1. Fixed prediction origin (``anchor``) per episode across all history strata so
         history length ``H`` is not confounded with episode stage ``t_anchor``.
      2. True ``realized_hist_s`` reported in ``history_seconds`` (with ``H = 0.0`` genuinely
         repeating only ``anchor - 1`` so its span is 0.0 s), plus ``history_truncated`` flag.
      3. Transition-aligned actions ``actions[anchor - 1 : anchor - 1 + horizon_frames]``
         matching ``o_{t+1} = T(o_t, a_t)``.
    """

    def __init__(
        self,
        episodes: list[dict[str, Tensor | float]],
        *,
        context_frames: int = 4,
        horizon_frames: int = 4,
        frame_stride: int = 1,
        strata_seconds: tuple[float, ...] = HISTORY_STRATA_SECONDS,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if context_frames < 1 or horizon_frames < 1 or frame_stride < 1:
            raise ValueError("context_frames, horizon_frames, and frame_stride must be positive")
        fut_span = (horizon_frames - 1) * frame_stride + 1
        min_len = context_frames + fut_span
        for idx, ep in enumerate(episodes):
            if "frames" not in ep or "actions" not in ep or "proprio" not in ep:
                raise ValueError(f"episode {idx} missing frames/actions/proprio")
            if int(ep["frames"].shape[0]) < min_len:
                raise ValueError(f"episode {idx} shorter than required context + future span")
        self.episodes = episodes
        self.context_frames = context_frames
        self.horizon_frames = horizon_frames
        self.frame_stride = frame_stride
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

        # Fixed prediction origin per episode across all history strata (unconfounded H test)
        fut_span = (self.horizon_frames - 1) * self.frame_stride + 1
        anchor = total_t - fut_span
        last_ctx = anchor - 1
        if target_hist_s <= 0.0:
            ctx_indices = torch.full((self.context_frames,), last_ctx, dtype=torch.int64)
            padding_fraction = 0.0
        else:
            requested_span_frames = max(1, int(round(target_hist_s * fps)))
            raw_start = last_ctx - requested_span_frames
            unclamped = torch.linspace(raw_start, last_ctx, steps=self.context_frames, dtype=torch.float64).round()
            padding_fraction = float((unclamped < 0).to(dtype=torch.float64).mean().item())
            ctx_indices = unclamped.clamp(min=0, max=last_ctx).to(torch.int64)
        fut_indices = anchor + torch.arange(self.horizon_frames, dtype=torch.int64) * self.frame_stride
        act_indices = last_ctx + torch.arange(self.horizon_frames, dtype=torch.int64) * self.frame_stride

        f_ctx = frames[ctx_indices].to(dtype=torch.float32)
        f_fut = frames[fut_indices].to(dtype=torch.float32)
        if f_ctx.max() > 1.5:
            f_ctx = f_ctx / 255.0
            f_fut = f_fut / 255.0
        p_ctx = proprio[ctx_indices].to(dtype=torch.float32)
        a_fut = actions[act_indices].to(dtype=torch.float32)
        progress = (fut_indices.to(dtype=torch.float32) + 1.0) / float(total_t)
        realized_hist_s = float(ctx_indices[-1].item() - ctx_indices[0].item()) / max(fps, 1e-6)
        truncated = realized_hist_s + 1e-6 < target_hist_s
        return {
            "frames_context": f_ctx,
            "frames_future": f_fut,
            "proprio": p_ctx,
            "actions": a_fut,
            "history_seconds": torch.tensor(realized_hist_s, dtype=torch.float32),
            "requested_history_seconds": torch.tensor(target_hist_s, dtype=torch.float32),
            "history_truncated": torch.tensor(truncated, dtype=torch.bool),
            "padding_fraction": torch.tensor(padding_fraction, dtype=torch.float32),
            "anchor_index": torch.tensor(anchor, dtype=torch.int64),
            "target_progress": progress.clamp(0.0, 1.0),
        }
