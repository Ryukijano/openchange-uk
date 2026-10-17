"""Small ViT encoder and projector for synthetic clean-JEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from openchange.jepa.config import keep_count
from openchange.jepa.views import ViewBatch


def _projector_norm(kind: str, features: int) -> nn.Module:
    if kind in {"batch", "frozen"}:
        return nn.BatchNorm1d(features)
    return nn.Identity()


def sincos_pos_embed(
    grid: int,
    dim: int,
    ref_grid: int = 14,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if grid <= 0 or dim <= 0 or dim % 4:
        raise ValueError("grid and dim must be positive, and dim must be divisible by four")
    coordinates = (torch.arange(grid, dtype=torch.float64) + 0.5) * (ref_grid / grid)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = 1.0 / (10000.0 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4)))

    def encode(position: Tensor) -> Tensor:
        phase = position.reshape(-1, 1) * frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=1)

    embedding = torch.cat((encode(rows), encode(columns)), dim=1)
    return embedding.to(dtype=dtype)


class _Block(nn.Module):
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

    def forward(self, value: Tensor) -> Tensor:
        batch, length, dim = value.shape
        qkv = self.qkv(self.norm1(value)).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, content = qkv.permute(2, 0, 3, 1, 4)
        attended = functional.scaled_dot_product_attention(query, key, content)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        value = value + self.proj(attended)
        return value + self.mlp(self.norm2(value))


@dataclass
class EncoderOutput:
    b: Tensor
    z: Tensor
    tokens: Tensor


class JEPAEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        torch.manual_seed(cfg["seed"])
        model_cfg = cfg["model"]
        self.dim = model_cfg["dim"]
        self.patch_size = cfg["patch_size"]
        self.global_crop_pixels = cfg["global_crop_pixels"]
        self.local_crop_pixels = cfg["local_crop_pixels"]
        self.global_token_drop = cfg["global_token_drop"]
        self.local_token_drop = cfg["local_token_drop"]
        self.projector_norm_kind = cfg["projector_norm"]
        self._position_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
        if model_cfg["pos_embed"] != "sincos2d_ref14" or model_cfg["drop_path"] != 0:
            raise ValueError("only fixed sincos2d_ref14 positions and zero drop path are supported")

        self.stems = nn.ModuleDict(
            {
                "s1": nn.Linear(len(cfg["sensors"]["s1"]) * self.patch_size**2, self.dim),
                "s2": nn.Linear(len(cfg["sensors"]["s2"]) * self.patch_size**2, self.dim),
            }
        )
        self.modality_embed = nn.ParameterDict(
            {
                "s1": nn.Parameter(torch.empty(self.dim)),
                "s2": nn.Parameter(torch.empty(self.dim)),
            }
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.dim))
        self.blocks = nn.ModuleList(
            [_Block(self.dim, model_cfg["heads"], model_cfg["mlp_ratio"]) for _ in range(model_cfg["depth"])]
        )
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)

        hidden = cfg["projector_hidden"]
        output_dim = cfg["projector_dim"]
        self.projector = nn.Sequential(
            nn.Linear(self.dim, hidden[0]),
            _projector_norm(self.projector_norm_kind, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            _projector_norm(self.projector_norm_kind, hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], output_dim),
        )
        self._initialize()
        if self.projector_norm_kind == "frozen":
            self.train(self.training)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for embedding in self.modality_embed.values():
            nn.init.trunc_normal_(embedding, std=0.02)

    def train(self, mode: bool = True) -> JEPAEncoder:
        super().train(mode)
        if self.projector_norm_kind == "frozen":
            for module in self.projector.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        return self

    def embed(self, pixels: dict[str, Tensor]) -> Tensor:
        if not pixels or any(sensor not in self.stems for sensor in pixels):
            raise ValueError("pixels must contain one or more supported sensors")
        tokens: list[Tensor] = []
        grid_size: int | None = None
        batch_size: int | None = None
        for sensor in ("s1", "s2"):
            if sensor not in pixels:
                continue
            image = pixels[sensor].to(
                device=self.stems[sensor].weight.device,
                dtype=self.stems[sensor].weight.dtype,
            )
            if image.ndim != 4 or image.shape[1] != self.stems[sensor].in_features // (self.patch_size**2):
                raise ValueError(f"{sensor} pixels must have shape [B,C,H,W] with configured channels")
            batch, _, height, width = image.shape
            if height != width or height % self.patch_size:
                raise ValueError("sensor images must be square and divisible by patch_size")
            grid = height // self.patch_size
            if grid_size is not None and grid != grid_size:
                raise ValueError("all sensors in a view must have the same resolution")
            if batch_size is not None and batch != batch_size:
                raise ValueError("all sensor batches must have the same batch size")
            grid_size, batch_size = grid, batch
            patches = (
                image.reshape(batch, image.shape[1], grid, self.patch_size, grid, self.patch_size)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch, grid * grid, -1)
            )
            position_key = (grid, image.device, image.dtype)
            if position_key not in self._position_cache:
                if len(self._position_cache) >= 8:
                    self._position_cache.pop(next(iter(self._position_cache)))
                self._position_cache[position_key] = sincos_pos_embed(grid, self.dim, dtype=image.dtype).to(
                    device=image.device
                )
            positions = self._position_cache[position_key]
            tokens.append(
                self.stems[sensor](patches)
                + positions.unsqueeze(0)
                + self.modality_embed[sensor].to(dtype=image.dtype).view(1, 1, -1)
            )
        return torch.cat(tokens, dim=1)

    def backbone(self, pixels: dict[str, Tensor], keep_idx: Tensor, *, validate: bool = True) -> Tensor:
        embedded = self.embed(pixels)
        if keep_idx.ndim != 2 or keep_idx.dtype != torch.int64 or keep_idx.shape[0] != embedded.shape[0]:
            raise ValueError("keep_idx must be an int64 tensor of shape [B,keep]")
        if (
            validate
            and keep_idx.numel()
            and (int(keep_idx.min()) < 0 or int(keep_idx.max()) >= embedded.shape[1])
        ):
            raise ValueError("keep_idx contains an out-of-range token")
        resolution = next(iter(pixels.values())).shape[-1]
        if resolution == self.global_crop_pixels:
            rho = self.global_token_drop
        elif resolution == self.local_crop_pixels:
            rho = self.local_token_drop
        else:
            raise ValueError("resolution is not present in the configured view crops")
        expected_keep = keep_count(embedded.shape[1], rho)
        if keep_idx.shape[1] != expected_keep:
            raise ValueError(f"keep_idx width must equal keep_count={expected_keep}")
        indices = keep_idx.to(device=embedded.device, non_blocking=True)
        selected = torch.gather(embedded, 1, indices.unsqueeze(-1).expand(-1, -1, self.dim))
        cls = self.cls_token.to(dtype=selected.dtype).expand(selected.shape[0], -1, -1)
        value = torch.cat((cls, selected), dim=1)
        for block in self.blocks:
            value = block(value)
        return self.norm(value[:, 0])

    def forward(self, batch: ViewBatch, *, validate: bool = True) -> EncoderOutput:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        b = parameter.new_zeros((batch.n_views, batch.n_samples, self.dim))
        filled = (
            torch.zeros((batch.n_views, batch.n_samples), dtype=torch.int64, device=device)
            if validate
            else None
        )
        tokens = torch.zeros((batch.n_views, batch.n_samples), dtype=torch.int64, device=device)
        for bucket in batch.buckets:
            pixels = {
                sensor: image.to(device=device, dtype=dtype, non_blocking=True)
                for sensor, image in bucket.pixels.items()
            }
            view_idx = bucket.view_idx.to(device=device, non_blocking=True)
            sample_idx = bucket.sample_idx.to(device=device, non_blocking=True)
            embeddings = self.backbone(pixels, bucket.keep_idx, validate=validate)
            b = b.index_put((view_idx, sample_idx), embeddings)
            if filled is not None:
                filled.index_put_(
                    (view_idx, sample_idx),
                    torch.ones_like(view_idx),
                    accumulate=True,
                )
            tokens.index_put_(
                (view_idx, sample_idx),
                torch.full_like(view_idx, bucket.keep_idx.shape[1]),
            )
        if validate and not torch.all(filled == 1):
            raise ValueError("each (view, sample) pair must be filled exactly once")
        z = self.projector(b.reshape(batch.n_views * batch.n_samples, self.dim)).reshape(
            batch.n_views, batch.n_samples, -1
        )
        return EncoderOutput(b=b, z=z, tokens=tokens)
