"""Distributed primitives used by the clean-JEPA loss."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as torch_dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    group: object | None = None


def get_context() -> DistContext:
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return DistContext(rank=0, world_size=1)
    return DistContext(
        rank=torch_dist.get_rank(),
        world_size=torch_dist.get_world_size(),
    )


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor, group: object | None) -> Tensor:
        ctx.group = group
        result = value.contiguous().clone()
        torch_dist.all_reduce(result, op=torch_dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        result = gradient.contiguous().clone()
        torch_dist.all_reduce(result, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        return result, None


def all_reduce_sum_autograd(value: Tensor, ctx: DistContext) -> Tensor:
    if ctx.world_size == 1:
        return value
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    return _AllReduceSum.apply(value, ctx.group)


def all_reduce_sum_nograd(value: Tensor, ctx: DistContext) -> Tensor:
    with torch.no_grad():
        result = value.detach().contiguous().clone()
        if ctx.world_size > 1:
            if not torch_dist.is_available() or not torch_dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            torch_dist.all_reduce(result, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        return result


def make_global_strata(quotas: Sequence[int]) -> Tensor:
    values = list(quotas)
    if not values:
        raise ValueError("at least one stratum quota is required")
    if any(not isinstance(quota, int) or isinstance(quota, bool) or quota < 0 for quota in values):
        raise ValueError("stratum quotas must be non-negative integers")
    labels = torch.arange(len(values), dtype=torch.int64)
    counts = torch.tensor(values, dtype=torch.int64)
    return torch.repeat_interleave(labels, counts)


def shard_global_batch(value: Tensor, ctx: DistContext, dim: int) -> Tensor:
    if not -value.ndim <= dim < value.ndim:
        raise IndexError(f"dimension out of range: {dim}")
    dim %= value.ndim
    if ctx.world_size < 1 or not 0 <= ctx.rank < ctx.world_size:
        raise ValueError("invalid distributed rank or world size")
    indices = torch.arange(ctx.rank, value.shape[dim], ctx.world_size, device=value.device)
    return value.index_select(dim, indices)


def assert_quotas(
    local_strata: Tensor,
    global_quotas: Sequence[int],
    ctx: DistContext,
) -> None:
    quotas = [int(quota) for quota in global_quotas]
    if not quotas or any(quota < 0 for quota in quotas):
        raise ValueError("global quotas must be a non-empty sequence of non-negative counts")
    if local_strata.ndim != 1 or local_strata.dtype != torch.int64:
        raise ValueError("local_strata must be a one-dimensional int64 tensor")
    if ctx.world_size < 1:
        raise ValueError("world_size must be positive")
    if sum(quotas) % ctx.world_size:
        raise ValueError("global batch size must be divisible by world_size")
    if any(quota % ctx.world_size for quota in quotas):
        raise ValueError("each stratum quota must be divisible by world_size")
    if local_strata.numel() and (
        int(local_strata.min()) < 0 or int(local_strata.max()) >= len(quotas)
    ):
        raise ValueError("local_strata contains an out-of-range stratum")
    local_counts = torch.bincount(local_strata, minlength=len(quotas))
    expected_local = torch.tensor(
        [quota // ctx.world_size for quota in quotas],
        dtype=local_counts.dtype,
        device=local_counts.device,
    )
    if not torch.equal(local_counts, expected_local):
        raise ValueError("local stratum counts do not match the equal per-rank quotas")
    global_counts = all_reduce_sum_nograd(local_counts, ctx)
    expected_global = torch.tensor(quotas, dtype=global_counts.dtype, device=global_counts.device)
    if not torch.equal(global_counts, expected_global):
        raise ValueError("reduced stratum counts do not match global_quotas")


def assert_no_grad_accumulation(steps: int) -> None:
    if steps != 1:
        raise ValueError("clean-JEPA currently requires exactly one accumulation step")


def assert_ddp_mean_reduction(model: object, ctx: DistContext) -> None:
    if ctx.world_size <= 1:
        return
    if not isinstance(model, DistributedDataParallel):
        raise ValueError("multi-rank clean-JEPA requires DistributedDataParallel")
    if getattr(model, "_comm_hooks", []):
        raise ValueError("custom DDP communication hooks are not supported")


def mean_for_logging(value: Tensor, ctx: DistContext) -> Tensor:
    return all_reduce_sum_nograd(value, ctx) / ctx.world_size
