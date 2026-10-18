import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("clean-JEPA tests require the optional torch dependency")

import json
import os
import tempfile
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as torch_dist
import torch.distributed.nn.functional as torch_dist_nn
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from openchange.jepa.distributed import (
    DistContext,
    all_reduce_sum_autograd,
    assert_ddp_mean_reduction,
    assert_no_grad_accumulation,
    assert_quotas,
    get_context,
    make_global_strata,
    mean_for_logging,
    shard_global_batch,
)
from openchange.jepa.losses import jepa_loss, sigreg_directions


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    values = 1.5 * torch.randn((4, 16, 5), generator=generator, dtype=torch.float64)
    strata = make_global_strata([4] * 4)
    directions = sigreg_directions(
        8,
        32,
        seed=44,
        step=2,
        device="cpu",
        dtype=torch.float64,
    )
    return values, strata, directions


def _loss_kwargs(mode: str, directions: torch.Tensor) -> dict[str, object]:
    return {
        "n_global": 2,
        "n_strata": 4,
        "weights": [0.25] * 4,
        "lam": 0.3,
        "lam_mu": 0.2,
        "directions": directions,
        "mode": mode,
    }


def _flat_gradients(model: nn.Module) -> list[float]:
    return torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()]).tolist()


def _flat_parameters(model: nn.Module) -> list[float]:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).tolist()


def _identity_backward_reduce(value: torch.Tensor, context: DistContext) -> torch.Tensor:
    if context.world_size == 1:
        return value
    total = value.contiguous().clone()
    torch_dist.all_reduce(total, op=torch_dist.ReduceOp.SUM, group=context.group)
    return value + (total - value).detach()


def _ddp_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch_dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    context = get_context()
    try:
        values, global_strata, directions = _batch()
        local_values = shard_global_batch(values, context, dim=1)
        local_strata = shard_global_batch(global_strata, context, dim=0)
        assert_quotas(local_strata, [4] * 4, context)
        records: dict[str, object] = {"runs": {}, "collective": {}}
        assert_ddp_mean_reduction(DistributedDataParallel(nn.Linear(5, 8)), context)

        probe = torch.tensor([rank + 1.0], requires_grad=True)
        reduced = all_reduce_sum_autograd(probe, context)
        reduced.sum().backward()
        records["collective"]["sum"] = reduced.item()
        records["collective"]["our_grad"] = probe.grad.item()
        if hasattr(torch_dist_nn, "all_reduce"):
            reference = torch.tensor([rank + 1.0], requires_grad=True)
            reference_sum = torch_dist_nn.all_reduce(reference, op=torch_dist.ReduceOp.SUM)
            reference_sum.sum().backward()
            records["collective"]["functional_sum"] = reference_sum.item()
            records["collective"]["functional_grad"] = reference.grad.item()

        for mode in ("pooled", "stratified"):
            for optimizer_name in ("adamw", "sgd"):
                torch.manual_seed(0)
                model = DistributedDataParallel(nn.Linear(5, 8))
                output = jepa_loss(
                    model(local_values),
                    local_strata,
                    **_loss_kwargs(mode, directions),
                    dist=context,
                )
                logged_loss = mean_for_logging(output["loss"], context)
                output["loss"].backward()
                if optimizer_name == "adamw":
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=1e-2,
                        betas=(0.9, 0.95),
                        weight_decay=0.05,
                    )
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
                optimizer.step()
                records["runs"][f"{mode}_{optimizer_name}"] = {
                    "loss": output["loss"].item(),
                    "logged_loss": logged_loss.item(),
                    "gradients": _flat_gradients(model),
                    "parameters": _flat_parameters(model),
                }

        if world_size > 1:
            torch.manual_seed(0)
            wrong_model = DistributedDataParallel(nn.Linear(5, 8))
            with mock.patch(
                "openchange.jepa.losses.all_reduce_sum_autograd",
                _identity_backward_reduce,
            ):
                wrong_loss = jepa_loss(
                    wrong_model(local_values),
                    local_strata,
                    **_loss_kwargs("stratified", directions),
                    dist=context,
                )["loss"]
                wrong_loss.backward()
            torch.manual_seed(0)
            reference_model = nn.Linear(5, 8)
            reference_loss = jepa_loss(
                reference_model(values),
                global_strata,
                **_loss_kwargs("stratified", directions),
            )["loss"]
            reference_loss.backward()
            reference_gradient = torch.cat(
                [parameter.grad.reshape(-1) for parameter in reference_model.parameters()]
            )
            wrong_gradient = torch.cat(
                [parameter.grad.reshape(-1) for parameter in wrong_model.parameters()]
            )
            records["wrong_reducer_relative_error"] = (
                torch.linalg.vector_norm(wrong_gradient - reference_gradient)
                / torch.linalg.vector_norm(reference_gradient)
            ).item()

        with open(os.path.join(output_dir, f"rank-{rank}.json"), "w", encoding="utf-8") as stream:
            json.dump(records, stream)
    finally:
        torch_dist.destroy_process_group()


class DistributedHelperTests(unittest.TestCase):
    def test_stratum_blocks_shard_evenly_and_bad_layout_is_rejected(self) -> None:
        quotas = [4, 4, 4, 4]
        global_strata = make_global_strata(quotas)
        self.assertEqual(global_strata.dtype, torch.int64)
        self.assertEqual(global_strata.tolist(), [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
        for rank in range(2):
            context = DistContext(rank=rank, world_size=2)
            local = shard_global_batch(global_strata, context, dim=0)
            counts = torch.bincount(local, minlength=4)
            self.assertEqual(counts.tolist(), [2] * 4)
            with mock.patch("openchange.jepa.distributed.all_reduce_sum_nograd", side_effect=lambda x, _: x * 2):
                assert_quotas(local, quotas, context)

        bad_layout = torch.arange(16, dtype=torch.int64) % 4
        bad_local = shard_global_batch(
            bad_layout,
            DistContext(rank=0, world_size=2),
            dim=0,
        )
        with self.assertRaises(ValueError):
            assert_quotas(bad_local, quotas, DistContext(rank=0, world_size=2))

    def test_accumulation_and_ddp_conventions(self) -> None:
        assert_no_grad_accumulation(1)
        with self.assertRaises(ValueError):
            assert_no_grad_accumulation(2)
        assert_ddp_mean_reduction(nn.Linear(2, 2), DistContext(rank=0, world_size=1))
        with self.assertRaises(ValueError):
            assert_ddp_mean_reduction(nn.Linear(2, 2), DistContext(rank=0, world_size=2))

    def test_non_distributed_context_and_logging_mean(self) -> None:
        self.assertEqual(get_context(), DistContext(rank=0, world_size=1))
        value = torch.tensor(3.0, requires_grad=True)
        logged = mean_for_logging(value, DistContext(rank=0, world_size=1))
        self.assertEqual(logged.item(), 3.0)
        self.assertFalse(logged.requires_grad)


class SpawnedGlooTests(unittest.TestCase):
    def _run_world(self, world_size: int) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="jepa-gloo-") as directory:
            init_path = os.path.join(directory, "init")
            init_method = f"file://{init_path}"
            mp.spawn(
                _ddp_worker,
                args=(world_size, init_method, directory),
                nprocs=world_size,
                join=True,
            )
            results = []
            for rank in range(world_size):
                with open(
                    os.path.join(directory, f"rank-{rank}.json"),
                    encoding="utf-8",
                ) as result_file:
                    results.append(json.load(result_file))
            return results

    def _reference(self, mode: str, optimizer_name: str) -> dict[str, object]:
        torch.set_default_dtype(torch.float64)
        values, strata, directions = _batch()
        torch.manual_seed(0)
        model = nn.Linear(5, 8)
        loss = jepa_loss(model(values), strata, **_loss_kwargs(mode, directions))["loss"]
        loss.backward()
        gradients = _flat_gradients(model)
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-2,
                betas=(0.9, 0.95),
                weight_decay=0.05,
            )
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        optimizer.step()
        return {
            "loss": loss.item(),
            "gradients": gradients,
            "parameters": _flat_parameters(model),
        }

    def test_spawned_gloo_matches_single_process_for_world_one_and_two(self) -> None:
        for world_size in (1, 2):
            with self.subTest(world_size=world_size):
                records = self._run_world(world_size)
                first_collective = records[0]["collective"]
                self.assertEqual(first_collective["sum"], sum(range(1, world_size + 1)))
                self.assertEqual(first_collective["our_grad"], world_size)
                if "functional_grad" in first_collective:
                    self.assertEqual(
                        first_collective["functional_sum"],
                        first_collective["sum"],
                    )
                    self.assertEqual(first_collective["functional_grad"], world_size)
                for mode in ("pooled", "stratified"):
                    for optimizer_name in ("adamw", "sgd"):
                        name = f"{mode}_{optimizer_name}"
                        reference = self._reference(mode, optimizer_name)
                        rank_runs = [record["runs"][name] for record in records]
                        mean_loss = sum(run["loss"] for run in rank_runs) / world_size
                        torch.testing.assert_close(
                            torch.tensor(mean_loss),
                            torch.tensor(reference["loss"]),
                            atol=1e-12,
                            rtol=0,
                        )
                        for run in rank_runs:
                            torch.testing.assert_close(
                                torch.tensor(run["logged_loss"]),
                                torch.tensor(reference["loss"]),
                                atol=1e-12,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                torch.tensor(run["gradients"]),
                                torch.tensor(reference["gradients"]),
                                atol=1e-10,
                                rtol=1e-8,
                            )
                            torch.testing.assert_close(
                                torch.tensor(run["parameters"]),
                                torch.tensor(reference["parameters"]),
                                atol=1e-10,
                                rtol=1e-8,
                            )
                if world_size == 2:
                    self.assertGreater(records[0]["wrong_reducer_relative_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
