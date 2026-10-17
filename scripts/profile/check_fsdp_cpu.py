# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise real two-rank CPU FSDP2; report failures without changing PC versions.

This checks correctness and parameter lifetimes, not CPU/GPU performance or
Triton kernels. Each case gets fresh processes and a bounded timeout, since an
invalid saved parameter may crash a worker instead of raising a Python error.
"""

import argparse
import copy
import faulthandler
import itertools
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError:  # The repo's PyTorch 2.7 import path.
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile.benchmark_mlp import PCConfig, build_mlp
from scripts.profile.implementations import (
    REFERENCE, available_implementations, implementation_metadata,
    initialize_model, load_implementation,
)

WORLD_SIZE = 2


def full_tensor(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().clone()


def reference_name(name):
    return name.replace("_checkpoint_wrapped_module.", "")


def data_for(seed, step, rank, microbatch, dim):
    generator = torch.Generator().manual_seed(
        10000 + seed * 1000 + step * 100 + rank * 10 + microbatch
    )
    return (torch.randn(3, dim, generator=generator),
            torch.randn(3, dim, generator=generator))


def worker(rank, workdir, case):
    path = Path(workdir)
    fault_log = (path / f"rank{rank}.fault.log").open("w")
    faulthandler.enable(file=fault_log)
    torch.set_num_threads(1)
    report = {"rank": rank, "status": "failed", "stage": "init_process_group",
              "checks": 0, "max_abs_error": 0.0, "microbatches": []}

    def checkpoint(stage):
        report["stage"] = stage
        # A native storage access can crash before Python's exception handler.
        # Persist the last stage so the parent still has useful evidence.
        (path / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    def compare(label, actual, expected):
        if actual is None or expected is None:
            raise AssertionError(f"{label}: missing tensor/gradient")
        actual, expected = actual.detach(), expected.detach()
        if actual.shape != expected.shape:
            raise AssertionError(f"{label}: {actual.shape} != {expected.shape}")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{label}: non-finite tensor")
        error = (actual - expected).abs().max().item() if actual.numel() else 0.0
        report["max_abs_error"] = max(report["max_abs_error"], error)
        report["checks"] += 1
        rtol, atol = (0.02, 0.002) if case.get("mixed_precision") == "bfloat16" else (1e-4, 1e-5)
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            ratio = actual.float().norm().item() / max(expected.float().norm().item(), 1e-30)
            raise AssertionError(f"{label}: max_abs_error={error:.6g}, norm_ratio={ratio:.6g}")

    try:
        checkpoint("init_process_group")
        dist.init_process_group(
            "gloo", init_method=(path / "rendezvous").as_uri(), rank=rank,
            world_size=WORLD_SIZE, timeout=timedelta(seconds=case["timeout"] / 2),
        )
        # Some older Gloo builds accept AVG but perform SUM. Keep this evidence
        # in the report; never compensate by changing losses or gradients.
        probe = torch.full((WORLD_SIZE,), float(rank + 1))
        probe_output = torch.empty(1)
        report["gloo_avg_probe"] = {"expected": (WORLD_SIZE + 1) / 2}
        try:
            dist.reduce_scatter_tensor(probe_output, probe, op=dist.ReduceOp.AVG)
            report["gloo_avg_probe"]["actual"] = probe_output.item()
        except RuntimeError as error:
            report["gloo_avg_probe"]["error"] = str(error)
        checkpoint("build_and_shard")
        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("dp",))
        implementation = case["implementation"]
        version = load_implementation(implementation)
        reference_implementation = "plain" if version is None else REFERENCE
        reference_version = load_implementation(reference_implementation)
        config = PCConfig(power_iter=2, pc_checkpoint=case.get("pc_checkpoint", "full"))
        torch.manual_seed(case["seed"])
        reference = build_mlp(8, 16, 2, reference_implementation, config)
        initialize_model(reference, reference_implementation)
        if reference_version:
            reference_version.update_model_sn_state(reference)
        candidate = build_mlp(8, 16, 2, implementation, config)
        initialize_model(candidate, implementation)
        candidate.load_state_dict(reference.state_dict())
        if case.get("with_plain_head"):
            head = torch.nn.Linear(8, 8, bias=True)
            reference.add_module("head", head)
            candidate.add_module("head", copy.deepcopy(head))
        param_dtype = torch.bfloat16 if case.get("mixed_precision") == "bfloat16" else torch.float32
        reduce_dtype = getattr(torch, case.get("reduce_dtype", "float32"))
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        # Match apply_dp(): shard every block, retain the last block's weights
        # after forward, and fully_shard the root as well. No FSDP monkeypatches.
        for index in range(2):
            block = candidate[index]
            ac = case.get("activation_checkpoint", "none")
            if ac == "full" or (ac == "selective" and index == 1):
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                from torch.utils.checkpoint import checkpoint as activation_checkpoint
                block = checkpoint_wrapper(block, checkpoint_fn=activation_checkpoint,
                                           use_reentrant=False, preserve_rng_state=False)
                candidate[index] = block
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=index < 1)
        fully_shard(candidate, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        if not all(isinstance(p, DTensor) for p in candidate.parameters()):
            raise AssertionError("FSDP did not shard all parameters")
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        accum = case["grad_accum_steps"]

        for step in range(2):
            checkpoint(f"step_{step + 1}/reference")
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs, expected_input_grads, expected_losses = [], [], []
            # The unsharded reference averages over both ranks' distinct data.
            # Reference forward/backward never invokes FSDP or an optimized PC.
            for data_rank in range(WORLD_SIZE):
                for microbatch in range(accum):
                    x, target = data_for(case["seed"], step, data_rank, microbatch, 8)
                    x.requires_grad_(case.get("input_grad", True))
                    if param_dtype == torch.bfloat16:
                        params = {name: p.to(param_dtype) for name, p in reference.named_parameters()}
                        output = torch.func.functional_call(reference, params, (x.to(param_dtype),))
                    else:
                        output = reference(x)
                    loss = F.mse_loss(output.float(), target)
                    (loss / (accum * WORLD_SIZE)).backward()
                    if data_rank == rank:
                        expected_outputs.append(output.detach())
                        # FSDP averages parameter gradients, not input gradients.
                        expected_input_grads.append(x.grad.detach() * WORLD_SIZE if x.grad is not None else None)
                        expected_losses.append(loss.detach())

            optimizer.zero_grad(set_to_none=True)
            context_factory = getattr(version, "accumulation_context", None)
            if implementation == "pc_v4":
                from torchtitan.pc_layer.runtime import accumulation_context as context_factory
            context = context_factory(candidate, accum) if context_factory else nullcontext()
            with context:
                for microbatch in range(accum):
                    if case.get("gradient_sync") == "final":
                        candidate.set_requires_gradient_sync(microbatch == accum - 1)
                    checkpoint(f"step_{step + 1}/microbatch_{microbatch + 1}/forward")
                    x, target = data_for(case["seed"], step, rank, microbatch, 8)
                    x.requires_grad_(case.get("input_grad", True))
                    output = candidate(x)
                    compare("output", output, expected_outputs[microbatch])
                    loss = F.mse_loss(output.float(), target)
                    compare("loss", loss, expected_losses[microbatch])
                    checkpoint(f"step_{step + 1}/microbatch_{microbatch + 1}/backward")
                    (loss / accum).backward()
                    if case.get("input_grad", True):
                        compare("input gradient", x.grad, expected_input_grads[microbatch])
                    report["microbatches"].append({
                        "step": step + 1, "microbatch": microbatch + 1,
                        "parameter_grads_present": sum(p.grad is not None for p in candidate.parameters()),
                    })
                checkpoint(f"step_{step + 1}/context_exit_pc_backward")

            checkpoint(f"step_{step + 1}/parameter_gradients")
            missing = [name for name, p in candidate.named_parameters() if p.grad is None]
            all_missing = [None] * WORLD_SIZE
            dist.all_gather_object(all_missing, missing)
            if any(all_missing):
                report["missing_gradients_by_rank"] = all_missing
                raise AssertionError(f"Missing parameter gradients: {all_missing}")
            reference_params = dict(reference.named_parameters())
            for name, parameter in candidate.named_parameters():
                compare(f"{name} gradient", full_tensor(parameter.grad), reference_params[reference_name(name)].grad)
            if case.get("clip_grad", 0):
                checkpoint(f"step_{step + 1}/gradient_clipping")
                expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), case["clip_grad"], foreach=True)
                actual_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), case["clip_grad"], foreach=True)
                compare("global gradient norm", full_tensor(actual_norm), expected_norm)
            optimizer.step()
            reference_optimizer.step()
            checkpoint(f"step_{step + 1}/updated_parameters")
            for name, parameter in candidate.named_parameters():
                compare(f"{name} after SGD", full_tensor(parameter), reference_params[reference_name(name)])

            checkpoint(f"step_{step + 1}/sn_update")
            if version:
                version.update_model_sn_state(candidate)
                reference_version.update_model_sn_state(reference)
                reference_buffers = dict(reference.named_buffers())
                for name, buffer in candidate.named_buffers():
                    compare(f"{name} after SN update", buffer, reference_buffers[reference_name(name)])
                # Explicitly verify replicated SN buffers agree across ranks.
                for name, buffer in candidate.named_buffers():
                    copies = [torch.empty_like(buffer) for _ in range(WORLD_SIZE)]
                    dist.all_gather(copies, buffer)
                    compare(f"{name} cross-rank", copies[0], copies[1])
            for module in candidate.modules():
                if getattr(module, "_weight_leaf", None) is not None:
                    raise AssertionError("PC cache survived accumulation context exit")
                if getattr(module, "_step_cache", None) is not None:
                    raise AssertionError("PC step cache survived accumulation context exit")
        report.update(status="passed", stage="complete")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        # Save before teardown so a communication failure cannot erase evidence.
        (path / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


def run_case(case):
    result = {**case, **implementation_metadata(case["implementation"])}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pc-fsdp-cpu-") as directory:
        context = mp.spawn(worker, args=(directory, case), nprocs=WORLD_SIZE, join=False)
        try:
            while not context.join(timeout=1):
                if time.monotonic() - started > case["timeout"]:
                    raise TimeoutError(f"Case exceeded {case['timeout']} seconds")
        except Exception as error:
            result["worker_error"] = f"{type(error).__name__}: {error}"
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        result["ranks"] = [json.loads(path.read_text())
                           for path in sorted(Path(directory).glob("rank*.json"))]
        result["native_tracebacks"] = {
            path.name: path.read_text() for path in Path(directory).glob("rank*.fault.log")
            if path.stat().st_size
        }
        result["exit_codes"] = [process.exitcode for process in context.processes]
    result["status"] = "passed" if (
        len(result["ranks"]) == WORLD_SIZE
        and all(rank["status"] == "passed" for rank in result["ranks"])
        and all(code == 0 for code in result["exit_codes"])
        and "worker_error" not in result
    ) else "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", nargs="+", choices=available_implementations(),
                        default=["plain", "pc_baseline", "pc_v4"])
    parser.add_argument("--grad-accum-steps", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--pc-checkpoint", nargs="+", choices=("none", "polynomial", "full"), default=["full"])
    parser.add_argument("--activation-checkpoint", choices=("none", "full", "selective"), nargs="+", default=["none"])
    parser.add_argument("--mixed-precision", choices=("none", "bfloat16"), nargs="+", default=["none"])
    parser.add_argument("--reduce-dtype", choices=("float32", "float64"), default="float32",
                        help="float64 uses FSDP's explicit averaging path for older CPU/Gloo builds")
    parser.add_argument("--gradient-sync", choices=("each", "final"), default="each")
    parser.add_argument("--input-grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-plain-head", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Check global gradient clipping before SGD")
    parser.add_argument("--timeout", type=float, default=60, help="Seconds per isolated two-rank case")
    parser.add_argument("--output", type=Path, default=None, help="JSON report; never overwrite")
    args = parser.parse_args()
    if any(value < 1 for value in args.grad_accum_steps) or args.timeout <= 0 or args.clip_grad < 0:
        parser.error("accumulation counts and timeout must be positive; clip-grad must be nonnegative")
    if not dist.is_gloo_available():
        parser.error("This PyTorch build does not include Gloo")
    # Use only local communication unless the caller selected an interface.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "outputs" / "pc_profile" / f"fsdp_cpu_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"torch_version": torch.__version__, "device": "cpu", "backend": "gloo",
              "world_size": WORLD_SIZE, "master_dtype": "float32", "steps": 2,
              "dim": 8, "hidden_dim": 16, "layers": 2, "tokens": 3,
              "reshard_after_forward": [True, False], "cases": []}
    print(f"PyTorch {torch.__version__}; CPU/Gloo; 2 ranks; report: {output}", flush=True)
    with output.open("x") as stream:
        for implementation in args.implementation:
            supported = getattr(load_implementation(implementation), "CHECKPOINT_MODES", ())
            modes = args.pc_checkpoint if supported else ["none"]
            for accum, seed, mode, ac, precision in itertools.product(
                args.grad_accum_steps, args.seeds, modes, args.activation_checkpoint, args.mixed_precision,
            ):
                case = run_case({"implementation": implementation, "grad_accum_steps": accum,
                                 "seed": seed, "timeout": args.timeout, "pc_checkpoint": mode,
                                 "activation_checkpoint": ac, "mixed_precision": precision,
                                 "reduce_dtype": args.reduce_dtype,
                                 "gradient_sync": args.gradient_sync, "input_grad": args.input_grad,
                                 "with_plain_head": args.with_plain_head, "clip_grad": args.clip_grad})
                report["cases"].append(case)
                print(f"{case['status'].upper()}: {implementation}, accum={accum}, seed={seed}, "
                      f"PC={mode}, AC={ac}, MP={precision}, reduce={args.reduce_dtype}, "
                      f"sync={args.gradient_sync}", flush=True)
                if case["status"] == "failed":
                    failed = next((rank for rank in case["ranks"] if rank["status"] == "failed"), {})
                    print(f"  {failed.get('stage', 'worker')}: "
                          f"{failed.get('error', case.get('worker_error', 'worker exited'))}", flush=True)
                stream.seek(0)
                json.dump(report, stream, indent=2)
                stream.write("\n")
                stream.truncate()
                stream.flush()
    return int(any(case["status"] != "passed" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
