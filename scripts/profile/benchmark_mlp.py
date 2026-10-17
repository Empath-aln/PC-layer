#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark MLP -> FP32 MSE -> backward on fixed random data, one row per seed."""

import argparse
import csv
import gc
import itertools
import math
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile.correctness import check_equivalence
from scripts.profile.implementations import (
    REFERENCE, available_implementations, implementation_metadata, initialize_model,
    load_implementation,
)

PCLinear = load_implementation(REFERENCE).PCLinear
update_model_sn_state = load_implementation(REFERENCE).update_model_sn_state

PHASES = ("forward", "loss", "backward", "sn_update", "total")
MEMORY_FIELDS = (
    "start_allocated_mib", "peak_allocated_mib", "extra_peak_allocated_mib",
    "peak_reserved_mib",
)
FIELDS = (
    "run_id", "label", "implementation", "implementation_file", "implementation_sha256",
    "reference_implementation", "reference_sha256", "status", "error", "seed", "tokens",
    "dim", "hidden_dim", "layers", "grad_accum_steps", "effective_tokens",
    "dtype", "pc_level", "power_iter",
    "recover_w_norm", "learnable_gamma", "warmup", "steps", "device",
    "device_name", "device_total_mib", "device_sm_count", "torch_version",
    "cuda_version", "git_commit", "git_dirty", "loss", "parameter_mib",
    "buffer_mib", "wall_ms",
    "correctness_status", "correctness_error", "correctness_rtol", "correctness_atol",
    "correctness_checks", "correctness_max_abs_error", "correctness_max_rel_error",
    *(field for phase in PHASES for field in (f"{phase}_ms", f"{phase}_p95_ms")),
    *MEMORY_FIELDS,
)


@dataclass
class PCConfig:
    pc_level: int = 4
    power_iter: int = 10
    pc_norm_type: str = "sn"
    pc_norm_eps: float = 1e-7
    recover_w_norm: bool = True
    learnable_gamma: bool = True
    gamma_init_value: float = 1.0


def build_mlp(dim, hidden_dim, layers, implementation, config):
    version = load_implementation(implementation)
    # Create every raw weight before a version's constructor can consume RNG.
    projections = [
        (nn.Linear(dim, hidden_dim, bias=False), nn.Linear(hidden_dim, dim, bias=False))
        for _ in range(layers)
    ]
    blocks = []
    for layer_id, (up, down) in enumerate(projections):
        if version is not None:
            up = version.PCLinear(up, config, layer_id)
            down = version.PCLinear(down, config, layer_id)
        blocks.append(nn.Sequential(up, nn.GELU(), down))
    return nn.Sequential(*blocks)


def run_step(
    model, x, target, uses_pc, markers=None, grad_accum_steps=1, update_sn_state=None,
    accumulation_context=None,
):
    """One accumulated step; the fixed microbatch is reused without data copies."""
    model.zero_grad(set_to_none=True)

    def mark(index):
        if markers is not None:
            if x.is_cuda:
                markers[index].record()
            else:
                markers[index] = time.perf_counter()

    # Versions may share a weight transform across microbatches and defer its
    # backward until context exit. That work belongs to the final backward
    # interval, before the SN state can change. Context entry only sets up state;
    # transforms are evaluated lazily inside the measured forward.
    context = accumulation_context(model, grad_accum_steps) if accumulation_context else nullcontext()
    with context:
        for microbatch in range(grad_accum_steps):
            # Only parameter gradients accumulate. x stands in for a fresh input
            # leaf on each microbatch, so retain just its most recent gradient.
            x.grad = None
            offset = 4 * microbatch
            mark(offset)
            prediction = model(x)
            mark(offset + 1)
            unscaled_loss = F.mse_loss(prediction.float(), target)
            loss = unscaled_loss / grad_accum_steps if grad_accum_steps > 1 else unscaled_loss
            del prediction
            mark(offset + 2)
            loss.backward()
            if microbatch + 1 < grad_accum_steps:
                mark(offset + 3)
    mark(4 * grad_accum_steps - 1)
    mark(4 * grad_accum_steps)
    if uses_pc:
        (update_sn_state or update_model_sn_state)(model)
    mark(4 * grad_accum_steps + 1)
    # All microbatches reuse the same data and SN state, so their unscaled
    # losses are equal. Report that MSE, independent of accumulation count.
    return unscaled_loss.detach()


def percentile95(values):
    """Linearly interpolated 95th percentile, also defined for a single sample."""
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_case_model(args, implementation, seed, dim, hidden_dim, layers, device, reference_state=None):
    torch.manual_seed(seed)
    config = PCConfig(
        pc_level=args.pc_level, power_iter=args.power_iter,
        recover_w_norm=args.recover_w_norm, learnable_gamma=args.learnable_gamma,
    )
    model = build_mlp(dim, hidden_dim, layers, implementation, config)
    model = model.to(device=device, dtype=getattr(torch, args.dtype)).train()
    initialize_model(model, implementation)
    if reference_state is not None:
        model.load_state_dict(reference_state, strict=True)
    elif implementation == REFERENCE:
        load_implementation(implementation).update_model_sn_state(model)
    elif implementation != "plain":
        raise ValueError("Optimized PC versions require a checked reference state")
    return model


def make_inputs(args, seed, tokens, dim, device):
    # A separate generator keeps inputs/targets identical across implementations,
    # even when an implementation consumes additional randomness during setup.
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    x = torch.randn(
        tokens, dim, device=device, dtype=getattr(torch, args.dtype),
        generator=generator, requires_grad=True,
    )
    target = torch.randn(tokens, dim, device=device, generator=generator)
    return x, target


def check_case(args, implementation, seed, tokens, dim, hidden_dim, layers, device, grad_accum_steps):
    reference_update = load_implementation(REFERENCE).update_model_sn_state
    candidate = load_implementation(implementation)
    rtol = args.check_rtol if args.check_rtol is not None else (0.02 if args.dtype == "bfloat16" else 1e-4)
    atol = args.check_atol if args.check_atol is not None else (0.002 if args.dtype == "bfloat16" else 1e-5)
    return check_equivalence(
        lambda: make_case_model(args, REFERENCE, seed, dim, hidden_dim, layers, device),
        lambda state: make_case_model(args, implementation, seed, dim, hidden_dim, layers, device, state),
        lambda: make_inputs(args, seed, tokens, dim, device),
        lambda model, x, target: run_step(
            model, x, target, True, grad_accum_steps=grad_accum_steps, update_sn_state=reference_update,
        ),
        lambda model, x, target: run_step(
            model, x, target, True, grad_accum_steps=grad_accum_steps,
            update_sn_state=candidate.update_model_sn_state,
            accumulation_context=getattr(candidate, "accumulation_context", None),
        ),
        rtol=rtol, atol=atol,
    )


def measure_case(
    args, implementation, seed, tokens, dim, hidden_dim, layers, device,
    grad_accum_steps=1, reference_state=None,
):
    model = make_case_model(args, implementation, seed, dim, hidden_dim, layers, device, reference_state)
    version = load_implementation(implementation)
    uses_pc = version is not None
    updater = version.update_model_sn_state if uses_pc else None
    accumulation_context = getattr(version, "accumulation_context", None)
    x, target = make_inputs(args, seed, tokens, dim, device)
    for _ in range(args.warmup):
        run_step(
            model, x, target, uses_pc, grad_accum_steps=grad_accum_steps,
            update_sn_state=updater, accumulation_context=accumulation_context,
        )
    model.zero_grad(set_to_none=True)
    x.grad = None

    marker_count = 4 * grad_accum_steps + 2
    markers = [[0.0] * marker_count for _ in range(args.steps)]
    memory = {field: "" for field in MEMORY_FIELDS}
    if device.type == "cuda":
        markers = [
            [torch.cuda.Event(enable_timing=True) for _ in range(marker_count)]
            for _ in range(args.steps)
        ]
        # Materialize lazy event resources before recording measured steps.
        for step_markers in markers:
            for event in step_markers:
                event.record()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory["start_allocated_mib"] = torch.cuda.memory_allocated(device) / 2**20

    wall_start = time.perf_counter()
    for step_markers in markers:
        last_loss = run_step(
            model, x, target, uses_pc, step_markers, grad_accum_steps, updater,
            accumulation_context,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000 / args.steps

    if device.type == "cuda":
        memory["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        memory["extra_peak_allocated_mib"] = (
            memory["peak_allocated_mib"] - memory["start_allocated_mib"]
        )
        memory["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20

    samples = {phase: [] for phase in PHASES}
    for step_markers in markers:
        def elapsed(start, end):
            if device.type == "cuda":
                return step_markers[start].elapsed_time(step_markers[end])
            return (step_markers[end] - step_markers[start]) * 1000

        for i, phase in enumerate(PHASES[:3]):
            samples[phase].append(sum(
                elapsed(4 * microbatch + i, 4 * microbatch + i + 1)
                for microbatch in range(grad_accum_steps)
            ))
        samples["sn_update"].append(
            elapsed(4 * grad_accum_steps, marker_count - 1) if uses_pc else 0.0
        )
        samples["total"].append(elapsed(0, marker_count - 1))

    # Validate outside the measured region, after capturing the memory counters.
    gradients = [x.grad, *(parameter.grad for parameter in model.parameters())]
    valid = torch.isfinite(last_loss).item() and all(
        grad is not None and torch.isfinite(grad).all().item() for grad in gradients
    )
    result = {
        "status": "ok" if valid else "invalid",
        "error": "" if valid else "Non-finite loss/gradients or missing gradients",
        "loss": last_loss.item(), "wall_ms": wall_ms,
        "parameter_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
        "buffer_mib": sum(b.numel() * b.element_size() for b in model.buffers()) / 2**20,
        **memory,
    }
    for phase, values in samples.items():
        result[f"{phase}_ms"] = statistics.mean(values)
        result[f"{phase}_p95_ms"] = percentile95(values)
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", nargs="+", choices=available_implementations(),
        default=["plain", REFERENCE], help="PC versions are discovered from profile/pc_*.py",
    )
    parser.add_argument(
        "--tokens", nargs="+", type=positive_int, default=[1024],
        help="Rows N per microbatch (batch * sequence length, if applicable)",
    )
    parser.add_argument("--dim", nargs="+", type=positive_int, default=[1024])
    parser.add_argument("--hidden-dim", type=positive_int, help="Default: 4 * dim")
    parser.add_argument("--layers", nargs="+", type=positive_int, default=[1], help="Number of MLP blocks")
    parser.add_argument(
        "--grad-accum-steps", nargs="+", type=positive_int, default=[1],
        help="Microbatches per accumulated step; --tokens is per microbatch",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, etc.; cpu for smoke tests")
    parser.add_argument("--pc-level", type=int, choices=range(5), default=4)
    parser.add_argument("--power-iter", type=positive_int, default=10)
    parser.add_argument("--recover-w-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learnable-gamma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=positive_int, default=10, help="Full accumulated warmup steps")
    parser.add_argument("--steps", type=positive_int, default=30, help="Full accumulated measured steps")
    parser.add_argument("--label", help="Experiment name for plot legends; default: CSV filename")
    parser.add_argument("--check-rtol", type=float, help="Equivalence relative tolerance: FP32 1e-4, BF16 0.02")
    parser.add_argument("--check-atol", type=float, help="Equivalence absolute tolerance: FP32 1e-5, BF16 0.002")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=Path(f"outputs/pc_profile/mlp_{timestamp}.csv"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**63 - 1 for seed in args.seeds):
        parser.error("--seeds must be distinct integers in [0, 2**63 - 2]")
    for name in ("tokens", "dim", "layers", "grad_accum_steps", "implementation"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    for name in ("check_rtol", "check_atol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if any(name not in ("plain", REFERENCE) for name in args.implementation) and REFERENCE not in args.implementation:
        args.implementation.insert(0, REFERENCE)
    return args


def git_value(*arguments):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise SystemExit("Use a CUDA device, or --device cpu for a smoke test.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable. Use a CUDA PyTorch environment, or --device cpu for a smoke test.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    metadata = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "label": args.label or args.output.stem, "dtype": args.dtype,
        "pc_level": args.pc_level, "power_iter": args.power_iter,
        "recover_w_norm": args.recover_w_norm, "learnable_gamma": args.learnable_gamma,
        "warmup": args.warmup, "steps": args.steps, "device": str(device),
        "device_name": "CPU", "device_total_mib": "", "device_sm_count": "",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda or "",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": git_value("status", "--porcelain") != "",
        "reference_sha256": implementation_metadata(REFERENCE)["implementation_sha256"],
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            device_name=properties.name, device_total_mib=properties.total_memory / 2**20,
            device_sm_count=properties.multi_processor_count,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Output already exists; choose a new --output path: {args.output}")
    failed = 0
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for tokens, dim, layers, grad_accum_steps in itertools.product(
            args.tokens, args.dim, args.layers, args.grad_accum_steps,
        ):
            hidden_dim = args.hidden_dim or 4 * dim
            results = {name: [] for name in args.implementation}
            for seed_index, seed in enumerate(args.seeds):
                # Alternate the paired order to reduce systematic ordering bias.
                order = args.implementation if seed_index % 2 == 0 else args.implementation[::-1]
                for implementation in order:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {
                        **metadata, "implementation": implementation, "seed": seed,
                        **implementation_metadata(implementation),
                        "reference_implementation": REFERENCE if implementation != "plain" else "",
                        "correctness_status": "not_applicable" if implementation == "plain" else "reference",
                        "tokens": tokens, "dim": dim, "hidden_dim": hidden_dim, "layers": layers,
                        "grad_accum_steps": grad_accum_steps,
                        "effective_tokens": tokens * grad_accum_steps,
                    }
                    reference_state = None
                    try:
                        if implementation not in ("plain", REFERENCE):
                            check, reference_state = check_case(
                                args, implementation, seed, tokens, dim, hidden_dim, layers, device, grad_accum_steps,
                            )
                            row.update(check)
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        if row["correctness_status"] == "failed":
                            row.update(status="correctness_failed", error=row["correctness_error"])
                        else:
                            row.update(measure_case(
                                args, implementation, seed, tokens, dim, hidden_dim, layers, device,
                                grad_accum_steps, reference_state,
                            ))
                    except torch.cuda.OutOfMemoryError as error:
                        row.update(status="oom", error=str(error))
                    del reference_state
                    writer.writerow(row)
                    handle.flush()
                    results[implementation].append(row)
                    failed += row["status"] != "ok"
                    print(
                        f"{implementation:8s} seed={seed} N={tokens} d={dim} "
                        f"h={hidden_dim} layers={layers} accum={grad_accum_steps}: "
                        f"{row['status']}", flush=True,
                    )
                    if row.get("correctness_status") in ("passed", "failed"):
                        print(
                            f"  vs {REFERENCE}: {row['correctness_status']} "
                            f"(max_abs={row['correctness_max_abs_error']:.3g}, "
                            f"max_rel={row['correctness_max_rel_error']:.3g}) "
                            f"{row.get('correctness_error', '')}", flush=True,
                        )
            for implementation, rows in results.items():
                successful = [row for row in rows if row["status"] == "ok"]
                if not successful:
                    continue
                values = [row["total_ms"] for row in successful]
                spread = statistics.stdev(values) if len(values) > 1 else 0.0
                message = (
                    f"  {implementation}: total {statistics.mean(values):.3f} +/- {spread:.3f} ms "
                    f"({len(values)}/{len(args.seeds)} seeds)"
                )
                if device.type == "cuda":
                    peaks = [row["peak_allocated_mib"] for row in successful]
                    spread = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
                    message += f", peak {statistics.mean(peaks):.1f} +/- {spread:.1f} MiB"
                print(message, flush=True)
            reference_rows = {
                row["seed"]: row for row in results.get(REFERENCE, []) if row["status"] == "ok"
            }
            for implementation, rows in results.items():
                if implementation in ("plain", REFERENCE):
                    continue
                pairs = [
                    (reference_rows[row["seed"]], row) for row in rows
                    if row["status"] == "ok" and row["seed"] in reference_rows
                ]
                if not pairs:
                    continue
                speedups = [reference["total_ms"] / row["total_ms"] for reference, row in pairs]
                spread = statistics.stdev(speedups) if len(speedups) > 1 else 0.0
                message = (
                    f"  {implementation} vs {REFERENCE}: "
                    f"{statistics.mean(speedups):.3f} +/- {spread:.3f}x speedup ({len(pairs)} paired seeds)"
                )
                if device.type == "cuda":
                    savings = [reference["peak_allocated_mib"] - row["peak_allocated_mib"] for reference, row in pairs]
                    message += f", mean peak memory saving {statistics.mean(savings):.1f} MiB"
                print(message, flush=True)
    print(f"Saved {args.output}", flush=True)
    if failed:
        raise SystemExit(f"{failed} case(s) failed; see status/error in the CSV.")


if __name__ == "__main__":
    main()
