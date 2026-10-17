#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot per-seed MLP benchmark CSVs as cross-seed means and variability bands."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (
    ("forward_ms", "Forward", "Time (ms)"),
    ("loss_ms", "MSE loss", "Time (ms)"),
    ("backward_ms", "Backward", "Time (ms)"),
    ("sn_update_ms", "SN state update", "Time (ms)"),
    ("total_ms", "Forward + loss + backward + SN update", "Time (ms)"),
    ("peak_allocated_mib", "Peak allocated CUDA memory", "Memory (MiB)"),
)
CONFIG_FIELDS = (
    "implementation", "implementation_sha256", "reference_sha256",
    "tokens", "dim", "hidden_dim", "layers", "grad_accum_steps", "dtype",
    "pc_level", "pc_checkpoint", "power_iter", "recover_w_norm", "learnable_gamma", "warmup",
    "steps", "device", "device_name", "device_total_mib", "device_sm_count",
    "torch_version", "cuda_version",
)
NAMES = {
    "tokens": "N", "dim": "d", "hidden_dim": "h", "hidden_ratio": "h/d",
    "grad_accum_steps": "accum",
    "pc_checkpoint": "checkpoint",
}
CHECKPOINTS = ("none", "polynomial", "full")


def load_rows(paths, labels=None):
    rows = []
    for i, path in enumerate(paths):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"seed", "status", "label", *CONFIG_FIELDS, *(metric[0] for metric in METRICS)}
            # CSVs from before accumulation support represent accum=1.
            for field in ("grad_accum_steps", "implementation_sha256", "reference_sha256", "pc_checkpoint"):
                required.discard(field)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                row["grad_accum_steps"] = row.get("grad_accum_steps") or "1"
                row["implementation_sha256"] = row.get("implementation_sha256") or ""
                row["reference_sha256"] = row.get("reference_sha256") or ""
                row["pc_checkpoint"] = row.get("pc_checkpoint") or "none"
                if row["status"] != "ok" or row.get("correctness_status") == "failed":
                    print(
                        f"Skipping {path}: {row['implementation']} seed={row['seed']} "
                        f"N={row['tokens']} d={row['dim']} ({row['status']})", file=sys.stderr,
                    )
                    continue
                row["source"] = str(path.resolve())
                row["series"] = labels[i] if labels else row["label"] or path.stem
                row["hidden_ratio"] = f"{int(row['hidden_dim']) / int(row['dim']):g}"
                rows.append(row)
    if not rows:
        raise ValueError("No successful benchmark rows to plot.")
    return rows


def group_rows(rows, x):
    fields = ["source", "series", *(field for field in CONFIG_FIELDS if field != x)]
    if x == "dim":
        # h normally scales with d; group a width sweep by expansion ratio.
        fields[fields.index("hidden_dim")] = "hidden_ratio"
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[field] for field in fields)
        value = CHECKPOINTS.index(row[x]) if x == "pc_checkpoint" else float(row[x])
        point = groups[key][value]
        if any(previous["seed"] == row["seed"] for previous in point):
            raise ValueError("Duplicate seed for the same curve/x value; pass each CSV only once and use distinct seeds.")
        point.append(row)
    return fields, groups


def mean_band(rows, metric, band):
    values = [float(row[metric]) for row in rows if row[metric] != ""]
    if not values:
        return None
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"Invalid values for {metric}: {values}")
    mean = statistics.mean(values)
    if band == "range":
        return mean, min(values), max(values), len(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, mean - spread, mean + spread, len(values)


def curve_label(config, varying, rows):
    parts = [config["series"], config["implementation"]]
    for field in varying:
        if field in ("source", "series", "implementation", "implementation_sha256", "reference_sha256"):
            continue
        parts.append(f"{NAMES.get(field, field)}={config[field]}")
    label = " / ".join(parts)
    sources = {row["source"] for row in rows if row["series"] == config["series"]}
    if len(sources) > 1:
        label += f" ({Path(config['source']).stem})"
    hashes = {row["implementation_sha256"] for row in rows if row["implementation"] == config["implementation"]}
    if len(hashes) > 1:
        label += f" [{config['implementation_sha256'][:8] or 'legacy'}]"
    return label


def plot_tradeoff(rows, band, output):
    if any(not row["peak_allocated_mib"] for row in rows):
        raise ValueError("Tradeoff plots require CUDA memory measurements; CPU rows have no CUDA memory value")
    fields, groups = group_rows(rows, "pc_checkpoint")
    varying = [field for field in fields if len({row[field] for row in rows}) > 1]
    fig, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    seed_counts = set()
    for index, (key, points) in enumerate(groups.items()):
        config = dict(zip(fields, key))
        label = curve_label(config, varying, rows)
        times, memories, xerr_low, xerr_high, yerr_low, yerr_high, annotations = [], [], [], [], [], [], []
        for mode, point in sorted(points.items()):
            time_mean, time_low, time_high, count = mean_band(point, "total_ms", band)
            mem_mean, mem_low, mem_high, _ = mean_band(point, "peak_allocated_mib", band)
            seed_counts.add(count)
            times.append(time_mean)
            memories.append(mem_mean)
            xerr_low.append(time_mean - time_low)
            xerr_high.append(time_high - time_mean)
            yerr_low.append(mem_mean - mem_low)
            yerr_high.append(mem_high - mem_mean)
            annotations.append(CHECKPOINTS[mode])
        color = plt.get_cmap("tab10")(index % 10)
        axis.errorbar(times, memories, xerr=[xerr_low, xerr_high], yerr=[yerr_low, yerr_high],
                      fmt="o", linestyle="--" if len(times) > 1 else "none", capsize=3,
                      color=color, label=label)
        if len(times) > 1 or config["implementation"] == "pc_v3":
            for x, y, annotation in zip(times, memories, annotations):
                axis.annotate(annotation, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
    band_text = "+/- 1 sample SD" if band == "std" else "min-max range"
    context = ", ".join(
        f"{NAMES.get(field, field)}={rows[0][field]}" for field in (
            "tokens", "dim", "hidden_dim", "layers", "grad_accum_steps", "dtype", "pc_level", "device_name",
        ) if len({row[field] for row in rows}) == 1
    )
    axis.set(xlabel="Time per accumulated step (ms)", ylabel="Peak allocated CUDA memory (MiB)",
             title=f"PC checkpoint tradeoff: lower and left are better\n{context}\n"
                   f"Cross-seed mean and {band_text}; seeds per point: {', '.join(map(str, sorted(seed_counts)))}")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", help="One experiment label per CSV")
    parser.add_argument(
        "--x", choices=("tokens", "dim", "hidden_dim", "layers", "pc_level", "grad_accum_steps", "pc_checkpoint"),
        default="dim",
    )
    parser.add_argument("--band", choices=("std", "range"), default="std", help="Across seeds: mean +/- one sample SD, or min/max")
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--tradeoff", action="store_true",
                        help="Plot step time versus peak CUDA memory, with checkpoint modes and seed intervals")
    parser.add_argument("--output", type=Path, default=Path("outputs/pc_profile/mlp_profile.png"))
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.csv_files):
        parser.error("--labels must contain one label per CSV")
    try:
        rows = load_rows(args.csv_files, args.labels)
        if args.tradeoff:
            plot_tradeoff(rows, args.band, args.output)
            print(f"Saved {args.output}")
            return
        fields, groups = group_rows(rows, args.x)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.log_x and args.x == "pc_checkpoint":
        parser.error("--log-x is unavailable for checkpoint modes")
    if args.log_x and any(float(row[args.x]) <= 0 for row in rows):
        parser.error("--log-x requires positive x values")

    varying = [field for field in fields if len({row[field] for row in rows}) > 1]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    seed_counts = set()
    handles, legends = [], []
    colors = plt.get_cmap("tab10")
    for index, (key, points) in enumerate(groups.items()):
        config = dict(zip(fields, key))
        label = curve_label(config, varying, rows)
        color = colors(index % 10)
        line = None
        for axis, (metric, title, ylabel) in zip(axes.flat, METRICS):
            data = [(x, mean_band(point_rows, metric, args.band)) for x, point_rows in sorted(points.items())]
            data = [(x, stats) for x, stats in data if stats is not None]
            if not data:
                continue
            xs = [item[0] for item in data]
            means, low, high, counts = zip(*(item[1] for item in data))
            seed_counts.update(counts)
            line, = axis.plot(xs, means, marker="o", markersize=4, color=color)
            if any(count > 1 for count in counts):
                axis.fill_between(xs, low, high, color=color, alpha=0.16)
                # A sweep with a single x value still needs a visible interval.
                axis.vlines(xs, low, high, color=color, linewidth=1.4)
        if line is not None:
            handles.append(line)
            legends.append(label)

    for axis, (metric, title, ylabel) in zip(axes.flat, METRICS):
        axis.set(title=title, xlabel=NAMES.get(args.x, args.x), ylabel=ylabel)
        axis.set_ylim(bottom=0)
        if args.log_x:
            axis.set_xscale("log", base=2)
        if args.x == "pc_checkpoint":
            axis.set_xticks(range(len(CHECKPOINTS)), CHECKPOINTS)
        if not axis.lines:
            axis.text(0.5, 0.5, "Unavailable on CPU", ha="center", va="center", transform=axis.transAxes)
    band_text = "+/- 1 sample SD" if args.band == "std" else "min-max range"
    counts_text = ", ".join(str(count) for count in sorted(seed_counts))
    context = ", ".join(
        f"{NAMES.get(field, field)}={rows[0][field]}"
        for field in fields
        if field in (
            "tokens", "dim", "hidden_dim", "hidden_ratio", "layers", "grad_accum_steps",
            "dtype", "pc_level", "device_name",
        )
        and field not in varying
    )
    fig.suptitle(
        f"MLP benchmark: cross-seed mean and {band_text}\n{context}\n"
        f"Seeds per point: {counts_text}; time per accumulated step (sum over microbatches)",
        fontsize=11,
    )
    fig.legend(handles, legends, loc="outside lower center", ncol=min(3, len(handles)), fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    if 1 in seed_counts:
        print("Some points have only one seed; their variability cannot be estimated.", file=sys.stderr)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
