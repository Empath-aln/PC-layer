# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting module for singular value histograms."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "grid.linestyle": (0, (8, 6)),
    "ytick.major.width": 1.2,
    "ytick.major.size": 5,
    "ytick.direction": "in",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = [
    "#5C6BC0",  # indigo
    "#ff7f0e",  # orange (matplotlib default)
    "#1f77b4",  # blue (matplotlib default)
    "#2ca02c",  # green (matplotlib default)
    "#d62728",  # red (matplotlib default)
    "#7B2D8E",  # purple
    "#E76F51",  # coral
    "#264653",  # dark teal
    "#A8DADC",  # light blue
    "#C2185B",  # deep pink
    "#4CAF50",  # green
    "#FF7043",  # deep orange
    "#8D6E63",  # brown
    "#00ACC1",  # cyan
    "#AFB42B",  # lime
    "#E91E63",  # pink
]


def load_json_records(filepaths: List[str]) -> List[Dict[str, Any]]:
    """Load multiple JSON files containing singular values.

    Args:
        filepaths: List of paths to JSON files

    Returns:
        List of loaded records
    """
    records = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            records.append(json.load(f))
    return records


def sanitize_path_component(value: str) -> str:
    """Convert an arbitrary string into a filesystem-safe path component."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "item"


def extract_layer_id(layer_name: str) -> int:
    """Extract layer/block ID from layer name.

    Args:
        layer_name: e.g., "layers.0.attention.wq"

    Returns:
        Layer ID (integer), or -1 if not found
    """
    match = re.search(r'layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return -1


def get_record_label(record: Dict[str, Any], idx: int) -> str:
    """Get label for a record (use swanlab_comment if available)."""
    meta = record.get("metadata", {})
    comment = meta.get("swanlab_comment", "")
    if comment:
        return comment
    return f"Record {idx+1}"


def _draw_series(ax, data, *, color, label, bins, hist_alpha, mode):
    """Draw one singular-value series as a histogram, ECDF, or quantile curve.

    Singular values are σ/σ_max-normalized by the caller before being passed in,
    so they live in [0, 1].

    mode == "quantile": empirical quantile function — sort ascending, x = fraction q
        in [0, 1] of values counted from the smallest, y = the value at that fraction.
        A higher curve in the low-fraction region means the lower tail of the spectrum
        is lifted.
    mode == "ecdf": stepped cumulative curve, F(t) = Pr(x <= t).
    mode == "hist": histogram (default).
    """
    if mode == "quantile":
        ys = np.sort(np.asarray(data, dtype=float))
        qs = np.linspace(0.0, 1.0, ys.size)
        ax.plot(qs, ys, color=color, linewidth=2.0, label=label)
        return

    if mode != "ecdf":
        ax.hist(data, bins=bins, color=color, edgecolor="white",
                alpha=hist_alpha, linewidth=0.6, rwidth=0.8, label=label)
        return

    # Empirical CDF: sort values, y goes 1/n .. 1, drawn as a post-step curve.
    xs = np.sort(np.asarray(data, dtype=float))
    ys = np.arange(1, xs.size + 1) / xs.size
    # Prepend a point at the left so the curve starts from y=0.
    xs = np.concatenate([xs[:1], xs])
    ys = np.concatenate([[0.0], ys])
    ax.plot(xs, ys, drawstyle="steps-post", color=color, linewidth=2.0, label=label)



def plot_singular_values(records: List[Dict[str, Any]],
                         output_dir: str = None,
                         log_y: bool = False,
                         labels: List[str] = None,
                         fmt: str = "pdf",
                         mode: str = "hist"):
    """Plot singular-value distributions, σ/σ_max-normalized to [0, 1].

    Creates folders by weight type (wq, wk, wv, wo, w1, w2, w3),
    each containing layer{N} plots for all layers.

    Args:
        records: List of loaded JSON records
        output_dir: Directory to save plots (defaults to sibling dir with combined swanlab_comments)
        log_y: Use a log-scale y-axis (histogram mode only)
        labels: Custom legend labels, one per record
        fmt: Output image format ("pdf" or "png")
        mode: "hist" (default), "ecdf", or "quantile"
    """
    if not records:
        print("No records to plot")
        return

    # Determine output directory
    if output_dir is None:
        json_dirs = [Path(r.get("_source_path", ".")).parent for r in records]
        if len(records) > 1:
            # Multiple JSONs: common parent / combined sanitized dir names
            common_parent = Path(os.path.commonpath([str(d) for d in json_dirs]))
            dir_name = "__".join(sanitize_path_component(d.name) for d in json_dirs)
            output_dir = common_parent / dir_name
        else:
            # Single JSON: use same dir as before
            output_dir = json_dirs[0]
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique layer names
    layer_names = set()
    for record in records:
        layer_names.update(record["layers"].keys())

    # Define weight types
    weight_types = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']

    # Group layers by (layer_id, weight_type)
    layer_weight_map = {}  # (layer_id, weight_type) -> [layer_names]

    for layer_name in layer_names:
        layer_id = extract_layer_id(layer_name)
        for wt in weight_types:
            if wt in layer_name:
                key = (layer_id, wt)
                if key not in layer_weight_map:
                    layer_weight_map[key] = []
                layer_weight_map[key].append(layer_name)
                break

    # Create plots: one folder per weight type, one plot per layer
    for (layer_id, weight_type), layer_names_list in sorted(layer_weight_map.items()):
        if not layer_names_list:
            continue

        # Create weight type folder
        wt_dir = output_dir / weight_type
        wt_dir.mkdir(parents=True, exist_ok=True)

        _, ax = plt.subplots(figsize=(10, 6))

        multi_json_mode = len(records) > 1

        # Pre-collect all data to compute shared bin edges (avoids per-record
        # independent binning, which causes bars at different positions/widths
        # and makes one histogram visually swallow the other).
        per_record_svs = []
        per_record_svs_pc = []
        for record in records:
            svs, svs_pc = [], []
            for layer_name in layer_names_list:
                if layer_name in record["layers"]:
                    layer_data = record["layers"][layer_name]
                    svs.extend(layer_data["singular_values"])
                    if "singular_values_pc" in layer_data:
                        svs_pc.extend(layer_data["singular_values_pc"])
            per_record_svs.append(svs)
            per_record_svs_pc.append(svs_pc)

        # Per-record σ/σ₁ normalization (computed within this (layer_id, weight_type)
        # group), so every distribution lives in [0, 1] and shapes compare directly.
        normalized_svs, normalized_svs_pc = [], []
        for svs, svs_pc in zip(per_record_svs, per_record_svs_pc):
            if svs:
                s1 = max(svs)
                svs = [v / s1 for v in svs] if s1 > 0 else svs
            if svs_pc:
                s1_pc = max(svs_pc)
                svs_pc = [v / s1_pc for v in svs_pc] if s1_pc > 0 else svs_pc
            normalized_svs.append(svs)
            normalized_svs_pc.append(svs_pc)
        per_record_svs, per_record_svs_pc = normalized_svs, normalized_svs_pc

        # Build shared bin edges from the union of whichever series will be plotted
        if multi_json_mode:
            plot_data_for_bins = [svs_pc if svs_pc else svs
                                  for svs, svs_pc in zip(per_record_svs, per_record_svs_pc)]
        else:
            # Single record: may plot both svs and svs_pc
            plot_data_for_bins = per_record_svs + per_record_svs_pc
        combined = [v for series in plot_data_for_bins for v in series]
        shared_bins = np.linspace(-0.02, 1.02, 151) if combined else 50

        # Determine how many series will be plotted (for alpha selection)
        n_series = 0
        for all_svs, all_svs_pc in zip(per_record_svs, per_record_svs_pc):
            if all_svs:
                if multi_json_mode:
                    n_series += 1
                elif all_svs_pc:
                    n_series += 2  # original + PC
                else:
                    n_series += 1
        hist_alpha = 0.85 if n_series <= 1 else 0.6

        color_idx = 0
        for record_idx, (all_svs, all_svs_pc) in enumerate(zip(per_record_svs, per_record_svs_pc)):
            base_label = (labels[record_idx] if labels and record_idx < len(labels)
                          else get_record_label(records[record_idx], record_idx))

            if all_svs:
                color = COLORS[color_idx % len(COLORS)]

                if multi_json_mode and all_svs_pc:
                    _draw_series(ax, all_svs_pc, color=color, label=base_label,
                                 bins=shared_bins, hist_alpha=hist_alpha, mode=mode)
                elif all_svs_pc:
                    _draw_series(ax, all_svs, color=color, label=base_label,
                                 bins=shared_bins, hist_alpha=hist_alpha, mode=mode)
                    color_idx += 1
                    pc_color = COLORS[color_idx % len(COLORS)]
                    _draw_series(ax, all_svs_pc, color=pc_color, label=f"{base_label} (PC)",
                                 bins=shared_bins, hist_alpha=hist_alpha, mode=mode)
                else:
                    _draw_series(ax, all_svs, color=color, label=base_label,
                                 bins=shared_bins, hist_alpha=hist_alpha, mode=mode)

                color_idx += 1

        label_fs, tick_fs, legend_fs = 27, 22, 24
        if mode == "quantile":
            # Quantile plot: x = fraction of singular values (smallest→largest),
            # y = the (normalized) singular value at that fraction. x ticks as %.
            ax.set_xlabel("Smallest singular values (fraction)", fontsize=label_fs)
            ax.set_ylabel(r"$\sigma / \sigma_{\max}$", fontsize=label_fs)
        else:
            ax.set_xlabel(r"$\sigma / \sigma_{\max}$", fontsize=label_fs)
            if mode == "ecdf":
                ax.set_ylabel("ECDF", fontsize=label_fs)
            else:
                ax.set_ylabel("Frequency (log)" if log_y else "Frequency", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

        ax.legend(fontsize=legend_fs)

        if mode == "quantile":
            ax.set_xlim(0, 1.0)
            # Label x ticks as percentages: 0% 20% ... 100%
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.set_xlim(-0.02, 1.02)

        if log_y and mode == "hist":
            ax.set_yscale("log")

        if mode == "ecdf":
            # ECDF: keep a normal y-axis in [0, 1] with a visible bottom spine.
            ax.set_ylim(0, 1.02)
        elif mode == "quantile":
            pass  # quantile keeps both spines and normal ticks
        else:
            # Histogram: no x ticks, dashed baseline at y=0 (original styling).
            ax.tick_params(axis="x", which="both", length=0)
            ax.spines["bottom"].set_visible(False)
            baseline_y = ax.get_ylim()[0] if log_y else 0
            ax.axhline(y=baseline_y, color="black", linewidth=1.0, linestyle=(0, (8, 6)))

        # y-axis minor ticks only
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", length=3, width=0.8, direction="in")

        plt.tight_layout()

        # Save to weight type folder
        output_path = wt_dir / f"layer{layer_id}.{fmt}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


def main():
    """CLI entry point for standalone plotting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot singular-value distributions (σ/σ_max-normalized) from JSON files."
    )
    parser.add_argument("json_files", nargs="+", help="Path(s) to singular_values JSON file(s)")
    parser.add_argument("--mode", choices=["hist", "ecdf", "quantile"], default="hist",
                        help="Plot style: 'hist' (histogram, default), 'ecdf' (stepped "
                             "cumulative curve), or 'quantile' (empirical quantile function — "
                             "x = singular-value percentile smallest→largest, y = value at that "
                             "percentile).")
    parser.add_argument("--logy", action="store_true",
                        help="Use log scale for y-axis (histogram mode only)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Custom legend labels, one per JSON file")
    parser.add_argument("--fmt", choices=["pdf", "png"], default="pdf",
                        help="Output image format (default: pdf)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. If set, overrides the auto-derived path. "
                             "A step_{N} subdir is still appended when step can be parsed.")
    args = parser.parse_args()

    # Load records and store source path
    records = []
    for filepath in args.json_files:
        with open(filepath, 'r') as f:
            record = json.load(f)
            record["_source_path"] = filepath
            records.append(record)

    # Extract step from each JSON filename and validate they match
    steps = []
    for filepath in args.json_files:
        p = Path(filepath)
        match = re.search(r"step_(\d+)", p.stem)
        steps.append(int(match.group(1)) if match else None)

    if len(records) > 1:
        valid_steps = [s for s in steps if s is not None]
        if valid_steps and len(set(valid_steps)) > 1:
            raise ValueError(f"Multiple JSONs must be from the same step, got steps: {valid_steps}")

    # Determine output directory with step subdirectory
    output_dir = None
    step = next((s for s in steps if s is not None), None)
    if args.output_dir is not None:
        base = Path(args.output_dir)
        output_dir = str(base / f"step_{step}") if step is not None else str(base)
    elif len(records) == 1:
        if step is not None:
            output_dir = str(Path(args.json_files[0]).parent / f"step_{step}")
    else:
        json_dirs = [Path(fp).parent for fp in args.json_files]
        common_parent = Path(os.path.commonpath([str(d) for d in json_dirs]))
        dir_name = "__".join(sanitize_path_component(d.name) for d in json_dirs)
        base = common_parent / dir_name
        if step is not None:
            output_dir = str(base / f"step_{step}")
        else:
            output_dir = str(base)

    plot_singular_values(records, output_dir=output_dir,
                         log_y=args.logy, labels=args.labels, fmt=args.fmt,
                         mode=args.mode)


if __name__ == "__main__":
    main()
