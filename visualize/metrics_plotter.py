# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot GMCN-vs-step curves from saved singular value JSON records."""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 21,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
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

from .plotter import extract_layer_id, get_record_label, sanitize_path_component

COLORS = [
    "#1f77b4",  # blue (matplotlib default)
    "#ff7f0e",  # orange (matplotlib default)
    "#2ca02c",  # green (matplotlib default)
    "#d62728",  # red (matplotlib default)
    "#7B2D8E",  # purple
    "#E76F51",  # coral
    "#264653",  # dark teal
    "#A8DADC",  # light blue
    "#C2185B",  # deep pink
    "#4CAF50",  # green
    "#FF7043",  # deep orange
    "#5C6BC0",  # indigo
    "#8D6E63",  # brown
    "#00ACC1",  # cyan
    "#AFB42B",  # lime
    "#E91E63",  # pink
]


MARKERS = ["s", "D", "^", "o", "v", "P", "*", "X", "p", "h"]

_WEIGHT_TYPES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
_WEIGHT_TYPE_ORDER = {weight_type: idx for idx, weight_type in enumerate(_WEIGHT_TYPES)}


def extract_weight_type(layer_name: str) -> Optional[str]:
    """Extract the weight/module type from an exact layer name."""
    last_part = layer_name.split(".")[-1]
    if last_part in _WEIGHT_TYPE_ORDER:
        return last_part

    for weight_type in _WEIGHT_TYPES:
        if f".{weight_type}" in layer_name or layer_name.endswith(weight_type):
            return weight_type
    return None


def select_singular_values(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> List[float]:
    """Select the singular values to use for a layer according to PC rules."""
    if record_metadata.get("pc_enabled") and "singular_values_pc" in layer_data:
        return layer_data["singular_values_pc"]
    return layer_data.get("singular_values", [])


def layer_uses_pc(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> bool:
    """Return True if PC was actually applied to this layer in this record.

    Decided per-layer by the presence of a post-PC spectrum (`singular_values_pc`)
    rather than by weight-type name. Used to infer, across all experiments, the set
    of layer NAMES that PC touches (see _collect_pc_layer_names) -- no need to
    hardcode which blocks (qkv / o / ffn) are preconditioned.
    """
    return bool(record_metadata.get("pc_enabled") and "singular_values_pc" in layer_data)


# Default groups for the "by_group" plot level. A group is defined by a predicate
# over (layer_name, pc_layer_names) -- where pc_layer_names is the set of layers PC
# touches, inferred across ALL experiments. Selecting by layer name (not by each
# record's own pc_enabled) is what lets a group plot the SAME layers for every
# experiment: e.g. the `pc` group plots the ffn+o layers for both the PC run and the
# baseline, so the two are directly comparable. Members are aggregated with the
# metric's global aggregator (geometric mean for GMCN). `pc` and `no_pc` are
# complementary subsets; the full-model curve is the separate `global` plot level
# (a "full" group here would be identical to it). Format: key -> (predicate, label).
_DEFAULT_GROUPS: Dict[str, Tuple[Callable[[str, "set"], bool], str]] = {
    "no_pc": (lambda layer_name, pc_layer_names: layer_name not in pc_layer_names, "non-PC blocks"),
    "pc": (lambda layer_name, pc_layer_names: layer_name in pc_layer_names, "PC blocks"),
}


def compute_modified_condition_number(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> float:
    """Compute the GMCN per-layer metric: largest singular value (sigma_1)
    divided by the mean of the bottom 10% of singular values."""
    singular_values = select_singular_values(layer_data, record_metadata)
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    values = np.sort(values)[::-1]
    k = max(1, math.ceil(values.size * 0.1))
    numerator = float(values[0])
    denominator = float(np.mean(values[-k:]))

    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")

    result = numerator / denominator
    return result if np.isfinite(result) else float("nan")


def aggregate_mean(values: Sequence[float]) -> float:
    """Aggregate with arithmetic mean after filtering invalid values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def aggregate_geometric_mean(values: Sequence[float]) -> float:
    """Aggregate with geometric mean over positive finite values only."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(array))))


AGGREGATOR_REGISTRY: Dict[str, Callable[[Sequence[float]], float]] = {
    "mean": aggregate_mean,
    "geometric_mean": aggregate_geometric_mean,
}

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "modified_condition_number": {
        "function": compute_modified_condition_number,
        "display_name": "Modified Condition Number",
        "y_label": "Modified Condition Number",
        "default_global_aggregator": "geometric_mean",
        "global_display_name": "GMCN",
        "global_y_label": "GMCN",
    },
}


def _extract_step_from_filename(filepath: Path) -> Optional[int]:
    match = re.search(r"singular_values_step_(\d+)\.json$", filepath.name)
    if match:
        return int(match.group(1))
    return None


def _get_record_step(record: Dict[str, Any], filepath: Path) -> int:
    metadata = record.get("metadata", {})
    metadata_step = metadata.get("checkpoint_step")
    if metadata_step is not None:
        return int(metadata_step)

    filename_step = _extract_step_from_filename(filepath)
    if filename_step is not None:
        return filename_step

    raise ValueError(f"Could not determine checkpoint step for {filepath}")


def _load_experiment_records(experiment_dir: str) -> Dict[str, Any]:
    exp_path = Path(experiment_dir).resolve()
    if not exp_path.is_dir():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

    json_paths = sorted(exp_path.glob("singular_values_step_*.json"))
    if not json_paths:
        raise ValueError(f"No singular_values_step_*.json found in {experiment_dir}")

    records: List[Dict[str, Any]] = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            record = json.load(f)
        record["_source_path"] = str(json_path)
        record["_step"] = _get_record_step(record, json_path)
        records.append(record)

    records.sort(key=lambda record: (record["_step"], record["_source_path"]))
    label = get_record_label(records[0], 0) if records else exp_path.name

    return {
        "path": exp_path,
        "label": label,
        "records": records,
    }


def _resolve_output_root(experiment_paths: Sequence[Path], output_dir: Optional[str]) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()

    if len(experiment_paths) == 1:
        return experiment_paths[0] / "metrics_vs_step"

    parent_paths = [str(path.parent) for path in experiment_paths]
    common_parent = Path(os.path.commonpath(parent_paths))
    combined_name = "__".join(sanitize_path_component(path.name) for path in experiment_paths)
    return common_parent / combined_name / "metrics_vs_step"


def _layer_sort_key(layer_name: str) -> Tuple[int, int, str]:
    weight_type = extract_weight_type(layer_name)
    return (
        extract_layer_id(layer_name),
        _WEIGHT_TYPE_ORDER.get(weight_type or "", len(_WEIGHT_TYPE_ORDER)),
        layer_name,
    )


def _make_unique_labels(experiments: Sequence[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    seen_counts: Dict[str, int] = defaultdict(int)

    for experiment in experiments:
        base_label = experiment["label"]
        fallback = sanitize_path_component(experiment["path"].name)
        count = seen_counts[base_label]
        seen_counts[base_label] += 1
        if count == 0:
            labels.append(base_label)
        else:
            labels.append(f"{base_label} ({fallback})")

    return labels


def _collect_pc_layer_names(experiments: Sequence[Dict[str, Any]]) -> set:
    """Infer the set of layer NAMES that PC touches, across ALL experiments.

    A layer name is included if PC was applied to it in any record of any
    experiment (i.e. that record carries a post-PC spectrum for it). This set is
    then used to select the SAME layers in every experiment for the pc / no_pc
    groups -- so e.g. the `pc` group plots the ffn+o layers for both the PC run
    and the baseline, making them directly comparable. Returns an empty set when
    no experiment uses PC (then `pc` is empty and `no_pc` == `full`).
    """
    pc_layer_names: set = set()
    for experiment in experiments:
        for record in experiment["records"]:
            metadata = record.get("metadata", {})
            for layer_name, layer_data in record.get("layers", {}).items():
                if layer_uses_pc(layer_data, metadata):
                    pc_layer_names.add(layer_name)
    return pc_layer_names


def _extract_metric_series(experiment: Dict[str, Any], metric_name: str,
                           groups: Optional[Dict[str, Tuple[Callable[..., bool], str]]] = None,
                           pc_layer_names: Optional[set] = None) -> Dict[str, Any]:
    metric_spec = METRIC_REGISTRY[metric_name]
    metric_function = metric_spec["function"]
    block_aggregator = AGGREGATOR_REGISTRY["mean"]
    global_aggregator = AGGREGATOR_REGISTRY[metric_spec["default_global_aggregator"]]
    if pc_layer_names is None:
        pc_layer_names = set()

    per_layer: Dict[str, Dict[int, float]] = defaultdict(dict)
    block_values_by_step: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    global_values_by_step: Dict[int, List[float]] = defaultdict(list)
    # by_group: each group selects layers by NAME via its predicate (see _DEFAULT_GROUPS),
    # using the PC-layer-name set inferred across all experiments.
    group_values_by_step: Dict[str, Dict[int, List[float]]] = {
        group_key: defaultdict(list) for group_key in (groups or {})
    }

    for record in experiment["records"]:
        step = record["_step"]
        metadata = record.get("metadata", {})
        for layer_name, layer_data in record.get("layers", {}).items():
            metric_value = metric_function(layer_data, metadata)
            per_layer[layer_name][step] = metric_value
            global_values_by_step[step].append(metric_value)

            weight_type = extract_weight_type(layer_name)
            if weight_type is not None:
                block_values_by_step[weight_type][step].append(metric_value)

            if groups:
                for group_key, (predicate, _group_label) in groups.items():
                    if predicate(layer_name, pc_layer_names):
                        group_values_by_step[group_key][step].append(metric_value)

    per_block = {
        weight_type: {
            step: block_aggregator(values)
            for step, values in sorted(step_map.items())
        }
        for weight_type, step_map in block_values_by_step.items()
    }
    global_series = {
        step: global_aggregator(values)
        for step, values in sorted(global_values_by_step.items())
    }

    # Aggregate each group with the metric's global aggregator (so a group curve
    # is comparable to the `global` curve). Drop groups with no matching layers.
    by_group: Dict[str, Dict[int, float]] = {
        group_key: {
            step: global_aggregator(values)
            for step, values in sorted(step_map.items())
        }
        for group_key, step_map in group_values_by_step.items()
        if step_map
    }

    return {
        "label": experiment["label"],
        "per_layer": dict(per_layer),
        "per_block": per_block,
        "by_group": by_group,
        "global": global_series,
    }


def _plot_series(series_by_label: Dict[str, Dict[int, float]],
                 title: str,
                 y_label: str,
                 output_path: Path,
                 x_label: str = "Step",
                 total_tokens: Optional[float] = None,
                 max_step: Optional[int] = None) -> bool:
    if not series_by_label:
        return False

    def _convert_x(steps: List[int]) -> List[float]:
        if total_tokens is not None and max_step is not None and max_step > 0:
            return [s / max_step * total_tokens for s in steps]
        return [float(s) for s in steps]

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for idx, (label, step_map) in enumerate(series_by_label.items()):
        if not step_map:
            continue

        steps = sorted(step_map)
        x_values = _convert_x(steps)
        values = [step_map[step] for step in steps]
        if not np.isfinite(np.asarray(values, dtype=float)).any():
            continue

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(x_values, values, marker=marker, linewidth=2, markersize=7, color=color, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    # y-axis minor ticks only
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8, direction="in")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def plot_metrics_vs_step(experiment_dirs: Sequence[str],
                         metric_names: Optional[Sequence[str]] = None,
                         output_dir: Optional[str] = None,
                         labels: Optional[Sequence[str]] = None,
                         x_unit: str = "step",
                         total_tokens: Optional[float] = None,
                         plot_levels: Optional[Sequence[str]] = None,
                         groups: Optional[Dict[str, Tuple[Callable[..., bool], str]]] = None,
                         fmt: str = "pdf") -> List[str]:
    """Load experiment directories and plot metric-vs-step curves.

    Args:
        experiment_dirs: Paths to experiment directories.
        metric_names: Which metrics to plot (default: all).
        output_dir: Override output directory.
        labels: Custom legend labels, one per experiment directory.
        x_unit: "step" (default) or "token".  When "token", *total_tokens*
            must be provided and the x-axis shows Total Tokens Trained (B).
        total_tokens: Total tokens (in billions) at the max step.  Required
            when *x_unit* is "token".
        plot_levels: Subset of {"per_layer", "per_block", "by_group", "global"}
            to plot. Default: all levels.
        groups: Groups for the "by_group" level, mapping
            group_key -> (predicate, label), where predicate(layer_name,
            pc_layer_names) selects member layers by name. Defaults to
            _DEFAULT_GROUPS (no_pc / pc / full). The pc_layer_names set is
            inferred across all experiments, so the pc / no_pc groups select
            the same layers for every experiment (baseline included) and are
            directly comparable.
    """
    if not experiment_dirs:
        raise ValueError("At least one experiment directory is required")

    if x_unit not in ("step", "token"):
        raise ValueError(f"x_unit must be 'step' or 'token', got '{x_unit}'")
    if x_unit == "token" and total_tokens is None:
        raise ValueError("total_tokens is required when x_unit='token'")

    if groups is None:
        groups = _DEFAULT_GROUPS

    _VALID_LEVELS = {"per_layer", "per_block", "by_group", "global"}
    if plot_levels is not None:
        user_levels = set(plot_levels)
        unknown_levels = user_levels - _VALID_LEVELS
        if unknown_levels:
            raise ValueError(
                f"Unknown plot levels: {unknown_levels}. Available: {sorted(_VALID_LEVELS)}"
            )
    else:
        user_levels = set(_VALID_LEVELS)

    if metric_names is None:
        metric_names = list(METRIC_REGISTRY)

    unknown_metrics = [metric_name for metric_name in metric_names if metric_name not in METRIC_REGISTRY]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metrics: {unknown_metrics}. Available metrics: {sorted(METRIC_REGISTRY)}"
        )

    experiments = [_load_experiment_records(experiment_dir) for experiment_dir in experiment_dirs]
    # Infer the PC-touched layer names once, across all experiments, so the
    # pc / no_pc groups select the same layers for every experiment (baseline included).
    pc_layer_names = _collect_pc_layer_names(experiments)
    output_root = _resolve_output_root([experiment["path"] for experiment in experiments], output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Custom labels override auto-generated ones
    if labels is not None:
        if len(labels) != len(experiments):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of experiments ({len(experiments)})"
            )
        unique_labels = list(labels)
    else:
        unique_labels = _make_unique_labels(experiments)

    # Compute max step across all experiments for token conversion
    max_step: Optional[int] = None
    if x_unit == "token":
        max_step = max(
            record["_step"]
            for experiment in experiments
            for record in experiment["records"]
        )

    x_label = "Step" if x_unit == "step" else "Total Tokens Trained (B)"
    x_unit_label = "Step" if x_unit == "step" else "Tokens"

    saved_paths: List[str] = []

    plot_kwargs: Dict[str, Any] = {
        "x_label": x_label,
        "total_tokens": total_tokens if x_unit == "token" else None,
        "max_step": max_step,
    }

    for metric_name in metric_names:
        metric_spec = METRIC_REGISTRY[metric_name]
        metric_dir = output_root / metric_name
        per_layer_dir = metric_dir / "per_layer"
        per_block_dir = metric_dir / "per_block"

        experiment_series = {
            label: _extract_metric_series(experiment, metric_name, groups, pc_layer_names)
            for label, experiment in zip(unique_labels, experiments)
        }

        if "per_layer" in user_levels:
            all_layers = sorted(
                {
                    layer_name
                    for series in experiment_series.values()
                    for layer_name in series["per_layer"].keys()
                },
                key=_layer_sort_key,
            )
            for layer_name in all_layers:
                output_path = per_layer_dir / f"{sanitize_path_component(layer_name)}.{fmt}"
                series_by_label = {
                    label: series["per_layer"][layer_name]
                    for label, series in experiment_series.items()
                    if layer_name in series["per_layer"]
                }
                if _plot_series(
                    series_by_label,
                    title=f"{metric_spec['display_name']} vs {x_unit_label} - {layer_name}",
                    y_label=metric_spec["y_label"],
                    output_path=output_path,
                    **plot_kwargs,
                ):
                    saved_paths.append(str(output_path))

        if "per_block" in user_levels:
            all_blocks = sorted(
                {
                    block_name
                    for series in experiment_series.values()
                    for block_name in series["per_block"].keys()
                },
                key=lambda block_name: (_WEIGHT_TYPE_ORDER.get(block_name, len(_WEIGHT_TYPE_ORDER)), block_name),
            )
            for block_name in all_blocks:
                output_path = per_block_dir / f"{sanitize_path_component(block_name)}.{fmt}"
                series_by_label = {
                    label: series["per_block"][block_name]
                    for label, series in experiment_series.items()
                    if block_name in series["per_block"]
                }
                if _plot_series(
                    series_by_label,
                    title=f"{metric_spec['display_name']} vs {x_unit_label} - {block_name.upper()}",
                    y_label=metric_spec["y_label"],
                    output_path=output_path,
                    **plot_kwargs,
                ):
                    saved_paths.append(str(output_path))

        if "by_group" in user_levels:
            # One plot per group (pc / no_pc / full), written directly under the
            # metric dir alongside global. Each group is aggregated with the global
            # aggregator (so a group curve is comparable to the `global` curve).
            # Title/y-label use the global display name (e.g. GMCN).
            group_display = metric_spec.get("global_display_name", metric_spec["display_name"])
            group_y = metric_spec.get("global_y_label", metric_spec["y_label"])
            for group_key, (_predicate, group_label) in groups.items():
                output_path = metric_dir / f"{sanitize_path_component(group_key)}.{fmt}"
                series_by_label = {
                    label: series["by_group"][group_key]
                    for label, series in experiment_series.items()
                    if group_key in series["by_group"]
                }
                if _plot_series(
                    series_by_label,
                    title=f"{group_display} ({group_label}) vs {x_unit_label}",
                    y_label=group_y,
                    output_path=output_path,
                    **plot_kwargs,
                ):
                    saved_paths.append(str(output_path))

        if "global" in user_levels:
            global_output_path = metric_dir / f"global.{fmt}"
            global_display = metric_spec.get("global_display_name", metric_spec["display_name"])
            global_y = metric_spec.get("global_y_label", metric_spec["y_label"])
            global_series_by_label = {
                label: series["global"]
                for label, series in experiment_series.items()
                if series["global"]
            }
            if _plot_series(
                global_series_by_label,
                title=f"{global_display} vs {x_unit_label}",
                y_label=global_y,
                output_path=global_output_path,
                **plot_kwargs,
            ):
                saved_paths.append(str(global_output_path))

    return saved_paths


def main() -> None:
    """CLI entry point for plotting metric-vs-step curves."""
    parser = argparse.ArgumentParser(
        description="Plot metric-vs-step curves from experiment directories containing singular_values_step_*.json"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="One or more experiment directories under visualization_output",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Metrics to plot. Available: {', '.join(METRIC_REGISTRY)}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <exp_dir>/metrics_vs_step or a combined sibling directory.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom legend labels, one per experiment directory.",
    )
    parser.add_argument(
        "--x-unit",
        choices=["step", "token"],
        default="step",
        help="X-axis unit: 'step' (default) or 'token'.",
    )
    parser.add_argument(
        "--total-tokens",
        type=float,
        default=None,
        help="Total tokens trained (in billions) at the max step. Required when --x-unit=token.",
    )
    parser.add_argument(
        "--plot-levels",
        nargs="+",
        choices=["per_layer", "per_block", "by_group", "global"],
        default=None,
        help="Which granularity levels to plot (default: all). Choose from: "
             "per_layer, per_block, by_group, global. 'by_group' splits layers by "
             "whether PC was applied (no_pc / pc / full) and aggregates each group "
             "with the geometric mean.",
    )
    parser.add_argument(
        "--fmt",
        choices=["pdf", "png"],
        default="pdf",
        help="Output image format (default: pdf).",
    )
    args = parser.parse_args()

    plot_metrics_vs_step(
        experiment_dirs=args.experiment_dirs,
        metric_names=args.metrics,
        output_dir=args.output_dir,
        labels=args.labels,
        x_unit=args.x_unit,
        total_tokens=args.total_tokens,
        plot_levels=args.plot_levels,
        fmt=args.fmt,
    )


if __name__ == "__main__":
    main()
