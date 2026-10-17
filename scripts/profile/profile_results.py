# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSV case identity, provenance, and deduplication; no torch dependency."""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

PROVENANCE_FIELDS = ("source_csv", "source_label")
CASE_FIELDS = (
    "implementation", "implementation_sha256", "reference_sha256", "seed",
    "tokens", "dim", "hidden_dim", "layers", "grad_accum_steps", "dtype",
    "pc_level", "pc_checkpoint", "power_iter", "recover_w_norm", "learnable_gamma",
    "warmup", "steps", "device", "device_name", "device_total_mib", "device_sm_count",
    "torch_version", "cuda_version", "correctness_rtol", "correctness_atol",
)
NUMERIC_FIELDS = {
    "seed", "tokens", "dim", "hidden_dim", "layers", "grad_accum_steps", "pc_level",
    "power_iter", "warmup", "steps", "device_total_mib", "device_sm_count",
    "correctness_rtol", "correctness_atol",
}
OPTIONAL_FIELDS = {
    "implementation_sha256", "reference_sha256", "grad_accum_steps", "pc_checkpoint",
    "correctness_rtol", "correctness_atol",
}
METRICS = (
    "forward_ms", "loss_ms", "backward_ms", "sn_update_ms", "total_ms",
    "peak_allocated_mib",
)


def normalized_row(row):
    result = dict(row)
    result["grad_accum_steps"] = result.get("grad_accum_steps") or "1"
    result["pc_checkpoint"] = result.get("pc_checkpoint") or "none"
    for field in OPTIONAL_FIELDS - {"grad_accum_steps", "pc_checkpoint"}:
        result[field] = result.get(field) or ""
    if not result.get("effective_tokens"):
        result["effective_tokens"] = str(int(result["tokens"]) * int(result["grad_accum_steps"]))
    return result


def canonical(field, value):
    value = str(value).strip() if value is not None else ""
    if field in ("recover_w_norm", "learnable_gamma"):
        if value.lower() in ("true", "1"):
            return "true"
        if value.lower() in ("false", "0"):
            return "false"
        raise ValueError(f"Invalid boolean {field}={value!r}")
    if field in NUMERIC_FIELDS and value:
        try:
            number = Decimal(value)
        except InvalidOperation as error:
            raise ValueError(f"Invalid number {field}={value!r}") from error
        if not number.is_finite():
            raise ValueError(f"Non-finite number {field}={value!r}")
        return str(number.normalize())
    return value


def case_key(row):
    row = normalized_row(row)
    return tuple(canonical(field, row.get(field, "")) for field in CASE_FIELDS)


def successful(row):
    if row.get("status") != "ok" or row.get("correctness_status") == "failed":
        return False
    if row["implementation"] not in ("plain", "pc_baseline"):
        return row.get("correctness_status") == "passed"
    return True


def read_results(paths):
    rows = []
    fields = []
    for path in paths:
        path = Path(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"status", "label", *CASE_FIELDS, *METRICS} - OPTIONAL_FIELDS
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
            for field in reader.fieldnames:
                if field not in fields:
                    fields.append(field)
            for line, raw in enumerate(reader, 2):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"{path}:{line}: incomplete or malformed CSV row")
                try:
                    row = normalized_row(raw)
                    case_key(row)
                    if successful(row):
                        for metric in METRICS:
                            value = row.get(metric, "")
                            if metric == "peak_allocated_mib" and not value and row["device"] == "cpu":
                                continue
                            number = Decimal(str(value))
                            if not number.is_finite() or number < 0:
                                raise ValueError(f"Invalid {metric}={value!r}")
                except (ValueError, InvalidOperation) as error:
                    raise ValueError(f"{path}:{line}: {error}") from error
                row["source_csv"] = row.get("source_csv") or str(path.resolve())
                row["source_label"] = row.get("source_label") or row.get("label", "")
                rows.append(row)
    return rows, fields


def deduplicate(rows, on_conflict="error"):
    if on_conflict not in ("error", "first", "last"):
        raise ValueError(f"Unknown conflict policy: {on_conflict}")
    selected = {}
    counts = {"duplicates": 0, "conflicts": 0, "superseded_failures": 0}
    for raw in rows:
        row = normalized_row(raw)
        key = case_key(row)
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        if successful(previous) != successful(row):
            counts["superseded_failures"] += 1
            if successful(row):
                selected[key] = row
            continue
        # Relabeling/copying a result must not create a second sample. Preserve
        # the first provenance when all original measurement fields agree.
        ignored = {"label", *PROVENANCE_FIELDS}
        comparable = set(previous) | set(row)
        if all(str(previous.get(k, "")) == str(row.get(k, "")) for k in comparable - ignored):
            counts["duplicates"] += 1
            continue
        counts["conflicts"] += 1
        if on_conflict == "error":
            raise ValueError(
                f"Conflicting results for {row['implementation']} checkpoint={row['pc_checkpoint']} "
                f"seed={row['seed']} N={row['tokens']} d={row['dim']} layers={row['layers']} "
                f"accum={row['grad_accum_steps']}: {previous.get('source_csv', '?')} and "
                f"{row.get('source_csv', '?')}; choose --on-conflict first or last"
            )
        if on_conflict == "last":
            selected[key] = row
    return selected, counts


def output_fields(rows, initial=()):
    fields = list(dict.fromkeys([*initial, *PROVENANCE_FIELDS]))
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    return fields


def relabel(row, label):
    row = dict(row)
    row["source_label"] = row.get("source_label") or row.get("label", "")
    row["label"] = label
    return row
