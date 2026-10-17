#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Merge MLP benchmark CSVs without counting repeated seeds as new samples."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile.profile_results import deduplicate, output_fields, read_results, relabel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="New CSV; inputs are never overwritten")
    parser.add_argument("--label", help="Unified plot label; defaults to the output filename")
    parser.add_argument("--on-conflict", choices=("error", "first", "last"), default="error",
                        help="How to select repeated measurements of the same case/seed; never average them")
    args = parser.parse_args()
    try:
        rows, fields = read_results(args.csv_files)
        selected, counts = deduplicate(rows, args.on_conflict)
        if not selected:
            raise ValueError("No benchmark rows to merge")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=output_fields(selected.values(), fields))
            writer.writeheader()
            writer.writerows(relabel(row, args.label or args.output.stem) for row in selected.values())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {len(rows)} input rows -> {len(selected)} distinct cases; "
          f"{counts['duplicates']} duplicate copies, {counts['conflicts']} conflicts "
          f"(policy={args.on_conflict}), {counts['superseded_failures']} superseded failures")


if __name__ == "__main__":
    main()
