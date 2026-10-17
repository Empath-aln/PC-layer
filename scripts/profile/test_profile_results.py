# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check provenance, duplicate seed handling, and real incremental benchmarks."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.profile import benchmark_mlp as benchmark
from scripts.profile import profile_results as results


class ResultCSVTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def row(self, **changes):
        row = {field: "" for field in results.CASE_FIELDS}
        row.update(implementation="pc_v2", implementation_sha256="a" * 64,
                   reference_sha256="b" * 64, seed="0", tokens="4", dim="8",
                   hidden_dim="32", layers="2", grad_accum_steps="3", dtype="float32",
                   pc_level="4", pc_checkpoint="none", power_iter="2", recover_w_norm="True",
                   learnable_gamma="True", warmup="1", steps="2", device="cpu", device_name="CPU",
                   torch_version="2.14.0", correctness_rtol="0.0001", correctness_atol="0.00001",
                   status="ok", correctness_status="passed", run_id="first", label="experiment")
        row.update({field: "1.25" for field in results.METRICS})
        row["peak_allocated_mib"] = ""
        row.update(changes)
        return row

    def write(self, name, rows):
        path = self.root / name
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_case_identity_ignores_provenance_and_normalizes_numbers(self):
        first = self.row()
        second = self.row(label="new label", run_id="second", source_csv="copy.csv",
                          git_commit="different", recover_w_norm=True, layers=2,
                          correctness_rtol="1e-4", correctness_atol=0.00001)
        self.assertEqual(results.case_key(first), results.case_key(second))

    def test_changed_config_hash_runtime_or_tolerance_is_distinct(self):
        first = self.row()
        for field, value in (
            ("implementation_sha256", "changed"), ("reference_sha256", "changed"),
            ("pc_checkpoint", "full"), ("seed", "1"), ("dim", "16"),
            ("grad_accum_steps", "4"), ("warmup", "10"), ("steps", "50"),
            ("device_name", "other device"), ("torch_version", "other version"),
            ("cuda_version", "other version"), ("correctness_rtol", "0.02"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(results.case_key(first), results.case_key(self.row(**{field: value})))

    def test_duplicate_copy_does_not_increase_seed_count(self):
        selected, counts = results.deduplicate([self.row(), self.row(label="copy", source_csv="copy.csv")])
        self.assertEqual(len(selected), 1)
        self.assertEqual(counts["duplicates"], 1)

    def test_conflicting_measurements_require_explicit_selection(self):
        first, last = self.row(), self.row(run_id="second", total_ms="2.5")
        with self.assertRaisesRegex(ValueError, "Conflicting results"):
            results.deduplicate([first, last])
        for policy, expected in (("first", "1.25"), ("last", "2.5")):
            selected, counts = results.deduplicate([first, last], policy)
            self.assertEqual(next(iter(selected.values()))["total_ms"], expected)
            self.assertEqual(counts["conflicts"], 1)

    def test_success_supersedes_failure_in_either_input_order(self):
        good, bad = self.row(), self.row(status="correctness_failed", correctness_status="failed")
        for rows in ([good, bad], [bad, good]):
            selected, counts = results.deduplicate(rows)
            self.assertTrue(results.successful(next(iter(selected.values()))))
            self.assertEqual(counts["superseded_failures"], 1)
        self.assertFalse(results.successful(self.row(correctness_status="not_applicable")))

    def test_legacy_defaults_provenance_and_unknown_columns_survive(self):
        row = self.row(custom_note="keep this")
        del row["grad_accum_steps"], row["pc_checkpoint"], row["implementation_sha256"]
        path = self.write("legacy.csv", [row])
        rows, fields = results.read_results([path])
        self.assertEqual(rows[0]["grad_accum_steps"], "1")
        self.assertEqual(rows[0]["pc_checkpoint"], "none")
        self.assertEqual(rows[0]["implementation_sha256"], "")
        self.assertEqual(rows[0]["source_csv"], str(path.resolve()))
        self.assertEqual(rows[0]["custom_note"], "keep this")
        self.assertIn("custom_note", results.output_fields(rows, fields))
        relabeled = results.relabel(rows[0], "combined")
        self.assertEqual(relabeled["source_label"], "experiment")
        self.assertEqual(relabeled["label"], "combined")

    def test_invalid_success_metrics_are_rejected(self):
        path = self.write("invalid.csv", [self.row(total_ms="nan")])
        with self.assertRaisesRegex(ValueError, "Invalid total_ms"):
            results.read_results([path])

    def test_reference_timing_can_be_disabled_without_changing_selection(self):
        with patch("sys.argv", ["benchmark", "--implementation", "pc_v3", "--no-reference-benchmark"]):
            args = benchmark.parse_args()
        self.assertEqual(args.implementation, ["pc_v3"])
        with patch("sys.argv", ["benchmark", "--implementation", "pc_v3"]):
            args = benchmark.parse_args()
        self.assertEqual(args.implementation, ["pc_baseline", "pc_v3"])

    def run_benchmark(self, name, extra=()):
        path = self.root / name
        argv = ["benchmark", "--device", "cpu", "--dtype", "float32",
                "--implementation", "plain", "--tokens", "4", "--dim", "8",
                "--layers", "1", "--grad-accum-steps", "3", "--seeds", "0", "1",
                "--warmup", "1", "--steps", "1", "--output", str(path), *map(str, extra)]
        output = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            benchmark.main()
        return path, output.getvalue()

    def test_resume_performs_no_forward_backward_or_measurement_for_complete_cases(self):
        previous, _ = self.run_benchmark("previous.csv")
        with patch.object(benchmark, "measure_case", side_effect=AssertionError("must reuse")), \
             patch.object(benchmark, "check_case", side_effect=AssertionError("must reuse")):
            output, text = self.run_benchmark("resumed.csv", ["--resume-from", previous])
        rows, _ = results.read_results([output])
        self.assertEqual(len(rows), 2)
        self.assertEqual(text.count("REUSED plain"), 2)

    def test_only_missing_seed_is_measured(self):
        previous, _ = self.run_benchmark("previous.csv", ["--seeds", "0"])
        with patch.object(benchmark, "measure_case", wraps=benchmark.measure_case) as measure:
            output, text = self.run_benchmark("resumed.csv", ["--resume-from", previous])
        self.assertEqual(measure.call_count, 1)
        self.assertEqual(text.count("REUSED plain"), 1)
        rows, _ = results.read_results([output])
        self.assertEqual({row["seed"] for row in rows}, {"0", "1"})

    def test_old_failure_is_retried_and_replaced(self):
        previous, _ = self.run_benchmark("previous.csv")
        rows, _ = results.read_results([previous])
        rows[0].update(status="oom", error="old failure")
        failed = self.write("failed.csv", rows)
        with patch.object(benchmark, "measure_case", wraps=benchmark.measure_case) as measure:
            output, _ = self.run_benchmark("retried.csv", ["--resume-from", failed])
        self.assertEqual(measure.call_count, 1)
        retried, _ = results.read_results([output])
        self.assertEqual(len(retried), 2)
        self.assertTrue(all(results.successful(row) for row in retried))

    def test_new_checkpoint_mode_still_gets_correctness_checked(self):
        common = ["--implementation", "pc_v3", "--no-reference-benchmark", "--seeds", "0"]
        previous, _ = self.run_benchmark("none.csv", [*common, "--pc-checkpoint", "none"])
        with patch.object(benchmark, "measure_case", wraps=benchmark.measure_case) as measure, \
             patch.object(benchmark, "check_case", wraps=benchmark.check_case) as check:
            output, text = self.run_benchmark("full.csv", [*common, "--pc-checkpoint", "none", "full",
                                                          "--resume-from", previous])
        self.assertEqual(measure.call_count, 1)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(text.count("REUSED pc_v3"), 1)
        rows, _ = results.read_results([output])
        self.assertEqual({row["pc_checkpoint"] for row in rows}, {"none", "full"})
        self.assertEqual({row["implementation"] for row in rows}, {"pc_v3"})

    def test_reused_baseline_is_paired_with_new_candidate_seeds(self):
        previous, _ = self.run_benchmark("baseline.csv", ["--implementation", "plain", "pc_baseline"])
        with patch.object(benchmark, "measure_case", wraps=benchmark.measure_case) as measure:
            output, text = self.run_benchmark("candidate.csv", ["--implementation", "pc_v3",
                                                               "--resume-from", previous])
        self.assertEqual(measure.call_count, 2)
        self.assertEqual(text.count("REUSED pc_baseline"), 2)
        self.assertIn("2 paired seeds", text)
        rows, _ = results.read_results([output])
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
