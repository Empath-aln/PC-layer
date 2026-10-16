# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regression checks for gradient accumulation and measurement boundaries."""

import copy
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.profile import benchmark_mlp as benchmark


class AccumulationTest(unittest.TestCase):
    def make_model(self, implementation):
        torch.manual_seed(7)
        model = benchmark.build_mlp(8, 16, 1, implementation, benchmark.PCConfig())
        for module in model.modules():
            if isinstance(module, benchmark.PCLinear):
                module.sn_u.zero_()
                module.sn_v.zero_()
                module._maybe_init_gamma()
        if implementation == "pc_baseline":
            benchmark.update_model_sn_state(model)
        return model

    def test_accumulated_gradients_match_full_batch(self):
        for implementation in ("plain", "pc_baseline"):
            for accum in (1, 3):
                with self.subTest(implementation=implementation, accum=accum):
                    model = self.make_model(implementation)
                    reference = copy.deepcopy(model)
                    x = torch.randn(4, 8, requires_grad=True)
                    target = torch.randn(4, 8)
                    full_x = x.detach().repeat(accum, 1).requires_grad_(True)
                    expected = torch.nn.functional.mse_loss(
                        reference(full_x), target.repeat(accum, 1),
                    )
                    expected.backward()
                    forwards = []
                    hook = model.register_forward_pre_hook(
                        lambda module, inputs: forwards.append(None)
                    )
                    with patch.object(
                        benchmark, "update_model_sn_state",
                        wraps=benchmark.update_model_sn_state,
                    ) as update:
                        actual = benchmark.run_step(
                            model, x, target, implementation == "pc_baseline",
                            grad_accum_steps=accum,
                        )
                    hook.remove()
                    self.assertEqual(len(forwards), accum)
                    self.assertEqual(update.call_count, int(implementation == "pc_baseline"))
                    torch.testing.assert_close(actual, expected.detach())
                    for parameter, full_parameter in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(parameter.grad, full_parameter.grad, rtol=1e-4, atol=1e-6)
                    # The input leaf is reset per microbatch; only its final
                    # scaled gradient should remain, not the sum for all inputs.
                    torch.testing.assert_close(x.grad, full_x.grad[-len(x):], rtol=1e-4, atol=1e-6)

    def test_parameter_gradients_reset_between_accumulated_steps(self):
        model = self.make_model("plain")
        x = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8)
        benchmark.run_step(model, x, target, False, grad_accum_steps=4)
        expected = [parameter.grad.clone() for parameter in model.parameters()]
        benchmark.run_step(model, x, target, False, grad_accum_steps=4)
        for parameter, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, gradient)

    def test_every_microbatch_has_separate_timing_boundaries(self):
        for accum in (1, 4):
            with self.subTest(accum=accum):
                model = self.make_model("plain")
                x = torch.randn(4, 8, requires_grad=True)
                target = torch.randn(4, 8)
                markers = [None] * (4 * accum + 2)
                with patch.object(benchmark.time, "perf_counter", side_effect=range(len(markers))):
                    benchmark.run_step(model, x, target, False, markers, accum)
                self.assertEqual(markers, list(range(len(markers))))


class VersionCorrectnessTest(unittest.TestCase):
    def test_baseline_snapshot_is_frozen(self):
        path = Path(__file__).with_name("pc_baseline.py")
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "91204116c4204c73400c28053ca2d98ae2d800faba096da7f847156e5b94b87b",
        )
        self.assertEqual(benchmark.PCLinear.__module__, "scripts.profile.pc_baseline")

    def check_variant(self, linear=None, updater=None, dtype="float32", level=4, accum=3):
        from scripts.profile import implementations

        baseline = implementations.load_implementation("pc_baseline")
        candidate = SimpleNamespace(
            PCLinear=linear or baseline.PCLinear,
            update_model_sn_state=updater or baseline.update_model_sn_state,
        )
        original_loader = implementations.load_implementation

        def resolve(name):
            return candidate if name == "pc_test" else original_loader(name)

        args = SimpleNamespace(
            pc_level=level, power_iter=1, recover_w_norm=True, learnable_gamma=True,
            dtype=dtype, check_rtol=None, check_atol=None,
        )
        with patch.object(benchmark, "load_implementation", side_effect=resolve), patch.object(
            implementations, "load_implementation", side_effect=resolve,
        ):
            return benchmark.check_case(args, "pc_test", 7, 4, 8, 16, 2, torch.device("cpu"), accum)

    def test_identical_versions_pass(self):
        for dtype in ("float32", "bfloat16"):
            for level in (0, 4):
                for accum in (1, 3):
                    with self.subTest(dtype=dtype, level=level, accum=accum):
                        result, state = self.check_variant(dtype=dtype, level=level, accum=accum)
                        self.assertEqual(result["correctness_status"], "passed", result)
                        self.assertEqual(result["correctness_max_abs_error"], 0)
                        self.assertGreater(result["correctness_checks"], 10)
                        self.assertTrue(all(value.device.type == "cpu" for value in state.values()))

    def test_wrong_output_is_rejected(self):
        class WrongOutput(benchmark.PCLinear):
            def forward(self, x):
                return super().forward(x) + 1

        result, _ = self.check_variant(linear=WrongOutput)
        self.assertEqual(result["correctness_status"], "failed")
        self.assertIn("output", result["correctness_error"])

    def test_equal_outputs_with_wrong_backward_are_rejected(self):
        class WrongBackward(benchmark.PCLinear):
            def forward(self, x):
                output = super().forward(x)
                return output.detach() + 0 * output

        result, _ = self.check_variant(linear=WrongBackward)
        self.assertEqual(result["correctness_status"], "failed")
        self.assertIn("gradient", result["correctness_error"])

    def test_missing_sn_update_is_rejected(self):
        result, _ = self.check_variant(updater=lambda model: None)
        self.assertEqual(result["correctness_status"], "failed")
        self.assertIn("output", result["correctness_error"])


if __name__ == "__main__":
    unittest.main()
