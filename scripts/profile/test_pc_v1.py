# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the shared PC VJP with distinct microbatches and changing weights."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.profile import benchmark_mlp as benchmark
from scripts.profile import pc_baseline, pc_v1
from scripts.profile.implementations import initialize_model


class SharedPCTransformTest(unittest.TestCase):
    def make_pair(self, dtype=torch.float32, **options):
        torch.manual_seed(17)
        config = benchmark.PCConfig(power_iter=2, **options)
        reference = benchmark.build_mlp(8, 16, 2, "pc_baseline", config).to(dtype)
        initialize_model(reference, "pc_baseline")
        pc_baseline.update_model_sn_state(reference)
        candidate = benchmark.build_mlp(8, 16, 2, "pc_v1", config).to(dtype)
        initialize_model(candidate, "pc_v1")
        candidate.load_state_dict(reference.state_dict())
        return reference, candidate

    def assert_cache_empty(self, model):
        for layer in pc_v1.iter_pc_linear_modules(model):
            self.assertFalse(layer._accumulation_active)
            self.assertIsNone(layer._transformed_weight)
            self.assertIsNone(layer._weight_leaf)

    def step(self, model, data, version):
        model.zero_grad(set_to_none=True)
        context = (
            pc_v1.accumulation_context(model, len(data))
            if version is pc_v1 else nullcontext()
        )
        outputs, input_grads = [], []
        with context:
            for value, target in data:
                x = value.detach().clone().requires_grad_(True)
                output = model(x)
                (F.mse_loss(output.float(), target) / len(data)).backward()
                outputs.append(output.detach())
                input_grads.append(x.grad.detach())
        return outputs, input_grads

    def test_distinct_microbatches_and_weight_updates(self):
        # Distinct data catches mistakes hidden by repeating the same batch.
        # Updating W/gamma and SN between steps catches stale cached transforms.
        for dtype in (torch.float32, torch.bfloat16):
            for accum in (1, 3, 4):
                for level, recover, gamma in ((4, True, True), (2, False, True), (0, True, False)):
                    with self.subTest(dtype=dtype, accum=accum, level=level, recover=recover, gamma=gamma):
                        reference, candidate = self.make_pair(
                            dtype, pc_level=level, recover_w_norm=recover,
                            learnable_gamma=gamma,
                        )
                        rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-5)
                        for _ in range(3):
                            data = [
                                (torch.randn(4, 8).to(dtype), torch.randn(4, 8))
                                for _ in range(accum)
                            ]
                            expected = self.step(reference, data, pc_baseline)
                            actual = self.step(candidate, data, pc_v1)
                            for actual_values, expected_values in zip(actual, expected):
                                for value, other in zip(actual_values, expected_values):
                                    torch.testing.assert_close(value, other, rtol=rtol, atol=atol)
                            for parameter, other in zip(candidate.parameters(), reference.parameters()):
                                torch.testing.assert_close(parameter.grad, other.grad, rtol=rtol, atol=atol)
                            self.assert_cache_empty(candidate)
                            # Keep both versions at exactly the same evaluation
                            # point. Independently applying rounded BF16 grads
                            # can produce different training trajectories, even
                            # when each individual VJP passes the tolerance.
                            with torch.no_grad():
                                for parameter, other in zip(reference.parameters(), candidate.parameters()):
                                    parameter.add_(parameter.grad, alpha=-0.01)
                                    other.copy_(parameter)
                            for model, version in ((reference, pc_baseline), (candidate, pc_v1)):
                                version.update_model_sn_state(model)
                            with torch.no_grad():
                                torch.testing.assert_close(
                                    candidate(data[0][0]), reference(data[0][0]),
                                    rtol=rtol, atol=atol,
                                )

    def test_one_pc_forward_and_backward_per_step(self):
        reference, candidate = self.make_pair()
        data = [(torch.randn(4, 8), torch.randn(4, 8)) for _ in range(4)]
        for model, version, expected_calls in ((reference, pc_baseline, 4), (candidate, pc_v1, 1)):
            counts, handles = [], []
            for layer in version.iter_pc_linear_modules(model):
                count = {"forward": 0, "backward": 0}
                counts.append(count)

                def count_transform(module, inputs, output, count=count):
                    count["forward"] += 1

                    def count_backward(gradient):
                        count["backward"] += 1

                    output.register_hook(count_backward)

                handles.append(layer.pc.register_forward_hook(count_transform))
            try:
                for _ in range(2):
                    self.step(model, data, version)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(counts, [{"forward": 2 * expected_calls, "backward": 2 * expected_calls}] * 4)

    def test_deferred_backward_is_inside_backward_timing(self):
        _, model = self.make_pair()
        accum = 3
        markers = [None] * (4 * accum + 2)
        final_backward_finished = False

        @contextmanager
        def measured_context(model, grad_accum_steps):
            nonlocal final_backward_finished
            with pc_v1.accumulation_context(model, grad_accum_steps):
                yield
                self.assertIsNone(markers[4 * accum - 1])
            # Original W/gamma gradients only exist after the shared PC VJP.
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
            self.assertIsNone(markers[4 * accum - 1])
            final_backward_finished = True

        def update(model):
            self.assertTrue(final_backward_finished)
            self.assertIsNotNone(markers[4 * accum - 1])
            self.assertIsNotNone(markers[4 * accum])
            self.assertIsNone(markers[4 * accum + 1])
            self.assert_cache_empty(model)
            pc_v1.update_model_sn_state(model)

        benchmark.run_step(
            model, torch.randn(4, 8, requires_grad=True), torch.randn(4, 8), True,
            markers, accum, update, measured_context,
        )
        self.assertEqual(markers, sorted(markers))

    def test_failed_step_releases_cache(self):
        _, model = self.make_pair()
        with self.assertRaisesRegex(RuntimeError, "aborted step"):
            with pc_v1.accumulation_context(model, 4):
                model(torch.randn(4, 8)).sum().backward()
                raise RuntimeError("aborted step")
        self.assert_cache_empty(model)
        self.step(model, [(torch.randn(4, 8), torch.randn(4, 8))] * 4, pc_v1)
        self.assert_cache_empty(model)

    def test_benchmark_correctness_gate(self):
        for dtype in ("float32", "bfloat16"):
            for accum in (1, 3, 8):
                with self.subTest(dtype=dtype, accum=accum):
                    args = SimpleNamespace(
                        pc_level=4, power_iter=2, recover_w_norm=True,
                        learnable_gamma=True, dtype=dtype,
                        check_rtol=None, check_atol=None,
                    )
                    report, _ = benchmark.check_case(
                        args, "pc_v1", 7, 4, 8, 16, 2, torch.device("cpu"), accum,
                    )
                    self.assertEqual(report["correctness_status"], "passed", report)


if __name__ == "__main__":
    unittest.main()
