# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regressions for shared PC in the normal backward graph and training plugin."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint

from scripts.profile import benchmark_mlp as benchmark
from scripts.profile import pc_baseline, pc_v4
from scripts.profile.implementations import initialize_model
from torchtitan.pc_layer import runtime


class FSDPReadyPCTest(unittest.TestCase):
    def make_pair(self, mode, dtype=torch.float32, **options):
        torch.manual_seed(7)
        config = benchmark.PCConfig(power_iter=2, pc_checkpoint=mode, **options)
        reference = benchmark.build_mlp(8, 16, 2, "pc_baseline", config).to(dtype)
        if config.pc_norm_type == "sn":
            initialize_model(reference, "pc_baseline")
        else:
            for layer in pc_baseline.iter_pc_linear_modules(reference):
                layer._maybe_init_gamma()
        pc_baseline.update_model_sn_state(reference)
        candidate = benchmark.build_mlp(8, 16, 2, "pc_v4", config).to(dtype)
        if config.pc_norm_type == "sn":
            initialize_model(candidate, "pc_v4")
        else:
            for layer in pc_v4.iter_pc_linear_modules(candidate):
                layer._maybe_init_gamma()
        candidate.load_state_dict(reference.state_dict())
        return reference, candidate

    def test_grads_complete_in_final_backward_and_context_exit_does_not_backward(self):
        for mode in pc_v4.CHECKPOINT_MODES:
            _, model = self.make_pair(mode)
            x = torch.randn(4, 8, requires_grad=True)
            target = torch.randn_like(x)
            with patch.object(torch.autograd, "backward", wraps=torch.autograd.backward) as backward:
                with pc_v4.accumulation_context(model, 3):
                    for i in range(3):
                        (F.mse_loss(model(x), target) / 3).backward()
                        self.assertEqual(all(p.grad is None for p in model.parameters()), i < 2)
                        if i == 2:
                            self.assertTrue(all(p.grad is not None for p in model.parameters()))
                    self.assertEqual(backward.call_count, 3)
                self.assertEqual(backward.call_count, 3)
            self.assertTrue(all(m._step_cache is None for m in pc_v4.iter_pc_linear_modules(model)))

    def test_cache_holds_no_original_parameter_or_old_autograd_graph(self):
        _, model = self.make_pair("none")
        x = torch.randn(4, 8, requires_grad=True)
        parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
        with pc_v4.accumulation_context(model, 3):
            for microbatch in range(3):
                (model(x).square().mean() / 3).backward()
                for layer in pc_v4.iter_pc_linear_modules(model):
                    cache = layer._step_cache
                    values = [cache.output, cache.gradient, cache.norm, *cache.saved]
                    for value in values:
                        if isinstance(value, torch.Tensor):
                            self.assertIsNone(value.grad_fn)
                            self.assertNotIn(value.untyped_storage().data_ptr(), parameter_storage)
                    self.assertEqual(cache.completed, microbatch + 1)

    def test_one_transform_and_one_vjp_with_activation_recompute(self):
        for mode in pc_v4.CHECKPOINT_MODES:
            for ac in (False, True):
                _, model = self.make_pair(mode)
                if ac:
                    for i, block in enumerate(model):
                        model[i] = checkpoint_wrapper(block, checkpoint_fn=checkpoint,
                                                      use_reentrant=False, preserve_rng_state=False)
                x = torch.randn(4, 8, requires_grad=True)
                with patch.object(pc_v4, "_forward_values", wraps=pc_v4._forward_values) as forward, \
                     patch.object(pc_v4, "_backward_values", wraps=pc_v4._backward_values) as backward:
                    with pc_v4.accumulation_context(model, 3):
                        for _ in range(3):
                            (model(x).square().mean() / 3).backward()
                    self.assertEqual(forward.call_count, 4, (mode, ac))
                    self.assertEqual(backward.call_count, 4, (mode, ac))

    def test_distinct_microbatches_multiple_optimizer_steps_and_gamma_edges(self):
        for mode in pc_v4.CHECKPOINT_MODES:
            for dtype in (torch.float32, torch.bfloat16):
                for gamma in (0.0, -0.7, 1.0):
                    reference, candidate = self.make_pair(mode, dtype, gamma_init_value=gamma)
                    tolerance = dict(rtol=0.02, atol=0.002) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-5)
                    for _ in range(2):
                        data = [(torch.randn(4, 8).to(dtype), torch.randn(4, 8)) for _ in range(3)]
                        reference.zero_grad(set_to_none=True)
                        candidate.zero_grad(set_to_none=True)
                        outputs = []
                        for value, target in data:
                            x = value.detach().clone().requires_grad_()
                            output = reference(x)
                            (F.mse_loss(output.float(), target) / 3).backward()
                            outputs.append((output.detach(), x.grad.clone()))
                        with pc_v4.accumulation_context(candidate, 3):
                            for (value, target), (expected, expected_grad) in zip(data, outputs):
                                x = value.detach().clone().requires_grad_()
                                output = candidate(x)
                                (F.mse_loss(output.float(), target) / 3).backward()
                                torch.testing.assert_close(output, expected, **tolerance)
                                torch.testing.assert_close(x.grad, expected_grad, **tolerance)
                        with torch.no_grad():
                            for actual, expected in zip(candidate.parameters(), reference.parameters()):
                                torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
                                expected.add_(expected.grad, alpha=-0.01)
                                actual.copy_(expected)
                        pc_baseline.update_model_sn_state(reference)
                        pc_v4.update_model_sn_state(candidate)

    def test_context_cleanup_after_failure_and_incomplete_step(self):
        _, model = self.make_pair("full")
        x = torch.randn(4, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "expected 3 backwards"):
            with pc_v4.accumulation_context(model, 3):
                model(x).sum().backward()
        self.assertTrue(all(m._step_cache is None for m in pc_v4.iter_pc_linear_modules(model)))
        with self.assertRaisesRegex(RuntimeError, "forced failure"):
            with pc_v4.accumulation_context(model, 3):
                model(x).sum().backward()
                raise RuntimeError("forced failure")
        with pc_v4.accumulation_context(model, 3):
            for _ in range(3):
                model(x).sum().backward()
        self.assertTrue(all(m._step_cache is None for m in pc_v4.iter_pc_linear_modules(model)))

    def test_shared_levels_disabled_gamma_and_frozen_parameters(self):
        for mode in pc_v4.CHECKPOINT_MODES:
            settings = [(level, "sn", False, True, False) for level in range(5)] + [
                (0, "none", False, True, False),
                (4, "sn", True, False, True),
                (4, "sn", True, True, False),
                (4, "sn", True, False, False),
            ]
            for level, norm, gamma, train_weight, train_gamma in settings:
                reference, candidate = self.make_pair(
                    mode, pc_level=level, pc_norm_type=norm, learnable_gamma=gamma,
                )
                for model in (reference, candidate):
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_(train_gamma if "gamma" in name else train_weight)
                expected = []
                data = [torch.randn(4, 8) for _ in range(3)]
                for value in data:
                    x = value.detach().clone().requires_grad_()
                    output = reference(x)
                    (output.square().mean() / 3).backward()
                    expected.append((output.detach(), x.grad))
                with pc_v4.accumulation_context(candidate, 3):
                    for value, (wanted, wanted_grad) in zip(data, expected):
                        x = value.detach().clone().requires_grad_()
                        output = candidate(x)
                        (output.square().mean() / 3).backward()
                        torch.testing.assert_close(output, wanted, rtol=1e-4, atol=1e-5)
                        torch.testing.assert_close(x.grad, wanted_grad, rtol=1e-4, atol=1e-5)
                for actual, wanted in zip(candidate.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-4, atol=1e-5)

    def test_runtime_plugin_preserves_baseline_default_and_state_names(self):
        config = benchmark.PCConfig(power_iter=2)
        old = runtime.make_pc_linear(torch.nn.Linear(8, 16), config, 0)
        self.assertIsInstance(old, runtime.BaselinePCLinear)
        config.pc_implementation = "pc_v4"
        new = runtime.make_pc_linear(torch.nn.Linear(8, 16), config, 0)
        self.assertIsInstance(new, pc_v4.PCLinear)
        self.assertTrue(runtime.is_pc_linear(old) and runtime.is_pc_linear(new))
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        initialize_model(new, "pc_v4")
        model = torch.nn.Sequential(new)
        runtime.update_model_sn_state(model)
        self.assertTrue(runtime.model_uses_sn_norm(model))
        with runtime.accumulation_context(model, 3):
            for _ in range(3):
                model(torch.randn(4, 8)).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def check_gate(self, device):
        for mode in pc_v4.CHECKPOINT_MODES:
            for dtype in ("float32", "bfloat16"):
                for accum in (1, 3):
                    args = SimpleNamespace(pc_level=4, power_iter=2, recover_w_norm=True,
                                           learnable_gamma=True, dtype=dtype, check_rtol=None,
                                           check_atol=None, pc_checkpoint=mode)
                    report, _ = benchmark.check_case(args, "pc_v4", 7, 8, 16, 64, 2,
                                                     torch.device(device), accum)
                    self.assertEqual(report["correctness_status"], "passed", report)

    def test_cpu_gate(self):
        self.check_gate("cpu")

    @unittest.skipUnless(torch.cuda.is_available() and pc_v4.triton is not None, "CUDA + Triton required")
    def test_cuda_gate(self):
        self.check_gate("cuda")


if __name__ == "__main__":
    unittest.main()
