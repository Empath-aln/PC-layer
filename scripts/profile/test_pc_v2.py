# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent checks of v2's VJP, saved storage, GEMM count, and CUDA kernels."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from scripts.profile import benchmark_mlp as benchmark
from scripts.profile import pc_v1, pc_v2
from scripts.profile.implementations import initialize_model


class CountGEMMs(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mm.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


class FusedPCTest(unittest.TestCase):
    def compare_transform(self, dtype, shape, level, recover, gamma_value=0.7, device="cpu"):
        torch.manual_seed(23)
        weight = (torch.randn(shape, device=device, dtype=dtype) * 0.1).requires_grad_()
        norm = torch.tensor(1.7, dtype=dtype, device=device, requires_grad=True)
        gamma = None if gamma_value is None else torch.tensor(
            [gamma_value], dtype=dtype, device=device, requires_grad=True,
        )
        config = benchmark.PCConfig(pc_level=level, recover_w_norm=recover, learnable_gamma=gamma is not None)
        inputs = (weight, norm) if gamma is None else (weight, norm, gamma)
        reference = pc_v1.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
        actual = pc_v2.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
        upstream = torch.randn_like(actual) / actual.numel()
        expected_grads = torch.autograd.grad(reference, inputs, upstream)
        actual_grads = torch.autograd.grad(actual, inputs, upstream)
        rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-5)
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        for value, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)

    def test_forward_and_vjp_tall_wide_square(self):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((9, 5), (5, 9), (5, 5)):
                for level in range(5):
                    for recover in (False, True):
                        with self.subTest(dtype=dtype, shape=shape, level=level, recover=recover):
                            self.compare_transform(dtype, shape, level, recover)

    def test_zero_negative_and_disabled_gamma(self):
        # Gamma=0 must still receive its derivative; dividing P by gamma would
        # be incorrect and is deliberately avoided by saving the pre-gamma value.
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((9, 5), (5, 9)):
                for gamma in (0.0, -0.5, None):
                    with self.subTest(dtype=dtype, shape=shape, gamma=gamma):
                        self.compare_transform(dtype, shape, 4, True, gamma)

    def test_finite_difference_gradcheck(self):
        for shape in ((3, 2), (2, 3)):
            for level in range(5):
                with self.subTest(shape=shape, level=level):
                    torch.manual_seed(0)
                    weight = (torch.randn(shape, dtype=torch.float64) * 0.1).requires_grad_()
                    norm = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
                    gamma = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
                    # Recovery intentionally stops the norm derivative. Disable
                    # it for finite differences; enabled recovery is checked
                    # against the baseline's actual autograd VJP above.
                    config = benchmark.PCConfig(pc_level=level, recover_w_norm=False)
                    op = pc_v2.PCTransform(config)
                    self.assertTrue(torch.autograd.gradcheck(
                        lambda w, s, g: op(w, gamma=g, sn_norm=s),
                        (weight, norm, gamma),
                    ))

    def test_frozen_inputs_and_noncontiguous_weight(self):
        for freeze_weight, freeze_norm, freeze_gamma in (
            (True, False, False), (False, True, False), (False, False, True), (True, True, False),
        ):
            config = benchmark.PCConfig()
            torch.manual_seed(7)
            weight = (torch.randn(5, 9).T * 0.1).requires_grad_(not freeze_weight)
            norm = torch.tensor(1.7, requires_grad=not freeze_norm)
            gamma = torch.tensor([0.7], requires_grad=not freeze_gamma)
            inputs = tuple(t for t in (weight, norm, gamma) if t.requires_grad)
            outputs = [version.PCTransform(config)(weight, gamma=gamma, sn_norm=norm) for version in (pc_v1, pc_v2)]
            grads = [torch.autograd.grad(output.sum(), inputs) for output in outputs]
            for actual, expected in zip(grads[1], grads[0]):
                torch.testing.assert_close(actual, expected)

    def test_saved_storage_is_smaller(self):
        shape = (128, 32)
        config = benchmark.PCConfig()
        weight = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        norm = torch.tensor(16.0, dtype=torch.bfloat16, requires_grad=True)
        gamma = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
        original = {t.untyped_storage().data_ptr() for t in (weight, norm, gamma)}
        sizes = []
        for version in (pc_v1, pc_v2):
            unique = {}

            def pack(tensor):
                storage = tensor.untyped_storage()
                if storage.data_ptr() not in original:
                    unique[storage.data_ptr()] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = version.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
            sizes.append(sum(unique.values()))
            del output
        # Original: two weight-size + five Gram-size storages.
        # v2: one pre-gamma weight-size + four Gram-size storages.
        self.assertEqual(sizes[0], 2 * (2 * 128 * 32 + 5 * 32 * 32))
        self.assertEqual(sizes[1], 2 * (128 * 32 + 4 * 32 * 32))
        self.assertLess(sizes[1], sizes[0])

    def test_no_extra_gemms(self):
        for level in range(5):
            counts = []
            for version in (pc_v1, pc_v2):
                weight = torch.randn(9, 5, requires_grad=True) * 0.1
                norm = torch.tensor(1.7, requires_grad=True)
                gamma = torch.tensor([0.7], requires_grad=True)
                with CountGEMMs() as counter:
                    output = version.PCTransform(benchmark.PCConfig(pc_level=level))(
                        weight, gamma=gamma, sn_norm=norm,
                    )
                    forward_count = counter.count
                    output.sum().backward()
                counts.append((forward_count, counter.count - forward_count))
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[1], (level + 1, 2 * (level + 1)) if level else (0, 0))

    def test_distinct_microbatches_and_weight_changes(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(9)
            config = benchmark.PCConfig(power_iter=2)
            reference = benchmark.build_mlp(8, 16, 2, "pc_v1", config).to(dtype)
            initialize_model(reference, "pc_v1")
            pc_v1.update_model_sn_state(reference)
            candidate = benchmark.build_mlp(8, 16, 2, "pc_v2", config).to(dtype)
            initialize_model(candidate, "pc_v2")
            candidate.load_state_dict(reference.state_dict())
            rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-5)
            for _ in range(3):
                data = [(torch.randn(4, 8).to(dtype), torch.randn(4, 8)) for _ in range(3)]
                results = []
                for model, version in ((reference, pc_v1), (candidate, pc_v2)):
                    model.zero_grad(set_to_none=True)
                    records = []
                    with version.accumulation_context(model, 3):
                        for value, target in data:
                            x = value.detach().clone().requires_grad_()
                            output = model(x)
                            (torch.nn.functional.mse_loss(output.float(), target) / 3).backward()
                            records.append((output.detach(), x.grad))
                    results.append(records)
                for actual, expected in zip(results[1], results[0]):
                    for value, other in zip(actual, expected):
                        torch.testing.assert_close(value, other, rtol=rtol, atol=atol)
                for parameter, other in zip(candidate.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=rtol, atol=atol)
                for layer in pc_v2.iter_pc_linear_modules(candidate):
                    self.assertIsNone(layer._transformed_weight)
                    self.assertIsNone(layer._weight_leaf)
                    self.assertFalse(layer._accumulation_active)
                # Advance both operators to the same new evaluation point.
                with torch.no_grad():
                    for parameter, other in zip(reference.parameters(), candidate.parameters()):
                        parameter.add_(parameter.grad, alpha=-0.01)
                        other.copy_(parameter)
                pc_v1.update_model_sn_state(reference)
                pc_v2.update_model_sn_state(candidate)

    def check_benchmark(self, device):
        for dtype in ("float32", "bfloat16"):
            for accum in (1, 3, 8):
                args = SimpleNamespace(
                    pc_level=4, power_iter=2, recover_w_norm=True,
                    learnable_gamma=True, dtype=dtype,
                    check_rtol=None, check_atol=None,
                )
                report, _ = benchmark.check_case(
                    args, "pc_v2", 7, 8, 16, 64, 2, torch.device(device), accum,
                )
                self.assertEqual(report["correctness_status"], "passed", report)

    def test_benchmark_gate_cpu(self):
        self.check_benchmark("cpu")

    def test_backend_compiler_errors_are_recorded(self):
        class KernelCompilationError(Exception):
            pass

        args = SimpleNamespace(
            pc_level=4, power_iter=2, recover_w_norm=True, learnable_gamma=True,
            dtype="float32", check_rtol=None, check_atol=None,
        )
        with patch.object(pc_v2, "_mm_diagonal", side_effect=KernelCompilationError("kernel did not compile")):
            report, _ = benchmark.check_case(
                args, "pc_v2", 7, 4, 8, 16, 1, torch.device("cpu"), 3,
            )
        self.assertEqual(report["correctness_status"], "failed")
        self.assertIn("KernelCompilationError", report["correctness_error"])

    @unittest.skipUnless(torch.cuda.is_available() and pc_v2.triton is not None, "CUDA + Triton required")
    def test_cuda_benchmark_gate(self):
        self.check_benchmark("cuda")

    @unittest.skipUnless(torch.cuda.is_available() and pc_v2.triton is not None, "CUDA + Triton required")
    def test_cuda_kernels_against_torch(self):
        for dtype in (torch.float32, torch.bfloat16):
            for d in (5, 33, 128):
                torch.manual_seed(5)
                a = torch.randn(d, d, device="cuda", dtype=dtype) / d**0.5
                b = torch.randn_like(a)
                actual_affine = pc_v2._affine_diagonal(a, 2.890, -10.351)
                actual_mm = pc_v2._mm_diagonal(a, b, 14.097)
                with patch.object(pc_v2, "_use_cuda_kernels", return_value=False):
                    expected_affine = pc_v2._affine_diagonal(a, 2.890, -10.351)
                    expected_mm = pc_v2._mm_diagonal(a, b, 14.097)
                rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-5)
                torch.testing.assert_close(actual_affine, expected_affine, rtol=rtol, atol=atol)
                torch.testing.assert_close(actual_mm, expected_mm, rtol=rtol, atol=atol)
            for shape in ((9, 5), (5, 9), (128, 32), (32, 128)):
                for level in range(5):
                    for gamma in (0.0, 0.7, None):
                        self.compare_transform(dtype, shape, level, True, gamma, device="cuda")


if __name__ == "__main__":
    unittest.main()
