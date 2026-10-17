# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check checkpointed PC derivatives, storage, recomputation, and accumulation."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.profile import benchmark_mlp as benchmark
from scripts.profile import pc_baseline, pc_v2, pc_v3
from scripts.profile.implementations import initialize_model
from scripts.profile.test_pc_v2 import CountGEMMs


class CheckpointPCTest(unittest.TestCase):
    def compare_transform(self, mode, dtype, shape, level, recover, gamma_value, device="cpu"):
        torch.manual_seed(23)
        weight = (torch.randn(shape, device=device, dtype=dtype) * 0.1).requires_grad_()
        norm = torch.tensor(1.7, device=device, dtype=dtype, requires_grad=True)
        gamma = None if gamma_value is None else torch.tensor(
            [gamma_value], device=device, dtype=dtype, requires_grad=True,
        )
        config = benchmark.PCConfig(pc_level=level, recover_w_norm=recover,
                                    learnable_gamma=gamma is not None, pc_checkpoint=mode)
        inputs = (weight, norm) if gamma is None else (weight, norm, gamma)
        expected = pc_baseline.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
        actual = pc_v3.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
        upstream = torch.randn_like(actual) / actual.numel()
        expected_grads = torch.autograd.grad(expected, inputs, upstream)
        actual_grads = torch.autograd.grad(actual, inputs, upstream)
        tolerance = dict(rtol=0.02, atol=0.002) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual, expected, **tolerance)
        for value, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, reference, **tolerance)

    def test_forward_and_vjp(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for dtype in (torch.float32, torch.bfloat16):
                for shape in ((9, 5), (5, 9), (5, 5)):
                    for level in range(5):
                        for recover in (False, True):
                            with self.subTest(mode=mode, dtype=dtype, shape=shape, level=level, recover=recover):
                                self.compare_transform(mode, dtype, shape, level, recover, 0.7)

    def test_zero_negative_disabled_gamma(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for dtype in (torch.float32, torch.bfloat16):
                for shape in ((9, 5), (5, 9)):
                    for level in (0, 4):
                        for gamma in (0.0, -0.7, None):
                            with self.subTest(mode=mode, dtype=dtype, shape=shape, level=level, gamma=gamma):
                                self.compare_transform(mode, dtype, shape, level, True, gamma)

    def test_frozen_inputs_and_noncontiguous_weight(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for level in (0, 4):
                for trainable in ((True, False, False), (False, True, False),
                                  (False, False, True), (True, True, False)):
                    torch.manual_seed(7)
                    weight = (torch.randn(5, 9).T * 0.1).requires_grad_(trainable[0])
                    norm = torch.tensor(1.7, requires_grad=trainable[1])
                    gamma = torch.tensor([0.7], requires_grad=trainable[2])
                    config = benchmark.PCConfig(pc_level=level, pc_checkpoint=mode)
                    inputs = tuple(x for x in (weight, norm, gamma) if x.requires_grad)
                    expected = pc_baseline.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
                    actual = pc_v3.PCTransform(config)(weight, gamma=gamma, sn_norm=norm)
                    for value, reference in zip(torch.autograd.grad(actual.sum(), inputs),
                                                torch.autograd.grad(expected.sum(), inputs)):
                        torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)

    def test_finite_difference_gradcheck(self):
        for mode in ("polynomial", "full"):
            for level in (0, 1, 4):
                torch.manual_seed(0)
                weight = (torch.randn(3, 2, dtype=torch.float64) * 0.1).requires_grad_()
                norm = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
                gamma = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
                op = pc_v3.PCTransform(benchmark.PCConfig(
                    pc_level=level, recover_w_norm=False, pc_checkpoint=mode,
                ))
                self.assertTrue(torch.autograd.gradcheck(
                    lambda w, s, g: op(w, gamma=g, sn_norm=s), (weight, norm, gamma),
                ))

    def test_saved_storage_tradeoff(self):
        weight = torch.randn(128, 32, dtype=torch.bfloat16, requires_grad=True)
        norm = torch.tensor(16.0, dtype=torch.bfloat16, requires_grad=True)
        gamma = torch.ones(1, dtype=torch.bfloat16, requires_grad=True)
        originals = {x.untyped_storage().data_ptr() for x in (weight, norm, gamma)}
        sizes = []
        for mode in pc_v3.CHECKPOINT_MODES:
            saved = {}

            def pack(tensor):
                storage = tensor.untyped_storage()
                if storage.data_ptr() not in originals:
                    saved[storage.data_ptr()] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = pc_v3.PCTransform(benchmark.PCConfig(pc_checkpoint=mode))(
                    weight, gamma=gamma, sn_norm=norm,
                )
            sizes.append(sum(saved.values()))
            del output
        self.assertEqual(sizes, [2 * (128 * 32 + 4 * 32 * 32), 2 * (128 * 32 + 32 * 32), 0])

    def test_gemm_tradeoff(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for level in range(5):
                weight = (torch.randn(9, 5) * 0.1).requires_grad_()
                norm = torch.tensor(1.7, requires_grad=True)
                gamma = torch.tensor([0.7], requires_grad=True)
                with CountGEMMs() as counter:
                    output = pc_v3.PCTransform(benchmark.PCConfig(pc_level=level, pc_checkpoint=mode))(
                        weight, gamma=gamma, sn_norm=norm,
                    )
                    forward = counter.count
                    output.sum().backward()
                extra = {"none": 0, "polynomial": max(level - 1, 0), "full": level + 1}[mode]
                expected = (level + 1, 2 * (level + 1) + extra) if level else (0, 0)
                self.assertEqual((forward, counter.count - forward), expected, (mode, level))

    def test_distinct_microbatches_and_updated_weights(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(9)
                config = benchmark.PCConfig(power_iter=2, pc_checkpoint=mode)
                reference = benchmark.build_mlp(8, 16, 2, "pc_baseline", config).to(dtype)
                initialize_model(reference, "pc_baseline")
                pc_baseline.update_model_sn_state(reference)
                candidate = benchmark.build_mlp(8, 16, 2, "pc_v3", config).to(dtype)
                initialize_model(candidate, "pc_v3")
                candidate.load_state_dict(reference.state_dict())
                tolerance = dict(rtol=0.02, atol=0.002) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-5)
                for _ in range(3):
                    data = [(torch.randn(4, 8).to(dtype), torch.randn(4, 8)) for _ in range(3)]
                    results = []
                    for model, version in ((reference, pc_baseline), (candidate, pc_v3)):
                        model.zero_grad(set_to_none=True)
                        records = []
                        context = pc_v3.accumulation_context(model, 3) if version is pc_v3 else nullcontext()
                        with context:
                            for value, target in data:
                                x = value.detach().clone().requires_grad_()
                                output = model(x)
                                (F.mse_loss(output.float(), target) / 3).backward()
                                records.append((output.detach(), x.grad))
                        results.append(records)
                    for actual, expected in zip(results[1], results[0]):
                        for value, other in zip(actual, expected):
                            torch.testing.assert_close(value, other, **tolerance)
                    with torch.no_grad():
                        for actual, expected in zip(candidate.parameters(), reference.parameters()):
                            torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
                            expected.add_(expected.grad, alpha=-0.01)
                            actual.copy_(expected)
                    pc_baseline.update_model_sn_state(reference)
                    pc_v3.update_model_sn_state(candidate)
                    for module in pc_v3.iter_pc_linear_modules(candidate):
                        self.assertIsNone(module._weight_leaf)
                        self.assertIsNone(module._transformed_weight)

    def check_benchmark(self, device):
        for mode in pc_v3.CHECKPOINT_MODES:
            for dtype in ("float32", "bfloat16"):
                for accum in (1, 3):
                    args = SimpleNamespace(pc_level=4, power_iter=2, recover_w_norm=True,
                                           learnable_gamma=True, dtype=dtype, check_rtol=None,
                                           check_atol=None, pc_checkpoint=mode)
                    report, _ = benchmark.check_case(args, "pc_v3", 7, 8, 16, 64, 2,
                                                     torch.device(device), accum)
                    self.assertEqual(report["correctness_status"], "passed", report)

    def test_benchmark_gate_cpu(self):
        self.check_benchmark("cpu")

    def test_old_implementations_are_not_repeated_for_checkpoint_modes(self):
        args = SimpleNamespace(implementation=["plain", "pc_baseline", "pc_v2", "pc_v3"],
                               pc_checkpoint=list(pc_v3.CHECKPOINT_MODES))
        self.assertEqual(benchmark.experiment_variants(args), [
            ("plain", "none"), ("pc_baseline", "none"), ("pc_v2", "none"),
            ("pc_v3", "none"), ("pc_v3", "polynomial"), ("pc_v3", "full"),
        ])

    def test_checkpoint_does_not_add_bf16_accumulation_rounding(self):
        # This tiny shape can fail the baseline gate even for v2. Checkpoint
        # modes must reproduce v2 exactly, without weakening the baseline gate.
        for seed in (0, 1):
            args = SimpleNamespace(pc_level=4, power_iter=2, recover_w_norm=True,
                                   learnable_gamma=True, dtype="bfloat16", pc_checkpoint="none")
            reference = benchmark.make_case_model(args, "pc_baseline", seed, 8, 32, 1, torch.device("cpu"))
            state = reference.state_dict()
            records = []
            for name, mode, version in [("pc_v2", "none", pc_v2), *(
                ("pc_v3", mode, pc_v3) for mode in pc_v3.CHECKPOINT_MODES
            )]:
                args.pc_checkpoint = mode
                model = benchmark.make_case_model(args, name, seed, 8, 32, 1, torch.device("cpu"), state)
                x, target = benchmark.make_inputs(args, seed, 4, 8, torch.device("cpu"))
                steps = []
                for _ in range(2):
                    loss = benchmark.run_step(model, x, target, True, grad_accum_steps=3,
                                              update_sn_state=version.update_model_sn_state,
                                              accumulation_context=version.accumulation_context)
                    steps.append([loss.clone(), x.grad.clone(), *(p.grad.clone() for p in model.parameters())])
                records.append(steps)
            for result in records[1:]:
                for actual_step, expected_step in zip(result, records[0]):
                    for actual, expected in zip(actual_step, expected_step):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and pc_v3.triton is not None, "CUDA + Triton required")
    def test_cuda_benchmark_gate(self):
        self.check_benchmark("cuda")

    @unittest.skipUnless(torch.cuda.is_available() and pc_v3.triton is not None, "CUDA + Triton required")
    def test_cuda_recompute_kernels_against_baseline(self):
        for mode in pc_v3.CHECKPOINT_MODES:
            for dtype in (torch.float32, torch.bfloat16):
                for shape in ((9, 5), (5, 9), (128, 32), (32, 128)):
                    for level in (0, 1, 4):
                        for gamma in (0.0, 0.7, None):
                            self.compare_transform(mode, dtype, shape, level, True, gamma, "cuda")


if __name__ == "__main__":
    unittest.main()
