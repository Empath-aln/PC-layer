# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check a PC version against a fixed reference outside performance measurement."""

import gc

import torch


def cpu_copy(tensor):
    return tensor.detach().cpu().clone()


def capture_reference(model_factory, input_factory, step, check_steps):
    model = model_factory()
    x, target = input_factory()
    state = {name: cpu_copy(value) for name, value in model.state_dict().items()}
    records = []
    for _ in range(check_steps):
        outputs = []

        def capture(module, inputs, output):
            if not outputs:
                outputs.append(cpu_copy(output))

        hook = model.register_forward_hook(capture)
        try:
            loss = step(model, x, target)
        finally:
            hook.remove()
        records.append({
            "output": outputs[0], "loss": cpu_copy(loss),
            "input_grad": cpu_copy(x.grad) if x.grad is not None else None,
            "grads": {
                name: cpu_copy(parameter.grad) if parameter.grad is not None else None
                for name, parameter in model.named_parameters()
            },
        })
    with torch.no_grad():
        post_update = cpu_copy(model(x))
    # All GPU objects are local to this function; only CPU snapshots escape.
    return state, records, post_update


def check_equivalence(
    reference_factory, candidate_factory, input_factory, reference_step, candidate_step,
    rtol, atol, check_steps=2,
):
    """Return scalar diagnostics and the shared CPU initial state.

    Factories create one model at a time. candidate_factory receives the exact
    reference state (weights, gamma, and SN vectors) before any test steps.
    """
    report = {
        "correctness_status": "passed", "correctness_error": "",
        "correctness_rtol": rtol, "correctness_atol": atol,
        "correctness_checks": 0, "correctness_max_abs_error": 0.0,
        "correctness_max_rel_error": 0.0,
    }
    state = None

    def compare(label, actual, expected):
        if actual is None or expected is None:
            raise AssertionError(f"{label}: missing tensor/gradient")
        if actual.shape != expected.shape:
            raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
        actual = actual.detach().to(device="cpu", dtype=torch.float32)
        expected = expected.to(dtype=torch.float32)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{label}: non-finite reference or candidate")
        difference = (actual - expected).abs()
        absolute = difference.max().item()
        relative = (difference / expected.abs().clamp_min(max(atol, 1e-12))).max().item()
        report["correctness_checks"] += 1
        report["correctness_max_abs_error"] = max(report["correctness_max_abs_error"], absolute)
        report["correctness_max_rel_error"] = max(report["correctness_max_rel_error"], relative)
        if not torch.all(difference <= atol + rtol * expected.abs()):
            raise AssertionError(f"{label}: max_abs={absolute:.6g}, max_rel={relative:.6g}")

    try:
        state, records, post_update = capture_reference(
            reference_factory, input_factory, reference_step, check_steps,
        )
        gc.collect()
        # Reclaim unused reference allocator blocks before constructing a candidate.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        model = candidate_factory(state)
        x, target = input_factory()
        for step_index, expected in enumerate(records):
            microbatch = 0

            def check_output(module, inputs, output):
                nonlocal microbatch
                compare(f"step {step_index + 1} microbatch {microbatch + 1} output", output, expected["output"])
                microbatch += 1

            hook = model.register_forward_hook(check_output)
            try:
                loss = candidate_step(model, x, target)
            finally:
                hook.remove()
            compare(f"step {step_index + 1} loss", loss, expected["loss"])
            compare(f"step {step_index + 1} input gradient", x.grad, expected["input_grad"])
            parameters = dict(model.named_parameters())
            if parameters.keys() != expected["grads"].keys():
                raise AssertionError("Parameter names differ from the PC baseline")
            for name, parameter in parameters.items():
                compare(f"step {step_index + 1} {name} gradient", parameter.grad, expected["grads"][name])
        with torch.no_grad():
            compare("output after final SN update", model(x), post_update)
    except torch.cuda.OutOfMemoryError:
        raise
    except (AssertionError, RuntimeError, ValueError, TypeError, AttributeError) as error:
        report.update(correctness_status="failed", correctness_error=str(error))
    return report, state
