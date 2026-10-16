# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover independent PC version files beside the benchmark."""

import hashlib
import importlib
from functools import lru_cache
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
REFERENCE = "pc_baseline"


def available_implementations():
    return ["plain", *sorted(
        path.stem for path in DIRECTORY.glob("pc_*.py") if path.stem.isidentifier()
    )]


@lru_cache(maxsize=None)
def load_implementation(name):
    if name == "plain":
        return None
    if name not in available_implementations():
        raise ValueError(f"Unknown PC implementation: {name}")
    module = importlib.import_module(f"scripts.profile.{name}")
    for attribute in ("PCLinear", "update_model_sn_state"):
        if not callable(getattr(module, attribute, None)):
            raise ValueError(f"{name}.py must export {attribute}")
    return module


def implementation_metadata(name):
    if name == "plain":
        return {"implementation_file": "torch.nn.Linear", "implementation_sha256": ""}
    path = DIRECTORY / f"{name}.py"
    return {
        "implementation_file": f"scripts/profile/{path.name}",
        "implementation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def initialize_model(model, implementation):
    """Initialize empty SN buffers/gamma before loading or creating a state."""
    module = load_implementation(implementation)
    if module is None:
        return
    # A version with a different initialization strategy can provide this hook.
    if callable(getattr(module, "initialize_model", None)):
        module.initialize_model(model)
        return
    for linear in model.modules():
        if isinstance(linear, module.PCLinear):
            linear.sn_u.zero_()
            linear.sn_v.zero_()
            linear._maybe_init_gamma()
