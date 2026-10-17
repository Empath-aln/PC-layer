# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select a PC implementation without changing model or parameter names."""

import importlib
from contextlib import nullcontext
from functools import lru_cache

from .pc_layer import PCLinear as BaselinePCLinear


@lru_cache(maxsize=None)
def _backend(name):
    if name == "baseline":
        return importlib.import_module("torchtitan.pc_layer.pc_layer")
    if name == "pc_v4":
        # Version files remain independently reviewable profiling snapshots.
        return importlib.import_module("scripts.profile.pc_v4")
    raise ValueError(f"Unknown PC implementation: {name}; use baseline or pc_v4")


def make_pc_linear(linear, model_args, layer_id):
    return _backend(getattr(model_args, "pc_implementation", "baseline")).PCLinear(linear, model_args, layer_id)


def is_pc_linear(module):
    return isinstance(module, BaselinePCLinear) or getattr(type(module), "_pc_backend", None) == "pc_v4"


def _layers(model):
    return (module for module in model.modules() if is_pc_linear(module))


def model_uses_sn_norm(model):
    return any(layer._uses_sn_norm() for layer in _layers(model))


def update_model_sn_state(model):
    for layer in _layers(model):
        layer.update_sn_state()


def accumulation_context(model, grad_accum_steps):
    if any(getattr(type(layer), "_pc_backend", None) == "pc_v4" for layer in _layers(model)):
        return _backend("pc_v4").accumulation_context(model, grad_accum_steps)
    return nullcontext()
