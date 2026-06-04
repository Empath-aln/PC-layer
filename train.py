# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel
from torchtitan.checkpoint import CheckpointManager
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_schedulers
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.models.llama.model import Attention, FeedForward, ResidualAdd
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torch.distributed.tensor import DTensor
from torchtitan.parallelisms.pipelining_utils import build_pipeline_schedule
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.pc_layer.pc_layer import (
    model_uses_sn_norm,
    update_model_sn_state,
)
from torchtitan.utils import (
    Color,
    dist_max,
    dist_mean,
    get_metrics_rank,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)
import torch.distributed as dist
import swanlab
from matplotlib import pyplot as plt
import json
from torchtitan.optimizers import Muon
import math


def aggregate_rms_across_ranks(local_rms: float, world_size: int, global_rank: int) -> float:
    """
    Aggregate RMS values across all ranks (using all_reduce for better performance).

    Aggregation formula: sqrt(mean([rank_rms^2 for rank_rms in all_rank_rms]))

    Performance: use all_reduce instead of gather to reduce communication overhead
    and leverage collective communication optimizations.

    Args:
        local_rms: RMS value computed on the current rank
        world_size: total number of ranks
        global_rank: global index of the current rank

    Returns:
        The aggregated global RMS, identical across all ranks
    """
    if world_size == 1:
        return local_rms

    # Square the value on each rank
    local_rms_sq_tensor = torch.tensor(local_rms ** 2, dtype=torch.float32).cuda()

    # Use all_reduce(SUM) to sum the squares across all ranks
    dist.all_reduce(local_rms_sq_tensor, op=dist.ReduceOp.SUM)

    # Compute RMS: sqrt(sum_of_squares / world_size)
    global_rms = math.sqrt(local_rms_sq_tensor.item() / world_size)

    return global_rms


def aggregate_absmax_across_ranks(local_absmax: float, world_size: int, global_rank: int) -> float:
    """
    Aggregate AbsMax values across all ranks (using all_reduce for better performance).

    Aggregation formula: max(all_rank_absmax)

    Performance: use all_reduce instead of gather to reduce communication overhead
    and leverage collective communication optimizations.

    Args:
        local_absmax: AbsMax value computed on the current rank
        world_size: total number of ranks
        global_rank: global index of the current rank

    Returns:
        The aggregated global AbsMax, identical across all ranks
    """
    if world_size == 1:
        return local_absmax

    local_absmax_tensor = torch.tensor(local_absmax, dtype=torch.float32).cuda()

    # Use all_reduce(MAX) to obtain the global maximum directly
    dist.all_reduce(local_absmax_tensor, op=dist.ReduceOp.MAX)

    return local_absmax_tensor.item()


def log_norms(model, step, global_rank, job_config):
    """Log gradient norms and weight matrix norms of QKVO and MLP (W1, W2, W3) layers + learnable gamma monitoring.

    Supports the Llama model naming convention:
    - Llama: wq, wk, wv, wo, w1, w2, w3, layers.{i}.xxx
    """
    grad_norms = {}
    weight_norms = {}
    pc_gamma_vals = {}
    pc_gamma_grads = {}

    # List of supported weight names (Llama)
    weight_keys = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']

    # ----------------------------
    # 1) weight / gradient norm
    # ----------------------------
    for name, param in model.named_parameters():
        if any(key in name for key in weight_keys) and name.endswith('weight'):
            weight_norm = param.norm().item()
            parts = name.split('.')

            # Handle PCLinear wrapping: layers.0.attention.wq.linear.weight
            # For PCLinear, parts[-2] is 'linear', so parts[-3] should be used
            if parts[-2] == 'linear' and len(parts) >= 5:
                layer_name = parts[-3]  # wq, wk, wv, etc.
            else:
                layer_name = parts[-2]  # original naming (no PCLinear wrapping)

            # Extract layer_id (Llama's layers.{i})
            layer_id = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_id = parts[i + 1]
                    break

            if layer_id is None:
                layer_id = '0'  # fallback

            weight_key = f"weight_norms_{layer_name}/layers_{layer_id}"
            weight_norms[weight_key] = weight_norm

            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_key = f"grad_norms_{layer_name}/layers_{layer_id}"
                grad_norms[grad_key] = grad_norm


    # ----------------------------
    # 2) gamma monitoring (supports PCLinear wrapping)
    # ----------------------------
    # List of supported weight names (Llama)
    gamma_weight_keys = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']

    for module_name, module in model.named_modules():
        # Check whether this is a PCLinear module (by presence of gamma and layer_id attributes)
        if not hasattr(module, 'gamma'):
            continue
        if module.gamma is None:
            continue
        if not hasattr(module, 'layer_id'):
            continue

        g = module.gamma
        layer_id = module.layer_id
        layer_tag = f"layers_{layer_id}"

        # Extract weight type from the module name
        # PCLinear naming format (Llama): layers.0.attention.wq or layers.0.feed_forward.w1
        parts = module_name.split('.')
        if len(parts) < 3:
            continue

        # The last part is the weight name (wq, wk, wv, wo, w1, w2, w3)
        weight_name = parts[-1]
        if weight_name not in gamma_weight_keys:
            continue

        attr = f"gamma_{weight_name}"

        # gamma value & gradient
        if hasattr(g, "v") and g.v is not None:
            v = g.v.detach()
            if isinstance(v, DTensor):
                v = v.full_tensor()
            if v.numel() == 1:
                pc_gamma_vals[f"{attr}/{layer_tag}"] = float(v.item())
            else:
                # per-head gamma: expand into one metric per head
                v_flat = v.reshape(-1)
                for h in range(v_flat.numel()):
                    pc_gamma_vals[f"{attr}/{layer_tag}/h{h}"] = float(v_flat[h].item())
            if g.v.grad is not None:
                gv = g.v.grad.detach()
                if isinstance(gv, DTensor):
                    gv = gv.full_tensor()
                if gv.numel() == 1:
                    pc_gamma_grads[f"{attr}_grad/{layer_tag}"] = float(gv.item())
                else:
                    gv_flat = gv.reshape(-1)
                    for h in range(gv_flat.numel()):
                        pc_gamma_grads[f"{attr}_grad/{layer_tag}/h{h}"] = float(gv_flat[h].item())

    # ----------------------------
    # 3) log
    # ----------------------------
    if global_rank == 0 and job_config.metrics.enable_swanlab:
        metrics = {**grad_norms, **weight_norms, **pc_gamma_vals, **pc_gamma_grads}
        if metrics:
            swanlab.log(metrics, step=step)

    return grad_norms, weight_norms, pc_gamma_vals, pc_gamma_grads



    return norm_norms


def register_custom_hooks(model, model_config):
    """
    Register forward hooks on all key layers to monitor signal propagation statistics.

    Features:
    - Record per-layer output statistics (RMS and AbsMax)
    - Compute cross-layer averages
    - Monitor outputs of attention, FFN, and residual connections

    Supports the Llama model naming convention:
    - Llama: layers.{i}.attention, layers.{i}.feed_forward, tok_embeddings
    """
    output_norms = {}

    # --------- per-forward buffers (for cross-layer averages) ---------
    _buffer_keys = ["attn_absmax", "attn_rms",
                    "ffn_absmax", "ffn_rms",
                    "h_absmax", "h_rms",
                    "out_absmax", "out_rms"]
    _buffer = {k: {} for k in _buffer_keys}

    # ---------- helpers ----------
    def _to_tensor(x):
        """Convert output to tensor (simplified version)"""
        # Direct tensor or DTensor
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, DTensor):
            return x.to_local()

        # Tuple/list - take first element
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return _to_tensor(x[0])

        return None

    def _rms_hidden(t: torch.Tensor) -> float:
        """Compute the overall RMS (root mean square) of the tensor"""
        return torch.sqrt(torch.mean(t.float() ** 2)).item()

    def _absmax(t: torch.Tensor) -> float:
        return t.float().abs().max().item()

    def _extract_layer_id(name: str) -> str:
        """
        Extract layer_id from the module name.
        Supported format:
        - Llama: "layers.0.attention.xxx" -> "0"
        """
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                return parts[i + 1]
        return '0'

    # ---------- Model-level hooks to compute cross-layer average ----------
    def _model_forward_pre_hook(module, inputs):
        # Clear this round's buffers at the start of each forward
        for buffer in _buffer.values():
            buffer.clear()

    def _compute_avg(buffer_key: str) -> float:
        """Compute cross-layer average from buffer"""
        if len(_buffer[buffer_key]) > 0:
            vals = list(_buffer[buffer_key].values())
            return float(sum(vals) / len(vals))
        return None

    def _model_forward_hook(module, inputs, outputs):
        # After forward completes, compute cross-layer averages
        for buffer_key in _buffer.keys():
            output_norms[f"cross_layer_avg/{buffer_key}"] = _compute_avg(buffer_key)

    # Register on the whole model (ensures cross-layer avg is computed on every forward)
    model.register_forward_pre_hook(_model_forward_pre_hook)
    model.register_forward_hook(_model_forward_hook)

    # ---------- Forward Hook ----------
    def _record_metrics(t, layer_type: str, layer_id: str, name: str):
        """Record metrics for a layer (generalized pattern)"""
        absmax_val = _absmax(t)
        rms_val = _rms_hidden(t)

        # Determine buffer key based on layer_type
        if layer_type == "attention":
            buffer_keys = ["attn_absmax", "attn_rms"]
        elif layer_type == "ffn":
            buffer_keys = ["ffn_absmax", "ffn_rms"]
        elif layer_type == "attn_residual":
            buffer_keys = ["h_absmax", "h_rms"]
        elif layer_type == "ffn_residual":
            buffer_keys = ["out_absmax", "out_rms"]
        else:
            return

        # Record to output_norms and buffer
        output_norms[f"{buffer_keys[0]}/{name}"] = absmax_val
        output_norms[f"{buffer_keys[1]}/{name}"] = rms_val
        _buffer[buffer_keys[0]][str(layer_id)] = absmax_val
        _buffer[buffer_keys[1]][str(layer_id)] = rms_val

    def get_hook(layer_type, layer_id=None, sublayer=None):
        def hook(module, block_input, output):
            t = _to_tensor(output)
            if t is None:
                return

            # Build metric name
            name_parts = [layer_type]
            if layer_id is not None:
                name_parts.append(str(layer_id))
            if sublayer:
                name_parts.append(sublayer)
            name = ".".join(name_parts)

            # Record metrics based on layer type
            if layer_type in ("attention", "ffn", "attn_residual", "ffn_residual"):
                _record_metrics(t, layer_type, layer_id, name)
            elif layer_type == "embedding":
                output_norms[f"emb_absmax/{name}"] = _absmax(t)
                output_norms[f"emb_rmsnorm/{name}"] = _rms_hidden(t)

        return hook

    # ---------- register hooks ----------
    if model_config.log_signal_propagation:
        for name, module in model.named_modules():
            # Attention layers
            if isinstance(module, Attention):
                layer_id = _extract_layer_id(name)
                module.register_forward_hook(get_hook("attention", layer_id))

            # FFN layers
            elif isinstance(module, FeedForward):
                layer_id = _extract_layer_id(name)
                module.register_forward_hook(get_hook("ffn", layer_id))

            # ResidualAdd layers
            elif isinstance(module, ResidualAdd):
                # name format: "layers.3.attn_residual" or "layers.3.ffn_residual"
                layer_id = _extract_layer_id(name)
                if name.endswith("attn_residual"):
                    module.register_forward_hook(get_hook("attn_residual", layer_id))
                elif name.endswith("ffn_residual"):
                    module.register_forward_hook(get_hook("ffn_residual", layer_id))

            # Embedding layer
            elif isinstance(module, torch.nn.Embedding):
                if name.endswith("tok_embeddings"):
                    module.register_forward_hook(get_hook("embedding"))

    return output_norms, _buffer
                

@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizers(model_parts, job_config: JobConfig, world_mesh=None):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model, world_mesh=None):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        muon_lr_adjust = job_config.optimizer.muon_lr_adjust
        muon_coefficient_type = job_config.optimizer.muon_coefficient_type
        ns_steps = job_config.optimizer.ns_steps
        fused = False # job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95), 
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            logger.info(f"======>>>>> Using Adam optimizer, lr = {lr}")
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            logger.info(f"======>>>>> Using AdamW optimizer, lr = {lr}")
            
        elif name == "muon":
            # This for muon_fsdp file
            # Separate parameters
            muon_params = []
            adamw_params = []

            for p in model.parameters():
                if p.ndim >= 2:  # parameters with >=2D required by Muon
                    muon_params.append(p)
                else:  # low-dimensional parameters handled by AdamW
                    adamw_params.append(p)

            # Configure Muon parameters
            muon_config = {
                "lr": lr,
                "momentum": 0.95,  # Muon momentum parameter
                "nesterov": True,
                "ns_steps": ns_steps,     # number of Newton-Schulz iteration steps
                "adamw_params": adamw_params if adamw_params else None,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
                "adamw_wd": 0.1,    # AdamW weight decay
                "lr_adjust": muon_lr_adjust,
                "muon_coefficient_type": muon_coefficient_type,
                "use_bf16": job_config.optimizer.use_bf16
            }
            
            optimizer = Muon(muon_params, **muon_config)
            logger.info(f"======>>>>> Using Muon optimizer with auxiliary Adam, lr = {lr}")
            
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
                
        def __iter__(self):
            return iter(self.optimizers)

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])


def validate(job_config, model_config, model, data_loader_val, loss_fn, current_step):
    # Reset the dataset before starting validation
    if hasattr(data_loader_val.dataset, "reset"):
        data_loader_val.dataset.reset()

    model.eval()
    loss_list = []
    total_tokens = 0
    num_val_batch = job_config.training.num_val_batch # limit the number of data for validation
    current_batch_idx = 0
    logger.info(f"Calculating validation loss...")
    with torch.no_grad():
        # Original logic: for batch in data_loader_val:
        for batch in iter(data_loader_val): # Use iter() to get a new iterator
            current_batch_idx  += 1
            if current_batch_idx > num_val_batch:
                break
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
            loss_list.append( loss.item() )
            total_tokens += labels.numel()

    avg_loss = np.mean(loss_list)
    logger.info(f"Validation completed: step: {current_step}, val loss: {avg_loss} val token: {total_tokens}")
    model.train()

    return avg_loss


def save_config_files(job_config):
    """Save config files and source code to checkpoint base directory"""
    # Get base checkpoint directory
    base_ckpt_dir = Path(job_config.checkpoint.folder)
    
    # Save config files
    config_dir = base_ckpt_dir / "configs" / job_config.metrics.swanlab_comment
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy toml file 
    toml_path = Path(job_config.job.config_file)
    shutil.copy2(toml_path, config_dir / toml_path.name)
    
    # Save source code
    code_dir = base_ckpt_dir / "code" / job_config.metrics.swanlab_comment
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train.py and model.py
    shutil.copy2("train.py", code_dir / "train.py")
    shutil.copy2("torchtitan/models/llama/model.py", code_dir / "model.py")


# List of parameters that must be passed in from job_config (Llama-specific research features)
MODEL_CONFIG_KEYS = {
    'norm_type',
    'precondition_ffn',
    'precondition_o',
    'precondition_qk',
    'precondition_v',
    'power_iter',
    'pc_norm_type',
    'pc_norm_eps',
    'pc_level',
    'recover_w_norm',
    'learnable_gamma',
    'gamma_init_value',
    'log_signal_propagation',
    'log_gradients',
}

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = Color if job_config.metrics.enable_color_printing else NoColor

    # take control of garbage collection to avoid stragglers
    _gc_freq = job_config.training.gc_freq
    gc.disable()
    gc.collect(1)

    # init distributed
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(job_config)
    
    # ====== Set the random seed for distributed training ======
    seed = 42  # or read the seed configuration from job_config
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure the seed is synchronized across all processes
    if world_size > 1:
        torch.distributed.barrier()
    # ====== End of random seed setup ======

    # Save config files at the start of training
    # =======================================
    if global_rank == 0:  # only save config files on the main process
        save_config_files(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
        dp_mesh = None # is this correct?

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_train_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
        # num_workers=4,          # or 8, depending on the machine
        # prefetch_factor=4,      # each worker prefetches 4 batches
        # pin_memory=True,
        # persistent_workers=True,
    )

    # validation dataloader use c4 mini

    data_loader_val = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_val_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
        infinite=False, # this causes a strange bug
        # num_workers=4,          # or 8, depending on the machine
        # prefetch_factor=4,      # each worker prefetches 4 batches
        # pin_memory=True,
        # persistent_workers=True,
    )

    # loss_parallel enables dispatching to efficient loss operators
    loss_parallel_ctx = (
        loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    global model_config
    model_config = models_config[model_name][job_config.model.flavor]

    # Override model config with values from the toml [model] section.
    # Keys absent from the toml keep their default from model_config.
    for key in MODEL_CONFIG_KEYS:
        if hasattr(job_config.model, key):
            setattr(model_config, key, getattr(job_config.model, key))

    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len


    run_id = job_config.job.description + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr) + job_config.metrics.swanlab_comment
    logger.info(f"=========> Currently running: {run_id}")
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")

    
    with torch.device("meta"):
        whole_model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(whole_model, job_config)

    # log model size
    model_param_count = get_num_params(whole_model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(whole_model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset} "
        f"num_flop_per_token: {num_flop_per_token:,} "
    )


    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

    if parallel_dims.pp_enabled:
        stages, model_parts = models_pipelining_fns[model_name](
            whole_model, world_mesh, parallel_dims, job_config, device, model_config
        )
    else:
        # In 1D/2D cases or PP with simple schedules, model_parts is just one item
        # for PP with looped schedules, each item is one stage-model-chunk
        # we iterate all model_parts for applying SPMD parallelism, compilation, optimizer, and checkpointing
        model_parts = [whole_model]

    # apply PT-D DP/TP parallelisms and activation checkpointing
    model_parts = [
        models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
        for m in model_parts
    ]

    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    for model in model_parts:
        model.to_empty(device=init_device)

    if parallel_dims.pp_enabled:
        pp_schedule = build_pipeline_schedule(
            job_config, parallel_dims, stages, loss_fn
        )
    else:
        # If PP is enabled, we can't rely on init_weights, because some layers are missing.
        # In the future, we may make init_weights handle missing layers, but also have to consider RNG seed propagation.
        # allocate sharded model on GPU and initialize weights via DTensor
        whole_model.init_weights()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config, world_mesh=dp_mesh)
    lr_schedulers = get_lr_schedulers(optimizers.optimizers, job_config)


    metric_logger = build_metric_logger(
        job_config, metrics_log_rank=get_metrics_rank(world_mesh, parallel_dims), run_id = run_id
    )

    if job_config.metrics.enable_swanlab and global_rank == 0:
        # if torch.distributed.get_rank() == 0
        logger.info("Initializing swanlab")
        run_id = job_config.model.name + job_config.model.flavor + job_config.optimizer.name + str(job_config.optimizer.lr) + job_config.metrics.swanlab_comment
        swanlab.init(
            workspace=job_config.job.workspace,
            project=job_config.job.description,
            name=run_id
        )


    train_state = TrainState()
    
#     timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # train loop
    for model in model_parts:
        model.train()
        
        # module_list = [name for name, _ in model.named_parameters()]
        # print("Model parameters:", module_list)
        
    # Initialize hooks (run once before training starts)
    if model_config.log_signal_propagation:
        # output_norms, grad_output_norms, grad_input_norms, layernorm_features, attention_logits = register_custom_hooks(model, model_config)
        output_norms, _buffer = register_custom_hooks(model, model_config)

        # Used to store statistics for each microbatch
        # key: metric name, value: list of per-microbatch values
        microbatch_rms_values = {}  # store RMS-type metrics
        microbatch_absmax_values = {}  # store AbsMax-type metrics

    # load initial checkpoint
    job_config.checkpoint.folder +=  f"/{job_config.model.name}_{job_config.model.flavor}/{job_config.optimizer.name}/{job_config.metrics.swanlab_comment}"
    
    checkpoint = CheckpointManager(
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        dataloader=data_loader,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load(resume = job_config.checkpoint.resume)

    if parallel_dims.pp_enabled and model_uses_sn_norm(whole_model):
        raise RuntimeError(
            "Deferred SN state commit currently supports only non-PP training. "
            "Please disable pipeline parallelism when pc_norm_type='sn'."
        )

    if parallel_dims.pp_enabled and not checkpoint_loaded:
        raise RuntimeError(
            "Pipeline Parallelism requires meta-initialization and loading seed checkpoint. "
            "Please run `./create_seed_checkpoint.sh` and rerun training with `--checkpoint.enable_checkpoint`"
        )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    if not parallel_dims.pp_enabled and model_uses_sn_norm(whole_model):
        update_model_sn_state(whole_model)

    checkpoint.reset()

    # variables used to keep info for metrics logging
    losses_since_last_log: List[float] = []
    ntokens_since_last_log = 0
    ntokens_total_train = 0
    data_loading_times: List[float] = []
    time_last_log = timer()
    gpu_memory_monitor.reset_peak_stats()

    # train loop
    logger.info(f"Training starts at step {train_state.step + 1}")
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            if train_state.step > 1 and train_state.step % _gc_freq == 0:
                gc.collect(1)

            if train_state.step % job_config.training.val_interval  == 0:
                val_loss = validate(job_config, model_config, model, data_loader_val, loss_fn, train_state.step)
                if global_rank == 0 and job_config.metrics.enable_swanlab:
                    swanlab.log({'val_loss': val_loss}, step=train_state.step)

            # # get batch
            # data_load_start = timer()
            # batch = next(data_iterator)
            # input_ids, labels = batch
            # ntokens_since_last_log += labels.numel()
            # data_loading_times.append(timer() - data_load_start)
            # input_ids = input_ids.cuda()
            # labels = labels.cuda()

            optimizers.zero_grad()

            if parallel_dims.pp_enabled:
                logger.info('are we here? pipeline parallel forward / backward inside step() call') # False
                # pipeline parallel forward / backward inside step() call
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                with loss_parallel_ctx():
                    if pp_mesh.get_local_rank() == 0:
                        pp_schedule.step(input_ids)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=labels, losses=losses)
                    else:
                        pp_schedule.step()

                # accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses))
                    if is_last_stage
                    else torch.Tensor([-1.0])
                )
            else:
                # Non-PP forward / backward
                # with loss_parallel_ctx():
                #     logger.info('are we here? Non-PP forward / backward') # True
                #     pred = model(input_ids)
                #     loss = loss_fn(pred, labels)
                #     # pred.shape=(bs, seq_len, vocab_size)
                #     # need to free to before bwd to avoid peaking memory
                #     del pred
                #     loss.backward()
                
                for microbatch_idx in range(job_config.training.grad_accumulation_steps):

                    # get batch
                    data_load_start = timer()
                    batch = next(data_iterator)
                    input_ids, labels = batch
                    ntokens_since_last_log += labels.numel() * world_size
                    ntokens_total_train  += labels.numel() * world_size
                    data_loading_times.append(timer() - data_load_start)
                    
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()

                    # model.set_requires_gradient_sync(microbatch_idx==(job_config.training.grad_accumulation_steps-1)) # OOM error

                    with loss_parallel_ctx():
                        # Clear the previous step's records (_buffer is cleared automatically at the start of each forward, but clear output_norms here)
                        if model_config.log_signal_propagation:
                            output_norms.clear()

                        # print(f" train_state.step {train_state.step}, microbatch_idx: {microbatch_idx}, ntokens_since_last_log {ntokens_since_last_log}, rank {torch.distributed.get_rank()}")

                        pred = model(input_ids)

                        # Accumulate statistics for the current microbatch
                        if model_config.log_signal_propagation and output_norms:
                            for key, value in output_norms.items():
                                if value is None:
                                    continue

                                # Determine the type by the key name
                                if '_rms' in key or 'rmsnorm' in key:
                                    if key not in microbatch_rms_values:
                                        microbatch_rms_values[key] = []
                                    microbatch_rms_values[key].append(value)
                                else:  # absmax type
                                    if key not in microbatch_absmax_values:
                                        microbatch_absmax_values[key] = []
                                    microbatch_absmax_values[key].append(value)

                        loss_unnormalized = loss_fn(pred, labels)
                        del pred

                        loss = loss_unnormalized / job_config.training.grad_accumulation_steps

                        losses_since_last_log.append(loss_unnormalized) # need to log the un-normalized loss. It will be normalized later in the log function
         
                        loss.backward()

            # clip gradients
            for model in model_parts:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), job_config.training.max_norm, foreach=True
                )

            # Aggregate and log signal propagation statistics (after all microbatches complete)
            if model_config.log_signal_propagation and microbatch_rms_values and microbatch_absmax_values:
                aggregated_metrics = {}

                # ============ Use all_reduce to aggregate 1 value at a time ============
                # Aggregate RMS type: sqrt(mean([rms^2 for rms in rms_list]))
                for key, rms_list in microbatch_rms_values.items():
                    if rms_list:
                        # Aggregated RMS on the current rank
                        mean_sq_rms = sum([r**2 for r in rms_list]) / len(rms_list)
                        local_rms = math.sqrt(mean_sq_rms)
                        # Aggregate across ranks
                        global_rms = aggregate_rms_across_ranks(local_rms, world_size, global_rank)

                        # Only rank 0 records
                        if global_rank == 0:
                            aggregated_metrics[key] = global_rms

                # Aggregate AbsMax type: max(all_absmax)
                for key, absmax_list in microbatch_absmax_values.items():
                    if absmax_list:
                        # Aggregated AbsMax on the current rank
                        local_absmax = max(absmax_list)
                        # Aggregate across ranks
                        global_absmax = aggregate_absmax_across_ranks(local_absmax, world_size, global_rank)

                        # Only rank 0 records
                        if global_rank == 0:
                            aggregated_metrics[key] = global_absmax
                # =============================================================

                # Log to SwanLab (rank 0 only)
                if global_rank == 0 and job_config.metrics.enable_swanlab and aggregated_metrics:
                    swanlab.log(aggregated_metrics, step=train_state.step)

                # Clear caches in preparation for the next step
                microbatch_rms_values.clear()
                microbatch_absmax_values.clear()

            # Record Grad Norm
            # NOTE: must run on all ranks — log_norms() calls DTensor.full_tensor()
            # which is an NCCL all_gather collective. Running on rank 0 only would
            # deadlock all other ranks. Internal swanlab.log is already gated on rank 0.
            if model_config.log_gradients:
                log_norms(model, train_state.step, global_rank, job_config)

            # optimizer step
            checkpoint.wait_for_staging()
            optimizers.step()
            if not parallel_dims.pp_enabled:
                update_model_sn_state(whole_model)
            lr_schedulers.step()

            # === Log current learning rate to SwanLab ===
            if global_rank == 0 and job_config.metrics.enable_swanlab:
                # Some optimizers have multiple param_groups; average them or log them separately here
                current_lrs = [group['lr'] for group in optimizers.optimizers[0].param_groups]
                # If you want to log multiple groups separately (e.g. different lr for different parts of the model)
                lr_logs = {f"lr/group_{i}": lr for i, lr in enumerate(current_lrs)}
                swanlab.log(lr_logs, step=train_state.step)

            # losses_since_last_log.append(loss)

            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = (
                    np.mean(losses),
                    np.max(losses),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh).item(),
                        dist_max(max_loss, dp_mesh).item(),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = timer() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = np.mean(data_loading_times)
                time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                # logger.info(
                #     f"optimizer: {job_config.optimizer.name}"
                #     f"{color.cyan}step: {train_state.step:2}  "
                #     f"{color.green}loss: {global_avg_loss:7.4f}  "
                #     f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                #     f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                #     f"{color.blue}wps: {round(wps):,}  "
                #     f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                # )

                logger.info(f"optimizer: {job_config.optimizer.name}, Currently running '{job_config.metrics.swanlab_comment}' ")
                logger.info(f"{color.cyan}step: {train_state.step} "
                            f"{color.green}loss: {global_avg_loss:7.4f} "
                            f"{color.blue}memory/max_active(GiB): {gpu_mem_stats.max_active_gib} " 
                            f"memory/max_active(%): {gpu_mem_stats.max_active_pct} "
                            f"memory/max_reserved(GiB): {gpu_mem_stats.max_reserved_gib} "
                            f"memory/max_reserved(%): {gpu_mem_stats.max_reserved_pct} "
                            f"wps: {round(wps):,}  " # this is throughput tokens per second
                            f"{color.magenta}mfu: {mfu:.2f}%{color.reset} "
                            f"total_time since last log (s): {time_end_to_end:7.4f} "
                            f"total_token trained: {ntokens_total_train  / 1e9} B " # assume 2 gpu
                            # f"parallel_dims.model_parallel_size: {parallel_dims.model_parallel_size} " This is 1
                        )
    
                
                if global_rank == 0 and job_config.metrics.enable_swanlab:
                    # Only log the actual pc_level when any precondition is enabled, otherwise log 0
                    pc_level_value = (
                        model_config.pc_level
                        if (model_config.precondition_ffn or model_config.precondition_o
                            or model_config.precondition_qk or model_config.precondition_v)
                        else 0
                    )
                    swanlab.log({
                        # "test": 1,
                        "step": train_state.step,
                        "loss": global_avg_loss,
#                         "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
#                         "memory/max_active(%)": gpu_mem_stats.max_active_pct,
#                         "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
#                         "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
#                         "wps": wps, # this is throughput tokens per second
#                         "mfu": mfu,
#                         "total_time since last log (s)": time_end_to_end,
                        "total_token trained (B)": ntokens_total_train / 1e9,
                        "pc_level": pc_level_value,
                        },
                        step = train_state.step)
            
                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = timer()
                gpu_memory_monitor.reset_peak_stats()
                
            if train_state.step == 1:
                checkpoint.save(train_state.step, force=True)
                logger.info(f"Saved checkpoint after first step into {job_config.checkpoint.folder}")

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            ) # save in the end?

            # signals the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()

            if memory_profiler:
                memory_profiler.step()

            # Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()

    # Check for visualization mode
    if getattr(config.visualize, 'enable', False):
        from visualize import run_visualize
        run_visualize(config)
    else:
        main(config)
        destroy_process_group()
