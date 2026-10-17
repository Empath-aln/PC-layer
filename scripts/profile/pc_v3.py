"""PC v3: selectable PC checkpointing on top of v2's fusion and accumulation.

none saves v2's tensors; polynomial recomputes Horner stages; full recomputes
Gram, Horner stages, and the pre-gamma output. All recomputation is inside the
first-order VJP, preserving the original dtype rounding and detached norm
recovery. This is operator-local checkpointing, without activation checkpointing
or training-loop changes. Kernels, SN helpers, and accumulation are copied from
v2; this file does not import implementation code from previous versions.
"""

import warnings
from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import Replicate
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _affine_diagonal_kernel(A, OUT, D: tl.constexpr, SCALE: tl.constexpr,
                                DIAGONAL: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(A + offsets, offsets < D * D, other=0).to(tl.float32)
        dtype = OUT.dtype.element_ty
        value = (value * SCALE).to(dtype).to(tl.float32)
        diagonal = tl.full((), DIAGONAL, tl.float32).to(dtype).to(tl.float32)
        value += tl.where(offsets // D == offsets % D, diagonal, 0.0)
        tl.store(OUT + offsets, value, offsets < D * D)

    @triton.jit
    def _mm_diagonal_kernel(A, B, OUT, D: tl.constexpr, DIAGONAL: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # Horner's square matrices are contiguous. Keep the baseline's BF16/FP32
        # rounding between GEMM and diagonal addition, without a global write.
        pid = tl.program_id(0)
        rows = (pid // tl.cdiv(D, BN)) * BM + tl.arange(0, BM)
        cols = (pid % tl.cdiv(D, BN)) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(D, BK)):
            k = block * BK + ks
            left = tl.load(A + rows[:, None] * D + k[None, :],
                           (rows[:, None] < D) & (k[None, :] < D), other=0)
            right = tl.load(B + k[:, None] * D + cols[None, :],
                            (k[:, None] < D) & (cols[None, :] < D), other=0)
            accumulator = tl.dot(left, right, accumulator, input_precision="ieee")
        dtype = OUT.dtype.element_ty
        result = accumulator.to(dtype).to(tl.float32)
        diagonal = tl.full((), DIAGONAL, tl.float32).to(dtype).to(tl.float32)
        result += tl.where(rows[:, None] == cols[None, :], diagonal, 0.0)
        tl.store(OUT + rows[:, None] * D + cols[None, :], result,
                 (rows[:, None] < D) & (cols[None, :] < D))

    @triton.jit
    def _finish_kernel(VALUE, NORM, GAMMA, OUT, N: tl.constexpr,
                       RECOVER: tl.constexpr, HAS_GAMMA: tl.constexpr,
                       SAVE_GAMMA_INPUT: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        value = tl.load(VALUE + offsets, mask, other=0).to(tl.float32)
        dtype = VALUE.dtype.element_ty
        if RECOVER:
            norm = tl.load(NORM).to(tl.float32)
            value = (value * norm).to(dtype).to(tl.float32)
        if SAVE_GAMMA_INPUT:
            tl.store(VALUE + offsets, value, mask)
        if HAS_GAMMA:
            value *= tl.load(GAMMA).to(tl.float32)
        tl.store(OUT + offsets, value, mask)

    @triton.jit
    def _scale_backward_kernel(GRAD, GAMMA_INPUT, NORM, GAMMA, OUT, PARTIAL,
                               N: tl.constexpr, RECOVER: tl.constexpr,
                               HAS_GAMMA: tl.constexpr, GAMMA_GRAD: tl.constexpr,
                               BLOCK: tl.constexpr):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        grad = tl.load(GRAD + offsets, mask, other=0).to(tl.float32)
        dtype = GRAD.dtype.element_ty
        if GAMMA_GRAD:
            gamma_input = tl.load(GAMMA_INPUT + offsets, mask, other=0).to(tl.float32)
            product = (grad * gamma_input).to(dtype).to(tl.float32)
            tl.store(PARTIAL + block, tl.sum(product, 0))
        if HAS_GAMMA:
            grad = (grad * tl.load(GAMMA).to(tl.float32)).to(dtype).to(tl.float32)
        if RECOVER:
            grad *= tl.load(NORM).to(tl.float32)
        tl.store(OUT + offsets, grad, mask)

    @triton.jit
    def _normalize_backward_kernel(GRAD, NORMALIZED, NORM, OUT, PARTIAL,
                                   N: tl.constexpr, NORM_GRAD: tl.constexpr,
                                   WEIGHT_GRAD: tl.constexpr, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        grad = tl.load(GRAD + offsets, mask, other=0).to(tl.float32)
        norm = tl.load(NORM).to(tl.float32)
        dtype = GRAD.dtype.element_ty
        if NORM_GRAD:
            normalized = tl.load(NORMALIZED + offsets, mask, other=0).to(tl.float32)
            ratio = (normalized / norm).to(dtype).to(tl.float32)
            product = (-grad * ratio).to(dtype).to(tl.float32)
            tl.store(PARTIAL + block, tl.sum(product, 0))
        if WEIGHT_GRAD:
            tl.store(OUT + offsets, grad / norm, mask)


def _use_cuda_kernels(tensor):
    if not tensor.is_cuda or tensor.dtype not in (torch.float32, torch.bfloat16):
        return False
    if triton is None:
        raise RuntimeError("pc_v3 on CUDA requires Triton; PyTorch 2.7 Linux wheels include triton==3.3.0")
    return True


COEFFICIENTS = (
    (1.0,),
    (1.507, -0.507),
    (2.083, -1.643, 0.560),
    (2.909, -4.649, 4.023, -1.283),
    (3.625, -9.261, 14.097, -10.351, 2.890),
)
CHECKPOINT_MODES = ("none", "polynomial", "full")


def _affine_diagonal(matrix, scale, diagonal):
    """Round scale*A and c*I separately, matching the baseline's dtype."""
    if _use_cuda_kernels(matrix):
        result = torch.empty_like(matrix)
        _affine_diagonal_kernel[(triton.cdiv(matrix.numel(), 1024),)](
            matrix, result, matrix.size(0), scale, diagonal, 1024, enable_fp_fusion=False,
        )
        return result
    result = matrix * scale
    result.diagonal().add_(torch.as_tensor(diagonal, device=matrix.device, dtype=matrix.dtype))
    return result


def _mm_diagonal(left, right, diagonal):
    if _use_cuda_kernels(left):
        result = torch.empty_like(left)
        d = left.size(0)
        _mm_diagonal_kernel[(triton.cdiv(d, 64) * triton.cdiv(d, 64),)](
            left, right, result, d, diagonal, 64, 64, 32, num_warps=4,
        )
        return result
    result = torch.mm(left, right)
    result.diagonal().add_(torch.as_tensor(diagonal, device=left.device, dtype=left.dtype))
    return result


def _finish_transform(value, norm, gamma, recover, save_gamma_input):
    if _use_cuda_kernels(value):
        if gamma is None and not recover:
            return value, None
        output = torch.empty_like(value) if save_gamma_input else value
        _finish_kernel[(triton.cdiv(value.numel(), 1024),)](
            value, norm, gamma if gamma is not None else norm, output, value.numel(),
            recover, gamma is not None, save_gamma_input, 1024,
        )
        return output, value if save_gamma_input else None
    if recover:
        value.mul_(norm)
    if gamma is None:
        return value, None
    if save_gamma_input:
        return value * gamma, value
    return value.mul_(gamma), None


def _scale_backward(grad_output, gamma_input, norm, gamma, recover, needs_gamma_grad):
    if _use_cuda_kernels(grad_output):
        if gamma is None and not recover:
            return grad_output, None
        # Expanded gradients (e.g. output.sum().backward()) are not contiguous.
        grad_output = grad_output.contiguous()
        grad = torch.empty_like(grad_output)
        blocks = triton.cdiv(grad.numel(), 1024)
        partial = torch.empty(blocks, device=grad.device, dtype=torch.float32) if needs_gamma_grad else norm
        _scale_backward_kernel[(blocks,)](
            grad_output, gamma_input if gamma_input is not None else norm, norm,
            gamma if gamma is not None else norm, grad, partial, grad.numel(),
            recover, gamma is not None, needs_gamma_grad, 1024,
        )
        grad_gamma = partial.sum().to(gamma.dtype).reshape_as(gamma) if needs_gamma_grad else None
        return grad, grad_gamma
    grad_gamma = (grad_output * gamma_input).sum().reshape_as(gamma) if needs_gamma_grad else None
    grad = grad_output * gamma if gamma is not None else grad_output
    if recover:
        grad = grad * norm
    return grad, grad_gamma


def _normalize_backward(grad, normalized, norm, needs_weight_grad, needs_norm_grad):
    if _use_cuda_kernels(grad):
        grad = grad.contiguous()
        normalized = normalized.contiguous()
        grad_weight = torch.empty_like(grad) if needs_weight_grad else norm
        blocks = triton.cdiv(grad.numel(), 1024)
        partial = torch.empty(blocks, device=grad.device, dtype=torch.float32) if needs_norm_grad else norm
        _normalize_backward_kernel[(blocks,)](
            grad, normalized, norm, grad_weight, partial, grad.numel(),
            needs_norm_grad, needs_weight_grad, 1024,
        )
        grad_norm = partial.sum().to(norm.dtype).reshape_as(norm) if needs_norm_grad else None
        return grad_weight if needs_weight_grad else None, grad_norm
    grad_norm = -(grad * (normalized / norm)).sum().reshape_as(norm) if needs_norm_grad else None
    grad_weight = grad / norm if needs_weight_grad else None
    return grad_weight, grad_norm


def _polynomial(normalized, level, tall, gram=None, collect_stages=True, keep_first=True):
    if gram is None:
        gram = torch.mm(normalized.T, normalized) if tall else torch.mm(normalized, normalized.T)
    coefficients = COEFFICIENTS[level]
    first = _affine_diagonal(gram, coefficients[-1], coefficients[-2])
    polynomial, stages = first, []
    if not keep_first:
        first = None
    for coefficient in reversed(coefficients[:-2]):
        polynomial = _mm_diagonal(gram, polynomial, coefficient)
        if collect_stages:
            stages.append(polynomial)
    return gram, first, stages, polynomial


class _PCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, norm, gamma, level, recover, checkpoint):
        normalized = (weight / norm).contiguous()
        tall = weight.size(0) >= weight.size(1)
        gram, stages = None, []
        if level:
            gram, first, stages, polynomial = _polynomial(
                normalized, level, tall, collect_stages=checkpoint == "none", keep_first=False,
            )
            del first
            value = torch.mm(normalized, polynomial) if tall else torch.mm(polynomial, normalized)
        else:
            value = normalized
        del normalized
        output, gamma_input = _finish_transform(
            value, norm, gamma, recover,
            gamma is not None and ctx.needs_input_grad[2] and checkpoint != "full",
        )
        ctx.level, ctx.recover, ctx.tall, ctx.checkpoint = level, recover, tall, checkpoint
        needs_pc_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        needs_recompute_gamma = checkpoint == "full" and gamma is not None and ctx.needs_input_grad[2]
        ctx.save_for_backward(
            weight if needs_pc_grad or needs_recompute_gamma else None, norm, gamma,
            gamma_input if checkpoint != "full" else None,
            gram if needs_pc_grad and checkpoint != "full" else None,
            *(stages if needs_pc_grad and checkpoint == "none" else []),
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        weight, norm, gamma, gamma_input, gram, *stages = ctx.saved_tensors
        needs_pc_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        needs_gamma_grad = gamma is not None and ctx.needs_input_grad[2]
        normalized = None
        if needs_pc_grad or (ctx.checkpoint == "full" and needs_gamma_grad):
            normalized = (weight / norm).contiguous()
            if ctx.level:
                coefficients = COEFFICIENTS[ctx.level]
                if ctx.checkpoint == "none":
                    first = _affine_diagonal(gram, coefficients[-1], coefficients[-2])
                    polynomial = stages[-1] if stages else first
                else:
                    gram, first, stages, polynomial = _polynomial(
                        normalized, ctx.level, ctx.tall, gram, collect_stages=needs_pc_grad,
                    )
        if ctx.checkpoint == "full" and needs_gamma_grad:
            if ctx.level:
                value = torch.mm(normalized, polynomial) if ctx.tall else torch.mm(polynomial, normalized)
            else:
                # Recovery is in-place: do not overwrite X needed by norm VJP.
                value = normalized.clone() if ctx.recover else normalized
            gamma_input, _ = _finish_transform(value, norm, None, ctx.recover, False)
            del value
        # Norm recovery deliberately has no derivative with respect to norm.
        grad, grad_gamma = _scale_backward(
            grad_output, gamma_input, norm, gamma, ctx.recover, ctx.needs_input_grad[2],
        )
        del gamma_input
        if not needs_pc_grad:
            return None, None, grad_gamma, None, None, None
        if ctx.level:
            if ctx.tall:
                grad_normalized = torch.mm(grad, polynomial.T)
                grad_polynomial = torch.mm(normalized.T, grad)
            else:
                grad_normalized = torch.mm(polynomial.T, grad)
                grad_polynomial = torch.mm(grad, normalized.T)
            del grad
            grad_gram = None
            for previous in reversed([first, *stages[:-1]] if stages else []):
                contribution = torch.mm(grad_polynomial, previous.T)
                if grad_gram is None:
                    grad_gram = contribution
                else:
                    grad_gram.add_(contribution)
                del contribution
                grad_polynomial = torch.mm(gram.T, grad_polynomial)
            contribution = grad_polynomial * coefficients[-1]
            if grad_gram is None:
                grad_gram = contribution
            else:
                grad_gram.add_(contribution)
            if stages:
                del previous
            del contribution, grad_polynomial, first, polynomial
            # Preserve both Gram input contributions (no symmetry approximation).
            if ctx.tall:
                grad_normalized.add_(torch.mm(normalized, grad_gram.T))
                grad_normalized.add_(torch.mm(normalized, grad_gram))
            else:
                grad_normalized.add_(torch.mm(grad_gram, normalized))
                grad_normalized.add_(torch.mm(grad_gram.T, normalized))
            del grad_gram
        else:
            grad_normalized = grad
        grad_weight, grad_norm = _normalize_backward(
            grad_normalized, normalized, norm, ctx.needs_input_grad[0], ctx.needs_input_grad[1],
        )
        return grad_weight, grad_norm, grad_gamma, None, None, None


class LearnableGamma(nn.Module):
    def __init__(self, shape=(1,)):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(*shape))

    @torch.no_grad()
    def reset_parameters(self, init_gamma: float):
        if self.v is None or self.v.is_meta:
            return
        self.v.fill_(float(init_gamma))

    def value(self) -> torch.Tensor:
        return self.v


class PCTransform(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

    def forward(self, weight, gamma=None, sn_norm=None, return_norm=False):
        config = self.model_config
        if isinstance(weight, DTensor):
            raise ValueError("pc_v3 currently supports single-device tensors, not DTensor")
        if weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("pc_v3 requires a real floating-point matrix")
        if not 0 <= config.pc_level < len(COEFFICIENTS):
            raise ValueError(f"Unsupported PC level: {config.pc_level}")
        if config.pc_norm_type == "sn":
            if sn_norm is None:
                raise ValueError("pc_norm_type='sn' requires sn_norm")
            norm = sn_norm
        elif config.pc_norm_type == "none":
            if config.pc_level:
                warnings.warn("PC without normalization can be unstable", UserWarning)
            norm = weight.new_tensor(1.0)
        else:
            raise ValueError(f"Unknown pc_norm_type: {config.pc_norm_type}")
        if not config.learnable_gamma:
            gamma = None
        elif gamma is not None:
            gamma = gamma.to(dtype=weight.dtype, device=weight.device)
        if norm.numel() != 1 or (gamma is not None and gamma.numel() != 1):
            raise ValueError("pc_v3 requires scalar norm and gamma")
        checkpoint = getattr(config, "pc_checkpoint", "full")
        if checkpoint not in CHECKPOINT_MODES:
            raise ValueError(f"Unknown PC checkpoint mode: {checkpoint}")
        result = _PCFunction.apply(weight, norm, gamma, config.pc_level, config.recover_w_norm, checkpoint)
        return (result, norm) if return_norm else result


class PCLinear(nn.Module):
    def __init__(self, linear: nn.Linear, model_args, layer_id: int):
        super().__init__()
        self.linear = linear
        self.model_args = model_args
        self.layer_id = layer_id
        self.pc = PCTransform(model_args)

        if model_args.pc_norm_type == "sn":
            out_features, in_features = linear.weight.shape
            self.register_buffer("sn_u", torch.empty(out_features), persistent=True)
            self.register_buffer("sn_v", torch.empty(in_features), persistent=True)

        if model_args.learnable_gamma:
            self.gamma = LearnableGamma(shape=(1,))
        else:
            self.gamma = None

        # Safeguard: avoid the case where a meta parameter cannot be fill_'ed during __init__
        self._gamma_inited_after_materialize = False

        # Transient tensors are deliberately absent from state_dict. The leaf
        # shares storage with the transformed weight but disconnects its graph
        # from microbatch backward; its .grad collects the upstream gradients.
        self._accumulation_active = False
        self._transformed_weight = None
        self._weight_leaf = None

    @torch.no_grad()
    def _maybe_init_gamma(self):
        if self.gamma is None or self._gamma_inited_after_materialize:
            return
        v = self.gamma.v
        if v is not None and (not v.is_meta):
            self.gamma.reset_parameters(self.model_args.gamma_init_value)
            self._gamma_inited_after_materialize = True

    # ── Spectral-norm helpers ──────────────────────────────────────────

    def _uses_sn_norm(self):
        return self.model_args.pc_norm_type == "sn"

    def _normalize_vector(self, vec):
        return vec / (vec.norm() + self.model_args.pc_norm_eps)

    @torch.no_grad()
    def update_sn_state(self):
        if not self._uses_sn_norm():
            return
        weight = self.linear.weight
        sn_weight = self._get_weight_for_sn(weight)
        self._initialize_sn_state_if_needed(sn_weight)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            u, v = self._compute_updated_sn_state(sn_weight)
            self.sn_u.copy_(u)
            self.sn_v.copy_(v)
        self._broadcast_sn_state()

    @torch.no_grad()
    def _broadcast_sn_state(self):
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.sn_u, src=0)
            dist.broadcast(self.sn_v, src=0)

    def _get_weight_for_sn(self, weight):
        if isinstance(weight, DTensor):
            if all(isinstance(p, Replicate) for p in weight.placements):
                return weight.to_local()
            return weight.full_tensor()
        return weight

    def _wrap_scalar_like_weight(self, scalar, weight):
        if isinstance(weight, DTensor):
            return DTensor.from_local(
                scalar,
                device_mesh=weight.device_mesh,
                placements=[Replicate()],
                run_check=False,
            )
        return scalar

    @torch.no_grad()
    def _random_unit_vector(self, size, weight):
        vec = torch.randn(size, device=weight.device, dtype=weight.dtype)
        return self._normalize_vector(vec)

    def _has_valid_sn_state(self, weight):
        # dtype is intentionally excluded: sn_u/sn_v are stored in the update
        # dtype (e.g. float32) but forward may see a cast weight (e.g. bfloat16
        # from FSDP mixed-precision).  Casting happens lazily in
        # _compute_sn_norm_from_state.
        if (
            self.sn_u.numel() != weight.size(0)
            or self.sn_v.numel() != weight.size(1)
            or self.sn_u.device != weight.device
            or self.sn_v.device != weight.device
        ):
            return False
        # After meta-device materialization, buffers may contain garbage
        # (NaN/Inf/zero). Detect this so _initialize_sn_state_if_needed
        # properly reinitializes them.
        if (
            not torch.isfinite(self.sn_u).all()
            or not torch.isfinite(self.sn_v).all()
            or self.sn_u.norm() == 0
            or self.sn_v.norm() == 0
        ):
            return False
        return True

    @torch.no_grad()
    def _initialize_sn_state_if_needed(self, weight):
        if not self._uses_sn_norm() or self._has_valid_sn_state(weight):
            return
        self.sn_u = self._random_unit_vector(weight.size(0), weight)
        self.sn_v = self._random_unit_vector(weight.size(1), weight)

    def _ensure_sn_state(self, weight):
        if self._uses_sn_norm() and not self._has_valid_sn_state(weight):
            raise RuntimeError(
                f"Spectral-norm state is not initialized for layer {self.layer_id}. "
                "Call update_sn_state() or update_model_sn_state() before forward."
            )

    @torch.no_grad()
    def _compute_updated_sn_state(self, weight):
        # Power iteration, warm-started from the persistent sn_u/sn_v buffers.
        # The buffers are random-initialized once (_initialize_sn_state_if_needed)
        # and refined in place on every update.
        u = self.sn_u
        v = self.sn_v
        for _ in range(self.model_args.power_iter):
            v = self._normalize_vector(torch.mv(weight.T, u))
            u = self._normalize_vector(torch.mv(weight, v))
        return u, v

    def _compute_sn_norm_from_state(self, weight):
        sn_weight = self._get_weight_for_sn(weight)
        self._ensure_sn_state(sn_weight)
        # Cast sn_u/sn_v to match the forward weight dtype (e.g. bfloat16 under
        # FSDP mixed precision) so torch.mv/dot don't error on dtype mismatch.
        sn_u = self.sn_u.to(dtype=sn_weight.dtype)
        sn_v = self.sn_v.to(dtype=sn_weight.dtype)
        wv = torch.mv(sn_weight, sn_v)
        W_norm_local = torch.dot(sn_u, wv) + self.model_args.pc_norm_eps
        return self._wrap_scalar_like_weight(W_norm_local, weight)

    # ── forward ───────────────────────────────────────────────────────

    def _transform_weight(self):
        self._maybe_init_gamma()
        g = self.gamma.value() if self.gamma is not None else None
        if self._uses_sn_norm():
            sn_norm = self._compute_sn_norm_from_state(self.linear.weight)
            w = self.pc(self.linear.weight, gamma=g, sn_norm=sn_norm)
        else:
            w = self.pc(self.linear.weight, gamma=g)
        return w

    def forward(self, x):
        if not self._accumulation_active or not torch.is_grad_enabled():
            return F.linear(x, self._transform_weight(), self.linear.bias)
        if self._weight_leaf is None:
            self._transformed_weight = self._transform_weight()
            self._weight_leaf = self._transformed_weight.detach().requires_grad_(
                self._transformed_weight.requires_grad
            )
        return F.linear(x, self._weight_leaf, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# ── Module-level utilities ────────────────────────────────────────────

def iter_pc_linear_modules(module):
    for submodule in module.modules():
        if isinstance(submodule, PCLinear):
            yield submodule


def model_uses_sn_norm(module):
    return any(submodule._uses_sn_norm() for submodule in iter_pc_linear_modules(module))


@torch.no_grad()
def update_model_sn_state(module):
    for submodule in iter_pc_linear_modules(module):
        if submodule._uses_sn_norm():
            submodule.update_sn_state()


@contextmanager
def accumulation_context(model, grad_accum_steps):
    """Share PC within one step, completing parameter gradients before exit.

    Keep weights, gamma, and SN buffers unchanged throughout this context. Call
    backward on each scaled microbatch loss, then update SN/optimizer only after
    leaving it. This is a single-device, first-order profiling implementation;
    it does not integrate DDP/FSDP reducers, higher-order grads, or GradScaler.
    Outside this context (including G=1), there is no accumulation cache;
    v3's fused transform, checkpointing, and custom VJP still apply.
    """
    if grad_accum_steps == 1:
        yield
        return
    layers = tuple(iter_pc_linear_modules(model))
    if any(layer._accumulation_active for layer in layers):
        raise RuntimeError("PC accumulation contexts cannot be nested")
    try:
        for layer in layers:
            layer._accumulation_active = True
        yield
        outputs, gradients = [], []
        for layer in layers:
            leaf = layer._weight_leaf
            if leaf is not None and leaf.grad is not None:
                outputs.append(layer._transformed_weight)
                gradients.append(leaf.grad)
        if outputs:
            # Linearity of this VJP allows all microbatch contributions to pass
            # through PC once, including sigma and learnable-gamma derivatives.
            torch.autograd.backward(outputs, gradients)
    finally:
        # Also release graphs if forward/backward raises or a case runs OOM.
        for layer in layers:
            layer._accumulation_active = False
            if layer._weight_leaf is not None:
                layer._weight_leaf.grad = None
            layer._transformed_weight = None
            layer._weight_leaf = None
