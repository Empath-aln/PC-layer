import warnings
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import Replicate


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
        return self.apply_preconditioner(weight=weight, model_config=self.model_config, gamma=gamma, sn_norm=sn_norm, return_norm=return_norm)

    def apply_preconditioner(self, weight=None, model_config=None, gamma=None, sn_norm=None, return_norm=False):
        W_normalized, W_norm = self.pc_normalize(weight=weight, model_config=model_config, sn_norm=sn_norm)
        r, c = W_normalized.shape

        if r >= c:
            W_preconditioned = self.preconditionertall(
                weight=W_normalized, model_config=model_config
            )
        else:
            W_preconditioned = self.preconditionerwide(
                weight=W_normalized, model_config=model_config
            )

        if model_config.recover_w_norm:
            norm_for_recover = W_norm.detach()
            W_preconditioned = W_preconditioned * norm_for_recover
        if model_config.learnable_gamma and gamma is not None:
            gamma = gamma.to(dtype=W_preconditioned.dtype, device=W_preconditioned.device)
            W_preconditioned = W_preconditioned * gamma

        if return_norm:
            return W_preconditioned, W_norm
        return W_preconditioned

    def pc_normalize(self, weight=None, model_config=None, sn_norm=None):
        if weight.ndim != 2:
            raise ValueError("Weight must be a 2D tensor")

        if model_config.pc_norm_type == 'none':
            if model_config.pc_level != 0:
                warnings.warn(
                    "pc_norm_type is None but pc_level != 0: weight is not normalized before applying preconditioner. "
                    "This may lead to unexpected behavior.",
                    UserWarning
                )
            W_norm = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)

        elif model_config.pc_norm_type == "sn":
            if sn_norm is None:
                raise ValueError(
                    "pc_norm_type='sn' requires sn_norm to be pre-computed by PCLinear."
                )
            W_norm = sn_norm

        else:
            raise ValueError(f"Unknown pc_norm_type: {model_config.pc_norm_type}")

        normalized_weight = weight / W_norm
        return normalized_weight, W_norm

    def preconditionertall(self, weight=None, model_config=None):
        pc_level = model_config.pc_level
        if pc_level == 0:
            return weight

        _, c = weight.shape
        I = torch.eye(c, device=weight.device, dtype=weight.dtype)
        wtw = weight.t().mm(weight)

        if pc_level == 1:
            weight = weight.mm(1.507 * I - 0.507 * wtw)
        elif pc_level == 2:
            weight = weight.mm(2.083 * I + wtw.mm(-1.643 * I + 0.560 * wtw))
        elif pc_level == 3:
            weight = weight.mm(2.909 * I + wtw.mm(-4.649 * I + wtw.mm(4.023 * I - 1.283 * wtw)))
        elif pc_level == 4:
            weight = weight.mm(3.625 * I + wtw.mm(-9.261 * I + wtw.mm(14.097 * I + wtw.mm(-10.351 * I + 2.890 * wtw))))
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def preconditionerwide(self, weight=None, model_config=None):
        pc_level = model_config.pc_level
        if pc_level == 0:
            return weight

        r, _ = weight.shape
        I = torch.eye(r, device=weight.device, dtype=weight.dtype)
        wwt = weight.mm(weight.t())

        if pc_level == 1:
            weight = (1.507 * I - 0.507 * wwt).mm(weight)
        elif pc_level == 2:
            weight = (2.083 * I + wwt.mm(-1.643 * I + 0.560 * wwt)).mm(weight)
        elif pc_level == 3:
            weight = (2.909 * I + wwt.mm(-4.649 * I + wwt.mm(4.023 * I - 1.283 * wwt))).mm(weight)
        elif pc_level == 4:
            weight = (3.625 * I + wwt.mm(-9.261 * I + wwt.mm(14.097 * I + wwt.mm(-10.351 * I + 2.890 * wwt)))).mm(weight)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight


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

    def forward(self, x):
        self._maybe_init_gamma()
        g = self.gamma.value() if self.gamma is not None else None
        if self._uses_sn_norm():
            sn_norm = self._compute_sn_norm_from_state(self.linear.weight)
            w = self.pc(self.linear.weight, gamma=g, sn_norm=sn_norm)
        else:
            w = self.pc(self.linear.weight, gamma=g)
        return F.linear(x, w, self.linear.bias)

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

