from __future__ import annotations

import importlib.util
import json
import random
from dataclasses import dataclass
from typing import Any, Callable

import torch

from optimizers.constraint_consensus_optimizer import ConstraintConsensusOptimizer, PhysicsRecoveryOptimizer
from optimizers.diffusion_adam import DiffusionAdam
from optimizers.direction_recovery_optimizer import DirectionRecoveryOptimizer, RecoveryDirectionOptimizer
from optimizers.coherent_momentum_real_baseline import (
    CoherentMomentumAdaptiveMassBaseline,
    CoherentMomentumPhysicalBaseline,
    CoherentMomentumRealBaseline,
)
from optimizers.coherent_direction_reference import CoherentDirectionReferenceOptimizer
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer
from optimizers.coherent_momentum_optimizer_improved import CoherentMomentumOptimizerImproved
from optimizers.observation_recovery_optimizer import ObservationRecoveryOptimizer
from optimizers.sds_adam import SDSAdam
from optimizers.thermodynamic_adam import ThermodynamicAdam
from optimizers.topological_adam import BaselineTopologicalAdam
from optimizers.unified_physics_adam import UnifiedPhysicsAdam
from optimizers.uncertainty_adam import QuantumUncertaintyAdam


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()
                param.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class AdaBelief(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_var"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]
                state["step"] += 1
                step = state["step"]

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                m_hat = exp_avg / bias_correction1
                s_hat = exp_avg_var / bias_correction2
                update = m_hat / (s_hat.sqrt() + eps)
                param.add_(update, alpha=-lr)
        return loss


class SAMOptimizer(torch.optim.Optimizer):
    wants_step_closure = True

    def __init__(
        self,
        params,
        base_optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
        **base_optimizer_kwargs: Any,
    ) -> None:
        param_list = list(params)
        defaults = dict(rho=rho, adaptive=adaptive, eps=eps, **base_optimizer_kwargs)
        super().__init__(param_list, defaults)
        self.base_optimizer = base_optimizer_cls(param_list, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        base_state = self.base_optimizer.state_dict()
        return {
            "base_optimizer": base_state,
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict):
        self.defaults.update(state_dict.get("defaults", {}))
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.state.update(state_dict.get("state", {}))
        self.param_groups = self.base_optimizer.param_groups
        return self

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms: list[torch.Tensor] = []
        shared_device = None
        for group in self.param_groups:
            adaptive = bool(group.get("adaptive", self.defaults["adaptive"]))
            for param in group["params"]:
                if param.grad is None:
                    continue
                shared_device = param.device
                if adaptive:
                    norms.append((param.abs() * param.grad).norm(p=2))
                else:
                    norms.append(param.grad.norm(p=2))
        if not norms:
            device = shared_device or torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _first_step(self) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = float(group.get("rho", self.defaults["rho"]))
            adaptive = bool(group.get("adaptive", self.defaults["adaptive"]))
            eps = float(group.get("eps", self.defaults["eps"]))
            scale = rho / (grad_norm + eps)
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if adaptive:
                    e_w = param.grad * scale * param.abs().clamp_min(eps)
                else:
                    e_w = param.grad * scale
                param.add_(e_w)
                state["e_w"] = e_w

    @torch.no_grad()
    def _second_step(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                e_w = self.state[param].pop("e_w", None)
                if e_w is not None:
                    param.sub_(e_w)
        self.base_optimizer.step()

    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAMOptimizer requires a closure.")
        self._first_step()
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
        self._second_step()
        return loss


def _build_schedulefree_adamw(params, **kwargs):
    if importlib.util.find_spec("schedulefree") is None:
        raise RuntimeError(
            "schedulefree is not installed. Install it with `pip install schedulefree` to enable Schedule-Free AdamW."
        )
    from schedulefree import AdamWScheduleFree  # type: ignore

    return AdamWScheduleFree(params, **kwargs)


def optimizer_availability() -> dict[str, dict[str, str | bool]]:
    has_muon = hasattr(torch.optim, "Muon")
    has_schedulefree = importlib.util.find_spec("schedulefree") is not None
    return {
        "muon_hybrid": {
            "available": has_muon,
            "reason": "" if has_muon else "requires torch.optim.Muon in the installed PyTorch build",
        },
        "schedulefree_adamw": {
            "available": has_schedulefree,
            "reason": "" if has_schedulefree else "optional dependency `schedulefree` is not installed",
        },
        "sam_adamw": {"available": True, "reason": ""},
        "asam_adamw": {"available": True, "reason": ""},
        "adabelief": {"available": True, "reason": ""},
    }


def available_optimizer_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
    availability = optimizer_availability()
    available: list[str] = []
    skipped: dict[str, str] = {}
    for name in names:
        status = availability.get(name)
        if status is None or bool(status["available"]):
            available.append(name)
        else:
            skipped[name] = str(status["reason"])
    return available, skipped


class HybridMuon(torch.optim.Optimizer):
    """Practical Muon wrapper for mixed-parameter models.

    Muon only accepts 2D tensors, so this routes matrix parameters to Muon and
    non-matrix parameters to AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adamw_lr_scale: float = 0.3,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        param_list = list(params)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adamw_lr_scale=adamw_lr_scale,
            betas=betas,
        )
        super().__init__(param_list, defaults)
        matrix_params = [param for param in param_list if param.requires_grad and param.ndim == 2]
        residual_params = [param for param in param_list if param.requires_grad and param.ndim != 2]
        self._muon = (
            torch.optim.Muon(
                matrix_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_coefficients=ns_coefficients,
                eps=eps,
                ns_steps=ns_steps,
            )
            if matrix_params
            else None
        )
        self._adamw = (
            torch.optim.AdamW(
                residual_params,
                lr=lr * adamw_lr_scale,
                weight_decay=weight_decay,
                betas=betas,
            )
            if residual_params
            else None
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def state_dict(self):
        return {
            "param_groups": self.param_groups,
            "defaults": self.defaults,
            "muon": None if self._muon is None else self._muon.state_dict(),
            "adamw": None if self._adamw is None else self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if self._muon is not None and state_dict.get("muon") is not None:
            self._muon.load_state_dict(state_dict["muon"])
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])
        self.defaults.update(state_dict.get("defaults", {}))
        return self


class LBFGSWithClosure(torch.optim.Optimizer):
    wants_step_closure = True

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 8,
        history_size: int = 20,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        param_list = list(params)
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )
        super().__init__(param_list, defaults)
        self._inner = torch.optim.LBFGS(
            param_list,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("LBFGSWithClosure requires a closure.")
        return self._inner.step(closure)

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)


class AdamWLBFGSHybrid(torch.optim.Optimizer):
    wants_step_closure = True

    def __init__(
        self,
        params,
        adamw_lr: float = 1e-3,
        lbfgs_lr: float = 0.8,
        weight_decay: float = 1e-4,
        warmup_steps: int = 48,
        history_size: int = 20,
        max_iter: int = 8,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        param_list = list(params)
        defaults = dict(
            adamw_lr=adamw_lr,
            lbfgs_lr=lbfgs_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            history_size=history_size,
            max_iter=max_iter,
            betas=betas,
        )
        super().__init__(param_list, defaults)
        self._adamw = torch.optim.AdamW(param_list, lr=adamw_lr, weight_decay=weight_decay, betas=betas)
        self._lbfgs = torch.optim.LBFGS(
            param_list,
            lr=lbfgs_lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )
        self._step_count = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._adamw.zero_grad(set_to_none=set_to_none)
        self._lbfgs.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self._step_count += 1
        warmup_steps = int(self.defaults["warmup_steps"])
        if self._step_count <= warmup_steps:
            if closure is not None:
                with torch.enable_grad():
                    closure()
            self._adamw.step()
            return None
        if closure is None:
            raise ValueError("AdamWLBFGSHybrid requires a closure after the AdamW warmup phase.")
        return self._lbfgs.step(closure)

    def state_dict(self):
        return {
            "step_count": self._step_count,
            "defaults": self.defaults,
            "adamw": self._adamw.state_dict(),
            "lbfgs": self._lbfgs.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._step_count = int(state_dict.get("step_count", 0))
        self.defaults.update(state_dict.get("defaults", {}))
        self._adamw.load_state_dict(state_dict["adamw"])
        self._lbfgs.load_state_dict(state_dict["lbfgs"])
        return self


@dataclass(slots=True)
class OptimizerSpec:
    name: str
    factory: Callable[..., torch.optim.Optimizer]
    default_params: dict[str, Any]
    search_space: dict[str, list[Any]]
    neutral_params: dict[str, Any] | None = None


def build_optimizer_registry() -> dict[str, OptimizerSpec]:
    return {
        "sgd": OptimizerSpec(
            "sgd",
            torch.optim.SGD,
            {"lr": 0.05, "weight_decay": 0.0},
            {"lr": [0.1, 0.05, 0.02, 0.01], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "sgd_momentum": OptimizerSpec(
            "sgd_momentum",
            torch.optim.SGD,
            {"lr": 0.03, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
            {
                "lr": [0.08, 0.05, 0.03, 0.015],
                "momentum": [0.85, 0.9, 0.95],
                "weight_decay": [0.0, 1e-4, 1e-3],
            },
        ),
        "rmsprop": OptimizerSpec(
            "rmsprop",
            torch.optim.RMSprop,
            {"lr": 0.005, "alpha": 0.99, "weight_decay": 0.0, "momentum": 0.0},
            {
                "lr": [0.01, 0.005, 0.002],
                "alpha": [0.95, 0.99],
                "weight_decay": [0.0, 1e-4, 1e-3],
            },
        ),
        "adam": OptimizerSpec(
            "adam",
            torch.optim.Adam,
            {"lr": 1e-3, "weight_decay": 0.0},
            {"lr": [3e-3, 1e-3, 5e-4], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "adamw": OptimizerSpec(
            "adamw",
            torch.optim.AdamW,
            {"lr": 1e-3, "weight_decay": 1e-4},
            {"lr": [3e-3, 1e-3, 5e-4], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "lbfgs": OptimizerSpec(
            "lbfgs",
            LBFGSWithClosure,
            {"lr": 0.8, "max_iter": 8, "history_size": 20},
            {
                "lr": [0.4, 0.8, 1.0],
                "max_iter": [4, 8, 12],
                "history_size": [10, 20],
            },
        ),
        "adamw_lbfgs_hybrid": OptimizerSpec(
            "adamw_lbfgs_hybrid",
            AdamWLBFGSHybrid,
            {
                "adamw_lr": 1e-3,
                "lbfgs_lr": 0.8,
                "weight_decay": 1e-4,
                "warmup_steps": 48,
                "history_size": 20,
                "max_iter": 8,
            },
            {
                "adamw_lr": [3e-3, 1e-3, 5e-4],
                "lbfgs_lr": [0.4, 0.8, 1.0],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "warmup_steps": [24, 48, 72],
                "max_iter": [4, 8],
                "history_size": [10, 20],
            },
        ),
        "nadam": OptimizerSpec(
            "nadam",
            torch.optim.NAdam,
            {"lr": 2e-3, "weight_decay": 0.0},
            {"lr": [3e-3, 2e-3, 1e-3], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "radam": OptimizerSpec(
            "radam",
            torch.optim.RAdam,
            {"lr": 1e-3, "weight_decay": 0.0},
            {"lr": [3e-3, 1e-3, 5e-4], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "lion": OptimizerSpec(
            "lion",
            Lion,
            {"lr": 5e-4, "weight_decay": 0.0},
            {"lr": [1e-3, 5e-4, 2e-4], "weight_decay": [0.0, 1e-4, 1e-3]},
        ),
        "adabelief": OptimizerSpec(
            "adabelief",
            AdaBelief,
            {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8},
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
            },
        ),
        "muon_hybrid": OptimizerSpec(
            "muon_hybrid",
            HybridMuon,
            {"lr": 2e-3, "weight_decay": 0.05, "momentum": 0.95, "adamw_lr_scale": 0.3},
            {
                "lr": [3e-3, 2e-3, 1e-3],
                "weight_decay": [0.01, 0.05, 0.1],
                "momentum": [0.9, 0.95],
                "adamw_lr_scale": [0.2, 0.3, 0.5],
            },
        ),
        "sam_adamw": OptimizerSpec(
            "sam_adamw",
            SAMOptimizer,
            {"base_optimizer_cls": torch.optim.AdamW, "lr": 1e-3, "weight_decay": 1e-4, "rho": 0.05, "adaptive": False},
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "rho": [0.02, 0.05, 0.1],
            },
        ),
        "asam_adamw": OptimizerSpec(
            "asam_adamw",
            SAMOptimizer,
            {"base_optimizer_cls": torch.optim.AdamW, "lr": 1e-3, "weight_decay": 1e-4, "rho": 0.05, "adaptive": True},
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "rho": [0.02, 0.05, 0.1],
            },
        ),
        "schedulefree_adamw": OptimizerSpec(
            "schedulefree_adamw",
            _build_schedulefree_adamw,
            {"lr": 2.5e-3, "weight_decay": 1e-4},
            {
                "lr": [3e-3, 2.5e-3, 1e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
            },
        ),
        "topological_adam": OptimizerSpec(
            "topological_adam",
            BaselineTopologicalAdam,
            {
                "lr": 1e-3,
                "eta": 0.03,
                "w_topo": 0.08,
                "target_energy": 1e-3,
                "max_topo_ratio": 0.05,
                "deterministic_init": True,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "eta": [0.02, 0.03, 0.05],
                "w_topo": [0.05, 0.08, 0.12],
                "target_energy": [5e-4, 1e-3, 2e-3],
                "max_topo_ratio": [0.02, 0.05, 0.1],
            },
        ),
        "sds_adam": OptimizerSpec(
            "sds_adam",
            SDSAdam,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "inner_horizon": 5e-4,
                "outer_horizon": 2.5e-2,
                "horizon_sharpness": 12.0,
                "cooling_strength": 0.35,
                "reheating_strength": 0.15,
                "entropy_weight": 0.1,
                "min_scale": 0.6,
                "max_scale": 1.4,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "inner_horizon": [1e-4, 5e-4, 1e-3],
                "outer_horizon": [1e-2, 2.5e-2, 5e-2],
                "cooling_strength": [0.15, 0.35, 0.6],
                "reheating_strength": [0.05, 0.15, 0.3],
                "entropy_weight": [0.0, 0.1, 0.2],
            },
            {
                "inner_horizon": 5e-4,
                "outer_horizon": 2.5e-2,
                "horizon_sharpness": 12.0,
                "cooling_strength": 0.0,
                "reheating_strength": 0.0,
                "entropy_weight": 0.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
            },
        ),
        "coherent_direction_reference": OptimizerSpec(
            "coherent_direction_reference",
            CoherentDirectionReferenceOptimizer,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "alignment_strength": 0.15,
                "damping_strength": 0.2,
                "coherence_strength": 0.15,
                "rotation_penalty": 0.25,
                "field_clip": 2.0,
                "min_scale": 0.6,
                "max_scale": 1.4,
                "layerwise_mode": True,
                "global_mode": True,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "alignment_strength": [0.0, 0.15, 0.3],
                "damping_strength": [0.1, 0.2, 0.35],
                "coherence_strength": [0.0, 0.15, 0.3],
                "rotation_penalty": [0.1, 0.25, 0.4],
                "layerwise_mode": [True, False],
                "global_mode": [True, False],
            },
            {
                "alignment_strength": 0.0,
                "damping_strength": 0.0,
                "coherence_strength": 0.0,
                "rotation_penalty": 0.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
                "layerwise_mode": True,
                "global_mode": True,
            },
        ),
        "thermodynamic_adam": OptimizerSpec(
            "thermodynamic_adam",
            ThermodynamicAdam,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "eps": 1e-8,
                "entropy_weight": 0.15,
                "energy_weight": 0.25,
                "temperature_decay": 0.96,
                "cooling_strength": 0.12,
                "reheating_strength": 0.12,
                "max_temperature": 1.5,
                "min_temperature": 0.05,
                "min_scale": 0.55,
                "max_scale": 1.35,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "entropy_weight": [0.0, 0.15, 0.3],
                "energy_weight": [0.1, 0.25, 0.5],
                "cooling_strength": [0.0, 0.12, 0.25],
                "reheating_strength": [0.0, 0.12, 0.25],
                "max_temperature": [1.0, 1.5, 2.0],
            },
            {
                "entropy_weight": 0.0,
                "energy_weight": 0.0,
                "cooling_strength": 0.0,
                "reheating_strength": 0.0,
                "min_temperature": 1.0,
                "max_temperature": 1.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
            },
        ),
        "diffusion_adam": OptimizerSpec(
            "diffusion_adam",
            DiffusionAdam,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "diffusion_strength": 0.04,
                "diffusion_decay": 0.98,
                "entropy_scaled_noise": True,
                "stagnation_trigger": 8,
                "min_noise": 0.0,
                "max_noise": 0.25,
                "noise_to_update_cap": 0.35,
                "aligned_noise_weight": 0.25,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "diffusion_strength": [0.02, 0.05, 0.1],
                "diffusion_decay": [0.95, 0.98, 0.995],
                "entropy_scaled_noise": [False, True],
                "max_noise": [0.15, 0.25, 0.35],
                "noise_to_update_cap": [0.15, 0.35, 0.5],
                "aligned_noise_weight": [0.0, 0.25, 0.5],
            },
            {
                "diffusion_strength": 0.0,
                "diffusion_decay": 1.0,
                "entropy_scaled_noise": False,
                "stagnation_trigger": 9999,
                "min_noise": 0.0,
                "max_noise": 0.0,
                "noise_to_update_cap": 0.0,
                "aligned_noise_weight": 0.0,
            },
        ),
        "coherent_momentum_physical_baseline": OptimizerSpec(
            "coherent_momentum_physical_baseline",
            CoherentMomentumPhysicalBaseline,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "friction": 0.08,
                "energy_correction_strength": 0.15,
                "oscillation_damping": 0.2,
                "momentum_coupling": 0.25,
                "energy_decay": 0.95,
                "min_scale": 0.55,
                "max_scale": 1.35,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "friction": [0.02, 0.08, 0.15],
                "energy_correction_strength": [0.0, 0.15, 0.3],
                "oscillation_damping": [0.0, 0.2, 0.4],
                "momentum_coupling": [0.0, 0.25, 0.5],
                "energy_decay": [0.9, 0.95, 0.98],
            },
            {
                "friction": 0.0,
                "energy_correction_strength": 0.0,
                "oscillation_damping": 0.0,
                "momentum_coupling": 0.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
            },
        ),
        "coherent_momentum_adaptive_mass_baseline": OptimizerSpec(
            "coherent_momentum_adaptive_mass_baseline",
            CoherentMomentumAdaptiveMassBaseline,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "friction": 0.06,
                "energy_correction_strength": 0.18,
                "oscillation_damping": 0.16,
                "momentum_coupling": 0.35,
                "loss_ema_decay": 0.95,
                "energy_ema_decay": 0.96,
                "drift_ema_decay": 0.90,
                "drift_threshold": 0.025,
                "correction_floor": 0.0,
                "correction_cap": 0.35,
                "predictive_damping_strength": 0.14,
                "alignment_boost": 0.12,
                "misalignment_damping": 0.18,
                "alignment_min_scale": 0.85,
                "alignment_max_scale": 1.18,
                "min_step_scale": 0.55,
                "max_step_scale": 1.42,
                "normalized_energy_weight": 1.0,
                "update_energy_weight": 0.55,
                "force_energy_weight": 0.35,
                "loss_change_weight": 0.65,
                "force_mode": "adamw",
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
                "friction": [0.02, 0.06, 0.1],
                "energy_correction_strength": [0.08, 0.18, 0.3],
                "oscillation_damping": [0.08, 0.16, 0.28],
                "momentum_coupling": [0.2, 0.35, 0.5],
                "energy_ema_decay": [0.92, 0.96, 0.98],
                "drift_ema_decay": [0.85, 0.9, 0.95],
                "drift_threshold": [0.01, 0.025, 0.05],
                "alignment_boost": [0.0, 0.12, 0.22],
                "misalignment_damping": [0.08, 0.18, 0.3],
                "min_step_scale": [0.6, 0.8],
                "max_step_scale": [1.2, 1.42, 1.7],
            },
            {
                "friction": 0.0,
                "energy_correction_strength": 0.0,
                "oscillation_damping": 0.0,
                "momentum_coupling": 0.0,
                "predictive_damping_strength": 0.0,
                "alignment_boost": 0.0,
                "misalignment_damping": 0.0,
                "min_step_scale": 1.0,
                "max_step_scale": 1.0,
                "alignment_min_scale": 1.0,
                "alignment_max_scale": 1.0,
                "normalized_energy_weight": 0.0,
                "update_energy_weight": 0.0,
                "force_energy_weight": 0.0,
                "loss_change_weight": 0.0,
                "use_normalized_energy": False,
                "use_energy_trend": False,
                "use_predictive_damping": False,
                "use_alignment_scaling": False,
                "use_adaptive_step_scale": False,
                "use_symplectic_correction": False,
                "use_oscillation_damping": False,
                "use_friction": False,
                "force_mode": "adamw",
            },
        ),
        "coherent_momentum_real_baseline": OptimizerSpec(
            "coherent_momentum_real_baseline",
            CoherentMomentumRealBaseline,
            {
                "lr": 0.02,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "mode": "dissipative_hamiltonian",
                "mass_mode": "adaptive",
                "fixed_mass": 1.0,
                "use_adam_preconditioning": True,
                "adaptive_mass_trust": 0.08,
                "mass_smoothing": 0.985,
                "mass_anisotropy_cap": 1.6,
                "mass_change_cap": 1.05,
                "mass_warmup_steps": 40,
                "mass_alignment_strength": 0.25,
                "mass_shock_penalty": 1.4,
                "min_inverse_mass": 0.55,
                "max_inverse_mass": 1.45,
                "friction": 0.035,
                "use_friction": True,
                "energy_correction_strength": 0.07,
                "use_energy_correction": True,
                "drift_threshold": 0.035,
                "max_energy_correction": 0.16,
                "relative_energy_floor": 0.1,
                "use_decoupled_weight_decay": True,
            },
            {
                "lr": [0.01, 0.015, 0.02, 0.03],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
                "mode": [
                    "symplectic_euler",
                    "leapfrog_with_closure",
                    "dissipative_hamiltonian",
                    "adam_preconditioned_hamiltonian",
                ],
                "mass_mode": ["adaptive", "fixed"],
                "fixed_mass": [0.75, 1.0, 1.5],
                "adaptive_mass_trust": [0.04, 0.08, 0.15],
                "mass_smoothing": [0.95, 0.975, 0.985],
                "mass_anisotropy_cap": [1.4, 1.8, 2.4],
                "mass_change_cap": [1.03, 1.05, 1.08],
                "mass_warmup_steps": [16, 32, 48],
                "mass_alignment_strength": [0.0, 0.25, 0.45],
                "mass_shock_penalty": [1.0, 1.4, 1.8],
                "min_inverse_mass": [0.5, 0.55, 0.65],
                "max_inverse_mass": [1.25, 1.45, 1.7],
                "friction": [0.0, 0.02, 0.035, 0.05],
                "energy_correction_strength": [0.0, 0.04, 0.07, 0.12],
                "drift_threshold": [0.02, 0.035, 0.05],
                "max_energy_correction": [0.08, 0.16, 0.24],
            },
            {
                "mode": "adam_preconditioned_hamiltonian",
                "mass_mode": "adaptive",
                "use_adam_preconditioning": True,
                "adaptive_mass_trust": 0.0,
                "mass_smoothing": 1.0,
                "mass_anisotropy_cap": 1.0,
                "mass_change_cap": 1.0,
                "mass_warmup_steps": 1,
                "mass_alignment_strength": 0.0,
                "mass_shock_penalty": 0.0,
                "min_inverse_mass": 1.0,
                "max_inverse_mass": 1.0,
                "friction": 0.0,
                "use_friction": False,
                "energy_correction_strength": 0.0,
                "use_energy_correction": False,
                "drift_threshold": 1.0,
                "max_energy_correction": 0.0,
                "use_decoupled_weight_decay": True,
            },
        ),
        "coherent_momentum_optimizer": OptimizerSpec(
            "coherent_momentum_optimizer",
            CoherentMomentumOptimizer,
            {
                "lr": 0.02,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "mode": "dissipative_hamiltonian",
                "mass_mode": "fixed",
                "fixed_mass": 1.0,
                "use_adam_preconditioning": True,
                "adaptive_mass_trust": 0.08,
                "mass_smoothing": 0.98,
                "mass_anisotropy_cap": 1.5,
                "mass_change_cap": 1.05,
                "mass_warmup_steps": 32,
                "mass_alignment_strength": 0.25,
                "mass_shock_penalty": 1.4,
                "min_inverse_mass": 0.55,
                "max_inverse_mass": 1.45,
                "friction": 0.03,
                "use_friction": True,
                "energy_correction_strength": 0.06,
                "use_energy_correction": True,
                "drift_threshold": 0.035,
                "max_energy_correction": 0.16,
                "relative_energy_floor": 0.1,
                "alignment_strength": 0.14,
                "coherence_strength": 0.12,
                "conflict_damping": 0.20,
                "rotation_penalty": 0.22,
                "projection_strength": 0.12,
                "max_projection": 0.25,
                "conflict_gate_threshold": 0.28,
                "activation_rotation_threshold": 0.38,
                "activation_conflict_weight": 0.7,
                "activation_rotation_weight": 0.5,
                "stable_coherence_bonus": 0.24,
                "projection_activation_threshold": 0.18,
                "min_alignment_scale": 0.88,
                "max_alignment_scale": 1.12,
                "field_clip": 2.0,
                "use_decoupled_weight_decay": True,
            },
            {
                "lr": [0.01, 0.015, 0.02, 0.03],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
                "mode": [
                    "symplectic_euler",
                    "leapfrog_with_closure",
                    "dissipative_hamiltonian",
                    "adam_preconditioned_hamiltonian",
                ],
                "mass_mode": ["adaptive", "fixed"],
                "fixed_mass": [0.75, 1.0, 1.5],
                "adaptive_mass_trust": [0.04, 0.08, 0.15],
                "mass_smoothing": [0.96, 0.98, 0.99],
                "mass_anisotropy_cap": [1.3, 1.6, 2.0],
                "mass_change_cap": [1.03, 1.05, 1.08],
                "mass_warmup_steps": [16, 32, 48],
                "mass_alignment_strength": [0.0, 0.25, 0.45],
                "mass_shock_penalty": [1.0, 1.4, 1.8],
                "min_inverse_mass": [0.5, 0.55, 0.65],
                "max_inverse_mass": [1.25, 1.45, 1.7],
                "friction": [0.02, 0.03, 0.05],
                "energy_correction_strength": [0.0, 0.06, 0.12],
                "drift_threshold": [0.02, 0.035, 0.05],
                "alignment_strength": [0.08, 0.14, 0.22],
                "coherence_strength": [0.06, 0.12, 0.18],
                "conflict_damping": [0.12, 0.20, 0.32],
                "rotation_penalty": [0.12, 0.22, 0.35],
                "projection_strength": [0.0, 0.08, 0.14],
                "max_projection": [0.18, 0.25, 0.35],
                "conflict_gate_threshold": [0.12, 0.25, 0.4],
                "activation_rotation_threshold": [0.25, 0.38, 0.55],
                "activation_conflict_weight": [0.5, 0.7, 1.0],
                "activation_rotation_weight": [0.25, 0.5, 0.75],
                "stable_coherence_bonus": [0.0, 0.18, 0.32],
                "projection_activation_threshold": [0.0, 0.18, 0.35],
                "min_alignment_scale": [0.85, 0.9, 0.95],
                "max_alignment_scale": [1.08, 1.12, 1.18],
            },
            {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "conflict_damping": 0.0,
                "rotation_penalty": 0.0,
                "projection_strength": 0.0,
                "max_projection": 0.0,
                "conflict_gate_threshold": 1.0,
                "activation_conflict_weight": 0.0,
                "activation_rotation_weight": 0.0,
                "stable_coherence_bonus": 0.0,
                "min_alignment_scale": 1.0,
                "max_alignment_scale": 1.0,
            },
        ),
        "coherent_momentum_optimizer_improved": OptimizerSpec(
            "coherent_momentum_optimizer_improved",
            CoherentMomentumOptimizerImproved,
            {
                "lr": 0.02,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "mode": "dissipative_hamiltonian",
                "mass_mode": "adaptive",
                "fixed_mass": 1.0,
                "use_adam_preconditioning": True,
                "adaptive_mass_trust": 0.08,
                "mass_smoothing": 0.98,
                "mass_anisotropy_cap": 1.5,
                "mass_change_cap": 1.05,
                "mass_warmup_steps": 32,
                "mass_alignment_strength": 0.25,
                "mass_shock_penalty": 1.4,
                "min_inverse_mass": 0.55,
                "max_inverse_mass": 1.45,
                "friction": 0.03,
                "use_friction": True,
                "energy_correction_strength": 0.06,
                "use_energy_correction": True,
                "drift_threshold": 0.035,
                "max_energy_correction": 0.16,
                "relative_energy_floor": 0.1,
                "alignment_strength": 0.12,
                "coherence_strength": 0.10,
                "conflict_damping": 0.0,
                "rotation_penalty": 0.18,
                "projection_strength": 0.12,
                "max_projection": 0.22,
                "conflict_gate_threshold": 0.28,
                "activation_rotation_threshold": 0.34,
                "activation_conflict_weight": 0.7,
                "activation_rotation_weight": 0.45,
                "stable_coherence_bonus": 0.18,
                "projection_activation_threshold": 0.16,
                "min_alignment_scale": 0.92,
                "max_alignment_scale": 1.10,
                "field_clip": 2.0,
                "preset": "balanced",
                "projection_mode": "conflict_only",
                "standard_safe_strength": 0.7,
                "soft_conflict_correction": 0.12,
                "soft_conflict_max": 0.22,
                "conv_safe_mode": True,
                "conv_update_ratio_cap": 0.018,
                "conv_projection_scale": 0.65,
                "conv_friction_scale": 0.72,
                "conv_alignment_scale": 0.75,
                "conv_support_weight": 0.35,
                "use_decoupled_weight_decay": True,
                "diagnostics_every_n_steps": 4,
            },
            {
                "lr": [0.01, 0.015, 0.02, 0.03],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
                "mode": [
                    "symplectic_euler",
                    "leapfrog_with_closure",
                    "dissipative_hamiltonian",
                    "adam_preconditioned_hamiltonian",
                ],
                "mass_mode": ["adaptive", "fixed"],
                "adaptive_mass_trust": [0.04, 0.08, 0.12],
                "mass_smoothing": [0.97, 0.98, 0.99],
                "friction": [0.02, 0.03, 0.05],
                "energy_correction_strength": [0.0, 0.06, 0.12],
                "alignment_strength": [0.08, 0.12, 0.18],
                "coherence_strength": [0.06, 0.10, 0.14],
                "rotation_penalty": [0.12, 0.18, 0.26],
                "projection_strength": [0.0, 0.08, 0.12],
                "max_projection": [0.16, 0.22, 0.28],
                "preset": ["balanced", "standard_safe", "stress_specialist", "cnn_safe"],
                "projection_mode": ["conflict_only", "always", "rotation_only"],
                "standard_safe_strength": [0.5, 0.7, 0.9],
                "soft_conflict_correction": [0.08, 0.12, 0.16],
                "soft_conflict_max": [0.16, 0.22, 0.28],
                "conv_update_ratio_cap": [0.012, 0.018, 0.024],
                "conv_projection_scale": [0.5, 0.65, 0.8],
                "conv_friction_scale": [0.6, 0.72, 0.85],
                "conv_alignment_scale": [0.65, 0.75, 0.9],
                "conv_support_weight": [0.2, 0.35, 0.5],
                "diagnostics_every_n_steps": [1, 4, 8],
            },
            {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "conflict_damping": 0.0,
                "rotation_penalty": 0.0,
                "projection_strength": 0.0,
                "max_projection": 0.0,
                "soft_conflict_correction": 0.0,
                "soft_conflict_max": 0.0,
                "min_alignment_scale": 1.0,
                "max_alignment_scale": 1.0,
                "conv_safe_mode": False,
            },
        ),
        "uncertainty_adam": OptimizerSpec(
            "uncertainty_adam",
            QuantumUncertaintyAdam,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "uncertainty_weight": 0.2,
                "interference_weight": 0.2,
                "reliability_strength": 0.15,
                "exploration_strength": 0.08,
                "min_scale": 0.6,
                "max_scale": 1.4,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "uncertainty_weight": [0.0, 0.2, 0.4],
                "interference_weight": [0.0, 0.2, 0.4],
                "reliability_strength": [0.0, 0.15, 0.3],
                "exploration_strength": [0.0, 0.08, 0.16],
            },
            {
                "uncertainty_weight": 0.0,
                "interference_weight": 0.0,
                "reliability_strength": 0.0,
                "exploration_strength": 0.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
            },
        ),
        "unified_physics_adam": OptimizerSpec(
            "unified_physics_adam",
            UnifiedPhysicsAdam,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "enable_sds": True,
                "enable_coherence": True,
                "enable_thermodynamic": True,
                "enable_diffusion": True,
                "enable_hamiltonian": True,
                "enable_uncertainty": True,
                "inner_horizon": 5e-4,
                "outer_horizon": 2.5e-2,
                "horizon_sharpness": 12.0,
                "sds_cooling_strength": 0.25,
                "sds_reheating_strength": 0.12,
                "sds_entropy_weight": 0.08,
                "alignment_strength": 0.12,
                "coherence_strength": 0.10,
                "rotation_penalty": 0.18,
                "misalignment_damping": 0.18,
                "layerwise_mode": True,
                "global_mode": True,
                "thermodynamic_entropy_weight": 0.10,
                "thermodynamic_energy_weight": 0.20,
                "temperature_decay": 0.96,
                "thermodynamic_cooling_strength": 0.08,
                "thermodynamic_reheating_strength": 0.08,
                "max_temperature": 1.5,
                "min_temperature": 0.05,
                "diffusion_strength": 0.02,
                "diffusion_decay": 0.985,
                "entropy_scaled_noise": True,
                "stagnation_trigger": 8,
                "min_noise": 0.0,
                "max_noise": 0.20,
                "noise_to_update_cap": 0.20,
                "aligned_noise_weight": 0.2,
                "friction": 0.06,
                "energy_correction_strength": 0.12,
                "oscillation_damping": 0.12,
                "momentum_coupling": 0.28,
                "loss_ema_decay": 0.95,
                "energy_ema_decay": 0.96,
                "drift_ema_decay": 0.90,
                "drift_threshold": 0.02,
                "correction_floor": 0.0,
                "correction_cap": 0.25,
                "uncertainty_weight": 0.15,
                "interference_weight": 0.15,
                "reliability_strength": 0.12,
                "exploration_strength": 0.05,
                "min_step_scale": 0.60,
                "max_step_scale": 1.35,
            },
            {
                "lr": [3e-3, 1e-3, 5e-4],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "betas": [(0.9, 0.999), (0.9, 0.99)],
                "sds_cooling_strength": [0.0, 0.25, 0.45],
                "sds_reheating_strength": [0.0, 0.12, 0.24],
                "alignment_strength": [0.0, 0.12, 0.24],
                "rotation_penalty": [0.0, 0.18, 0.30],
                "thermodynamic_energy_weight": [0.0, 0.20, 0.35],
                "thermodynamic_cooling_strength": [0.0, 0.08, 0.16],
                "diffusion_strength": [0.0, 0.02, 0.05],
                "noise_to_update_cap": [0.10, 0.20, 0.35],
                "friction": [0.0, 0.06, 0.12],
                "energy_correction_strength": [0.0, 0.12, 0.24],
                "oscillation_damping": [0.0, 0.12, 0.24],
                "uncertainty_weight": [0.0, 0.15, 0.30],
                "interference_weight": [0.0, 0.15, 0.30],
                "min_step_scale": [0.70, 0.85, 1.0],
                "max_step_scale": [1.10, 1.35, 1.55],
            },
            {
                "enable_sds": False,
                "enable_coherence": False,
                "enable_thermodynamic": False,
                "enable_diffusion": False,
                "enable_hamiltonian": False,
                "enable_uncertainty": False,
                "sds_cooling_strength": 0.0,
                "sds_reheating_strength": 0.0,
                "sds_entropy_weight": 0.0,
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "rotation_penalty": 0.0,
                "misalignment_damping": 0.0,
                "thermodynamic_entropy_weight": 0.0,
                "thermodynamic_energy_weight": 0.0,
                "thermodynamic_cooling_strength": 0.0,
                "thermodynamic_reheating_strength": 0.0,
                "diffusion_strength": 0.0,
                "max_noise": 0.0,
                "noise_to_update_cap": 0.0,
                "aligned_noise_weight": 0.0,
                "friction": 0.0,
                "energy_correction_strength": 0.0,
                "oscillation_damping": 0.0,
                "momentum_coupling": 0.0,
                "uncertainty_weight": 0.0,
                "interference_weight": 0.0,
                "reliability_strength": 0.0,
                "exploration_strength": 0.0,
                "min_step_scale": 1.0,
                "max_step_scale": 1.0,
            },
        ),
        "direction_recovery_optimizer": OptimizerSpec(
            "direction_recovery_optimizer",
            DirectionRecoveryOptimizer,
            {
                "lr": 3e-3,
                "weight_decay": 1e-4,
                "memory_decay": 0.92,
                "grad_decay": 0.95,
                "recovery_strength": 0.65,
                "coherence_strength": 0.35,
                "rotation_penalty": 0.12,
                "perturb_scale": 0.05,
                "perturb_samples": 2,
                "trust_smoothing": 0.9,
                "dimension_power": 0.25,
                "max_update_ratio": 0.05,
                "min_scale": 0.55,
                "max_scale": 1.35,
                "use_recovery": True,
                "use_coherence": True,
                "use_projection": True,
            },
            {
                "lr": [1e-2, 5e-3, 3e-3, 1e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "memory_decay": [0.85, 0.92, 0.97],
                "grad_decay": [0.9, 0.95, 0.98],
                "recovery_strength": [0.35, 0.65, 0.9],
                "coherence_strength": [0.15, 0.35, 0.55],
                "rotation_penalty": [0.05, 0.12, 0.25],
                "perturb_scale": [0.02, 0.05, 0.1],
                "perturb_samples": [1, 2, 3],
                "dimension_power": [0.0, 0.125, 0.25, 0.5],
                "max_update_ratio": [0.02, 0.05, 0.1],
                "min_scale": [0.55, 0.7, 1.0],
                "max_scale": [1.1, 1.35, 1.6],
                "use_projection": [False, True],
            },
            {
                "memory_decay": 1.0,
                "grad_decay": 1.0,
                "recovery_strength": 0.0,
                "coherence_strength": 0.0,
                "rotation_penalty": 0.0,
                "perturb_scale": 0.0,
                "perturb_samples": 0,
                "trust_smoothing": 1.0,
                "dimension_power": 0.25,
                "max_update_ratio": 1.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
                "use_recovery": False,
                "use_coherence": False,
                "use_projection": False,
            },
        ),
        "constraint_consensus_optimizer": OptimizerSpec(
            "constraint_consensus_optimizer",
            ConstraintConsensusOptimizer,
            {
                "lr": 3e-3,
                "weight_decay": 1e-4,
                "agreement_strength": 0.55,
                "recoverability_strength": 0.60,
                "balance_strength": 0.25,
                "conflict_penalty": 0.18,
                "memory_decay": 0.90,
                "memory_strength": 0.18,
                "projection_strength": 0.14,
                "dimension_power": 0.15,
                "max_update_ratio": 0.08,
                "min_scale": 0.55,
                "max_scale": 1.45,
                "use_memory": True,
                "use_projection": True,
            },
            {
                "lr": [1e-2, 5e-3, 3e-3, 1e-3],
                "weight_decay": [0.0, 1e-4, 1e-3],
                "agreement_strength": [0.3, 0.55, 0.8],
                "recoverability_strength": [0.3, 0.6, 0.9],
                "balance_strength": [0.1, 0.25, 0.45],
                "conflict_penalty": [0.0, 0.18, 0.35],
                "memory_decay": [0.8, 0.9, 0.96],
                "memory_strength": [0.0, 0.18, 0.3],
                "projection_strength": [0.0, 0.14, 0.25],
                "dimension_power": [0.0, 0.15, 0.3],
                "max_update_ratio": [0.04, 0.08, 0.12],
                "min_scale": [0.55, 0.75, 1.0],
                "max_scale": [1.15, 1.45, 1.7],
                "use_memory": [False, True],
                "use_projection": [False, True],
            },
            {
                "agreement_strength": 0.0,
                "recoverability_strength": 0.0,
                "balance_strength": 0.0,
                "conflict_penalty": 0.0,
                "memory_decay": 1.0,
                "memory_strength": 0.0,
                "projection_strength": 0.0,
                "max_update_ratio": 1.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
                "use_memory": False,
                "use_projection": False,
            },
        ),
    }


def sample_search_configs(spec: OptimizerSpec, budget: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{spec.name}")
    configs: list[dict[str, Any]] = [dict(spec.default_params)]
    seen = {json.dumps(spec.default_params, sort_keys=True, default=str)}
    keys = list(spec.search_space.keys())
    max_attempts = max(50, budget * 20)

    attempts = 0
    while len(configs) < max(1, budget) and attempts < max_attempts:
        candidate = dict(spec.default_params)
        for key in keys:
            candidate[key] = rng.choice(spec.search_space[key])
        encoded = json.dumps(candidate, sort_keys=True, default=str)
        if encoded not in seen:
            configs.append(candidate)
            seen.add(encoded)
        attempts += 1
    return configs[: max(1, budget)]


def instantiate_optimizer(
    optimizer_name: str,
    parameters,
    overrides: dict[str, Any] | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    registry = build_optimizer_registry()
    spec = registry[optimizer_name]
    hyperparameters = dict(spec.default_params)
    if overrides:
        hyperparameters.update(overrides)
    optimizer = spec.factory(parameters, **hyperparameters)
    return optimizer, hyperparameters


def benchmark_optimizer_names() -> list[str]:
    return [
        "sgd",
        "sgd_momentum",
        "rmsprop",
        "adam",
        "adamw",
        "nadam",
        "radam",
        "lion",
        "muon_hybrid",
        "topological_adam",
        "sds_adam",
        "coherent_direction_reference",
        "thermodynamic_adam",
        "diffusion_adam",
        "coherent_momentum_physical_baseline",
        "coherent_momentum_adaptive_mass_baseline",
        "coherent_momentum_real_baseline",
        "coherent_momentum_optimizer",
        "coherent_momentum_optimizer_improved",
        "uncertainty_adam",
        "unified_physics_adam",
        "direction_recovery_optimizer",
        "observation_recovery_optimizer",
        "constraint_consensus_optimizer",
    ]
