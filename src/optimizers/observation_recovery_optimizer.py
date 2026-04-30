from __future__ import annotations

import math
from typing import Any

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import average, bounded_scale, clamp_scalar, cosine_similarity, safe_float


def _normalize(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    norm = tensor.detach().float().norm()
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(tensor)
    return tensor / (norm + eps)


class ObservationRecoveryOptimizer(PhysicalOptimizerBase, torch.optim.Optimizer):
    """Optimizer using masked gradient observations to score reconstructible directions.

    For each tensor block, this optimizer creates several masked/corrupted views of
    the current gradient, reconstructs a consensus direction from those partial
    observations, and trusts the update only if that direction survives masking.
    """

    def __init__(
        self,
        params,
        lr: float = 4e-3,
        weight_decay: float = 1e-4,
        eps: float = 1e-8,
        view_count: int = 3,
        keep_probability: float = 0.6,
        corruption_std: float = 0.03,
        recoverability_strength: float = 0.55,
        disagreement_penalty: float = 0.24,
        memory_strength: float = 0.20,
        memory_decay: float = 0.90,
        dimension_power: float = 0.20,
        max_update_ratio: float = 0.08,
        min_scale: float = 0.55,
        max_scale: float = 1.45,
        use_memory: bool = True,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            view_count=view_count,
            keep_probability=keep_probability,
            corruption_std=corruption_std,
            recoverability_strength=recoverability_strength,
            disagreement_penalty=disagreement_penalty,
            memory_strength=memory_strength,
            memory_decay=memory_decay,
            dimension_power=dimension_power,
            max_update_ratio=max_update_ratio,
            min_scale=min_scale,
            max_scale=max_scale,
            use_memory=use_memory,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("ObservationRecoveryOptimizer")

    def _init_state(self, state: dict[str, Any], param: torch.Tensor) -> None:
        if len(state) != 0:
            return
        state["step"] = 0
        state["direction_memory"] = torch.zeros_like(param)
        state["prev_update"] = torch.zeros_like(param)
        state["trust_ema"] = 1.0

    def _build_views(
        self,
        grad: torch.Tensor,
        *,
        keep_probability: float,
        corruption_std: float,
        view_count: int,
        eps: float,
    ) -> list[torch.Tensor]:
        grad_scale = float(grad.detach().float().pow(2).mean().sqrt().item()) if grad.numel() > 0 else 0.0
        views: list[torch.Tensor] = []
        for _ in range(max(2, view_count)):
            mask = (torch.rand_like(grad) < keep_probability).to(grad.dtype)
            masked = grad * mask / max(keep_probability, eps)
            if corruption_std > 0.0 and grad_scale > 0.0:
                masked = masked + torch.randn_like(grad) * (corruption_std * grad_scale)
            views.append(_normalize(masked, eps))
        return views

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, _ = self._prepare_closure(closure)

        recoverability_values: list[float] = []
        disagreement_values: list[float] = []
        trust_values: list[float] = []
        memory_values: list[float] = []
        update_ratio_values: list[float] = []
        view_support_values: list[float] = []
        active_params = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            view_count = int(group["view_count"])
            keep_probability = clamp_scalar(float(group["keep_probability"]), 0.05, 1.0)
            corruption_std = max(0.0, float(group["corruption_std"]))
            recoverability_strength = float(group["recoverability_strength"])
            disagreement_penalty = float(group["disagreement_penalty"])
            memory_strength = float(group["memory_strength"])
            memory_decay = clamp_scalar(float(group["memory_decay"]), 0.0, 0.9999)
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            use_memory = bool(group["use_memory"])
            maximize = bool(group["maximize"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    continue

                grad_norm = safe_float(grad.norm())
                if grad_norm <= eps:
                    continue

                state = self.state[param]
                self._init_state(state, param)
                state["step"] = int(state["step"]) + 1

                views = self._build_views(
                    grad,
                    keep_probability=keep_probability,
                    corruption_std=corruption_std,
                    view_count=view_count,
                    eps=eps,
                )
                consensus_raw = torch.zeros_like(param)
                for view in views:
                    consensus_raw.add_(view)
                consensus = _normalize(consensus_raw, eps)
                if safe_float(consensus.norm()) <= eps:
                    continue

                recoverability = average(cosine_similarity(consensus, view) for view in views)
                pairwise = []
                for index, first in enumerate(views):
                    for second in views[index + 1 :]:
                        pairwise.append(cosine_similarity(first, second))
                disagreement = average(max(0.0, 1.0 - value) for value in pairwise) if pairwise else 0.0

                memory = state["direction_memory"]
                memory_dir = _normalize(memory, eps)
                memory_alignment = cosine_similarity(consensus, memory_dir) if safe_float(memory.norm()) > eps else 0.0
                if use_memory and safe_float(memory_dir.norm()) > eps and memory_alignment > 0.0:
                    consensus = _normalize((1.0 - memory_strength) * consensus + memory_strength * memory_dir, eps)

                support = max(0.0, safe_float(torch.sum(grad * consensus))) / (grad_norm + eps)
                candidate_trust = bounded_scale(
                    1.0
                    + recoverability_strength * recoverability
                    + 0.15 * support
                    - disagreement_penalty * disagreement,
                    min_scale,
                    max_scale,
                )
                trust_ema = float(state["trust_ema"])
                trust_ema = 0.9 * trust_ema + 0.1 * candidate_trust
                trust_scale = bounded_scale(trust_ema, min_scale, max_scale)

                block_size = max(1, param.numel())
                dimension_scale = float(block_size) ** dimension_power
                step_norm = lr * trust_scale * grad_norm / max(dimension_scale, eps)
                if max_update_ratio > 0.0:
                    step_cap = max_update_ratio * (safe_float(param.detach().norm()) + eps)
                    step_norm = min(step_norm, step_cap)

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                update = consensus * step_norm
                param.add_(update, alpha=-1.0)

                memory.copy_(_normalize(memory * memory_decay + consensus * (1.0 - memory_decay), eps))
                state["prev_update"].copy_(update)
                state["trust_ema"] = trust_ema

                recoverability_values.append(recoverability)
                disagreement_values.append(disagreement)
                trust_values.append(trust_scale)
                memory_values.append(memory_alignment)
                view_support_values.append(support)
                update_ratio_values.append(step_norm / (safe_float(param.detach().norm()) + eps))
                active_params += 1

        self._record_step(
            {
                "optimizer": self.optimizer_name,
                "observation_recoverability": average(recoverability_values),
                "observation_disagreement": average(disagreement_values),
                "trust_scale": average(trust_values),
                "memory_alignment": average(memory_values),
                "view_support": average(view_support_values),
                "update_ratio": average(update_ratio_values),
                "active_params": float(active_params),
            }
        )
        return loss_tensor

