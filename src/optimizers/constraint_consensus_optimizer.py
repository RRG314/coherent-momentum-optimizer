from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import average, bounded_scale, clamp_scalar, cosine_similarity, safe_float


def _normalize(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    norm = tensor.detach().float().norm()
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(tensor)
    return tensor / (norm + eps)


class ConstraintConsensusOptimizer(PhysicalOptimizerBase, torch.optim.Optimizer):
    """PINN-oriented optimizer based on constraint consensus and collocation recoverability.

    The update is not built from Adam moments. Instead, each parameter tensor is treated
    as a block and updated along a normalized consensus direction formed from individual
    constraint gradients such as PDE residual, boundary, and initial/data terms.

    The block update direction is weighted by:
    - cross-constraint directional agreement
    - recoverability of the residual direction under collocation perturbation
    - balance of support across constraint projections onto the consensus direction
    """

    uses_component_gradients = True

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        eps: float = 1e-8,
        agreement_strength: float = 0.55,
        recoverability_strength: float = 0.60,
        balance_strength: float = 0.25,
        conflict_penalty: float = 0.18,
        memory_decay: float = 0.90,
        memory_strength: float = 0.18,
        projection_strength: float = 0.14,
        trust_smoothing: float = 0.90,
        dimension_power: float = 0.15,
        max_update_ratio: float = 0.08,
        min_scale: float = 0.55,
        max_scale: float = 1.45,
        use_memory: bool = True,
        use_projection: bool = True,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            agreement_strength=agreement_strength,
            recoverability_strength=recoverability_strength,
            balance_strength=balance_strength,
            conflict_penalty=conflict_penalty,
            memory_decay=memory_decay,
            memory_strength=memory_strength,
            projection_strength=projection_strength,
            trust_smoothing=trust_smoothing,
            dimension_power=dimension_power,
            max_update_ratio=max_update_ratio,
            min_scale=min_scale,
            max_scale=max_scale,
            use_memory=use_memory,
            use_projection=use_projection,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("ConstraintConsensusOptimizer")
        self._component_closures: dict[str, Callable[[], torch.Tensor]] = {}
        self._component_metadata: dict[str, Any] = {}

    def set_component_closures(
        self,
        component_closures: dict[str, Callable[[], torch.Tensor]] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._component_closures = dict(component_closures or {})
        self._component_metadata = dict(metadata or {})

    def clear_component_closures(self) -> None:
        self._component_closures = {}
        self._component_metadata = {}

    def _init_state(self, state: dict[str, Any], param: torch.Tensor) -> None:
        if len(state) != 0:
            return
        state["step"] = 0
        state["direction_memory"] = torch.zeros_like(param)
        state["prev_update"] = torch.zeros_like(param)
        state["trust_ema"] = 1.0

    def _trainable_entries(self) -> list[tuple[dict[str, Any], torch.Tensor]]:
        entries: list[tuple[dict[str, Any], torch.Tensor]] = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    entries.append((group, param))
        return entries

    def _compute_component_gradients(
        self,
        entries: list[tuple[dict[str, Any], torch.Tensor]],
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, float]]:
        params = [param for _, param in entries]
        component_grads: dict[str, list[torch.Tensor]] = {}
        component_losses: dict[str, float] = {}
        for name, closure in self._component_closures.items():
            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = closure()
            loss_value = safe_float(loss.detach().item())
            component_losses[name] = loss_value
            if not torch.isfinite(loss):
                continue
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            component_grads[name] = [
                torch.zeros_like(param) if grad is None else grad.detach().clone()
                for param, grad in zip(params, grads)
            ]
        self.zero_grad(set_to_none=True)
        return component_grads, component_losses

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor = None
        if closure is not None and not self._component_closures:
            loss_tensor, _ = self._prepare_closure(closure)

        entries = self._trainable_entries()
        if not entries:
            return loss_tensor

        component_grads: dict[str, list[torch.Tensor]] = {}
        component_losses: dict[str, float] = {}
        if self._component_closures:
            component_grads, component_losses = self._compute_component_gradients(entries)

        agreement_values: list[float] = []
        recoverability_values: list[float] = []
        balance_values: list[float] = []
        consensus_values: list[float] = []
        conflict_values: list[float] = []
        trust_values: list[float] = []
        memory_alignment_values: list[float] = []
        residual_alignment_values: list[float] = []
        update_ratio_values: list[float] = []
        active_params = 0

        residual_name = str(self._component_metadata.get("residual_name", "residual"))
        perturbed_name = str(self._component_metadata.get("perturbed_residual_name", "residual_perturbed"))
        training_components = set(self._component_metadata.get("training_components", []))

        for index, (group, param) in enumerate(entries):
            if not self._component_closures:
                grad = None if param.grad is None else param.grad.detach()
                if grad is None or not torch.isfinite(grad).all():
                    continue
                component_views = {"total": grad}
            else:
                component_views = {}
                for name, grads in component_grads.items():
                    if index >= len(grads):
                        continue
                    grad = grads[index]
                    if grad.numel() == 0 or not torch.isfinite(grad).all():
                        continue
                    if safe_float(grad.norm()) <= float(group["eps"]):
                        continue
                    component_views[name] = grad
                if not component_views:
                    continue

            eps = float(group["eps"])
            agreement_strength = float(group["agreement_strength"])
            recoverability_strength = float(group["recoverability_strength"])
            balance_strength = float(group["balance_strength"])
            conflict_penalty = float(group["conflict_penalty"])
            memory_decay = clamp_scalar(float(group["memory_decay"]), 0.0, 0.9999)
            memory_strength = float(group["memory_strength"])
            projection_strength = float(group["projection_strength"])
            trust_smoothing = clamp_scalar(float(group["trust_smoothing"]), 0.0, 0.9999)
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            use_memory = bool(group["use_memory"])
            use_projection = bool(group["use_projection"])
            maximize = bool(group["maximize"])
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])

            state = self.state[param]
            self._init_state(state, param)
            state["step"] = int(state["step"]) + 1

            directions = {
                name: _normalize(-grad if maximize else grad, eps)
                for name, grad in component_views.items()
            }
            active_names = [name for name, direction in directions.items() if safe_float(direction.norm()) > eps]
            if not active_names:
                continue

            pairwise_cosines: list[float] = []
            component_weights: dict[str, float] = {}
            for name in active_names:
                other_cosines = [
                    cosine_similarity(directions[name], directions[other])
                    for other in active_names
                    if other != name
                ]
                agreement = average(other_cosines) if other_cosines else 1.0
                pairwise_cosines.extend(other_cosines)
                weight = max(0.0, agreement)
                if name == residual_name and perturbed_name in directions:
                    weight *= 1.0 + recoverability_strength * max(
                        0.0,
                        cosine_similarity(directions[residual_name], directions[perturbed_name]),
                    )
                elif name in training_components or not training_components:
                    weight *= 1.0 + 0.5 * agreement_strength * max(0.0, agreement)
                if name == perturbed_name:
                    weight *= 0.5
                component_weights[name] = max(weight, 1e-4)

            consensus_raw = torch.zeros_like(param)
            for name in active_names:
                consensus_raw.add_(directions[name], alpha=component_weights[name])
            consensus_strength = safe_float(consensus_raw.norm()) / max(sum(component_weights.values()), eps)
            consensus_direction = _normalize(consensus_raw, eps)
            if safe_float(consensus_direction.norm()) <= eps:
                continue

            state_memory = state["direction_memory"]
            memory_direction = _normalize(state_memory, eps)
            memory_alignment = cosine_similarity(consensus_direction, memory_direction)
            if use_memory and safe_float(memory_direction.norm()) > eps and memory_alignment > 0.0:
                consensus_direction = _normalize(
                    (1.0 - memory_strength) * consensus_direction + memory_strength * memory_direction,
                    eps,
                )

            residual_alignment = cosine_similarity(consensus_direction, directions[residual_name]) if residual_name in directions else float("nan")
            if use_projection and residual_name in directions and math.isfinite(residual_alignment) and residual_alignment < 0.0:
                consensus_direction = _normalize(
                    consensus_direction + projection_strength * directions[residual_name],
                    eps,
                )
                residual_alignment = cosine_similarity(consensus_direction, directions[residual_name])

            support_values: list[float] = []
            normalized_supports: list[float] = []
            for name in active_names:
                if training_components and name not in training_components:
                    continue
                grad = component_views[name]
                projected = max(0.0, safe_float(torch.sum(grad * consensus_direction)))
                support_values.append(projected)
                normalized_supports.append(projected / (safe_float(grad.norm()) + eps))
            if not support_values:
                support_values = [max(0.0, safe_float(torch.sum(component_views[active_names[0]] * consensus_direction)))]
                normalized_supports = [support_values[0] / (safe_float(component_views[active_names[0]].norm()) + eps)]

            support_mean = average(support_values)
            support_balance = 1.0 / (1.0 + float(np.var(normalized_supports)) if normalized_supports else 1.0)
            agreement_score = average(pairwise_cosines) if pairwise_cosines else 1.0
            conflict_score = average(max(0.0, -value) for value in pairwise_cosines) if pairwise_cosines else 0.0
            recoverability = (
                cosine_similarity(directions[residual_name], directions[perturbed_name])
                if residual_name in directions and perturbed_name in directions
                else max(0.0, agreement_score)
            )

            candidate_trust = bounded_scale(
                1.0
                + agreement_strength * max(0.0, agreement_score)
                + recoverability_strength * max(0.0, recoverability)
                + balance_strength * (support_balance - 0.5)
                - conflict_penalty * conflict_score,
                min_scale,
                max_scale,
            )
            trust_ema = float(state["trust_ema"])
            trust_ema = trust_smoothing * trust_ema + (1.0 - trust_smoothing) * candidate_trust
            trust_scale = bounded_scale(trust_ema, min_scale, max_scale)

            block_size = max(1, param.numel())
            dimension_scale = float(block_size) ** dimension_power
            step_norm = lr * trust_scale * support_mean / max(dimension_scale, eps)
            if max_update_ratio > 0.0:
                step_cap = max_update_ratio * (safe_float(param.detach().norm()) + eps)
                step_norm = min(step_norm, step_cap)
            if weight_decay > 0.0:
                param.mul_(1.0 - lr * weight_decay)
            update = consensus_direction * step_norm
            param.add_(update, alpha=-1.0)

            state_memory.copy_(_normalize(state_memory * memory_decay + consensus_direction * (1.0 - memory_decay), eps))
            state["prev_update"].copy_(update)
            state["trust_ema"] = trust_ema

            agreement_values.append(agreement_score)
            recoverability_values.append(recoverability)
            balance_values.append(support_balance)
            consensus_values.append(consensus_strength)
            conflict_values.append(conflict_score)
            trust_values.append(trust_scale)
            memory_alignment_values.append(memory_alignment)
            if math.isfinite(residual_alignment):
                residual_alignment_values.append(residual_alignment)
            update_ratio_values.append(step_norm / (safe_float(param.detach().norm()) + eps))
            active_params += 1

        self._record_step(
            {
                "optimizer": self.optimizer_name,
                "constraint_agreement": average(agreement_values),
                "recoverability_score": average(recoverability_values),
                "support_balance": average(balance_values),
                "consensus_strength": average(consensus_values),
                "component_conflict": average(conflict_values),
                "trust_scale": average(trust_values),
                "memory_alignment": average(memory_alignment_values),
                "residual_alignment": average(residual_alignment_values),
                "update_ratio": average(update_ratio_values),
                "component_count": float(len(component_grads) if component_grads else (1 if active_params > 0 else 0)),
                "component_mode": 1.0 if component_grads else 0.0,
                "active_params": float(active_params),
                "residual_component_loss": component_losses.get(residual_name, float("nan")),
                "boundary_component_loss": component_losses.get("boundary", component_losses.get("boundary_value", float("nan"))),
                "initial_component_loss": component_losses.get("initial", float("nan")),
                "perturbed_residual_loss": component_losses.get(perturbed_name, float("nan")),
            }
        )
        self.clear_component_closures()
        return loss_tensor


PhysicsRecoveryOptimizer = ConstraintConsensusOptimizer
