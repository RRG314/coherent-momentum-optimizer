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


class DirectionRecoveryOptimizer(PhysicalOptimizerBase, torch.optim.Optimizer):
    """Direction-selection optimizer using recoverability under perturbation."""

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        eps: float = 1e-8,
        memory_decay: float = 0.92,
        grad_decay: float = 0.95,
        recovery_strength: float = 0.65,
        coherence_strength: float = 0.35,
        rotation_penalty: float = 0.12,
        perturb_scale: float = 0.05,
        perturb_samples: int = 2,
        trust_smoothing: float = 0.9,
        dimension_power: float = 0.25,
        max_update_ratio: float = 0.05,
        min_scale: float = 0.55,
        max_scale: float = 1.35,
        use_recovery: bool = True,
        use_coherence: bool = True,
        use_projection: bool = True,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            memory_decay=memory_decay,
            grad_decay=grad_decay,
            recovery_strength=recovery_strength,
            coherence_strength=coherence_strength,
            rotation_penalty=rotation_penalty,
            perturb_scale=perturb_scale,
            perturb_samples=perturb_samples,
            trust_smoothing=trust_smoothing,
            dimension_power=dimension_power,
            max_update_ratio=max_update_ratio,
            min_scale=min_scale,
            max_scale=max_scale,
            use_recovery=use_recovery,
            use_coherence=use_coherence,
            use_projection=use_projection,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("DirectionRecoveryOptimizer")

    def _init_state(self, state: dict[str, Any], param: torch.Tensor) -> None:
        if len(state) != 0:
            return
        state["step"] = 0
        state["direction_memory"] = torch.zeros_like(param)
        state["prev_grad"] = torch.zeros_like(param)
        state["prev_update"] = torch.zeros_like(param)
        state["grad_norm_ema"] = 0.0
        state["trust_ema"] = 0.0
        state["last_candidate"] = "current"

    def _candidate_scores(
        self,
        *,
        candidates: dict[str, torch.Tensor],
        grad: torch.Tensor,
        direction_memory: torch.Tensor,
        prev_grad: torch.Tensor,
        prev_update: torch.Tensor,
        perturb_scale: float,
        perturb_samples: int,
        eps: float,
        use_recovery: bool,
        use_coherence: bool,
        recovery_strength: float,
        coherence_strength: float,
        rotation_penalty: float,
    ) -> dict[str, dict[str, float]]:
        grad_rms = float(grad.detach().float().pow(2).mean().sqrt().item()) if grad.numel() > 0 else 0.0
        memory_dir = _normalize(direction_memory, eps)
        prev_grad_dir = _normalize(prev_grad, eps)
        prev_update_dir = _normalize(prev_update, eps)
        perturbed_dirs: list[torch.Tensor] = []
        if use_recovery and perturb_samples > 0 and grad_rms > eps:
            noise_scale = max(perturb_scale * grad_rms, eps)
            for _ in range(max(1, perturb_samples)):
                perturbed_dirs.append(_normalize(grad + noise_scale * torch.randn_like(grad), eps))

        scores: dict[str, dict[str, float]] = {}
        for name, candidate in candidates.items():
            coherence = 0.0
            if use_coherence:
                coherence = average(
                    [
                        cosine_similarity(candidate, memory_dir),
                        cosine_similarity(candidate, prev_grad_dir),
                        cosine_similarity(candidate, prev_update_dir),
                    ]
                )
            recovery = 1.0
            if perturbed_dirs:
                recovery = average(cosine_similarity(candidate, perturbed) for perturbed in perturbed_dirs)
            rotation = average(
                [
                    max(0.0, -cosine_similarity(candidate, prev_grad_dir)),
                    max(0.0, -cosine_similarity(candidate, prev_update_dir)),
                ]
            )
            scores[name] = {
                "score": recovery_strength * recovery + coherence_strength * coherence - rotation_penalty * rotation,
                "recovery": recovery,
                "coherence": coherence,
                "rotation": rotation,
            }
        return scores

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, _ = self._prepare_closure(closure)

        recovery_values: list[float] = []
        coherence_values: list[float] = []
        rotation_values: list[float] = []
        trust_values: list[float] = []
        relative_scale_values: list[float] = []
        counts = {"memory": 0.0, "blend": 0.0, "projected": 0.0, "current": 0.0}
        active_params = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            memory_decay = clamp_scalar(float(group["memory_decay"]), 0.0, 0.9999)
            grad_decay = clamp_scalar(float(group["grad_decay"]), 0.0, 0.9999)
            recovery_strength = float(group["recovery_strength"]) if bool(group["use_recovery"]) else 0.0
            coherence_strength = float(group["coherence_strength"]) if bool(group["use_coherence"]) else 0.0
            rotation_penalty = float(group["rotation_penalty"])
            perturb_scale = float(group["perturb_scale"])
            perturb_samples = int(group["perturb_samples"])
            trust_smoothing = clamp_scalar(float(group["trust_smoothing"]), 0.0, 0.9999)
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            use_recovery = bool(group["use_recovery"])
            use_coherence = bool(group["use_coherence"])
            use_projection = bool(group["use_projection"])
            maximize = bool(group["maximize"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[param]
                self._init_state(state, param)
                state["step"] = int(state["step"]) + 1
                direction_memory = state["direction_memory"]
                prev_grad = state["prev_grad"]
                prev_update = state["prev_update"]

                grad_norm = safe_float(grad.norm())
                if grad_norm <= eps:
                    continue
                grad_dir = _normalize(grad, eps)
                memory_dir = _normalize(direction_memory, eps)

                candidates: dict[str, torch.Tensor] = {"current": grad_dir}
                if safe_float(memory_dir.norm()) > eps:
                    candidates["memory"] = memory_dir
                    candidates["blend"] = _normalize(0.5 * (grad_dir + memory_dir), eps)
                    if use_projection and cosine_similarity(grad_dir, memory_dir) < 0.0:
                        projection = grad_dir - torch.sum(grad_dir * memory_dir) * memory_dir
                        projected = _normalize(projection, eps)
                        if safe_float(projected.norm()) > eps:
                            candidates["projected"] = projected

                scores = self._candidate_scores(
                    candidates=candidates,
                    grad=grad,
                    direction_memory=direction_memory,
                    prev_grad=prev_grad,
                    prev_update=prev_update,
                    perturb_scale=perturb_scale,
                    perturb_samples=perturb_samples,
                    eps=eps,
                    use_recovery=use_recovery,
                    use_coherence=use_coherence,
                    recovery_strength=recovery_strength,
                    coherence_strength=coherence_strength,
                    rotation_penalty=rotation_penalty,
                )
                chosen_name, chosen_stats = max(scores.items(), key=lambda item: item[1]["score"])
                chosen_dir = candidates[chosen_name]

                grad_norm_ema = float(state["grad_norm_ema"])
                if grad_norm_ema <= eps:
                    grad_norm_ema = grad_norm
                grad_norm_ema = grad_decay * grad_norm_ema + (1.0 - grad_decay) * grad_norm
                relative_scale = grad_norm / (grad_norm_ema + eps)
                trust_scale = bounded_scale(
                    1.0
                    + 0.35 * chosen_stats["recovery"]
                    + 0.20 * max(0.0, chosen_stats["coherence"])
                    - 0.25 * chosen_stats["rotation"]
                    + 0.10 * math.log1p(max(0.0, relative_scale - 1.0)),
                    min_scale,
                    max_scale,
                )

                supported_magnitude = max(0.0, safe_float(torch.sum(grad * chosen_dir)))
                block_size = max(1, grad.numel())
                dimension_scale = float(block_size) ** dimension_power
                step_norm = lr * trust_scale * supported_magnitude / max(dimension_scale, eps)
                if max_update_ratio > 0.0:
                    max_norm = max_update_ratio * (safe_float(param.detach().norm()) + eps)
                    step_norm = min(step_norm, max_norm)

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                update = chosen_dir * step_norm
                param.add_(update, alpha=-1.0)

                candidate_trust = 0.5 + 0.5 * clamp_scalar(chosen_stats["recovery"], -1.0, 1.0)
                trust_ema = float(state["trust_ema"])
                trust_ema = trust_smoothing * trust_ema + (1.0 - trust_smoothing) * candidate_trust
                effective_decay = clamp_scalar(memory_decay + 0.08 * (1.0 - trust_ema), 0.0, 0.995)
                direction_memory.copy_(_normalize(direction_memory * effective_decay + chosen_dir * (1.0 - effective_decay), eps))
                prev_grad.copy_(grad)
                prev_update.copy_(update)
                state["grad_norm_ema"] = grad_norm_ema
                state["trust_ema"] = trust_ema
                state["last_candidate"] = chosen_name

                recovery_values.append(chosen_stats["recovery"])
                coherence_values.append(chosen_stats["coherence"])
                rotation_values.append(chosen_stats["rotation"])
                trust_values.append(trust_scale)
                relative_scale_values.append(relative_scale)
                counts[chosen_name] += 1.0
                active_params += 1

        denom = max(1, active_params)
        metrics = {
            "optimizer": self.optimizer_name,
            "recovery_score": average(recovery_values),
            "direction_coherence": average(coherence_values),
            "rotation_score": average(rotation_values),
            "trust_scale": average(trust_values),
            "relative_gradient_scale": average(relative_scale_values),
            "memory_selection_fraction": counts["memory"] / denom,
            "blend_selection_fraction": counts["blend"] / denom,
            "projected_selection_fraction": counts["projected"] / denom,
            "current_selection_fraction": counts["current"] / denom,
        }
        self._record_step(metrics)
        return loss_tensor


RecoveryDirectionOptimizer = DirectionRecoveryOptimizer
