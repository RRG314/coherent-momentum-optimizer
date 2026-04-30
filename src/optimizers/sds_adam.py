from __future__ import annotations

import math

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import (
    DEFAULT_EPS,
    bounded_scale,
    gradient_norm,
    safe_float,
    smooth_sigmoid,
    tensor_entropy,
    update_ratio,
)


class SDSAdam(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        inner_horizon: float = 5e-4,
        outer_horizon: float = 2.5e-2,
        horizon_sharpness: float = 12.0,
        cooling_strength: float = 0.35,
        reheating_strength: float = 0.15,
        entropy_weight: float = 0.1,
        max_scale: float = 1.5,
        min_scale: float = 0.5,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            inner_horizon=inner_horizon,
            outer_horizon=outer_horizon,
            horizon_sharpness=horizon_sharpness,
            cooling_strength=cooling_strength,
            reheating_strength=reheating_strength,
            entropy_weight=entropy_weight,
            max_scale=max_scale,
            min_scale=min_scale,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("SDSAdam")

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)
        feedback = self.external_metrics
        validation_gap = float(feedback.get("validation_gap", 0.0) or 0.0)

        ratios: list[float] = []
        gradient_ratios: list[float] = []
        entropies: list[float] = []
        scales: list[float] = []
        coolings: list[float] = []
        reheatings: list[float] = []
        horizon_codes: list[float] = []
        divergence_flags: list[float] = []

        all_params = [param for group in self.param_groups for param in group["params"]]

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            inner_horizon = group["inner_horizon"]
            outer_horizon = group["outer_horizon"]
            sharpness = group["horizon_sharpness"]
            cooling_strength = group["cooling_strength"]
            reheating_strength = group["reheating_strength"]
            entropy_weight = group["entropy_weight"]
            max_scale = group["max_scale"]
            min_scale = group["min_scale"]
            maximize = bool(group["maximize"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    divergence_flags.append(1.0)
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["grad_norm_ema"] = 0.0

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                exp_avg_hat = exp_avg / max(bias_correction1, DEFAULT_EPS)
                exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)
                adam_direction = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                grad_norm_value = safe_float(grad.norm())
                state["grad_norm_ema"] = 0.95 * float(state["grad_norm_ema"]) + 0.05 * grad_norm_value
                gradient_ratio = grad_norm_value / (float(state["grad_norm_ema"]) + DEFAULT_EPS)
                entropy = tensor_entropy(grad)

                raw_ratio = update_ratio(adam_direction, param, lr=lr)
                inner_gate = smooth_sigmoid(inner_horizon - raw_ratio, sharpness)
                outer_gate = smooth_sigmoid(raw_ratio - outer_horizon, sharpness)
                stagnation_factor = min(1.5, self.stagnation_counter / 4.0)

                reheating = reheating_strength * inner_gate * (1.0 + 0.5 * stagnation_factor) * (1.0 - 0.5 * entropy)
                cooling = cooling_strength * outer_gate * (1.0 + max(0.0, gradient_ratio - 1.0) + max(0.0, validation_gap))
                cooling += entropy_weight * entropy * max(0.0, gradient_ratio - 1.0)
                if self.stagnation_counter > 0 and entropy < 0.25:
                    reheating += reheating_strength * 0.25 * min(1.0, self.stagnation_counter / 6.0)

                scale = bounded_scale(math.exp(reheating - cooling), min_scale, max_scale)
                actual_ratio = raw_ratio * scale

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(adam_direction, alpha=-(lr * scale))

                if actual_ratio < inner_horizon:
                    horizon_codes.append(-1.0)
                elif actual_ratio > outer_horizon:
                    horizon_codes.append(1.0)
                else:
                    horizon_codes.append(0.0)

                diverged = float((not math.isfinite(actual_ratio)) or actual_ratio > max(outer_horizon * 4.0, 1.0))
                divergence_flags.append(diverged)
                ratios.append(actual_ratio)
                gradient_ratios.append(gradient_ratio)
                entropies.append(entropy)
                scales.append(scale)
                coolings.append(cooling)
                reheatings.append(reheating)

        mean_horizon_code = sum(horizon_codes) / max(1, len(horizon_codes))
        if mean_horizon_code <= -0.33:
            horizon_state = "inner"
        elif mean_horizon_code >= 0.33:
            horizon_state = "outer"
        else:
            horizon_state = "stable"

        self._record_step(
            {
                "loss": current_loss,
                "update_ratio": sum(ratios) / max(1, len(ratios)),
                "gradient_ratio": sum(gradient_ratios) / max(1, len(gradient_ratios)),
                "entropy": sum(entropies) / max(1, len(entropies)),
                "horizon_state": horizon_state,
                "horizon_code": mean_horizon_code,
                "effective_lr_scale": sum(scales) / max(1, len(scales)),
                "damping_amount": sum(coolings) / max(1, len(coolings)),
                "reheating_amount": sum(reheatings) / max(1, len(reheatings)),
                "divergence_flag": sum(divergence_flags) / max(1, len(divergence_flags)),
                "global_gradient_norm": gradient_norm(all_params),
            }
        )
        return loss_tensor


HorizonAdam = SDSAdam
