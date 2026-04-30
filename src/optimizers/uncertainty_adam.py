from __future__ import annotations

import math

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import DEFAULT_EPS, average, bounded_scale, cosine_similarity, safe_float


class QuantumUncertaintyAdam(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        uncertainty_weight: float = 0.2,
        interference_weight: float = 0.2,
        reliability_strength: float = 0.15,
        exploration_strength: float = 0.08,
        min_scale: float = 0.6,
        max_scale: float = 1.4,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            uncertainty_weight=uncertainty_weight,
            interference_weight=interference_weight,
            reliability_strength=reliability_strength,
            exploration_strength=exploration_strength,
            min_scale=min_scale,
            max_scale=max_scale,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("QuantumUncertaintyAdam")

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)

        uncertainties: list[float] = []
        interferences: list[float] = []
        reliabilities: list[float] = []
        scales: list[float] = []
        explorations: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            uncertainty_weight = group["uncertainty_weight"]
            interference_weight = group["interference_weight"]
            reliability_strength = group["reliability_strength"]
            exploration_strength = group["exploration_strength"]
            min_scale = group["min_scale"]
            max_scale = group["max_scale"]
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
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)
                    state["prev_update"] = torch.zeros_like(param)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]
                prev_update = state["prev_update"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                exp_avg_hat = exp_avg / max(bias_correction1, DEFAULT_EPS)
                exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)
                adam_direction = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                mean_sq = safe_float(exp_avg_sq_hat.mean())
                mean = safe_float(exp_avg_hat.pow(2).mean())
                variance_proxy = max(0.0, mean_sq - mean)
                uncertainty_score = math.sqrt(variance_proxy) / (safe_float(exp_avg_hat.abs().mean()) + DEFAULT_EPS)
                interference_score = 0.5 * max(0.0, -cosine_similarity(grad, prev_grad))
                interference_score += 0.5 * max(0.0, -cosine_similarity(adam_direction, prev_update))
                reliability_score = max(0.0, cosine_similarity(grad, prev_grad)) * math.exp(-uncertainty_score)

                scale = bounded_scale(
                    1.0
                    + reliability_strength * reliability_score
                    - uncertainty_weight * uncertainty_score
                    - interference_weight * interference_score,
                    min_scale,
                    max_scale,
                )
                exploration_amount = exploration_strength * uncertainty_score * max(0.0, 1.0 - reliability_score)

                noise = torch.randn_like(adam_direction)
                noise_norm = noise.norm()
                if torch.isfinite(noise_norm) and noise_norm > 0:
                    noise = noise / (noise_norm + eps)
                else:
                    noise = torch.zeros_like(adam_direction)
                noise = noise * (exploration_amount * max(safe_float((adam_direction * lr).norm()), DEFAULT_EPS))

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                actual_update = adam_direction * scale + noise / max(lr, DEFAULT_EPS)
                param.add_(actual_update, alpha=-lr)

                prev_grad.copy_(grad)
                prev_update.copy_(actual_update.detach())

                uncertainties.append(uncertainty_score)
                interferences.append(interference_score)
                reliabilities.append(reliability_score)
                scales.append(scale)
                explorations.append(exploration_amount)

        self._record_step(
            {
                "loss": current_loss,
                "uncertainty_score": average(uncertainties),
                "interference_score": average(interferences),
                "reliability_score": average(reliabilities),
                "effective_lr_scale": average(scales),
                "exploration_amount": average(explorations),
            }
        )
        return loss_tensor


UncertaintyAdam = QuantumUncertaintyAdam
