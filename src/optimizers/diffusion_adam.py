from __future__ import annotations

import math

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import DEFAULT_EPS, average, bounded_scale, safe_float, tensor_entropy


class DiffusionAdam(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        diffusion_strength: float = 0.04,
        diffusion_decay: float = 0.98,
        entropy_scaled_noise: bool = True,
        stagnation_trigger: int = 8,
        min_noise: float = 0.0,
        max_noise: float = 0.25,
        noise_to_update_cap: float = 0.35,
        aligned_noise_weight: float = 0.25,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            diffusion_strength=diffusion_strength,
            diffusion_decay=diffusion_decay,
            entropy_scaled_noise=entropy_scaled_noise,
            stagnation_trigger=stagnation_trigger,
            min_noise=min_noise,
            max_noise=max_noise,
            noise_to_update_cap=noise_to_update_cap,
            aligned_noise_weight=aligned_noise_weight,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("DiffusionAdam")

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)

        noise_norms: list[float] = []
        update_norms: list[float] = []
        ratios: list[float] = []
        scales: list[float] = []
        entropies: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            diffusion_strength = group["diffusion_strength"]
            diffusion_decay = group["diffusion_decay"]
            entropy_scaled_noise = bool(group["entropy_scaled_noise"])
            stagnation_trigger = int(group["stagnation_trigger"])
            min_noise = group["min_noise"]
            max_noise = group["max_noise"]
            noise_to_update_cap = group["noise_to_update_cap"]
            aligned_noise_weight = group["aligned_noise_weight"]
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

                base_update = adam_direction * lr
                base_norm = max(safe_float(base_update.norm()), DEFAULT_EPS)
                entropy = tensor_entropy(grad)

                sigma = diffusion_strength * (diffusion_decay ** max(0, state["step"] - 1))
                if entropy_scaled_noise:
                    sigma *= 0.5 + entropy
                if state["step"] <= 4:
                    sigma *= 1.2
                if self.stagnation_counter >= stagnation_trigger:
                    sigma *= 1.0 + 0.25 * min(4.0, self.stagnation_counter - stagnation_trigger + 1)
                sigma = bounded_scale(sigma, min_noise, max_noise)

                iso_noise = torch.randn_like(adam_direction)
                grad_dir = grad / (grad.norm() + eps)
                aligned_noise = grad_dir * torch.randn((), device=grad.device, dtype=grad.dtype)
                noise = (1.0 - aligned_noise_weight) * iso_noise + aligned_noise_weight * aligned_noise
                noise_norm = noise.norm()
                if torch.isfinite(noise_norm) and noise_norm > 0:
                    noise = noise / (noise_norm + eps)
                else:
                    noise = torch.zeros_like(adam_direction)

                noise = noise * (sigma * base_norm)
                ratio = safe_float(noise.norm()) / base_norm
                if noise_to_update_cap >= 0.0 and ratio > noise_to_update_cap:
                    noise = noise * (noise_to_update_cap / (ratio + DEFAULT_EPS))
                    ratio = safe_float(noise.norm()) / base_norm

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                actual_update = base_update + noise
                param.add_(actual_update, alpha=-1.0)

                noise_norms.append(safe_float(noise.norm()))
                update_norms.append(safe_float(actual_update.norm()))
                ratios.append(ratio)
                scales.append(sigma)
                entropies.append(entropy)

        self._record_step(
            {
                "loss": current_loss,
                "noise_norm": average(noise_norms),
                "update_norm": average(update_norms),
                "noise_to_update_ratio": average(ratios),
                "diffusion_scale": average(scales),
                "effective_diffusion_scale": average(scales),
                "entropy": average(entropies),
                "stagnation_counter": self.stagnation_counter,
            }
        )
        return loss_tensor
