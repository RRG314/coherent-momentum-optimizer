from __future__ import annotations

import math

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import DEFAULT_EPS, average, bounded_scale, safe_float, tensor_energy, tensor_entropy


class ThermodynamicAdam(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        entropy_weight: float = 0.15,
        energy_weight: float = 0.25,
        temperature_decay: float = 0.96,
        cooling_strength: float = 0.12,
        reheating_strength: float = 0.12,
        max_temperature: float = 1.5,
        min_temperature: float = 0.05,
        min_scale: float = 0.55,
        max_scale: float = 1.35,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            entropy_weight=entropy_weight,
            energy_weight=energy_weight,
            temperature_decay=temperature_decay,
            cooling_strength=cooling_strength,
            reheating_strength=reheating_strength,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
            min_scale=min_scale,
            max_scale=max_scale,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("ThermodynamicAdam")

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)

        grad_energies: list[float] = []
        update_energies: list[float] = []
        entropies: list[float] = []
        temperatures: list[float] = []
        heat_spikes: list[float] = []
        scales: list[float] = []
        coolings: list[float] = []
        reheatings: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            entropy_weight = group["entropy_weight"]
            energy_weight = group["energy_weight"]
            temperature_decay = group["temperature_decay"]
            cooling_strength = group["cooling_strength"]
            reheating_strength = group["reheating_strength"]
            max_temperature = group["max_temperature"]
            min_temperature = group["min_temperature"]
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
                    state["temperature"] = float(min_temperature)
                    state["prev_signal"] = 0.0

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

                grad_energy = tensor_energy(grad)
                entropy = tensor_entropy(grad)
                signal = energy_weight * grad_energy + entropy_weight * entropy
                prev_signal = float(state["prev_signal"])
                state["temperature"] = temperature_decay * float(state["temperature"]) + (1.0 - temperature_decay) * signal
                temperature = max(min_temperature, min(max_temperature, float(state["temperature"])))
                heat_spike = max(0.0, signal - prev_signal)

                cooling = cooling_strength * heat_spike * (temperature / (max_temperature + DEFAULT_EPS))
                reheating = 0.0
                if self.stagnation_counter > 0:
                    reheating = reheating_strength * min(1.0, self.stagnation_counter / 6.0) * (1.0 - temperature / (max_temperature + DEFAULT_EPS))
                    reheating *= max(0.0, 1.0 - entropy)

                boltzmann = 1.0
                if energy_weight > 0.0:
                    boltzmann = math.exp(-energy_weight * heat_spike / (temperature + DEFAULT_EPS))
                raw_scale = boltzmann * max(0.05, 1.0 - cooling + reheating)
                scale = bounded_scale(raw_scale, min_scale, max_scale)

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(adam_direction, alpha=-(lr * scale))

                state["prev_signal"] = signal
                actual_update_energy = tensor_energy(adam_direction * scale)
                grad_energies.append(grad_energy)
                update_energies.append(actual_update_energy)
                entropies.append(entropy)
                temperatures.append(temperature)
                heat_spikes.append(heat_spike)
                scales.append(scale)
                coolings.append(cooling)
                reheatings.append(reheating)

        self._record_step(
            {
                "loss": current_loss,
                "gradient_energy": average(grad_energies),
                "update_energy": average(update_energies),
                "gradient_entropy": average(entropies),
                "entropy": average(entropies),
                "temperature": average(temperatures),
                "heat_spike": average(heat_spikes),
                "instability_score": average(heat_spikes),
                "effective_lr_scale": average(scales),
                "cooling_amount": average(coolings),
                "reheating_amount": average(reheatings),
            }
        )
        return loss_tensor
