from __future__ import annotations

import math

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import (
    DEFAULT_EPS,
    average,
    bounded_scale,
    clamp_scalar,
    cosine_similarity,
    safe_float,
    sign_flip_ratio,
    tensor_energy,
)


def _adamw_force(
    *,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    eps: float,
    force_mode: str,
) -> torch.Tensor:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    exp_avg_hat = exp_avg / max(bias_correction1, DEFAULT_EPS)
    exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)

    if force_mode == "rmsprop":
        return grad / (exp_avg_sq_hat.sqrt() + eps)
    return exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)


class CoherentMomentumPhysicalBaseline(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        friction: float = 0.08,
        energy_correction_strength: float = 0.15,
        oscillation_damping: float = 0.2,
        momentum_coupling: float = 0.25,
        energy_decay: float = 0.95,
        min_scale: float = 0.55,
        max_scale: float = 1.35,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            friction=friction,
            energy_correction_strength=energy_correction_strength,
            oscillation_damping=oscillation_damping,
            momentum_coupling=momentum_coupling,
            energy_decay=energy_decay,
            min_scale=min_scale,
            max_scale=max_scale,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("CoherentMomentumPhysicalBaseline")

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)
        potential = 0.0 if current_loss is None else float(current_loss)

        kinetics: list[float] = []
        potentials: list[float] = []
        totals: list[float] = []
        drifts: list[float] = []
        momentum_norms: list[float] = []
        oscillations: list[float] = []
        dampings: list[float] = []
        scales: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            friction = group["friction"]
            energy_correction_strength = group["energy_correction_strength"]
            oscillation_damping = group["oscillation_damping"]
            momentum_coupling = group["momentum_coupling"]
            energy_decay = group["energy_decay"]
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
                    state["prev_update"] = torch.zeros_like(param)
                    state["potential_ema"] = potential
                    state["prev_total_energy"] = 0.0

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_update = state["prev_update"]

                force = _adamw_force(
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    beta1=beta1,
                    beta2=beta2,
                    step=state["step"],
                    eps=eps,
                    force_mode="adamw",
                )
                exp_avg_hat = exp_avg / max(1.0 - beta1 ** state["step"], DEFAULT_EPS)

                kinetic = 0.5 * safe_float(exp_avg_hat.detach().float().pow(2).mean())
                state["potential_ema"] = energy_decay * float(state["potential_ema"]) + (1.0 - energy_decay) * potential
                potential_proxy = float(state["potential_ema"])
                total_energy = kinetic + potential_proxy
                energy_drift = total_energy - float(state["prev_total_energy"])

                oscillation = 0.5 * (1.0 - cosine_similarity(force, prev_update))
                oscillation += 0.5 * sign_flip_ratio(force, prev_update)
                oscillation = max(0.0, min(1.5, oscillation))

                effective_damping = friction + energy_correction_strength * max(0.0, energy_drift) + oscillation_damping * oscillation
                stability_drive = momentum_coupling * max(0.0, cosine_similarity(force, prev_update)) * max(0.0, -energy_drift)
                scale = bounded_scale(1.0 + stability_drive - effective_damping, min_scale, max_scale)

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                actual_update = force * scale
                param.add_(actual_update, alpha=-lr)

                prev_update.copy_(actual_update.detach())
                state["prev_total_energy"] = total_energy

                kinetics.append(kinetic)
                potentials.append(potential_proxy)
                totals.append(total_energy)
                drifts.append(energy_drift)
                momentum_norms.append(safe_float(exp_avg_hat.norm()))
                oscillations.append(oscillation)
                dampings.append(effective_damping)
                scales.append(scale)

        self._record_step(
            {
                "loss": current_loss,
                "kinetic_energy": average(kinetics),
                "potential_energy": average(potentials),
                "total_energy": average(totals),
                "energy_drift": average(drifts),
                "momentum_norm": average(momentum_norms),
                "oscillation_score": average(oscillations),
                "effective_damping": average(dampings),
                "effective_lr_scale": average(scales),
            }
        )
        return loss_tensor


class CoherentMomentumAdaptiveMassBaseline(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        friction: float = 0.06,
        energy_correction_strength: float = 0.18,
        oscillation_damping: float = 0.16,
        momentum_coupling: float = 0.35,
        loss_ema_decay: float = 0.95,
        energy_ema_decay: float = 0.96,
        drift_ema_decay: float = 0.90,
        drift_threshold: float = 0.025,
        correction_floor: float = 0.0,
        correction_cap: float = 0.35,
        predictive_damping_strength: float = 0.14,
        alignment_boost: float = 0.12,
        misalignment_damping: float = 0.18,
        alignment_min_scale: float = 0.85,
        alignment_max_scale: float = 1.18,
        min_step_scale: float = 0.55,
        max_step_scale: float = 1.42,
        normalized_energy_weight: float = 1.0,
        update_energy_weight: float = 0.55,
        force_energy_weight: float = 0.35,
        loss_change_weight: float = 0.65,
        force_mode: str = "adamw",
        reactive_baseline_mode: bool = False,
        use_normalized_energy: bool = True,
        use_energy_trend: bool = True,
        use_predictive_damping: bool = True,
        use_alignment_scaling: bool = True,
        use_adaptive_step_scale: bool = True,
        use_symplectic_correction: bool = True,
        use_oscillation_damping: bool = True,
        use_friction: bool = True,
        maximize: bool = False,
    ) -> None:
        if force_mode not in {"adamw", "rmsprop"}:
            raise ValueError("force_mode must be 'adamw' or 'rmsprop'")
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            friction=friction,
            energy_correction_strength=energy_correction_strength,
            oscillation_damping=oscillation_damping,
            momentum_coupling=momentum_coupling,
            loss_ema_decay=loss_ema_decay,
            energy_ema_decay=energy_ema_decay,
            drift_ema_decay=drift_ema_decay,
            drift_threshold=drift_threshold,
            correction_floor=correction_floor,
            correction_cap=correction_cap,
            predictive_damping_strength=predictive_damping_strength,
            alignment_boost=alignment_boost,
            misalignment_damping=misalignment_damping,
            alignment_min_scale=alignment_min_scale,
            alignment_max_scale=alignment_max_scale,
            min_step_scale=min_step_scale,
            max_step_scale=max_step_scale,
            normalized_energy_weight=normalized_energy_weight,
            update_energy_weight=update_energy_weight,
            force_energy_weight=force_energy_weight,
            loss_change_weight=loss_change_weight,
            force_mode=force_mode,
            reactive_baseline_mode=reactive_baseline_mode,
            use_normalized_energy=use_normalized_energy,
            use_energy_trend=use_energy_trend,
            use_predictive_damping=use_predictive_damping,
            use_alignment_scaling=use_alignment_scaling,
            use_adaptive_step_scale=use_adaptive_step_scale,
            use_symplectic_correction=use_symplectic_correction,
            use_oscillation_damping=use_oscillation_damping,
            use_friction=use_friction,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("CoherentMomentumAdaptiveMassBaseline")

    def _reactive_baseline_step_for_param(
        self,
        *,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, object],
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eps: float,
        friction: float,
        energy_correction_strength: float,
        oscillation_damping: float,
        momentum_coupling: float,
        loss_ema_decay: float,
        min_step_scale: float,
        max_step_scale: float,
        potential: float,
    ) -> dict[str, float]:
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
            state["prev_update"] = torch.zeros_like(param)
            state["loss_ema"] = potential
            state["prev_total_energy"] = 0.0

        state["step"] = int(state["step"]) + 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        prev_update = state["prev_update"]

        force = _adamw_force(
            grad=grad,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            beta1=beta1,
            beta2=beta2,
            step=int(state["step"]),
            eps=eps,
            force_mode="adamw",
        )
        exp_avg_hat = exp_avg / max(1.0 - beta1 ** int(state["step"]), DEFAULT_EPS)

        state["loss_ema"] = loss_ema_decay * float(state["loss_ema"]) + (1.0 - loss_ema_decay) * potential
        potential_proxy = float(state["loss_ema"])
        kinetic = 0.5 * safe_float(exp_avg_hat.detach().float().pow(2).mean())
        total_energy = kinetic + potential_proxy
        energy_drift = total_energy - float(state["prev_total_energy"])
        oscillation = 0.5 * (1.0 - cosine_similarity(force, prev_update))
        oscillation += 0.5 * sign_flip_ratio(force, prev_update)
        oscillation = clamp_scalar(oscillation, 0.0, 1.5)

        damping = friction + energy_correction_strength * max(0.0, energy_drift) + oscillation_damping * oscillation
        stability_drive = momentum_coupling * max(0.0, cosine_similarity(force, prev_update)) * max(0.0, -energy_drift)
        step_scale = bounded_scale(1.0 + stability_drive - damping, min_step_scale, max_step_scale)

        update_prev_cos = cosine_similarity(force, prev_update)
        if weight_decay > 0.0:
            param.mul_(1.0 - lr * weight_decay)
        update = force * step_scale
        param.add_(update, alpha=-lr)

        prev_update.copy_(update.detach())
        state["prev_total_energy"] = total_energy

        return {
            "kinetic_energy": kinetic,
            "potential_energy": potential_proxy,
            "total_energy": total_energy,
            "energy_drift": energy_drift,
            "raw_total_energy": total_energy,
            "normalized_total_energy": kinetic / (abs(potential_proxy) + DEFAULT_EPS),
            "kinetic_norm": kinetic / (abs(potential_proxy) + DEFAULT_EPS),
            "relative_loss_change": 0.0,
            "update_energy": tensor_energy(update),
            "force_energy": tensor_energy(force),
            "momentum_norm": safe_float(exp_avg_hat.norm()),
            "oscillation_score": oscillation,
            "effective_damping": damping,
            "predictive_damping": 0.0,
            "alignment_scale": 1.0,
            "adaptive_step_scale": step_scale,
            "effective_lr_scale": step_scale,
            "grad_momentum_cosine": cosine_similarity(grad, exp_avg_hat),
            "force_momentum_cosine": cosine_similarity(force, exp_avg_hat),
            "update_previous_update_cosine": update_prev_cos,
            "grad_previous_grad_cosine": 0.0,
            "gradient_acceleration": 0.0,
            "force_acceleration": 0.0,
            "sustained_drift_score": max(0.0, energy_drift),
            "heat_trend": max(0.0, energy_drift),
            "symplectic_correction_active": 0.0,
            "symplectic_correction_strength": 0.0,
            "force_mode_rmsprop": 0.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)
        potential = 0.0 if current_loss is None else float(current_loss)

        diagnostics: dict[str, list[float]] = {
            "kinetic_energy": [],
            "potential_energy": [],
            "total_energy": [],
            "raw_total_energy": [],
            "normalized_total_energy": [],
            "kinetic_norm": [],
            "relative_loss_change": [],
            "update_energy": [],
            "force_energy": [],
            "energy_drift": [],
            "momentum_norm": [],
            "oscillation_score": [],
            "effective_damping": [],
            "predictive_damping": [],
            "alignment_scale": [],
            "adaptive_step_scale": [],
            "effective_lr_scale": [],
            "grad_momentum_cosine": [],
            "force_momentum_cosine": [],
            "update_previous_update_cosine": [],
            "grad_previous_grad_cosine": [],
            "gradient_acceleration": [],
            "force_acceleration": [],
            "sustained_drift_score": [],
            "heat_trend": [],
            "symplectic_correction_active": [],
            "symplectic_correction_strength": [],
            "force_mode_rmsprop": [],
        }

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            friction = group["friction"]
            energy_correction_strength = group["energy_correction_strength"]
            oscillation_damping = group["oscillation_damping"]
            momentum_coupling = group["momentum_coupling"]
            loss_ema_decay = group["loss_ema_decay"]
            energy_ema_decay = group["energy_ema_decay"]
            drift_ema_decay = group["drift_ema_decay"]
            drift_threshold = group["drift_threshold"]
            correction_floor = group["correction_floor"]
            correction_cap = group["correction_cap"]
            predictive_damping_strength = group["predictive_damping_strength"]
            alignment_boost = group["alignment_boost"]
            misalignment_damping = group["misalignment_damping"]
            alignment_min_scale = group["alignment_min_scale"]
            alignment_max_scale = group["alignment_max_scale"]
            min_step_scale = group["min_step_scale"]
            max_step_scale = group["max_step_scale"]
            normalized_energy_weight = group["normalized_energy_weight"]
            update_energy_weight = group["update_energy_weight"]
            force_energy_weight = group["force_energy_weight"]
            loss_change_weight = group["loss_change_weight"]
            force_mode = str(group["force_mode"])
            reactive_baseline_mode = bool(group["reactive_baseline_mode"])
            use_normalized_energy = bool(group["use_normalized_energy"])
            use_energy_trend = bool(group["use_energy_trend"])
            use_predictive_damping = bool(group["use_predictive_damping"])
            use_alignment_scaling = bool(group["use_alignment_scaling"])
            use_adaptive_step_scale = bool(group["use_adaptive_step_scale"])
            use_symplectic_correction = bool(group["use_symplectic_correction"])
            use_oscillation_damping = bool(group["use_oscillation_damping"])
            use_friction = bool(group["use_friction"])
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
                if reactive_baseline_mode:
                    row = self._reactive_baseline_step_for_param(
                        param=param,
                        grad=grad,
                        state=state,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        eps=eps,
                        friction=friction,
                        energy_correction_strength=energy_correction_strength,
                        oscillation_damping=oscillation_damping,
                        momentum_coupling=momentum_coupling,
                        loss_ema_decay=loss_ema_decay,
                        min_step_scale=min_step_scale,
                        max_step_scale=max_step_scale,
                        potential=potential,
                    )
                    for key, value in row.items():
                        diagnostics[key].append(value)
                    continue

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["hamiltonian_momentum"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)
                    state["prev_force"] = torch.zeros_like(param)
                    state["prev_update"] = torch.zeros_like(param)
                    state["loss_ema"] = potential
                    state["energy_ema"] = 0.0
                    state["drift_ema"] = 0.0
                    state["oscillation_ema"] = 0.0
                    state["update_norm_ema"] = 0.0
                    state["prev_normalized_total_energy"] = 0.0

                state["step"] = int(state["step"]) + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                hamiltonian_momentum = state["hamiltonian_momentum"]
                prev_grad = state["prev_grad"]
                prev_force = state["prev_force"]
                prev_update = state["prev_update"]

                force = _adamw_force(
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    beta1=beta1,
                    beta2=beta2,
                    step=int(state["step"]),
                    eps=eps,
                    force_mode=force_mode,
                )

                state["loss_ema"] = loss_ema_decay * float(state["loss_ema"]) + (1.0 - loss_ema_decay) * potential
                loss_ema = float(state["loss_ema"])
                relative_loss_change = 0.0
                if current_loss is not None:
                    relative_loss_change = (potential - loss_ema) / (abs(loss_ema) + DEFAULT_EPS)

                grad_momentum_cos = cosine_similarity(grad, hamiltonian_momentum)
                force_momentum_cos = cosine_similarity(force, hamiltonian_momentum)
                grad_prev_grad_cos = cosine_similarity(grad, prev_grad)
                force_prev_update_cos = cosine_similarity(force, prev_update)
                oscillation = 0.5 * (1.0 - force_prev_update_cos) + 0.5 * sign_flip_ratio(force, prev_update)
                oscillation = clamp_scalar(oscillation, 0.0, 1.5)
                previous_oscillation_ema = float(state["oscillation_ema"])
                state["oscillation_ema"] = 0.9 * previous_oscillation_ema + 0.1 * oscillation
                oscillation_ema = float(state["oscillation_ema"])
                oscillation_rise = max(0.0, oscillation_ema - previous_oscillation_ema)

                grad_acceleration = safe_float((grad - prev_grad).norm()) / (safe_float(prev_grad.norm()) + DEFAULT_EPS)
                force_acceleration = safe_float((force - prev_force).norm()) / (safe_float(prev_force.norm()) + DEFAULT_EPS)

                provisional_momentum = hamiltonian_momentum + force
                provisional_update = (1.0 - momentum_coupling) * force + momentum_coupling * provisional_momentum

                force_energy = tensor_energy(force)
                update_energy = tensor_energy(provisional_update)
                kinetic = 0.5 * tensor_energy(provisional_momentum)
                kinetic_norm = kinetic / (abs(loss_ema) + DEFAULT_EPS) if current_loss is not None else kinetic / (force_energy + DEFAULT_EPS)
                normalized_total_energy = 0.0
                if use_normalized_energy:
                    normalized_total_energy = (
                        normalized_energy_weight * kinetic_norm
                        + update_energy_weight * (update_energy / (force_energy + DEFAULT_EPS))
                        + force_energy_weight * force_energy
                        + loss_change_weight * abs(relative_loss_change)
                    )
                raw_total_energy = kinetic + (potential if current_loss is not None else loss_ema)
                energy_drift = normalized_total_energy - float(state["prev_normalized_total_energy"])
                previous_energy_ema = float(state["energy_ema"])
                state["energy_ema"] = energy_ema_decay * previous_energy_ema + (1.0 - energy_ema_decay) * normalized_total_energy
                state["drift_ema"] = drift_ema_decay * float(state["drift_ema"]) + (1.0 - drift_ema_decay) * energy_drift
                sustained_drift_score = max(0.0, float(state["drift_ema"]) - drift_threshold) if use_energy_trend else 0.0
                heat_trend = max(0.0, float(state["energy_ema"]) - previous_energy_ema) + max(0.0, float(state["drift_ema"]))

                provisional_update_norm = safe_float((provisional_update * lr).norm())
                previous_update_norm_ema = float(state["update_norm_ema"])
                state["update_norm_ema"] = 0.92 * previous_update_norm_ema + 0.08 * provisional_update_norm
                update_norm_rise = max(
                    0.0,
                    provisional_update_norm / (float(state["update_norm_ema"]) + DEFAULT_EPS) - 1.0,
                )

                misalignment = average(
                    [
                        max(0.0, -grad_momentum_cos),
                        max(0.0, -force_momentum_cos),
                        max(0.0, -force_prev_update_cos),
                        max(0.0, -grad_prev_grad_cos),
                    ]
                )
                predictive_instability = 0.0
                if use_predictive_damping:
                    predictive_instability = average(
                        [
                            clamp_scalar(grad_acceleration, 0.0, 3.0),
                            clamp_scalar(force_acceleration, 0.0, 3.0),
                            clamp_scalar(oscillation_rise * 4.0, 0.0, 3.0),
                            clamp_scalar(update_norm_rise, 0.0, 3.0),
                            clamp_scalar(misalignment * 2.0, 0.0, 3.0),
                        ]
                    )

                friction_term = friction if use_friction else 0.0
                oscillation_term = oscillation_damping * oscillation_ema if use_oscillation_damping else 0.0
                energy_term = 0.0
                if use_normalized_energy:
                    energy_term += energy_correction_strength * max(0.0, normalized_total_energy - float(state["energy_ema"]))
                if use_energy_trend:
                    energy_term += energy_correction_strength * sustained_drift_score
                predictive_term = predictive_damping_strength * predictive_instability if use_predictive_damping else 0.0

                effective_damping = friction_term + oscillation_term + energy_term + predictive_term
                effective_damping = clamp_scalar(effective_damping, correction_floor, max(0.0, correction_cap))

                damped_momentum = hamiltonian_momentum * (1.0 - effective_damping) + force

                positive_alignment = average(
                    [
                        max(0.0, grad_momentum_cos),
                        max(0.0, force_momentum_cos),
                        max(0.0, force_prev_update_cos),
                        max(0.0, grad_prev_grad_cos),
                    ]
                )
                alignment_scale = 1.0
                if use_alignment_scaling:
                    stability_gate = max(0.0, 1.0 - sustained_drift_score - 0.25 * predictive_instability)
                    alignment_scale = bounded_scale(
                        1.0 + alignment_boost * positive_alignment * stability_gate - misalignment_damping * misalignment,
                        alignment_min_scale,
                        alignment_max_scale,
                    )

                adaptive_step_scale = 1.0
                if use_adaptive_step_scale:
                    stable_drive = momentum_coupling * max(0.0, cosine_similarity(damped_momentum, prev_update)) * max(0.0, 1.0 - predictive_instability)
                    adaptive_step_scale = bounded_scale(
                        1.0 + stable_drive - (energy_term + predictive_term + oscillation_term),
                        min_step_scale,
                        max_step_scale,
                    )

                symplectic_correction_strength = 0.0
                symplectic_correction_active = 0.0
                if use_symplectic_correction and (float(state["drift_ema"]) > drift_threshold or heat_trend > drift_threshold):
                    symplectic_correction_strength = clamp_scalar(
                        energy_correction_strength * (max(0.0, float(state["drift_ema"])) + max(0.0, heat_trend - drift_threshold)),
                        correction_floor,
                        correction_cap,
                    )
                    symplectic_correction_active = 1.0

                final_scale = bounded_scale(
                    alignment_scale * adaptive_step_scale / (1.0 + symplectic_correction_strength),
                    min_step_scale,
                    max_step_scale,
                )

                actual_update = ((1.0 - momentum_coupling) * force + momentum_coupling * damped_momentum) * final_scale
                update_prev_update_cos = cosine_similarity(actual_update, prev_update)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(actual_update, alpha=-lr)

                hamiltonian_momentum.copy_(damped_momentum.detach())
                prev_grad.copy_(grad)
                prev_force.copy_(force)
                prev_update.copy_(actual_update.detach())
                state["prev_normalized_total_energy"] = normalized_total_energy

                diagnostics["kinetic_energy"].append(kinetic)
                diagnostics["potential_energy"].append(loss_ema if current_loss is None else potential)
                diagnostics["total_energy"].append(raw_total_energy)
                diagnostics["raw_total_energy"].append(raw_total_energy)
                diagnostics["normalized_total_energy"].append(normalized_total_energy)
                diagnostics["kinetic_norm"].append(kinetic_norm)
                diagnostics["relative_loss_change"].append(relative_loss_change)
                diagnostics["update_energy"].append(update_energy)
                diagnostics["force_energy"].append(force_energy)
                diagnostics["energy_drift"].append(energy_drift)
                diagnostics["momentum_norm"].append(safe_float(damped_momentum.norm()))
                diagnostics["oscillation_score"].append(oscillation_ema)
                diagnostics["effective_damping"].append(effective_damping)
                diagnostics["predictive_damping"].append(predictive_term)
                diagnostics["alignment_scale"].append(alignment_scale)
                diagnostics["adaptive_step_scale"].append(adaptive_step_scale)
                diagnostics["effective_lr_scale"].append(final_scale)
                diagnostics["grad_momentum_cosine"].append(grad_momentum_cos)
                diagnostics["force_momentum_cosine"].append(force_momentum_cos)
                diagnostics["update_previous_update_cosine"].append(update_prev_update_cos)
                diagnostics["grad_previous_grad_cosine"].append(grad_prev_grad_cos)
                diagnostics["gradient_acceleration"].append(grad_acceleration)
                diagnostics["force_acceleration"].append(force_acceleration)
                diagnostics["sustained_drift_score"].append(sustained_drift_score)
                diagnostics["heat_trend"].append(heat_trend)
                diagnostics["symplectic_correction_active"].append(symplectic_correction_active)
                diagnostics["symplectic_correction_strength"].append(symplectic_correction_strength)
                diagnostics["force_mode_rmsprop"].append(1.0 if force_mode == "rmsprop" else 0.0)

        self._record_step({"loss": current_loss, **{key: average(values) for key, values in diagnostics.items()}})
        return loss_tensor


class CoherentMomentumRMSForceBaseline(CoherentMomentumAdaptiveMassBaseline):
    def __init__(self, params, **kwargs) -> None:
        kwargs.setdefault("force_mode", "rmsprop")
        super().__init__(params, **kwargs)


class CoherentMomentumRealBaseline(PhysicalOptimizerBase, torch.optim.Optimizer):
    _VALID_MODES = {
        "symplectic_euler",
        "leapfrog_with_closure",
        "dissipative_hamiltonian",
        "adam_preconditioned_hamiltonian",
        "reactive_baseline_compatibility",
    }
    _VALID_MASS_MODES = {"adaptive", "fixed"}

    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        mode: str = "dissipative_hamiltonian",
        mass_mode: str = "adaptive",
        fixed_mass: float = 1.0,
        use_adam_preconditioning: bool = True,
        adaptive_mass_trust: float = 0.08,
        mass_smoothing: float = 0.985,
        mass_anisotropy_cap: float = 1.6,
        mass_change_cap: float = 1.05,
        mass_warmup_steps: int = 40,
        mass_alignment_strength: float = 0.25,
        mass_shock_penalty: float = 1.4,
        min_inverse_mass: float = 0.55,
        max_inverse_mass: float = 1.45,
        friction: float = 0.035,
        use_friction: bool = True,
        energy_correction_strength: float = 0.07,
        use_energy_correction: bool = True,
        drift_threshold: float = 0.035,
        max_energy_correction: float = 0.16,
        relative_energy_floor: float = 0.1,
        use_decoupled_weight_decay: bool = True,
        maximize: bool = False,
        enable_step_diagnostics: bool = True,
        diagnostics_every_n_steps: int = 1,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")
        if mass_mode not in self._VALID_MASS_MODES:
            raise ValueError(f"mass_mode must be one of {sorted(self._VALID_MASS_MODES)}")
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            mode=mode,
            mass_mode=mass_mode,
            fixed_mass=fixed_mass,
            use_adam_preconditioning=use_adam_preconditioning,
            adaptive_mass_trust=adaptive_mass_trust,
            mass_smoothing=mass_smoothing,
            mass_anisotropy_cap=mass_anisotropy_cap,
            mass_change_cap=mass_change_cap,
            mass_warmup_steps=mass_warmup_steps,
            mass_alignment_strength=mass_alignment_strength,
            mass_shock_penalty=mass_shock_penalty,
            min_inverse_mass=min_inverse_mass,
            max_inverse_mass=max_inverse_mass,
            friction=friction,
            use_friction=use_friction,
            energy_correction_strength=energy_correction_strength,
            use_energy_correction=use_energy_correction,
            drift_threshold=drift_threshold,
            max_energy_correction=max_energy_correction,
            relative_energy_floor=relative_energy_floor,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self.enable_step_diagnostics = enable_step_diagnostics
        self.diagnostics_every_n_steps = diagnostics_every_n_steps
        self._initialize_physical_optimizer("CoherentMomentumRealBaseline")
        self._global_hamiltonian_state = {"prev_total_hamiltonian": None, "energy_ema": None, "last_mode_used": mode}
        self.wants_step_closure = True

    def state_dict(self):
        state = super().state_dict()
        state["physical_global_state"] = dict(self._global_hamiltonian_state)
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        physical_global_state = dict(state_dict.pop("physical_global_state", {}))
        result = super().load_state_dict(state_dict)
        self._global_hamiltonian_state = {
            "prev_total_hamiltonian": physical_global_state.get("prev_total_hamiltonian"),
            "energy_ema": physical_global_state.get("energy_ema"),
            "last_mode_used": physical_global_state.get("last_mode_used", self.param_groups[0].get("mode", "adam_preconditioned_hamiltonian")),
        }
        return result

    def _initialize_state(self, state: dict[str, object], param: torch.Tensor, potential: float) -> None:
        if len(state) != 0:
            return
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)
        state["hamiltonian_momentum"] = torch.zeros_like(param)
        state["prev_update"] = torch.zeros_like(param)
        state["potential_ema"] = potential
        state["prev_total_energy"] = 0.0
        state["inverse_mass_ema"] = torch.full_like(param, 1.0)
        state["last_mass_trust"] = 0.0
        state["last_mass_shock"] = 0.0

    def _inverse_mass_from_state(
        self,
        *,
        state: dict[str, object],
        grad: torch.Tensor,
        beta2: float,
        step: int,
        eps: float,
        use_adam_preconditioning: bool,
        mass_mode: str,
        fixed_mass: float,
        adaptive_mass_trust: float,
        mass_smoothing: float,
        mass_anisotropy_cap: float,
        mass_change_cap: float,
        mass_warmup_steps: int,
        mass_alignment_strength: float,
        mass_shock_penalty: float,
        min_inverse_mass: float,
        max_inverse_mass: float,
    ) -> torch.Tensor:
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        fixed = max(float(fixed_mass), DEFAULT_EPS)
        fixed_inverse_mass = torch.full_like(grad, 1.0 / fixed)
        if not use_adam_preconditioning or mass_mode == "fixed":
            state["inverse_mass_ema"] = fixed_inverse_mass
            return fixed_inverse_mass
        bias_correction2 = 1.0 - beta2**step
        exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)
        mass_diag = exp_avg_sq_hat.sqrt() + eps
        raw_inverse_mass = mass_diag.reciprocal()

        log_center = raw_inverse_mass.clamp_min(DEFAULT_EPS).log().mean().exp()
        normalized_inverse_mass = raw_inverse_mass / (log_center + DEFAULT_EPS)
        anisotropy_cap = max(1.0, float(mass_anisotropy_cap))
        normalized_inverse_mass = normalized_inverse_mass.clamp(1.0 / anisotropy_cap, anisotropy_cap)

        previous_inverse_mass = state.get("inverse_mass_ema", fixed_inverse_mass)
        if not isinstance(previous_inverse_mass, torch.Tensor):
            previous_inverse_mass = fixed_inverse_mass
        if step <= 1:
            previous_inverse_mass = fixed_inverse_mass

        warmup_steps = max(1, int(mass_warmup_steps))
        warmup_factor = min(1.0, step / warmup_steps)
        momentum = state.get("hamiltonian_momentum", torch.zeros_like(grad))
        previous_update = state.get("prev_update", torch.zeros_like(grad))
        momentum_alignment = max(
            0.0,
            cosine_similarity(grad, momentum if isinstance(momentum, torch.Tensor) else torch.zeros_like(grad)),
        )
        update_alignment = max(
            0.0,
            cosine_similarity(grad, previous_update if isinstance(previous_update, torch.Tensor) else torch.zeros_like(grad)),
        )
        alignment_gate = clamp_scalar(
            0.25 + float(mass_alignment_strength) * average([momentum_alignment, update_alignment]),
            0.1,
            1.0,
        )
        relative_mass_shock = safe_float(
            ((raw_inverse_mass - previous_inverse_mass).abs() / (previous_inverse_mass.abs() + eps)).mean()
        )
        shock_gate = 1.0 / (1.0 + float(mass_shock_penalty) * relative_mass_shock)
        trust = clamp_scalar(float(adaptive_mass_trust) * warmup_factor * alignment_gate * shock_gate, 0.0, 1.0)
        candidate_inverse_mass = fixed_inverse_mass * (1.0 - trust) + (fixed_inverse_mass * normalized_inverse_mass) * trust
        candidate_inverse_mass = candidate_inverse_mass.clamp(float(min_inverse_mass), float(max_inverse_mass))

        if mass_change_cap > 1.0:
            lower = previous_inverse_mass / float(mass_change_cap)
            upper = previous_inverse_mass * float(mass_change_cap)
            candidate_inverse_mass = torch.maximum(torch.minimum(candidate_inverse_mass, upper), lower)
        smoothing = clamp_scalar(float(mass_smoothing), 0.0, 0.9999)
        smoothing = clamp_scalar(max(smoothing, 1.0 - 0.35 * trust) + 0.15 * (1.0 - shock_gate), 0.0, 0.9999)
        smoothed_inverse_mass = previous_inverse_mass * smoothing + candidate_inverse_mass * (1.0 - smoothing)
        state["inverse_mass_ema"] = smoothed_inverse_mass
        state["last_mass_trust"] = trust
        state["last_mass_shock"] = relative_mass_shock
        return smoothed_inverse_mass

    def _v1_compatibility_step_for_param(
        self,
        *,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, object],
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eps: float,
        friction: float,
        energy_correction_strength: float,
        drift_threshold: float,
        max_energy_correction: float,
        potential: float,
        maximize: bool,
    ) -> dict[str, float]:
        self._initialize_state(state, param, potential)
        if maximize:
            grad = -grad
        state["step"] = int(state["step"]) + 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        momentum = state["hamiltonian_momentum"]
        prev_update = state["prev_update"]

        force = _adamw_force(
            grad=grad,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            beta1=beta1,
            beta2=beta2,
            step=int(state["step"]),
            eps=eps,
            force_mode="adamw",
        )
        inverse_mass = torch.ones_like(force)
        momentum.add_(force, alpha=-lr)
        if friction > 0.0:
            momentum.mul_(max(0.0, 1.0 - lr * friction))
        step_direction = inverse_mass * momentum
        if weight_decay > 0.0:
            param.mul_(1.0 - lr * weight_decay)
        param.add_(step_direction, alpha=lr)
        prev_update.copy_(step_direction.detach())

        kinetic = 0.5 * safe_float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).mean())
        potential_proxy = potential
        total_energy = kinetic + potential_proxy
        prev_total = float(state["prev_total_energy"])
        energy_drift = total_energy - prev_total
        relative_energy_drift = energy_drift / (abs(prev_total) + DEFAULT_EPS)
        damping_amount = 1.0 - max(0.0, 1.0 - lr * friction)
        if energy_correction_strength > 0.0 and relative_energy_drift > drift_threshold:
            correction = clamp_scalar(
                energy_correction_strength * (relative_energy_drift - drift_threshold),
                0.0,
                max_energy_correction,
            )
            momentum.mul_(1.0 - correction)
            damping_amount += correction
            kinetic = 0.5 * safe_float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).mean())
            total_energy = kinetic + potential_proxy
            energy_drift = total_energy - prev_total
            relative_energy_drift = energy_drift / (abs(prev_total) + DEFAULT_EPS)

        state["prev_total_energy"] = total_energy
        return {
            "potential_energy": potential_proxy,
            "kinetic_energy": kinetic,
            "total_hamiltonian": total_energy,
            "energy_drift": energy_drift,
            "relative_energy_drift": relative_energy_drift,
            "momentum_norm": safe_float(momentum.norm()),
            "parameter_step_norm": safe_float((step_direction * lr).norm()),
            "gradient_norm": safe_float(grad.norm()),
            "inverse_mass_mean": 1.0,
            "inverse_mass_std": 0.0,
            "damping_amount": damping_amount,
            "leapfrog_enabled": 0.0,
            "closure_recomputed_gradient": 0.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        current_loss = self.current_loss
        potential_before = 0.0 if current_loss is None else float(current_loss)

        diagnostics = {
            "potential_energy": [],
            "kinetic_energy": [],
            "total_hamiltonian": [],
            "energy_drift": [],
            "relative_energy_drift": [],
            "momentum_norm": [],
            "parameter_step_norm": [],
            "gradient_norm": [],
            "inverse_mass_mean": [],
            "inverse_mass_std": [],
            "mass_trust": [],
            "mass_shock": [],
            "damping_amount": [],
            "leapfrog_enabled": [],
            "closure_recomputed_gradient": [],
        }

        mode_used = None
        closure_recomputed = False
        updated_params: list[tuple[torch.Tensor, dict[str, object], torch.Tensor, torch.Tensor, float, float]] = []
        kinetic_sum = 0.0
        inverse_mass_sum = 0.0
        inverse_mass_sq_sum = 0.0
        total_elements = 0
        gradient_sq_sum = 0.0
        parameter_step_sq_sum = 0.0
        momentum_sq_sum = 0.0
        damping_amount = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            mode = str(group["mode"])
            mass_mode = str(group["mass_mode"])
            fixed_mass = float(group["fixed_mass"])
            use_adam_preconditioning = bool(group["use_adam_preconditioning"])
            adaptive_mass_trust = float(group["adaptive_mass_trust"])
            mass_smoothing = float(group["mass_smoothing"])
            mass_anisotropy_cap = float(group["mass_anisotropy_cap"])
            mass_change_cap = float(group["mass_change_cap"])
            mass_warmup_steps = int(group["mass_warmup_steps"])
            mass_alignment_strength = float(group["mass_alignment_strength"])
            mass_shock_penalty = float(group["mass_shock_penalty"])
            min_inverse_mass = float(group["min_inverse_mass"])
            max_inverse_mass = float(group["max_inverse_mass"])
            friction = float(group["friction"])
            use_friction = bool(group["use_friction"])
            energy_correction_strength = float(group["energy_correction_strength"])
            use_energy_correction = bool(group["use_energy_correction"])
            drift_threshold = float(group["drift_threshold"])
            max_energy_correction = float(group["max_energy_correction"])
            relative_energy_floor = float(group["relative_energy_floor"])
            use_decoupled_weight_decay = bool(group["use_decoupled_weight_decay"])
            maximize = bool(group["maximize"])
            mode_used = mode

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[param]
                self._initialize_state(state, param, potential_before)

                if mode == "reactive_baseline_compatibility":
                    row = self._v1_compatibility_step_for_param(
                        param=param,
                        grad=grad,
                        state=state,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        eps=eps,
                        friction=friction if use_friction else 0.0,
                        energy_correction_strength=energy_correction_strength if use_energy_correction else 0.0,
                        drift_threshold=drift_threshold,
                        max_energy_correction=max_energy_correction,
                        potential=potential_before,
                        maximize=False,
                    )
                    for key, value in row.items():
                        diagnostics[key].append(value)
                    continue

                state["step"] = int(state["step"]) + 1
                momentum = state["hamiltonian_momentum"]
                inverse_mass = self._inverse_mass_from_state(
                    state=state,
                    grad=grad,
                    beta2=beta2,
                    step=int(state["step"]),
                    eps=eps,
                    use_adam_preconditioning=use_adam_preconditioning,
                    mass_mode=mass_mode,
                    fixed_mass=fixed_mass,
                    adaptive_mass_trust=adaptive_mass_trust,
                    mass_smoothing=mass_smoothing,
                    mass_anisotropy_cap=mass_anisotropy_cap,
                    mass_change_cap=mass_change_cap,
                    mass_warmup_steps=mass_warmup_steps,
                    mass_alignment_strength=mass_alignment_strength,
                    mass_shock_penalty=mass_shock_penalty,
                    min_inverse_mass=min_inverse_mass,
                    max_inverse_mass=max_inverse_mass,
                )
                friction_factor = 1.0
                if use_friction and friction > 0.0:
                    friction_factor = math.exp(-lr * friction)
                use_leapfrog = mode == "leapfrog_with_closure" and closure is not None
                if mode in {"adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure is not None:
                    use_leapfrog = True
                if use_leapfrog:
                    momentum.mul_(math.sqrt(friction_factor)).add_(grad, alpha=-0.5 * lr)
                    step_direction = inverse_mass * momentum
                    if use_decoupled_weight_decay and weight_decay > 0.0:
                        param.mul_(1.0 - lr * weight_decay)
                    param.add_(step_direction, alpha=lr)
                    updated_params.append((param, state, step_direction.detach().clone(), inverse_mass.detach().clone(), friction_factor, lr))
                else:
                    momentum.mul_(friction_factor).add_(grad, alpha=-lr)
                    step_direction = inverse_mass * momentum
                    if use_decoupled_weight_decay and weight_decay > 0.0:
                        param.mul_(1.0 - lr * weight_decay)
                    param.add_(step_direction, alpha=lr)
                    updated_params.append((param, state, step_direction.detach().clone(), inverse_mass.detach().clone(), friction_factor, lr))

                total_elements += grad.numel()
                gradient_sq_sum += float(grad.detach().float().pow(2).sum().item())
                parameter_step_sq_sum += float((step_direction.detach().float() * lr).pow(2).sum().item())
                momentum_sq_sum += float(momentum.detach().float().pow(2).sum().item())
                inverse_mass_sum += float(inverse_mass.detach().float().sum().item())
                inverse_mass_sq_sum += float(inverse_mass.detach().float().pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).sum().item())
                damping_amount += 1.0 - friction_factor
                diagnostics["mass_trust"].append(float(state.get("last_mass_trust", 0.0)))
                diagnostics["mass_shock"].append(float(state.get("last_mass_shock", 0.0)))

        potential_after = potential_before
        loss_tensor = None
        if updated_params and closure is not None and mode_used in {"leapfrog_with_closure", "adam_preconditioned_hamiltonian", "dissipative_hamiltonian"}:
            with torch.enable_grad():
                loss_tensor = closure()
            potential_after = self.set_current_loss(loss_tensor)
            potential_after = potential_before if potential_after is None else float(potential_after)
            closure_recomputed = True
            kinetic_sum = 0.0
            momentum_sq_sum = 0.0
            gradient_sq_sum = 0.0
            total_elements = 0
            inverse_mass_sum = 0.0
            inverse_mass_sq_sum = 0.0
            for param, state, step_direction, inverse_mass, friction_factor, lr in updated_params:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if not torch.isfinite(grad).all():
                    continue
                momentum = state["hamiltonian_momentum"]
                momentum.mul_(math.sqrt(friction_factor)).add_(grad, alpha=-0.5 * lr)
                total_elements += grad.numel()
                gradient_sq_sum += float(grad.detach().float().pow(2).sum().item())
                momentum_sq_sum += float(momentum.detach().float().pow(2).sum().item())
                inverse_mass_sum += float(inverse_mass.detach().float().sum().item())
                inverse_mass_sq_sum += float(inverse_mass.detach().float().pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).sum().item())
                parameter_step_sq_sum += 0.0

        numel = max(total_elements, 1)
        kinetic_energy = kinetic_sum / numel
        inverse_mass_mean = inverse_mass_sum / numel
        inverse_mass_var = max(0.0, inverse_mass_sq_sum / numel - inverse_mass_mean**2)
        inverse_mass_std = math.sqrt(inverse_mass_var)
        total_hamiltonian = potential_after + kinetic_energy
        prev_total = self._global_hamiltonian_state.get("prev_total_hamiltonian")
        if prev_total is None:
            energy_drift = 0.0
            relative_energy_drift = 0.0
        else:
            energy_drift = total_hamiltonian - float(prev_total)
            drift_reference = max(abs(float(prev_total)), abs(total_hamiltonian), float(relative_energy_floor))
            relative_energy_drift = energy_drift / (drift_reference + DEFAULT_EPS)

        energy_correction_applied = 0.0
        if (
            updated_params
            and mode_used in {"dissipative_hamiltonian", "adam_preconditioned_hamiltonian", "leapfrog_with_closure"}
            and self.param_groups[0]["use_energy_correction"]
            and relative_energy_drift > float(self.param_groups[0]["drift_threshold"])
        ):
            energy_correction_applied = clamp_scalar(
                float(self.param_groups[0]["energy_correction_strength"]) * (relative_energy_drift - float(self.param_groups[0]["drift_threshold"])),
                0.0,
                float(self.param_groups[0]["max_energy_correction"]),
            )
            if energy_correction_applied > 0.0:
                for _, state, _, _, _, _ in updated_params:
                    momentum = state["hamiltonian_momentum"]
                    momentum.mul_(1.0 - energy_correction_applied)
                kinetic_energy *= (1.0 - energy_correction_applied) ** 2
                total_hamiltonian = potential_after + kinetic_energy
                if prev_total is None:
                    energy_drift = 0.0
                    relative_energy_drift = 0.0
                else:
                    energy_drift = total_hamiltonian - float(prev_total)
                    drift_reference = max(abs(float(prev_total)), abs(total_hamiltonian), float(relative_energy_floor))
                    relative_energy_drift = energy_drift / (drift_reference + DEFAULT_EPS)

        self._global_hamiltonian_state["prev_total_hamiltonian"] = total_hamiltonian
        self._global_hamiltonian_state["last_mode_used"] = mode_used

        momentum_norm = math.sqrt(max(momentum_sq_sum, 0.0))
        parameter_step_norm = math.sqrt(max(parameter_step_sq_sum, 0.0))
        gradient_norm_value = math.sqrt(max(gradient_sq_sum, 0.0))
        total_damping_amount = damping_amount / max(len(updated_params), 1) + energy_correction_applied

        diagnostics_row = {
            "loss": potential_after,
            "potential_energy": potential_after,
            "kinetic_energy": kinetic_energy,
            "total_hamiltonian": total_hamiltonian,
            "energy_drift": energy_drift,
            "relative_energy_drift": relative_energy_drift,
            "normalized_total_energy": kinetic_energy / (abs(potential_after) + DEFAULT_EPS),
            "momentum_norm": momentum_norm,
            "parameter_step_norm": parameter_step_norm,
            "gradient_norm": gradient_norm_value,
            "inverse_mass_mean": inverse_mass_mean,
            "inverse_mass_std": inverse_mass_std,
            "mass_trust": average(diagnostics["mass_trust"]),
            "mass_shock": average(diagnostics["mass_shock"]),
            "damping_amount": total_damping_amount,
            "effective_damping": total_damping_amount,
            "leapfrog_enabled": 1.0 if mode_used in {"leapfrog_with_closure", "adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure_recomputed else 0.0,
            "closure_recomputed_gradient": 1.0 if closure_recomputed else 0.0,
            "symplectic_euler_approximation": 0.0 if closure_recomputed else 1.0,
            "update_mode_code": {
                "symplectic_euler": 0.0,
                "leapfrog_with_closure": 1.0,
                "dissipative_hamiltonian": 2.0,
                "adam_preconditioned_hamiltonian": 3.0,
                "reactive_baseline_compatibility": 4.0,
            }.get(str(mode_used), -1.0),
        }
        recorded = self._record_step(diagnostics_row)

        if not diagnostics["potential_energy"]:
            for key in diagnostics:
                value = recorded.get(key)
                if value is not None:
                    diagnostics[key].append(float(value))
        else:
            for key, value in recorded.items():
                if key in diagnostics:
                    diagnostics[key].append(float(value))
        return loss_tensor


SymplecticAdam = CoherentMomentumRealBaseline
