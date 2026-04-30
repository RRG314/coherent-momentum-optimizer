from __future__ import annotations

import math
from collections import defaultdict

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
    smooth_sigmoid,
    tensor_energy,
    tensor_entropy,
    update_ratio,
)


def _adamw_direction(
    *,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    exp_avg_hat = exp_avg / max(bias_correction1, DEFAULT_EPS)
    exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)
    direction = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)
    return direction, exp_avg_hat


def _geometric_combine(scales: list[float], min_scale: float, max_scale: float) -> float:
    valid = [scale for scale in scales if math.isfinite(scale) and scale > 0.0]
    if not valid:
        return 1.0
    log_mean = sum(math.log(scale) for scale in valid) / len(valid)
    return bounded_scale(math.exp(log_mean), min_scale, max_scale)


class UnifiedPhysicsAdam(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        enable_sds: bool = True,
        enable_coherence: bool = True,
        enable_thermodynamic: bool = True,
        enable_diffusion: bool = False,
        enable_hamiltonian: bool = True,
        enable_uncertainty: bool = True,
        inner_horizon: float = 5e-4,
        outer_horizon: float = 2.5e-2,
        horizon_sharpness: float = 12.0,
        sds_cooling_strength: float = 0.25,
        sds_reheating_strength: float = 0.12,
        sds_entropy_weight: float = 0.08,
        alignment_strength: float = 0.12,
        coherence_strength: float = 0.10,
        rotation_penalty: float = 0.18,
        misalignment_damping: float = 0.18,
        layerwise_mode: bool = True,
        global_mode: bool = True,
        thermodynamic_entropy_weight: float = 0.10,
        thermodynamic_energy_weight: float = 0.20,
        temperature_decay: float = 0.96,
        thermodynamic_cooling_strength: float = 0.08,
        thermodynamic_reheating_strength: float = 0.08,
        max_temperature: float = 1.5,
        min_temperature: float = 0.05,
        diffusion_strength: float = 0.02,
        diffusion_decay: float = 0.985,
        entropy_scaled_noise: bool = True,
        stagnation_trigger: int = 8,
        min_noise: float = 0.0,
        max_noise: float = 0.20,
        noise_to_update_cap: float = 0.20,
        aligned_noise_weight: float = 0.2,
        friction: float = 0.06,
        energy_correction_strength: float = 0.12,
        oscillation_damping: float = 0.12,
        momentum_coupling: float = 0.28,
        loss_ema_decay: float = 0.95,
        energy_ema_decay: float = 0.96,
        drift_ema_decay: float = 0.90,
        drift_threshold: float = 0.02,
        correction_floor: float = 0.0,
        correction_cap: float = 0.25,
        uncertainty_weight: float = 0.15,
        interference_weight: float = 0.15,
        reliability_strength: float = 0.12,
        exploration_strength: float = 0.05,
        min_step_scale: float = 0.60,
        max_step_scale: float = 1.35,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            enable_sds=enable_sds,
            enable_coherence=enable_coherence,
            enable_thermodynamic=enable_thermodynamic,
            enable_diffusion=enable_diffusion,
            enable_hamiltonian=enable_hamiltonian,
            enable_uncertainty=enable_uncertainty,
            inner_horizon=inner_horizon,
            outer_horizon=outer_horizon,
            horizon_sharpness=horizon_sharpness,
            sds_cooling_strength=sds_cooling_strength,
            sds_reheating_strength=sds_reheating_strength,
            sds_entropy_weight=sds_entropy_weight,
            alignment_strength=alignment_strength,
            coherence_strength=coherence_strength,
            rotation_penalty=rotation_penalty,
            misalignment_damping=misalignment_damping,
            layerwise_mode=layerwise_mode,
            global_mode=global_mode,
            thermodynamic_entropy_weight=thermodynamic_entropy_weight,
            thermodynamic_energy_weight=thermodynamic_energy_weight,
            temperature_decay=temperature_decay,
            thermodynamic_cooling_strength=thermodynamic_cooling_strength,
            thermodynamic_reheating_strength=thermodynamic_reheating_strength,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
            diffusion_strength=diffusion_strength,
            diffusion_decay=diffusion_decay,
            entropy_scaled_noise=entropy_scaled_noise,
            stagnation_trigger=stagnation_trigger,
            min_noise=min_noise,
            max_noise=max_noise,
            noise_to_update_cap=noise_to_update_cap,
            aligned_noise_weight=aligned_noise_weight,
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
            uncertainty_weight=uncertainty_weight,
            interference_weight=interference_weight,
            reliability_strength=reliability_strength,
            exploration_strength=exploration_strength,
            min_step_scale=min_step_scale,
            max_step_scale=max_step_scale,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("UnifiedPhysicsAdam")

    def _init_state(self, state: dict[str, object], param: torch.Tensor, loss_value: float) -> None:
        if len(state) > 0:
            return
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)
        state["prev_grad"] = torch.zeros_like(param)
        state["prev_force"] = torch.zeros_like(param)
        state["prev_update"] = torch.zeros_like(param)
        state["grad_norm_ema"] = 0.0
        state["update_norm_ema"] = 0.0
        state["oscillation_ema"] = 0.0
        state["loss_ema"] = loss_value
        state["temperature"] = 0.0
        state["prev_thermo_signal"] = 0.0
        state["prev_normalized_total_energy"] = 0.0
        state["prev_raw_total_energy"] = 0.0
        state["energy_ema"] = 0.0
        state["drift_ema"] = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)
        potential = 0.0 if current_loss is None else float(current_loss)
        validation_gap = float(self.external_metrics.get("validation_gap", 0.0) or 0.0)

        diagnostics: dict[str, list[float]] = defaultdict(list)
        horizon_codes: list[float] = []

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            min_step_scale = float(group["min_step_scale"])
            max_step_scale = float(group["max_step_scale"])
            maximize = bool(group["maximize"])

            caches: list[dict[str, object]] = []
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    diagnostics["divergence_flag"].append(1.0)
                    continue

                state = self.state[param]
                self._init_state(state, param, potential)
                state["step"] = int(state["step"]) + 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]
                prev_force = state["prev_force"]
                prev_update = state["prev_update"]

                force, exp_avg_hat = _adamw_direction(
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    beta1=beta1,
                    beta2=beta2,
                    step=int(state["step"]),
                    eps=eps,
                )
                base_update = force * lr
                grad_norm_value = safe_float(grad.norm())
                prev_grad_norm_ema = float(state["grad_norm_ema"])
                gradient_ratio = grad_norm_value / (prev_grad_norm_ema + DEFAULT_EPS) if prev_grad_norm_ema > 0.0 else 1.0
                grad_entropy = tensor_entropy(grad)
                grad_energy = tensor_energy(grad)
                update_energy = tensor_energy(base_update)
                force_energy = tensor_energy(force)
                kinetic = 0.5 * safe_float(exp_avg_hat.detach().float().pow(2).mean())
                local_update_ratio = update_ratio(force, param, lr=lr)

                grad_momentum_cos = cosine_similarity(grad, exp_avg_hat)
                grad_prev_grad_cos = cosine_similarity(grad, prev_grad)
                force_momentum_cos = cosine_similarity(force, exp_avg_hat)
                update_prev_update_cos = cosine_similarity(force, prev_update)
                rotation_score = 0.5 * (1.0 - grad_prev_grad_cos) + 0.5 * sign_flip_ratio(force, prev_update)
                rotation_score = clamp_scalar(rotation_score, 0.0, 1.5)
                coherence_score = average([max(0.0, grad_momentum_cos), max(0.0, update_prev_update_cos)])

                mean_sq = safe_float((exp_avg_sq / max(1.0 - beta2 ** int(state["step"]), DEFAULT_EPS)).mean())
                mean_dir_sq = safe_float(exp_avg_hat.pow(2).mean())
                variance_proxy = max(0.0, mean_sq - mean_dir_sq)
                uncertainty_score = math.sqrt(variance_proxy) / (safe_float(exp_avg_hat.abs().mean()) + DEFAULT_EPS)
                interference_score = average(
                    [
                        max(0.0, -cosine_similarity(grad, prev_grad)),
                        max(0.0, -cosine_similarity(force, prev_update)),
                        max(0.0, -cosine_similarity(exp_avg_hat, prev_update)),
                    ]
                )
                reliability_score = average(
                    [max(0.0, grad_prev_grad_cos), max(0.0, grad_momentum_cos), max(0.0, update_prev_update_cos)]
                ) * math.exp(-uncertainty_score)

                prev_loss_ema = float(state["loss_ema"])
                loss_ema = float(group["loss_ema_decay"]) * prev_loss_ema + (1.0 - float(group["loss_ema_decay"])) * potential
                relative_loss_change = (potential - prev_loss_ema) / (abs(prev_loss_ema) + DEFAULT_EPS)
                raw_total_energy = kinetic + loss_ema
                kinetic_norm = kinetic / (abs(loss_ema) + DEFAULT_EPS)
                normalized_total_energy = (
                    kinetic_norm
                    + float(group["thermodynamic_energy_weight"]) * (update_energy / (force_energy + DEFAULT_EPS))
                    + 0.35 * force_energy
                    + 0.55 * abs(relative_loss_change)
                )
                prev_normalized_total = float(state["prev_normalized_total_energy"])
                energy_ema = float(group["energy_ema_decay"]) * float(state["energy_ema"]) + (1.0 - float(group["energy_ema_decay"])) * normalized_total_energy
                energy_drift = normalized_total_energy - prev_normalized_total
                drift_ema = float(group["drift_ema_decay"]) * float(state["drift_ema"]) + (1.0 - float(group["drift_ema_decay"])) * energy_drift
                sustained_drift_score = max(0.0, drift_ema - float(group["drift_threshold"]))

                thermo_signal = float(group["thermodynamic_energy_weight"]) * grad_energy + float(group["thermodynamic_entropy_weight"]) * grad_entropy
                temperature = float(group["temperature_decay"]) * float(state["temperature"]) + (1.0 - float(group["temperature_decay"])) * thermo_signal
                temperature = clamp_scalar(temperature, float(group["min_temperature"]), float(group["max_temperature"]))
                heat_spike = max(0.0, thermo_signal - float(state["prev_thermo_signal"]))

                gradient_acceleration = safe_float((grad - prev_grad).norm()) / (safe_float(prev_grad.norm()) + DEFAULT_EPS)
                force_acceleration = safe_float((force - prev_force).norm()) / (safe_float(prev_force.norm()) + DEFAULT_EPS)
                base_update_norm = max(safe_float(base_update.norm()), DEFAULT_EPS)
                update_norm_ema = 0.95 * float(state["update_norm_ema"]) + 0.05 * base_update_norm
                update_norm_rise = base_update_norm / (update_norm_ema + DEFAULT_EPS)
                oscillation_ema = 0.90 * float(state["oscillation_ema"]) + 0.10 * rotation_score
                heat_trend = max(0.0, temperature - float(state["temperature"])) + max(0.0, drift_ema)

                caches.append(
                    {
                        "param": param,
                        "grad": grad,
                        "state": state,
                        "force": force,
                        "exp_avg_hat": exp_avg_hat,
                        "base_update": base_update,
                        "base_update_norm": base_update_norm,
                        "grad_norm_value": grad_norm_value,
                        "gradient_ratio": gradient_ratio,
                        "grad_entropy": grad_entropy,
                        "grad_energy": grad_energy,
                        "update_energy": update_energy,
                        "force_energy": force_energy,
                        "kinetic": kinetic,
                        "local_update_ratio": local_update_ratio,
                        "grad_momentum_cos": grad_momentum_cos,
                        "grad_prev_grad_cos": grad_prev_grad_cos,
                        "force_momentum_cos": force_momentum_cos,
                        "update_prev_update_cos": update_prev_update_cos,
                        "rotation_score": rotation_score,
                        "coherence_score": coherence_score,
                        "uncertainty_score": uncertainty_score,
                        "interference_score": interference_score,
                        "reliability_score": reliability_score,
                        "loss_ema": loss_ema,
                        "relative_loss_change": relative_loss_change,
                        "raw_total_energy": raw_total_energy,
                        "kinetic_norm": kinetic_norm,
                        "normalized_total_energy": normalized_total_energy,
                        "energy_ema": energy_ema,
                        "energy_drift": energy_drift,
                        "drift_ema": drift_ema,
                        "sustained_drift_score": sustained_drift_score,
                        "temperature": temperature,
                        "thermo_signal": thermo_signal,
                        "heat_spike": heat_spike,
                        "gradient_acceleration": gradient_acceleration,
                        "force_acceleration": force_acceleration,
                        "update_norm_ema": update_norm_ema,
                        "update_norm_rise": update_norm_rise,
                        "oscillation_ema": oscillation_ema,
                        "heat_trend": heat_trend,
                    }
                )

            if not caches:
                continue

            global_metrics = {
                "update_ratio": average(item["local_update_ratio"] for item in caches),
                "gradient_ratio": average(item["gradient_ratio"] for item in caches),
                "entropy": average(item["grad_entropy"] for item in caches),
                "rotation_score": average(item["rotation_score"] for item in caches),
                "coherence_score": average(item["coherence_score"] for item in caches),
                "grad_momentum_cos": average(item["grad_momentum_cos"] for item in caches),
                "grad_prev_grad_cos": average(item["grad_prev_grad_cos"] for item in caches),
                "force_momentum_cos": average(item["force_momentum_cos"] for item in caches),
                "update_prev_update_cos": average(item["update_prev_update_cos"] for item in caches),
                "temperature": average(item["temperature"] for item in caches),
                "heat_spike": average(item["heat_spike"] for item in caches),
                "kinetic": average(item["kinetic"] for item in caches),
                "raw_total_energy": average(item["raw_total_energy"] for item in caches),
                "normalized_total_energy": average(item["normalized_total_energy"] for item in caches),
                "energy_drift": average(item["energy_drift"] for item in caches),
                "sustained_drift_score": average(item["sustained_drift_score"] for item in caches),
                "uncertainty_score": average(item["uncertainty_score"] for item in caches),
                "interference_score": average(item["interference_score"] for item in caches),
                "reliability_score": average(item["reliability_score"] for item in caches),
                "gradient_acceleration": average(item["gradient_acceleration"] for item in caches),
                "force_acceleration": average(item["force_acceleration"] for item in caches),
                "update_norm_rise": average(item["update_norm_rise"] for item in caches),
                "heat_trend": average(item["heat_trend"] for item in caches),
            }

            for item in caches:
                param = item["param"]
                grad = item["grad"]
                state = item["state"]
                force = item["force"]
                base_update = item["base_update"]
                base_update_norm = item["base_update_norm"]

                entropy_mix = average([float(item["grad_entropy"]), float(global_metrics["entropy"])])
                update_ratio_mix = average([float(item["local_update_ratio"]), float(global_metrics["update_ratio"])])
                gradient_ratio_mix = average([float(item["gradient_ratio"]), float(global_metrics["gradient_ratio"])])
                rotation_mix = average([float(item["rotation_score"]), float(global_metrics["rotation_score"])])
                coherence_mix = average([float(item["coherence_score"]), float(global_metrics["coherence_score"])])
                grad_momentum_mix = average([float(item["grad_momentum_cos"]), float(global_metrics["grad_momentum_cos"])])
                grad_prev_mix = average([float(item["grad_prev_grad_cos"]), float(global_metrics["grad_prev_grad_cos"])])
                force_momentum_mix = average([float(item["force_momentum_cos"]), float(global_metrics["force_momentum_cos"])])
                update_prev_mix = average([float(item["update_prev_update_cos"]), float(global_metrics["update_prev_update_cos"])])
                temperature_mix = average([float(item["temperature"]), float(global_metrics["temperature"])])
                heat_spike_mix = average([float(item["heat_spike"]), float(global_metrics["heat_spike"])])
                uncertainty_mix = average([float(item["uncertainty_score"]), float(global_metrics["uncertainty_score"])])
                interference_mix = average([float(item["interference_score"]), float(global_metrics["interference_score"])])
                reliability_mix = average([float(item["reliability_score"]), float(global_metrics["reliability_score"])])
                sustained_drift_mix = average([float(item["sustained_drift_score"]), float(global_metrics["sustained_drift_score"])])
                energy_drift_mix = average([float(item["drift_ema"]), float(global_metrics["energy_drift"])])
                heat_trend_mix = average([float(item["heat_trend"]), float(global_metrics["heat_trend"])])
                gradient_acc_mix = average([float(item["gradient_acceleration"]), float(global_metrics["gradient_acceleration"])])
                force_acc_mix = average([float(item["force_acceleration"]), float(global_metrics["force_acceleration"])])
                update_rise_mix = average([float(item["update_norm_rise"]), float(global_metrics["update_norm_rise"])])

                sds_scale = 1.0
                sds_cooling = 0.0
                sds_reheating = 0.0
                horizon_code = 0.0
                if bool(group["enable_sds"]):
                    inner_gate = smooth_sigmoid(float(group["inner_horizon"]) - update_ratio_mix, float(group["horizon_sharpness"]))
                    outer_gate = smooth_sigmoid(update_ratio_mix - float(group["outer_horizon"]), float(group["horizon_sharpness"]))
                    stagnation_factor = min(1.5, self.stagnation_counter / 4.0)
                    sds_reheating = (
                        float(group["sds_reheating_strength"])
                        * inner_gate
                        * (1.0 + 0.5 * stagnation_factor)
                        * (1.0 - 0.5 * entropy_mix)
                    )
                    sds_cooling = float(group["sds_cooling_strength"]) * outer_gate * (
                        1.0 + max(0.0, gradient_ratio_mix - 1.0) + max(0.0, validation_gap)
                    )
                    sds_cooling += float(group["sds_entropy_weight"]) * entropy_mix * max(0.0, gradient_ratio_mix - 1.0)
                    sds_scale = bounded_scale(math.exp(sds_reheating - sds_cooling), min_step_scale, max_step_scale)
                    controlled_ratio = update_ratio_mix * sds_scale
                    if controlled_ratio < float(group["inner_horizon"]):
                        horizon_code = -1.0
                    elif controlled_ratio > float(group["outer_horizon"]):
                        horizon_code = 1.0

                coherence_scale = 1.0
                if bool(group["enable_coherence"]):
                    coherence_input = 1.0
                    coherence_input += float(group["alignment_strength"]) * max(0.0, grad_momentum_mix)
                    coherence_input += float(group["coherence_strength"]) * coherence_mix
                    coherence_input -= float(group["misalignment_damping"]) * max(0.0, -grad_momentum_mix)
                    coherence_input -= float(group["rotation_penalty"]) * rotation_mix
                    if bool(group["global_mode"]) and not bool(group["layerwise_mode"]):
                        coherence_input -= 0.5 * float(group["rotation_penalty"]) * max(0.0, -global_metrics["grad_prev_grad_cos"])
                    coherence_scale = bounded_scale(coherence_input, min_step_scale, max_step_scale)

                thermodynamic_scale = 1.0
                thermo_cooling = 0.0
                thermo_reheating = 0.0
                if bool(group["enable_thermodynamic"]):
                    temp_norm = temperature_mix / (float(group["max_temperature"]) + DEFAULT_EPS)
                    thermo_cooling = float(group["thermodynamic_cooling_strength"]) * (
                        heat_spike_mix + max(0.0, temp_norm - 1.0) + 0.2 * heat_trend_mix
                    )
                    if self.stagnation_counter > 0:
                        thermo_reheating = (
                            float(group["thermodynamic_reheating_strength"])
                            * min(1.0, self.stagnation_counter / 6.0)
                            * max(0.0, 1.0 - temp_norm)
                            * max(0.0, 1.0 - entropy_mix)
                        )
                    boltzmann_scale = math.exp(
                        -float(group["thermodynamic_energy_weight"]) * heat_spike_mix / (temperature_mix + DEFAULT_EPS)
                    )
                    thermodynamic_scale = bounded_scale(
                        boltzmann_scale * max(0.05, 1.0 - thermo_cooling + thermo_reheating),
                        min_step_scale,
                        max_step_scale,
                    )

                predictive_signal = (
                    max(0.0, gradient_acc_mix - 1.0)
                    + max(0.0, force_acc_mix - 1.0)
                    + max(0.0, update_rise_mix - 1.0)
                    + max(0.0, rotation_mix - float(item["oscillation_ema"]))
                    + max(0.0, -grad_momentum_mix)
                )
                predictive_damping = 0.0
                if bool(group["enable_hamiltonian"]):
                    predictive_damping = 0.20 * predictive_signal

                hamiltonian_scale = 1.0
                effective_damping = 0.0
                correction_strength = 0.0
                if bool(group["enable_hamiltonian"]):
                    effective_damping += float(group["friction"])
                    effective_damping += float(group["energy_correction_strength"]) * sustained_drift_mix
                    effective_damping += float(group["oscillation_damping"]) * rotation_mix
                    effective_damping += predictive_damping
                    stability_drive = float(group["momentum_coupling"]) * max(0.0, force_momentum_mix) * max(0.0, -energy_drift_mix)
                    hamiltonian_scale = bounded_scale(1.0 + stability_drive - effective_damping, min_step_scale, max_step_scale)
                    correction_strength = bounded_scale(
                        float(group["energy_correction_strength"]) * sustained_drift_mix,
                        float(group["correction_floor"]),
                        float(group["correction_cap"]),
                    )

                uncertainty_scale = 1.0
                exploration_amount = 0.0
                if bool(group["enable_uncertainty"]):
                    uncertainty_scale = bounded_scale(
                        1.0
                        + float(group["reliability_strength"]) * reliability_mix
                        - float(group["uncertainty_weight"]) * uncertainty_mix
                        - float(group["interference_weight"]) * interference_mix,
                        min_step_scale,
                        max_step_scale,
                    )
                    exploration_amount = float(group["exploration_strength"]) * uncertainty_mix * max(0.0, 1.0 - reliability_mix)

                scale_factors = []
                if bool(group["enable_sds"]):
                    scale_factors.append(sds_scale)
                if bool(group["enable_coherence"]):
                    scale_factors.append(coherence_scale)
                if bool(group["enable_thermodynamic"]):
                    scale_factors.append(thermodynamic_scale)
                if bool(group["enable_hamiltonian"]):
                    scale_factors.append(hamiltonian_scale)
                if bool(group["enable_uncertainty"]):
                    scale_factors.append(uncertainty_scale)
                combined_scale = _geometric_combine(scale_factors, min_step_scale, max_step_scale)
                if bool(group["enable_hamiltonian"]) and correction_strength > 0.0:
                    combined_scale = bounded_scale(combined_scale * (1.0 - correction_strength), min_step_scale, max_step_scale)
                controlled_direction = force * combined_scale
                actual_update = controlled_direction * lr

                diffusion_scale = 0.0
                noise = torch.zeros_like(force)
                noise_ratio = 0.0
                if bool(group["enable_diffusion"]) and float(group["diffusion_strength"]) > 0.0:
                    diffusion_active = int(state["step"]) <= 4 or self.stagnation_counter >= int(group["stagnation_trigger"]) or uncertainty_mix >= 0.35
                    sigma = float(group["diffusion_strength"]) * (float(group["diffusion_decay"]) ** max(0, int(state["step"]) - 1))
                    if not diffusion_active:
                        sigma *= 0.15
                    if bool(group["entropy_scaled_noise"]):
                        sigma *= 0.5 + entropy_mix
                    sigma *= 1.0 + 0.5 * uncertainty_mix + 0.25 * min(1.0, temperature_mix / (float(group["max_temperature"]) + DEFAULT_EPS))
                    sigma = bounded_scale(sigma, float(group["min_noise"]), float(group["max_noise"]))

                    iso_noise = torch.randn_like(force)
                    grad_dir = grad / (grad.norm() + eps)
                    aligned_scalar = torch.randn((), device=grad.device, dtype=grad.dtype)
                    aligned_noise = grad_dir * aligned_scalar
                    mixed_noise = (1.0 - float(group["aligned_noise_weight"])) * iso_noise + float(group["aligned_noise_weight"]) * aligned_noise
                    mixed_norm = mixed_noise.norm()
                    if torch.isfinite(mixed_norm) and mixed_norm > 0:
                        mixed_noise = mixed_noise / (mixed_norm + eps)
                    else:
                        mixed_noise = torch.zeros_like(force)
                    noise = mixed_noise * (sigma * base_update_norm * (1.0 + exploration_amount))
                    noise_ratio = safe_float(noise.norm()) / max(base_update_norm * combined_scale, DEFAULT_EPS)
                    if float(group["noise_to_update_cap"]) >= 0.0 and noise_ratio > float(group["noise_to_update_cap"]):
                        noise = noise * (float(group["noise_to_update_cap"]) / (noise_ratio + DEFAULT_EPS))
                        noise_ratio = safe_float(noise.norm()) / max(base_update_norm * combined_scale, DEFAULT_EPS)
                    diffusion_scale = sigma

                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(actual_update + noise, alpha=-1.0)

                state["prev_grad"].copy_(grad)
                state["prev_force"].copy_(force.detach())
                state["prev_update"].copy_(controlled_direction.detach())
                state["grad_norm_ema"] = 0.95 * float(state["grad_norm_ema"]) + 0.05 * float(item["grad_norm_value"])
                state["update_norm_ema"] = float(item["update_norm_ema"])
                state["oscillation_ema"] = float(item["oscillation_ema"])
                state["loss_ema"] = float(item["loss_ema"])
                state["temperature"] = float(item["temperature"])
                state["prev_thermo_signal"] = float(item["thermo_signal"])
                state["prev_normalized_total_energy"] = float(item["normalized_total_energy"])
                state["prev_raw_total_energy"] = float(item["raw_total_energy"])
                state["energy_ema"] = float(item["energy_ema"])
                state["drift_ema"] = float(item["drift_ema"])

                diagnostics["update_ratio"].append(update_ratio_mix)
                diagnostics["gradient_ratio"].append(gradient_ratio_mix)
                diagnostics["entropy"].append(entropy_mix)
                diagnostics["grad_momentum_cosine"].append(grad_momentum_mix)
                diagnostics["grad_previous_grad_cosine"].append(grad_prev_mix)
                diagnostics["update_previous_update_cosine"].append(update_prev_mix)
                diagnostics["rotation_score"].append(rotation_mix)
                diagnostics["coherence_score"].append(coherence_mix)
                diagnostics["gradient_energy"].append(float(item["grad_energy"]))
                diagnostics["update_energy"].append(tensor_energy(actual_update + noise))
                diagnostics["temperature"].append(temperature_mix)
                diagnostics["heat_spike"].append(heat_spike_mix)
                diagnostics["noise_norm"].append(safe_float(noise.norm()))
                diagnostics["noise_to_update_ratio"].append(noise_ratio)
                diagnostics["diffusion_scale"].append(diffusion_scale)
                diagnostics["kinetic_energy"].append(float(item["kinetic"]))
                diagnostics["potential_energy"].append(float(item["loss_ema"]))
                diagnostics["total_energy"].append(float(item["raw_total_energy"]))
                diagnostics["energy_drift"].append(energy_drift_mix)
                diagnostics["normalized_total_energy"].append(float(item["normalized_total_energy"]))
                diagnostics["uncertainty_score"].append(uncertainty_mix)
                diagnostics["interference_score"].append(interference_mix)
                diagnostics["reliability_score"].append(reliability_mix)
                diagnostics["effective_lr_scale"].append(combined_scale)
                diagnostics["controller_scale_sds"].append(sds_scale if bool(group["enable_sds"]) else 1.0)
                diagnostics["controller_scale_coherence"].append(coherence_scale if bool(group["enable_coherence"]) else 1.0)
                diagnostics["controller_scale_thermodynamic"].append(thermodynamic_scale if bool(group["enable_thermodynamic"]) else 1.0)
                diagnostics["controller_scale_hamiltonian"].append(hamiltonian_scale if bool(group["enable_hamiltonian"]) else 1.0)
                diagnostics["controller_scale_uncertainty"].append(uncertainty_scale if bool(group["enable_uncertainty"]) else 1.0)
                diagnostics["predictive_damping"].append(predictive_damping)
                diagnostics["effective_damping"].append(effective_damping)
                diagnostics["reheating_amount"].append(sds_reheating + thermo_reheating)
                diagnostics["cooling_amount"].append(sds_cooling + thermo_cooling)
                diagnostics["exploration_amount"].append(exploration_amount)
                diagnostics["symplectic_correction_active"].append(1.0 if correction_strength > 0.0 else 0.0)
                diagnostics["symplectic_correction_strength"].append(correction_strength)
                diagnostics["divergence_flag"].append(0.0 if bool(torch.isfinite(param).all().item()) else 1.0)
                horizon_codes.append(horizon_code)

        mean_horizon_code = average(horizon_codes)
        if mean_horizon_code <= -0.33:
            horizon_state = "inner"
        elif mean_horizon_code >= 0.33:
            horizon_state = "outer"
        else:
            horizon_state = "stable"

        self._record_step(
            {
                "loss": current_loss,
                "update_ratio": average(diagnostics["update_ratio"]),
                "gradient_ratio": average(diagnostics["gradient_ratio"]),
                "entropy": average(diagnostics["entropy"]),
                "horizon_code": mean_horizon_code,
                "horizon_state": horizon_state,
                "grad_momentum_cosine": average(diagnostics["grad_momentum_cosine"]),
                "grad_previous_grad_cosine": average(diagnostics["grad_previous_grad_cosine"]),
                "update_previous_update_cosine": average(diagnostics["update_previous_update_cosine"]),
                "rotation_score": average(diagnostics["rotation_score"]),
                "coherence_score": average(diagnostics["coherence_score"]),
                "gradient_energy": average(diagnostics["gradient_energy"]),
                "update_energy": average(diagnostics["update_energy"]),
                "temperature": average(diagnostics["temperature"]),
                "heat_spike": average(diagnostics["heat_spike"]),
                "noise_norm": average(diagnostics["noise_norm"]),
                "noise_to_update_ratio": average(diagnostics["noise_to_update_ratio"]),
                "diffusion_scale": average(diagnostics["diffusion_scale"]),
                "kinetic_energy": average(diagnostics["kinetic_energy"]),
                "potential_energy": average(diagnostics["potential_energy"]),
                "total_energy": average(diagnostics["total_energy"]),
                "energy_drift": average(diagnostics["energy_drift"]),
                "normalized_total_energy": average(diagnostics["normalized_total_energy"]),
                "uncertainty_score": average(diagnostics["uncertainty_score"]),
                "interference_score": average(diagnostics["interference_score"]),
                "reliability_score": average(diagnostics["reliability_score"]),
                "effective_lr_scale": average(diagnostics["effective_lr_scale"]),
                "controller_scale_sds": average(diagnostics["controller_scale_sds"]),
                "controller_scale_coherence": average(diagnostics["controller_scale_coherence"]),
                "controller_scale_thermodynamic": average(diagnostics["controller_scale_thermodynamic"]),
                "controller_scale_hamiltonian": average(diagnostics["controller_scale_hamiltonian"]),
                "controller_scale_uncertainty": average(diagnostics["controller_scale_uncertainty"]),
                "predictive_damping": average(diagnostics["predictive_damping"]),
                "effective_damping": average(diagnostics["effective_damping"]),
                "reheating_amount": average(diagnostics["reheating_amount"]),
                "cooling_amount": average(diagnostics["cooling_amount"]),
                "exploration_amount": average(diagnostics["exploration_amount"]),
                "symplectic_correction_active": average(diagnostics["symplectic_correction_active"]),
                "symplectic_correction_strength": average(diagnostics["symplectic_correction_strength"]),
                "divergence_flag": average(diagnostics["divergence_flag"]),
            }
        )
        return loss_tensor


HorizonFieldAdam = UnifiedPhysicsAdam
