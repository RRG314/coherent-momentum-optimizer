from __future__ import annotations

import math

import torch

from .hamiltonian_adam import HamiltonianAdamReal
from .optimizer_utils import (
    DEFAULT_EPS,
    average,
    bounded_scale,
    clamp_scalar,
    cosine_similarity,
    safe_float,
    sign_flip_ratio,
)


class MagnetoHamiltonianAdam(HamiltonianAdamReal):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        mode: str = "dissipative_hamiltonian",
        mass_mode: str = "fixed",
        fixed_mass: float = 1.0,
        use_adam_preconditioning: bool = True,
        adaptive_mass_trust: float = 0.08,
        mass_smoothing: float = 0.98,
        mass_anisotropy_cap: float = 1.5,
        mass_change_cap: float = 1.05,
        mass_warmup_steps: int = 32,
        mass_alignment_strength: float = 0.25,
        mass_shock_penalty: float = 1.4,
        min_inverse_mass: float = 0.55,
        max_inverse_mass: float = 1.45,
        friction: float = 0.03,
        use_friction: bool = True,
        energy_correction_strength: float = 0.06,
        use_energy_correction: bool = True,
        drift_threshold: float = 0.035,
        max_energy_correction: float = 0.16,
        relative_energy_floor: float = 0.1,
        alignment_strength: float = 0.14,
        coherence_strength: float = 0.12,
        conflict_damping: float = 0.20,
        rotation_penalty: float = 0.22,
        projection_strength: float = 0.12,
        max_projection: float = 0.25,
        conflict_gate_threshold: float = 0.28,
        activation_rotation_threshold: float = 0.38,
        activation_conflict_weight: float = 0.7,
        activation_rotation_weight: float = 0.5,
        stable_coherence_bonus: float = 0.24,
        projection_activation_threshold: float = 0.18,
        min_alignment_scale: float = 0.88,
        max_alignment_scale: float = 1.12,
        field_clip: float = 2.0,
        use_decoupled_weight_decay: bool = True,
        maximize: bool = False,
        enable_step_diagnostics: bool = True,
        diagnostics_every_n_steps: int = 1,
    ) -> None:
        super().__init__(
            params,
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
            enable_step_diagnostics=enable_step_diagnostics,
            diagnostics_every_n_steps=diagnostics_every_n_steps,
        )
        for group in self.param_groups:
            group["alignment_strength"] = alignment_strength
            group["coherence_strength"] = coherence_strength
            group["conflict_damping"] = conflict_damping
            group["rotation_penalty"] = rotation_penalty
            group["projection_strength"] = projection_strength
            group["max_projection"] = max_projection
            group["conflict_gate_threshold"] = conflict_gate_threshold
            group["activation_rotation_threshold"] = activation_rotation_threshold
            group["activation_conflict_weight"] = activation_conflict_weight
            group["activation_rotation_weight"] = activation_rotation_weight
            group["stable_coherence_bonus"] = stable_coherence_bonus
            group["projection_activation_threshold"] = projection_activation_threshold
            group["min_alignment_scale"] = min_alignment_scale
            group["max_alignment_scale"] = max_alignment_scale
            group["field_clip"] = field_clip
        self._initialize_physical_optimizer("MagnetoHamiltonianAdam")

    def _compute_magneto_controls(
        self,
        *,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        prev_grad: torch.Tensor,
        prev_update: torch.Tensor,
        inverse_mass: torch.Tensor,
        alignment_strength: float,
        coherence_strength: float,
        conflict_damping: float,
        rotation_penalty: float,
        projection_strength: float,
        max_projection: float,
        conflict_gate_threshold: float,
        activation_rotation_threshold: float,
        activation_conflict_weight: float,
        activation_rotation_weight: float,
        stable_coherence_bonus: float,
        projection_activation_threshold: float,
        min_alignment_scale: float,
        max_alignment_scale: float,
        field_clip: float,
    ) -> dict[str, float]:
        force_direction = -(grad * inverse_mass)
        grad_momentum_cos = cosine_similarity(grad, momentum)
        force_momentum_cos = cosine_similarity(force_direction, momentum)
        grad_prev_grad_cos = cosine_similarity(grad, prev_grad)
        force_prev_update_cos = cosine_similarity(force_direction, prev_update)
        rotation_score = 0.5 * (1.0 - grad_prev_grad_cos) + 0.5 * sign_flip_ratio(force_direction, prev_update)
        rotation_score = clamp_scalar(rotation_score, 0.0, 1.5)

        coherence_score = average(
            [
                max(0.0, grad_momentum_cos),
                max(0.0, force_momentum_cos),
                max(0.0, force_prev_update_cos),
            ]
        )
        raw_conflict_score = average(
            [
                max(0.0, -grad_momentum_cos),
                max(0.0, -force_momentum_cos),
                clamp_scalar(rotation_score / 1.5, 0.0, 1.0),
            ]
        )
        severe_conflict = average(
            [
                max(0.0, -force_momentum_cos),
                max(0.0, -force_prev_update_cos),
                clamp_scalar(rotation_score / 1.5, 0.0, 1.0),
            ]
        )
        conflict_gate = clamp_scalar(
            (severe_conflict - float(conflict_gate_threshold)) / max(1.0 - float(conflict_gate_threshold), DEFAULT_EPS),
            0.0,
            1.0,
        )
        conflict_score = raw_conflict_score * conflict_gate
        rotation_gate = clamp_scalar(
            (rotation_score - float(activation_rotation_threshold)) / max(1.5 - float(activation_rotation_threshold), DEFAULT_EPS),
            0.0,
            1.0,
        )
        magneto_activation = clamp_scalar(
            float(activation_conflict_weight) * conflict_gate + float(activation_rotation_weight) * rotation_gate,
            0.0,
            1.0,
        )
        stable_gate = 1.0 - magneto_activation
        field_strength = clamp_scalar(abs(force_momentum_cos) + coherence_score, 0.0, field_clip)
        friction_multiplier = clamp_scalar(
            1.0
            + magneto_activation * (conflict_damping * conflict_score + 0.35 * rotation_penalty * rotation_gate)
            - stable_gate * stable_coherence_bonus * alignment_strength * coherence_score,
            0.86,
            1.55,
        )
        alignment_scale = bounded_scale(
            1.0
            + stable_gate * stable_coherence_bonus * coherence_strength * coherence_score
            + magneto_activation
            * (
                coherence_strength * coherence_score
                - rotation_penalty * rotation_score
                - 0.35 * conflict_damping * conflict_score
            ),
            min_alignment_scale,
            max_alignment_scale,
        )
        projection_gate = clamp_scalar(
            (magneto_activation - float(projection_activation_threshold)) / max(1.0 - float(projection_activation_threshold), DEFAULT_EPS),
            0.0,
            1.0,
        )
        projection = clamp_scalar(
            projection_strength
            * projection_gate
            * average([conflict_score, max(0.0, -force_momentum_cos), 0.5 * clamp_scalar(rotation_score / 1.5, 0.0, 1.0)]),
            0.0,
            max_projection,
        )
        return {
            "grad_momentum_cosine": grad_momentum_cos,
            "force_momentum_cosine": force_momentum_cos,
            "grad_previous_grad_cosine": grad_prev_grad_cos,
            "update_previous_update_cosine": force_prev_update_cos,
            "rotation_score": rotation_score,
            "rotation_gate": rotation_gate,
            "coherence_score": coherence_score,
            "conflict_score": conflict_score,
            "conflict_gate": conflict_gate,
            "magneto_activation": magneto_activation,
            "stable_gate": stable_gate,
            "field_strength": field_strength,
            "friction_multiplier": friction_multiplier,
            "alignment_scale": alignment_scale,
            "projection_strength": projection,
        }

    def _blend_step_direction(
        self,
        base_step: torch.Tensor,
        force_direction: torch.Tensor,
        projection_strength: float,
    ) -> torch.Tensor:
        if projection_strength <= 0.0:
            return base_step
        base_norm = safe_float(base_step.norm())
        if base_norm <= DEFAULT_EPS:
            return base_step
        blended = (1.0 - projection_strength) * base_step + projection_strength * force_direction
        blended_norm = safe_float(blended.norm())
        if blended_norm <= DEFAULT_EPS:
            return base_step
        return blended * (base_norm / (blended_norm + DEFAULT_EPS))

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
            "damping_amount": [],
            "leapfrog_enabled": [],
            "closure_recomputed_gradient": [],
            "grad_momentum_cosine": [],
            "force_momentum_cosine": [],
            "grad_previous_grad_cosine": [],
            "update_previous_update_cosine": [],
            "rotation_score": [],
            "rotation_gate": [],
            "coherence_score": [],
            "conflict_score": [],
            "conflict_gate": [],
            "magneto_activation": [],
            "stable_gate": [],
            "field_strength": [],
            "alignment_scale": [],
            "magneto_projection_strength": [],
            "magneto_friction_multiplier": [],
        }

        mode_used = None
        closure_recomputed = False
        updated_params: list[tuple[torch.Tensor, dict[str, object], torch.Tensor, torch.Tensor, float, float, dict[str, float]]] = []
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
            alignment_strength = float(group["alignment_strength"])
            coherence_strength = float(group["coherence_strength"])
            conflict_damping = float(group["conflict_damping"])
            rotation_penalty = float(group["rotation_penalty"])
            projection_strength = float(group["projection_strength"])
            max_projection = float(group["max_projection"])
            conflict_gate_threshold = float(group["conflict_gate_threshold"])
            activation_rotation_threshold = float(group["activation_rotation_threshold"])
            activation_conflict_weight = float(group["activation_conflict_weight"])
            activation_rotation_weight = float(group["activation_rotation_weight"])
            stable_coherence_bonus = float(group["stable_coherence_bonus"])
            projection_activation_threshold = float(group["projection_activation_threshold"])
            min_alignment_scale = float(group["min_alignment_scale"])
            max_alignment_scale = float(group["max_alignment_scale"])
            field_clip = float(group["field_clip"])
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
                prev_grad = state.setdefault("prev_grad", torch.zeros_like(param))

                state["step"] = int(state["step"]) + 1
                momentum = state["hamiltonian_momentum"]
                prev_update = state["prev_update"]
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
                controls = self._compute_magneto_controls(
                    grad=grad,
                    momentum=momentum,
                    prev_grad=prev_grad,
                    prev_update=prev_update,
                    inverse_mass=inverse_mass,
                    alignment_strength=alignment_strength,
                    coherence_strength=coherence_strength,
                    conflict_damping=conflict_damping,
                    rotation_penalty=rotation_penalty,
                    projection_strength=projection_strength,
                    max_projection=max_projection,
                    conflict_gate_threshold=conflict_gate_threshold,
                    activation_rotation_threshold=activation_rotation_threshold,
                    activation_conflict_weight=activation_conflict_weight,
                    activation_rotation_weight=activation_rotation_weight,
                    stable_coherence_bonus=stable_coherence_bonus,
                    projection_activation_threshold=projection_activation_threshold,
                    min_alignment_scale=min_alignment_scale,
                    max_alignment_scale=max_alignment_scale,
                    field_clip=field_clip,
                )

                friction_factor = 1.0
                if use_friction and friction > 0.0:
                    friction_factor = math.exp(-lr * friction * controls["friction_multiplier"])
                use_leapfrog = mode == "leapfrog_with_closure" and closure is not None
                if mode in {"adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure is not None:
                    use_leapfrog = True
                if use_leapfrog:
                    momentum.mul_(math.sqrt(friction_factor)).add_(grad, alpha=-0.5 * lr)
                else:
                    momentum.mul_(friction_factor).add_(grad, alpha=-lr)

                base_step = inverse_mass * momentum
                force_direction = -(grad * inverse_mass)
                step_direction = self._blend_step_direction(base_step, force_direction, controls["projection_strength"])
                step_direction = step_direction * controls["alignment_scale"]

                if use_decoupled_weight_decay and weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(step_direction, alpha=lr)

                updated_params.append(
                    (param, state, step_direction.detach().clone(), inverse_mass.detach().clone(), friction_factor, lr, controls)
                )

                total_elements += grad.numel()
                gradient_sq_sum += float(grad.detach().float().pow(2).sum().item())
                parameter_step_sq_sum += float((step_direction.detach().float() * lr).pow(2).sum().item())
                momentum_sq_sum += float(momentum.detach().float().pow(2).sum().item())
                inverse_mass_sum += float(inverse_mass.detach().float().sum().item())
                inverse_mass_sq_sum += float(inverse_mass.detach().float().pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).sum().item())
                damping_amount += 1.0 - friction_factor

                for key in [
                    "grad_momentum_cosine",
                    "force_momentum_cosine",
                    "grad_previous_grad_cosine",
                    "update_previous_update_cosine",
                    "rotation_score",
                    "rotation_gate",
                    "coherence_score",
                    "conflict_score",
                    "conflict_gate",
                    "magneto_activation",
                    "stable_gate",
                    "field_strength",
                    "alignment_scale",
                ]:
                    diagnostics[key].append(float(controls[key]))
                diagnostics["magneto_projection_strength"].append(float(controls["projection_strength"]))
                diagnostics["magneto_friction_multiplier"].append(float(controls["friction_multiplier"]))
                prev_grad.copy_(grad)
                prev_update.copy_(step_direction.detach())

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
            parameter_step_sq_sum = 0.0
            for param, state, step_direction, inverse_mass, friction_factor, lr, controls in updated_params:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if not torch.isfinite(grad).all():
                    continue
                momentum = state["hamiltonian_momentum"]
                momentum.mul_(math.sqrt(friction_factor)).add_(grad, alpha=-0.5 * lr)
                post_projection = 0.5 * controls["projection_strength"]
                if post_projection > 0.0:
                    force_direction = -(grad * inverse_mass)
                    momentum_norm = safe_float(momentum.norm())
                    force_norm = safe_float(force_direction.norm())
                    if momentum_norm > DEFAULT_EPS and force_norm > DEFAULT_EPS:
                        target_momentum = force_direction * (momentum_norm / (force_norm + DEFAULT_EPS))
                        momentum.mul_(1.0 - post_projection).add_(target_momentum, alpha=post_projection)
                total_elements += grad.numel()
                gradient_sq_sum += float(grad.detach().float().pow(2).sum().item())
                parameter_step_sq_sum += float((step_direction.detach().float() * lr).pow(2).sum().item())
                momentum_sq_sum += float(momentum.detach().float().pow(2).sum().item())
                inverse_mass_sum += float(inverse_mass.detach().float().sum().item())
                inverse_mass_sq_sum += float(inverse_mass.detach().float().pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum.detach().float().pow(2) * inverse_mass.detach().float()).sum().item())

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
            drift_reference = max(abs(float(prev_total)), abs(total_hamiltonian), float(self.param_groups[0]["relative_energy_floor"]))
            relative_energy_drift = energy_drift / (drift_reference + DEFAULT_EPS)

        energy_correction_applied = 0.0
        if (
            updated_params
            and mode_used in {"dissipative_hamiltonian", "adam_preconditioned_hamiltonian", "leapfrog_with_closure"}
            and bool(self.param_groups[0]["use_energy_correction"])
            and relative_energy_drift > float(self.param_groups[0]["drift_threshold"])
        ):
            energy_correction_applied = clamp_scalar(
                float(self.param_groups[0]["energy_correction_strength"]) * (relative_energy_drift - float(self.param_groups[0]["drift_threshold"])),
                0.0,
                float(self.param_groups[0]["max_energy_correction"]),
            )
            if energy_correction_applied > 0.0:
                for _, state, _, _, _, _, _ in updated_params:
                    momentum = state["hamiltonian_momentum"]
                    momentum.mul_(1.0 - energy_correction_applied)
                kinetic_energy *= (1.0 - energy_correction_applied) ** 2
                total_hamiltonian = potential_after + kinetic_energy
                if prev_total is None:
                    energy_drift = 0.0
                    relative_energy_drift = 0.0
                else:
                    energy_drift = total_hamiltonian - float(prev_total)
                    drift_reference = max(abs(float(prev_total)), abs(total_hamiltonian), float(self.param_groups[0]["relative_energy_floor"]))
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
            "damping_amount": total_damping_amount,
            "effective_damping": total_damping_amount,
            "leapfrog_enabled": 1.0 if mode_used in {"leapfrog_with_closure", "adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure_recomputed else 0.0,
            "closure_recomputed_gradient": 1.0 if closure_recomputed else 0.0,
            "symplectic_euler_approximation": 0.0 if closure_recomputed else 1.0,
            "effective_lr_scale": average(diagnostics["alignment_scale"]) if diagnostics["alignment_scale"] else 1.0,
            "alignment_scale": average(diagnostics["alignment_scale"]) if diagnostics["alignment_scale"] else 1.0,
            "magneto_projection_strength": average(diagnostics["magneto_projection_strength"]),
            "magneto_friction_multiplier": average(diagnostics["magneto_friction_multiplier"]),
        }
        for key in [
            "grad_momentum_cosine",
            "force_momentum_cosine",
            "grad_previous_grad_cosine",
            "update_previous_update_cosine",
            "rotation_score",
            "rotation_gate",
            "coherence_score",
            "conflict_score",
            "conflict_gate",
            "magneto_activation",
            "stable_gate",
            "field_strength",
        ]:
            diagnostics_row[key] = average(diagnostics[key])
        self._record_step(diagnostics_row)
        return loss_tensor
