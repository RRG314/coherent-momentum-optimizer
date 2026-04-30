from __future__ import annotations

import math

import torch

from .hamiltonian_adam import HamiltonianAdamReal
from .optimizer_utils import DEFAULT_EPS, average, bounded_scale, clamp_scalar, safe_float


def _clamp_tensor(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return value.clamp(min=low, max=high)


def _mean_tensors(*values: torch.Tensor) -> torch.Tensor:
    if not values:
        return torch.zeros(())
    return torch.stack(list(values)).mean()


def _cosine_similarity_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    a_flat = a.detach().reshape(-1).float()
    b_flat = b.detach().reshape(-1).float()
    dot = torch.dot(a_flat, b_flat)
    denom = a_flat.norm() * b_flat.norm()
    valid = torch.isfinite(denom) & (denom > eps)
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    value = dot / (safe_denom + eps)
    value = value.clamp(-1.0, 1.0)
    return torch.where(valid, value, torch.zeros_like(value))


def _sign_flip_ratio_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.detach().reshape(-1)
    b_flat = b.detach().reshape(-1)
    if a_flat.numel() == 0:
        return torch.zeros((), device=a.device, dtype=torch.float32)
    flips = torch.sign(a_flat) != torch.sign(b_flat)
    return flips.float().mean()


class MagnetoHamiltonianAdamImproved(HamiltonianAdamReal):
    _VALID_PRESETS = {"balanced", "standard_safe", "stress_specialist", "cnn_safe"}

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
        alignment_strength: float = 0.12,
        coherence_strength: float = 0.10,
        conflict_damping: float = 0.0,
        rotation_penalty: float = 0.18,
        projection_strength: float = 0.12,
        max_projection: float = 0.22,
        conflict_gate_threshold: float = 0.28,
        activation_rotation_threshold: float = 0.34,
        activation_conflict_weight: float = 0.7,
        activation_rotation_weight: float = 0.45,
        stable_coherence_bonus: float = 0.18,
        projection_activation_threshold: float = 0.16,
        min_alignment_scale: float = 0.92,
        max_alignment_scale: float = 1.10,
        field_clip: float = 2.0,
        preset: str = "balanced",
        projection_mode: str = "conflict_only",
        standard_safe_strength: float = 0.7,
        soft_conflict_correction: float = 0.12,
        soft_conflict_max: float = 0.22,
        conv_safe_mode: bool = True,
        conv_update_ratio_cap: float = 0.018,
        conv_projection_scale: float = 0.65,
        conv_friction_scale: float = 0.72,
        conv_alignment_scale: float = 0.75,
        conv_support_weight: float = 0.35,
        use_decoupled_weight_decay: bool = True,
        maximize: bool = False,
        enable_step_diagnostics: bool = True,
        diagnostics_every_n_steps: int = 1,
    ) -> None:
        if preset not in self._VALID_PRESETS:
            raise ValueError(f"preset must be one of {sorted(self._VALID_PRESETS)}")
        if projection_mode not in {"always", "conflict_only", "rotation_only"}:
            raise ValueError("projection_mode must be 'always', 'conflict_only', or 'rotation_only'")

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
            group["preset"] = preset
            group["projection_mode"] = projection_mode
            group["standard_safe_strength"] = standard_safe_strength
            group["soft_conflict_correction"] = soft_conflict_correction
            group["soft_conflict_max"] = soft_conflict_max
            group["conv_safe_mode"] = conv_safe_mode
            group["conv_update_ratio_cap"] = conv_update_ratio_cap
            group["conv_projection_scale"] = conv_projection_scale
            group["conv_friction_scale"] = conv_friction_scale
            group["conv_alignment_scale"] = conv_alignment_scale
            group["conv_support_weight"] = conv_support_weight
        self._initialize_physical_optimizer("MagnetoHamiltonianAdamImproved")
        self._apply_preset()

    def _apply_preset(self) -> None:
        for group in self.param_groups:
            preset = str(group["preset"])
            if preset == "standard_safe":
                group["alignment_strength"] = float(group["alignment_strength"]) * 0.8
                group["coherence_strength"] = float(group["coherence_strength"]) * 0.75
                group["rotation_penalty"] = float(group["rotation_penalty"]) * 0.8
                group["projection_strength"] = float(group["projection_strength"]) * 0.8
                group["soft_conflict_correction"] = float(group["soft_conflict_correction"]) * 0.85
                group["min_alignment_scale"] = max(0.95, float(group["min_alignment_scale"]))
                group["max_alignment_scale"] = min(1.08, float(group["max_alignment_scale"]))
                group["standard_safe_strength"] = max(0.75, float(group["standard_safe_strength"]))
            elif preset == "stress_specialist":
                group["alignment_strength"] = float(group["alignment_strength"]) * 1.15
                group["coherence_strength"] = float(group["coherence_strength"]) * 1.15
                group["rotation_penalty"] = float(group["rotation_penalty"]) * 1.15
                group["projection_strength"] = min(float(group["projection_strength"]) * 1.2, float(group["max_projection"]))
                group["soft_conflict_correction"] = float(group["soft_conflict_correction"]) * 1.15
                group["soft_conflict_max"] = min(0.3, float(group["soft_conflict_max"]) * 1.1)
                group["min_alignment_scale"] = min(float(group["min_alignment_scale"]), 0.9)
                group["max_alignment_scale"] = max(float(group["max_alignment_scale"]), 1.12)
            elif preset == "cnn_safe":
                group["projection_strength"] = float(group["projection_strength"]) * 0.75
                group["soft_conflict_correction"] = float(group["soft_conflict_correction"]) * 0.85
                group["conv_update_ratio_cap"] = min(float(group["conv_update_ratio_cap"]), 0.014)
                group["conv_projection_scale"] = min(float(group["conv_projection_scale"]), 0.55)
                group["conv_friction_scale"] = min(float(group["conv_friction_scale"]), 0.66)
                group["conv_alignment_scale"] = min(float(group["conv_alignment_scale"]), 0.7)

    def _compute_controls_tensor(
        self,
        *,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        prev_grad: torch.Tensor,
        prev_update: torch.Tensor,
        inverse_mass: torch.Tensor,
        group: dict[str, object],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        force_direction = -(grad * inverse_mass)
        grad_momentum_cos = _cosine_similarity_tensor(grad, momentum)
        force_momentum_cos = _cosine_similarity_tensor(force_direction, momentum)
        grad_prev_grad_cos = _cosine_similarity_tensor(grad, prev_grad)
        force_prev_update_cos = _cosine_similarity_tensor(force_direction, prev_update)
        rotation_score = 0.5 * (1.0 - grad_prev_grad_cos) + 0.5 * _sign_flip_ratio_tensor(force_direction, prev_update)
        rotation_score = _clamp_tensor(rotation_score, 0.0, 1.5)
        normalized_rotation = _clamp_tensor(rotation_score / 1.5, 0.0, 1.0)

        coherence_score = _mean_tensors(
            torch.relu(grad_momentum_cos),
            torch.relu(force_momentum_cos),
            torch.relu(force_prev_update_cos),
        )
        raw_conflict_score = _mean_tensors(
            torch.relu(-grad_momentum_cos),
            torch.relu(-force_momentum_cos),
            normalized_rotation,
        )
        severe_conflict = _mean_tensors(
            torch.relu(-force_momentum_cos),
            torch.relu(-force_prev_update_cos),
            normalized_rotation,
        )

        conflict_threshold = float(group["conflict_gate_threshold"])
        conflict_gate = _clamp_tensor(
            (severe_conflict - conflict_threshold) / max(1.0 - conflict_threshold, DEFAULT_EPS),
            0.0,
            1.0,
        )
        conflict_score = raw_conflict_score * conflict_gate

        rotation_threshold = float(group["activation_rotation_threshold"])
        rotation_gate = _clamp_tensor(
            (rotation_score - rotation_threshold) / max(1.5 - rotation_threshold, DEFAULT_EPS),
            0.0,
            1.0,
        )
        magneto_activation = _clamp_tensor(
            float(group["activation_conflict_weight"]) * conflict_gate
            + float(group["activation_rotation_weight"]) * rotation_gate,
            0.0,
            1.0,
        )
        stable_gate = 1.0 - magneto_activation
        field_strength = _clamp_tensor(torch.abs(force_momentum_cos) + coherence_score, 0.0, float(group["field_clip"]))

        clean_alignment = torch.relu(grad_prev_grad_cos)
        standard_safe_strength = float(group["standard_safe_strength"])
        clean_bonus = stable_gate * standard_safe_strength * float(group["stable_coherence_bonus"]) * _mean_tensors(coherence_score, clean_alignment)

        friction_multiplier = _clamp_tensor(
            1.0
            + magneto_activation
            * (
                0.45 * float(group["conflict_damping"]) * conflict_score
                + 0.30 * float(group["rotation_penalty"]) * rotation_gate
            )
            - clean_bonus * float(group["alignment_strength"]),
            0.90,
            1.45,
        )
        alignment_scale = _clamp_tensor(
            1.0
            + clean_bonus * float(group["coherence_strength"])
            + magneto_activation
            * (
                0.55 * float(group["coherence_strength"]) * coherence_score
                - 0.35 * float(group["rotation_penalty"]) * rotation_score
            ),
            float(group["min_alignment_scale"]),
            float(group["max_alignment_scale"]),
        )

        projection_activation_threshold = float(group["projection_activation_threshold"])
        projection_source = magneto_activation
        projection_mode = str(group["projection_mode"])
        if projection_mode == "conflict_only":
            projection_source = conflict_gate
        elif projection_mode == "rotation_only":
            projection_source = rotation_gate
        projection_gate = _clamp_tensor(
            (projection_source - projection_activation_threshold) / max(1.0 - projection_activation_threshold, DEFAULT_EPS),
            0.0,
            1.0,
        )
        projection_strength = _clamp_tensor(
            float(group["projection_strength"])
            * projection_gate
            * _mean_tensors(conflict_score, torch.relu(-force_momentum_cos), normalized_rotation),
            0.0,
            float(group["max_projection"]),
        )
        soft_conflict_correction = _clamp_tensor(
            float(group["soft_conflict_correction"])
            * _mean_tensors(conflict_score, torch.relu(-force_momentum_cos), 0.5 * normalized_rotation),
            0.0,
            float(group["soft_conflict_max"]),
        )
        controls = {
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
            "projection_strength": projection_strength,
            "soft_conflict_correction": soft_conflict_correction,
        }
        return controls, force_direction

    def _filter_support(self, grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if grad.ndim < 3 or grad.shape[0] <= 0:
            return torch.zeros((), device=grad.device, dtype=torch.float32)
        grad_rows = grad.detach().reshape(grad.shape[0], -1).float()
        ref_rows = reference.detach().reshape(reference.shape[0], -1).float()
        numerators = (grad_rows * ref_rows).sum(dim=1)
        denominators = grad_rows.norm(dim=1) * ref_rows.norm(dim=1)
        valid = torch.isfinite(denominators) & (denominators > DEFAULT_EPS)
        safe_denominators = torch.where(valid, denominators, torch.ones_like(denominators))
        cosines = numerators / (safe_denominators + DEFAULT_EPS)
        cosines = cosines.clamp(-1.0, 1.0)
        cosines = torch.where(valid, cosines, torch.zeros_like(cosines))
        support = torch.relu(cosines).mean()
        return support

    def _blend_step_direction_tensor(
        self,
        base_step: torch.Tensor,
        force_direction: torch.Tensor,
        projection_strength: torch.Tensor,
    ) -> torch.Tensor:
        base_norm = base_step.detach().float().norm()
        blend = (1.0 - projection_strength) * base_step + projection_strength * force_direction
        blend_norm = blend.detach().float().norm()
        safe_blend_norm = torch.where(blend_norm > DEFAULT_EPS, blend_norm, torch.ones_like(blend_norm))
        scaled = blend * (base_norm / (safe_blend_norm + DEFAULT_EPS))
        use_scaled = (base_norm > DEFAULT_EPS) & (blend_norm > DEFAULT_EPS)
        return torch.where(use_scaled, scaled, base_step)

    def _apply_conv_guard(
        self,
        *,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        prev_update: torch.Tensor,
        step_direction: torch.Tensor,
        controls: dict[str, torch.Tensor],
        lr: float,
        group: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(group["conv_safe_mode"]) or param.ndim < 3:
            zero = torch.zeros((), device=param.device, dtype=torch.float32)
            one = torch.ones((), device=param.device, dtype=torch.float32)
            return step_direction, zero, one

        filter_support = _mean_tensors(
            self._filter_support(grad, momentum),
            self._filter_support(grad, prev_update),
        )
        support_weight = float(group["conv_support_weight"])
        friction_scale = torch.full_like(filter_support, float(group["conv_friction_scale"])) + support_weight * filter_support
        alignment_scale = torch.full_like(filter_support, float(group["conv_alignment_scale"])) + support_weight * filter_support
        controls["friction_multiplier"] = 1.0 + (controls["friction_multiplier"] - 1.0) * friction_scale
        controls["projection_strength"] = controls["projection_strength"] * float(group["conv_projection_scale"]) * (0.7 + 0.3 * filter_support)
        controls["alignment_scale"] = 1.0 + (controls["alignment_scale"] - 1.0) * alignment_scale

        param_norm = param.detach().float().norm()
        update_norm = (step_direction.detach().float() * lr).norm()
        update_ratio = update_norm / (param_norm + DEFAULT_EPS)
        cap = torch.full_like(update_ratio, float(group["conv_update_ratio_cap"]))
        scale = torch.where(update_ratio > cap, cap / (update_ratio + DEFAULT_EPS), torch.ones_like(update_ratio))
        step_direction = step_direction * scale
        return step_direction, filter_support, scale

    @torch.no_grad()
    def step(self, closure=None):
        capture_diagnostics = self.enable_step_diagnostics and ((self._global_step + 1) % self.diagnostics_every_n_steps == 0)
        current_loss = self.current_loss
        potential_before = 0.0 if current_loss is None else float(current_loss)

        diagnostics: dict[str, list[float]] = {
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
            "filter_support": [],
            "conv_step_multiplier": [],
            "soft_conflict_correction": [],
        }

        mode_used = None
        closure_recomputed = False
        updated_params: list[tuple[torch.Tensor, dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor, float, dict[str, torch.Tensor]]] = []
        kinetic_sum = 0.0
        inverse_mass_sum = 0.0
        inverse_mass_sq_sum = 0.0
        total_elements = 0
        gradient_sq_sum = 0.0
        parameter_step_sq_sum = 0.0
        momentum_sq_sum = 0.0
        damping_amount = 0.0

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
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
                controls, force_direction = self._compute_controls_tensor(
                    grad=grad,
                    momentum=momentum,
                    prev_grad=prev_grad,
                    prev_update=prev_update,
                    inverse_mass=inverse_mass,
                    group=group,
                )

                friction_factor = torch.ones((), device=grad.device, dtype=grad.dtype)
                if use_friction and friction > 0.0:
                    friction_factor = torch.exp(
                        torch.full((), -lr * friction, device=grad.device, dtype=grad.dtype) * controls["friction_multiplier"].to(dtype=grad.dtype)
                    )

                use_leapfrog = mode == "leapfrog_with_closure" and closure is not None
                if mode in {"adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure is not None:
                    use_leapfrog = True
                if use_leapfrog:
                    momentum.mul_(torch.sqrt(friction_factor)).add_(grad, alpha=-0.5 * lr)
                else:
                    momentum.mul_(friction_factor).add_(grad, alpha=-lr)

                target_momentum = force_direction * (momentum.detach().float().norm() / (force_direction.detach().float().norm() + DEFAULT_EPS))
                conflict_correction = controls["soft_conflict_correction"].to(dtype=momentum.dtype)
                momentum.mul_(1.0 - conflict_correction).add_(target_momentum.to(dtype=momentum.dtype) * conflict_correction)

                base_step = inverse_mass * momentum
                step_direction = self._blend_step_direction_tensor(
                    base_step=base_step,
                    force_direction=force_direction,
                    projection_strength=controls["projection_strength"].to(dtype=base_step.dtype),
                )
                step_direction, filter_support, conv_step_multiplier = self._apply_conv_guard(
                    param=param,
                    grad=grad,
                    momentum=momentum,
                    prev_update=prev_update,
                    step_direction=step_direction,
                    controls=controls,
                    lr=lr,
                    group=group,
                )
                step_direction = step_direction * controls["alignment_scale"].to(dtype=step_direction.dtype)

                if use_decoupled_weight_decay and weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(step_direction, alpha=lr)

                updated_params.append(
                    (
                        param,
                        state,
                        step_direction.detach().clone(),
                        inverse_mass.detach().clone(),
                        friction_factor.detach().clone(),
                        lr,
                        {name: value.detach().clone() for name, value in controls.items()},
                    )
                )

                total_elements += grad.numel()
                grad_float = grad.detach().float()
                step_float = (step_direction.detach().float() * lr)
                momentum_float = momentum.detach().float()
                inverse_mass_float = inverse_mass.detach().float()
                gradient_sq_sum += float(grad_float.pow(2).sum().item())
                parameter_step_sq_sum += float(step_float.pow(2).sum().item())
                momentum_sq_sum += float(momentum_float.pow(2).sum().item())
                inverse_mass_sum += float(inverse_mass_float.sum().item())
                inverse_mass_sq_sum += float(inverse_mass_float.pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum_float.pow(2) * inverse_mass_float).sum().item())
                damping_amount += 1.0 - safe_float(friction_factor, default=1.0)

                if capture_diagnostics:
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
                        "soft_conflict_correction",
                    ]:
                        diagnostics[key].append(safe_float(controls[key]))
                    diagnostics["magneto_projection_strength"].append(safe_float(controls["projection_strength"]))
                    diagnostics["magneto_friction_multiplier"].append(safe_float(controls["friction_multiplier"]))
                    diagnostics["filter_support"].append(safe_float(filter_support))
                    diagnostics["conv_step_multiplier"].append(safe_float(conv_step_multiplier))

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
                momentum.mul_(torch.sqrt(friction_factor.to(dtype=momentum.dtype))).add_(grad, alpha=-0.5 * lr)
                post_projection = 0.5 * controls["projection_strength"]
                force_direction = -(grad * inverse_mass)
                target_momentum = force_direction * (momentum.detach().float().norm() / (force_direction.detach().float().norm() + DEFAULT_EPS))
                momentum.mul_(1.0 - post_projection.to(dtype=momentum.dtype)).add_(target_momentum.to(dtype=momentum.dtype) * post_projection.to(dtype=momentum.dtype))
                grad_float = grad.detach().float()
                total_elements += grad.numel()
                gradient_sq_sum += float(grad_float.pow(2).sum().item())
                parameter_step_sq_sum += float((step_direction.detach().float() * lr).pow(2).sum().item())
                momentum_sq_sum += float(momentum.detach().float().pow(2).sum().item())
                inverse_mass_float = inverse_mass.detach().float()
                inverse_mass_sum += float(inverse_mass_float.sum().item())
                inverse_mass_sq_sum += float(inverse_mass_float.pow(2).sum().item())
                kinetic_sum += 0.5 * float((momentum.detach().float().pow(2) * inverse_mass_float).sum().item())

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
                    state["hamiltonian_momentum"].mul_(1.0 - energy_correction_applied)
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

        diagnostics_row = {
            "loss": potential_after,
            "potential_energy": potential_after,
            "kinetic_energy": kinetic_energy,
            "total_hamiltonian": total_hamiltonian,
            "energy_drift": energy_drift,
            "relative_energy_drift": relative_energy_drift,
            "normalized_total_energy": kinetic_energy / (abs(potential_after) + DEFAULT_EPS),
            "momentum_norm": math.sqrt(max(momentum_sq_sum, 0.0)),
            "parameter_step_norm": math.sqrt(max(parameter_step_sq_sum, 0.0)),
            "gradient_norm": math.sqrt(max(gradient_sq_sum, 0.0)),
            "inverse_mass_mean": inverse_mass_mean,
            "inverse_mass_std": inverse_mass_std,
            "damping_amount": damping_amount / max(len(updated_params), 1) + energy_correction_applied,
            "effective_damping": damping_amount / max(len(updated_params), 1) + energy_correction_applied,
            "leapfrog_enabled": 1.0 if mode_used in {"leapfrog_with_closure", "adam_preconditioned_hamiltonian", "dissipative_hamiltonian"} and closure_recomputed else 0.0,
            "closure_recomputed_gradient": 1.0 if closure_recomputed else 0.0,
            "symplectic_euler_approximation": 0.0 if closure_recomputed else 1.0,
            "effective_lr_scale": average(diagnostics["alignment_scale"]) if diagnostics["alignment_scale"] else 1.0,
            "alignment_scale": average(diagnostics["alignment_scale"]) if diagnostics["alignment_scale"] else 1.0,
            "magneto_projection_strength": average(diagnostics["magneto_projection_strength"]),
            "magneto_friction_multiplier": average(diagnostics["magneto_friction_multiplier"]),
            "filter_support": average(diagnostics["filter_support"]),
            "conv_step_multiplier": average(diagnostics["conv_step_multiplier"]),
            "preset": str(self.param_groups[0]["preset"]),
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
            "soft_conflict_correction",
        ]:
            diagnostics_row[key] = average(diagnostics[key])
        self._record_step(diagnostics_row)
        return loss_tensor
