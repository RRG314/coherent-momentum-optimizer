from __future__ import annotations

import math
import time
from typing import Any

import torch

from .block_direction_optimizer_v2 import _normalize_rows, _row_cosine, _softmax_masked
from .block_direction_optimizer_v4_fast import V4_COST_PENALTIES
from .block_direction_optimizer_v44 import BlockDirectionOptimizerV44
from .optimizer_utils import average, clamp_scalar


class BlockDirectionOptimizerV45(BlockDirectionOptimizerV44):
    """Regime-routed typed block optimizer.

    V4.5 keeps the validated block-direction core intact:

    - the small V4Fast/V4.4 candidate set
    - winner-take-all selection
    - typed dense/conv parameter profiles
    - conv-safe scaling for convolutional tensors

    The new ingredient is a low-cost block policy router. Each block infers a
    local regime from coherence, structure support, and stress, then blends
    three scoring profiles:

    - dense_stable: reward trusted consensus under low oscillation
    - structured: reward matrix/conv-supported directions
    - stress: reward descent-aligned directions under oscillation or reversal

    The direction still comes from blockwise candidate selection rather than
    Adam moments. The router only changes how each block scores the existing
    candidates and how readily it leaves the raw-gradient fallback path.
    """

    def __init__(
        self,
        params,
        *,
        route_temperature: float = 0.72,
        stable_route_bias: float = 0.12,
        structured_route_bias: float = 0.0,
        stress_route_bias: float = -0.02,
        stable_route_weight: float = 1.0,
        structured_route_weight: float = 1.0,
        stress_route_weight: float = 1.05,
        route_stress_scale: float = 1.35,
        route_update_pressure_scale: float = 0.35,
        stable_route_bonus: float = 0.08,
        structured_route_bonus: float = 0.08,
        stress_route_bonus: float = 0.09,
        stress_memory_penalty: float = 0.04,
        route_threshold_relaxation: float = 0.04,
        route_step_gain: float = 0.04,
        route_stress_damping: float = 0.03,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("lr", 1.5e-2)
        kwargs.setdefault("weight_decay", 0.0)
        kwargs.setdefault("block_strategy", "smart_v4")
        kwargs.setdefault("selection_mode", "winner_take_all")
        kwargs.setdefault("selection_temperature", 0.25)
        kwargs.setdefault("memory_decay", 0.88)
        kwargs.setdefault("smooth_decay", 0.96)
        kwargs.setdefault("grad_decay", 0.97)
        kwargs.setdefault("trust_decay", 0.90)
        kwargs.setdefault("descent_weight", 0.52)
        kwargs.setdefault("coherence_weight", 0.18)
        kwargs.setdefault("improvement_weight", 0.04)
        kwargs.setdefault("recoverability_weight", 0.0)
        kwargs.setdefault("stability_weight", 0.16)
        kwargs.setdefault("oscillation_penalty", 0.12)
        kwargs.setdefault("conflict_penalty", 0.12)
        kwargs.setdefault("cost_penalty_weight", 0.02)
        kwargs.setdefault("recovery_threshold", 0.55)
        kwargs.setdefault("recoverability_keep_ratio", 0.5)
        kwargs.setdefault("recoverability_noise_scale", 0.02)
        kwargs.setdefault("recoverability_drop_fraction", 0.10)
        kwargs.setdefault("recoverability_samples", 0)
        kwargs.setdefault("dimension_power", 0.06)
        kwargs.setdefault("max_update_ratio", 0.16)
        kwargs.setdefault("min_scale", 0.82)
        kwargs.setdefault("max_scale", 1.45)
        kwargs.setdefault("fallback_threshold", 0.06)
        kwargs.setdefault("magnitude_mode", "energy_normalized")
        kwargs.setdefault("use_gradient_candidate", True)
        kwargs.setdefault("use_trusted_direction_candidate", True)
        kwargs.setdefault("use_low_rank_candidate", True)
        kwargs.setdefault("use_stable_consensus_candidate", True)
        kwargs.setdefault("consensus_memory_mix", 0.38)
        kwargs.setdefault("consensus_matrix_mix", 0.14)
        kwargs.setdefault("stable_consensus_bonus", 0.10)
        kwargs.setdefault("matrix_consensus_bonus", 0.05)
        kwargs.setdefault("matrix_candidate_mode", "row_column_consensus")
        kwargs.setdefault("row_column_mix", 0.6)
        kwargs.setdefault("row_column_energy_weight", 0.55)
        kwargs.setdefault("small_matrix_cutoff", 1024)
        kwargs.setdefault("energy_decay", 0.97)
        kwargs.setdefault("energy_power", 0.50)
        kwargs.setdefault("use_recoverability_gate", False)
        kwargs.setdefault("recoverability_interval", 8)
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["route_temperature"] = float(route_temperature)
            group["stable_route_bias"] = float(stable_route_bias)
            group["structured_route_bias"] = float(structured_route_bias)
            group["stress_route_bias"] = float(stress_route_bias)
            group["stable_route_weight"] = float(stable_route_weight)
            group["structured_route_weight"] = float(structured_route_weight)
            group["stress_route_weight"] = float(stress_route_weight)
            group["route_stress_scale"] = float(route_stress_scale)
            group["route_update_pressure_scale"] = float(route_update_pressure_scale)
            group["stable_route_bonus"] = float(stable_route_bonus)
            group["structured_route_bonus"] = float(structured_route_bonus)
            group["stress_route_bonus"] = float(stress_route_bonus)
            group["stress_memory_penalty"] = float(stress_memory_penalty)
            group["route_threshold_relaxation"] = float(route_threshold_relaxation)
            group["route_step_gain"] = float(route_step_gain)
            group["route_stress_damping"] = float(route_stress_damping)
        self._initialize_physical_optimizer("BlockDirectionOptimizerV4.5")

    def _route_weights(
        self,
        *,
        coherence: torch.Tensor,
        consensus: torch.Tensor,
        conflict: torch.Tensor,
        oscillation: torch.Tensor,
        structure_support: torch.Tensor,
        gradient_stress: torch.Tensor,
        block_profile: str,
        route_temperature: float,
        stable_route_bias: float,
        structured_route_bias: float,
        stress_route_bias: float,
        stable_route_weight: float,
        structured_route_weight: float,
        stress_route_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        oscillation_norm = torch.clamp(oscillation / 1.5, 0.0, 1.0)
        stable_logit = (
            stable_route_bias
            + stable_route_weight
            * (
                0.90 * coherence
                + 0.40 * consensus
                + 0.22 * (1.0 - gradient_stress)
                - 0.55 * oscillation_norm
                - 0.35 * conflict
            )
        )
        structured_logit = (
            structured_route_bias
            + structured_route_weight
            * (
                0.95 * structure_support
                + 0.35 * consensus
                + 0.15 * coherence
                - 0.20 * oscillation_norm
            )
        )
        stress_logit = (
            stress_route_bias
            + stress_route_weight
            * (
                0.90 * gradient_stress
                + 0.55 * oscillation_norm
                + 0.35 * conflict
                - 0.25 * coherence
            )
        )
        if block_profile == "conv":
            structured_logit = structured_logit + 0.18
        else:
            stable_logit = stable_logit + 0.12

        logits = torch.stack([stable_logit, structured_logit, stress_logit], dim=1)
        weights = torch.softmax(logits / max(route_temperature, 1e-6), dim=1)
        route_entropy = -(
            weights * torch.log(weights.clamp_min(1e-8))
        ).sum(dim=1) / math.log(3.0)
        return weights[:, 0], weights[:, 1], weights[:, 2], route_entropy

    @torch.no_grad()
    def step(self, closure=None):
        step_start = time.perf_counter()
        loss_tensor, _ = self._prepare_closure(closure)
        improvement = float(getattr(self._tracker, "loss_improvement", 0.0) or 0.0)
        loss_scale = max(abs(float(self.current_loss or 0.0)), 1e-3)
        reward = math.tanh(improvement / loss_scale) if math.isfinite(improvement) else 0.0

        selected_candidate_names: list[str] = []
        selected_route_names: list[str] = []
        selected_score_values: list[float] = []
        recoverability_values: list[float] = []
        coherence_values: list[float] = []
        conflict_values: list[float] = []
        oscillation_values: list[float] = []
        block_step_norm_values: list[float] = []
        fallback_values: list[float] = []
        consensus_values: list[float] = []
        energy_values: list[float] = []
        conv_support_values: list[float] = []
        conv_step_values: list[float] = []
        route_stable_values: list[float] = []
        route_structured_values: list[float] = []
        route_stress_values: list[float] = []
        route_entropy_values: list[float] = []
        route_structure_values: list[float] = []
        active_params = 0
        total_blocks = 0
        divergence_flag = 0.0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            block_strategy = str(group["block_strategy"])
            selection_mode = str(group["selection_mode"])
            selection_temperature = float(group["selection_temperature"])
            memory_decay = clamp_scalar(float(group["memory_decay"]), 0.0, 0.9999)
            smooth_decay = clamp_scalar(float(group["smooth_decay"]), 0.0, 0.9999)
            grad_decay = clamp_scalar(float(group["grad_decay"]), 0.0, 0.9999)
            trust_decay = clamp_scalar(float(group["trust_decay"]), 0.0, 0.9999)
            descent_weight = float(group["descent_weight"])
            coherence_weight = float(group["coherence_weight"])
            improvement_weight = float(group["improvement_weight"])
            recoverability_weight = float(group["recoverability_weight"])
            stability_weight = float(group["stability_weight"])
            oscillation_penalty = float(group["oscillation_penalty"])
            conflict_penalty = float(group["conflict_penalty"])
            cost_penalty_weight = float(group["cost_penalty_weight"])
            recovery_threshold = float(group["recovery_threshold"])
            recoverability_keep_ratio = float(group["recoverability_keep_ratio"])
            recoverability_noise_scale = float(group["recoverability_noise_scale"])
            recoverability_drop_fraction = float(group["recoverability_drop_fraction"])
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            fallback_threshold = float(group["fallback_threshold"])
            magnitude_mode = str(group["magnitude_mode"])
            maximize = bool(group["maximize"])
            small_matrix_cutoff = int(group.get("small_matrix_cutoff", 1024))
            energy_decay = clamp_scalar(float(group.get("energy_decay", 0.97)), 0.0, 0.9999)
            energy_power = clamp_scalar(float(group.get("energy_power", 0.5)), 0.0, 1.0)
            use_recoverability_gate = bool(group.get("use_recoverability_gate", False))
            recoverability_interval = max(1, int(group.get("recoverability_interval", 8)))
            use_stable_consensus_candidate = bool(group.get("use_stable_consensus_candidate", True))
            consensus_memory_mix = clamp_scalar(float(group.get("consensus_memory_mix", 0.38)), 0.0, 1.5)
            consensus_matrix_mix = clamp_scalar(float(group.get("consensus_matrix_mix", 0.14)), 0.0, 1.0)
            stable_consensus_bonus = float(group.get("stable_consensus_bonus", 0.10))
            matrix_consensus_bonus = float(group.get("matrix_consensus_bonus", 0.05))
            conv_max_update_ratio = max(0.0, float(group.get("conv_max_update_ratio", 0.10)))
            conv_energy_power_bonus = clamp_scalar(float(group.get("conv_energy_power_bonus", 0.18)), 0.0, 1.0)
            conv_step_floor = clamp_scalar(float(group.get("conv_step_floor", 0.80)), 0.0, 1.0)
            block_profile = str(group.get("block_profile", "dense"))

            route_temperature = clamp_scalar(float(group.get("route_temperature", 0.72)), 0.05, 5.0)
            stable_route_bias = float(group.get("stable_route_bias", 0.12))
            structured_route_bias = float(group.get("structured_route_bias", 0.0))
            stress_route_bias = float(group.get("stress_route_bias", -0.02))
            stable_route_weight = float(group.get("stable_route_weight", 1.0))
            structured_route_weight = float(group.get("structured_route_weight", 1.0))
            stress_route_weight = float(group.get("stress_route_weight", 1.05))
            route_stress_scale = max(1e-6, float(group.get("route_stress_scale", 1.35)))
            route_update_pressure_scale = max(1e-6, float(group.get("route_update_pressure_scale", 0.35)))
            stable_route_bonus = float(group.get("stable_route_bonus", 0.08))
            structured_route_bonus = float(group.get("structured_route_bonus", 0.08))
            stress_route_bonus = float(group.get("stress_route_bonus", 0.09))
            stress_memory_penalty = float(group.get("stress_memory_penalty", 0.04))
            route_threshold_relaxation = max(0.0, float(group.get("route_threshold_relaxation", 0.04)))
            route_step_gain = max(0.0, float(group.get("route_step_gain", 0.04)))
            route_stress_damping = max(0.0, float(group.get("route_stress_damping", 0.03)))

            enabled_candidates = {
                "gradient": bool(group["use_gradient_candidate"]),
                "stable_consensus": use_stable_consensus_candidate,
                "trusted_direction": bool(group["use_trusted_direction_candidate"]),
                "low_rank_matrix": bool(group["use_low_rank_candidate"]),
            }

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    divergence_flag = 1.0
                    continue

                state = self.state[param]
                layout, block_count = self._init_state_v4(state, param, block_strategy, small_matrix_cutoff)
                self._update_quality_memory(state, reward)
                state["step"] = int(state["step"]) + 1

                grad_blocks, _ = self._block_view_v4(grad, block_strategy, small_matrix_cutoff)
                param_blocks, _ = self._block_view_v4(param.detach(), block_strategy, small_matrix_cutoff)
                descent_blocks = -grad_blocks
                descent_dirs, grad_norms = _normalize_rows(descent_blocks, eps)
                if not bool((grad_norms > eps).any()):
                    continue

                trusted_blocks, _ = self._block_view_v4(state["trusted_direction"].reshape_as(param.detach()), block_strategy, small_matrix_cutoff)
                smoothed_blocks, _ = self._block_view_v4(state["smoothed_direction"].reshape_as(param.detach()), block_strategy, small_matrix_cutoff)
                prev_grad_blocks, _ = self._block_view_v4(state["prev_grad"].reshape_as(param.detach()), block_strategy, small_matrix_cutoff)
                prev_update_blocks, _ = self._block_view_v4(state["prev_update"].reshape_as(param.detach()), block_strategy, small_matrix_cutoff)

                trusted_dirs, trusted_norms = _normalize_rows(trusted_blocks, eps)
                smoothed_dirs, smoothed_norms = _normalize_rows(smoothed_blocks, eps)
                prev_grad_dirs, _ = _normalize_rows(prev_grad_blocks, eps)
                prev_update_dirs, _ = _normalize_rows(prev_update_blocks, eps)

                trusted_exists = trusted_norms > eps
                smoothed_exists = smoothed_norms > eps

                candidate_blocks: dict[str, torch.Tensor] = {}
                if enabled_candidates["gradient"]:
                    candidate_blocks["gradient"] = descent_dirs
                if enabled_candidates["trusted_direction"] and bool(trusted_exists.any()):
                    candidate_blocks["trusted_direction"] = trusted_dirs

                matrix_dirs = None
                matrix_candidates = self._matrix_candidate_blocks(
                    grad,
                    "tensor" if block_strategy == "smart_v4" and grad.ndim == 2 and grad.numel() < small_matrix_cutoff else block_strategy,
                    eps,
                    use_low_rank=enabled_candidates["low_rank_matrix"],
                    use_muon_like=False,
                )
                if "low_rank_matrix" in matrix_candidates:
                    matrix_dirs, _ = _normalize_rows(matrix_candidates["low_rank_matrix"], eps)
                    candidate_blocks["low_rank_matrix"] = matrix_dirs

                stable_candidate = self._build_stable_consensus_candidate(
                    descent_dirs=descent_dirs,
                    trusted_dirs=trusted_dirs,
                    smoothed_dirs=smoothed_dirs,
                    trusted_exists=trusted_exists,
                    smoothed_exists=smoothed_exists,
                    matrix_dirs=matrix_dirs,
                    memory_mix=consensus_memory_mix,
                    matrix_mix=consensus_matrix_mix,
                    eps=eps,
                )
                if enabled_candidates["stable_consensus"] and stable_candidate is not None:
                    candidate_blocks["stable_consensus"] = stable_candidate

                conv_support = self._conv_structure_support(
                    grad,
                    block_strategy,
                    eps,
                    small_matrix_cutoff=small_matrix_cutoff,
                )

                score_columns: list[torch.Tensor] = []
                valid_columns: list[torch.Tensor] = []
                direction_columns: list[torch.Tensor] = []
                selected_components: dict[str, torch.Tensor] = {}
                quality_matrix = state["candidate_quality"]
                grad_norm_ema = state["grad_norm_ema"]
                grad_norm_ema = torch.where(
                    grad_norm_ema > eps,
                    grad_decay * grad_norm_ema + (1.0 - grad_decay) * grad_norms.float(),
                    grad_norms.float(),
                )
                zero_metric = torch.zeros(block_count, dtype=torch.float32, device=param.device)

                param_norms = param_blocks.detach().float().norm(dim=1)
                base_trusted_alignment = (
                    ((_row_cosine(descent_dirs, trusted_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    if bool(trusted_exists.any())
                    else zero_metric
                )
                base_smoothed_alignment = (
                    ((_row_cosine(descent_dirs, smoothed_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    if bool(smoothed_exists.any())
                    else zero_metric
                )
                base_memory_coherence = torch.maximum(base_trusted_alignment, base_smoothed_alignment)
                base_conflict = (
                    0.5
                    * (
                        torch.clamp(-_row_cosine(descent_dirs, trusted_dirs, eps), min=0.0)
                        + torch.clamp(-_row_cosine(descent_dirs, prev_update_dirs, eps), min=0.0)
                    )
                    if bool(trusted_exists.any())
                    else torch.clamp(-_row_cosine(descent_dirs, prev_update_dirs, eps), min=0.0)
                )
                base_oscillation = (
                    0.5 * (1.0 - _row_cosine(descent_dirs, prev_grad_dirs, eps))
                    + 0.5 * torch.clamp(-_row_cosine(descent_dirs, prev_update_dirs, eps), min=0.0)
                ).clamp(0.0, 1.5)
                matrix_structure_support = (
                    ((_row_cosine(matrix_dirs, descent_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    if matrix_dirs is not None
                    else zero_metric
                )
                structure_support = torch.maximum(matrix_structure_support, conv_support.float()) if conv_support is not None else matrix_structure_support
                base_consensus = (0.55 + 0.25 * base_memory_coherence + 0.20 * structure_support).clamp(0.0, 1.0)
                grad_change = torch.abs(torch.log((grad_norms.float() + eps) / (grad_norm_ema + eps)))
                grad_pressure = torch.clamp(grad_change / route_stress_scale, 0.0, 1.0)
                update_pressure = torch.clamp((grad_norms.float() / (param_norms + eps)) / route_update_pressure_scale, 0.0, 1.0)
                gradient_stress = (0.65 * grad_pressure + 0.35 * update_pressure).clamp(0.0, 1.0)
                stable_route, structured_route, stress_route, route_entropy = self._route_weights(
                    coherence=base_memory_coherence,
                    consensus=base_consensus,
                    conflict=torch.clamp(base_conflict, 0.0, 1.0),
                    oscillation=base_oscillation,
                    structure_support=structure_support,
                    gradient_stress=gradient_stress,
                    block_profile=block_profile,
                    route_temperature=route_temperature,
                    stable_route_bias=stable_route_bias,
                    structured_route_bias=structured_route_bias,
                    stress_route_bias=stress_route_bias,
                    stable_route_weight=stable_route_weight,
                    structured_route_weight=structured_route_weight,
                    stress_route_weight=stress_route_weight,
                )

                for candidate_name in self.candidate_names:
                    candidate = candidate_blocks.get(candidate_name)
                    if candidate is None:
                        score_columns.append(torch.full((block_count,), -1e6, dtype=torch.float32, device=param.device))
                        valid_columns.append(torch.zeros(block_count, dtype=torch.bool, device=param.device))
                        direction_columns.append(torch.zeros_like(descent_dirs))
                        continue

                    candidate_dirs, candidate_norms = _normalize_rows(candidate, eps)
                    valid = candidate_norms > eps
                    descent_alignment = ((_row_cosine(candidate_dirs, descent_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    trusted_alignment = (
                        ((_row_cosine(candidate_dirs, trusted_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if bool(trusted_exists.any())
                        else zero_metric
                    )
                    smoothed_alignment = (
                        ((_row_cosine(candidate_dirs, smoothed_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if bool(smoothed_exists.any())
                        else zero_metric
                    )
                    matrix_alignment = (
                        ((_row_cosine(candidate_dirs, matrix_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if matrix_dirs is not None
                        else zero_metric
                    )
                    memory_coherence = torch.maximum(trusted_alignment, smoothed_alignment)
                    consensus_strength = (
                        0.50 * descent_alignment
                        + 0.30 * memory_coherence
                        + 0.20 * matrix_alignment
                    ).clamp(0.0, 1.0)
                    improvement_history = ((quality_matrix[:, self._candidate_index[candidate_name]].float() + 1.0) * 0.5).clamp(0.0, 1.0)
                    oscillation_score = (
                        0.5 * (1.0 - _row_cosine(candidate_dirs, prev_grad_dirs, eps))
                        + 0.5 * torch.clamp(-_row_cosine(candidate_dirs, prev_update_dirs, eps), min=0.0)
                    ).clamp(0.0, 1.5)
                    conflict_score = (
                        0.5
                        * (
                            torch.clamp(-_row_cosine(candidate_dirs, trusted_dirs, eps), min=0.0)
                            + torch.clamp(-_row_cosine(candidate_dirs, prev_update_dirs, eps), min=0.0)
                        )
                        if bool(trusted_exists.any())
                        else torch.clamp(-_row_cosine(candidate_dirs, prev_update_dirs, eps), min=0.0)
                    )
                    stability_score = 1.0 / (
                        1.0
                        + torch.abs(
                            torch.log(
                                (grad_norms.float() * descent_alignment.clamp_min(0.05) + eps)
                                / (grad_norm_ema + eps)
                            )
                        )
                    )

                    recovery_score = torch.ones(block_count, dtype=torch.float32, device=param.device)
                    if use_recoverability_gate and int(state["step"]) % recoverability_interval == 0:
                        recovery_score = self._recoverability_score(
                            candidate,
                            candidate_dirs,
                            keep_ratio=recoverability_keep_ratio,
                            noise_scale=recoverability_noise_scale,
                            drop_fraction=recoverability_drop_fraction,
                            samples=1,
                            eps=eps,
                        )
                    recovery_gate = torch.clamp((recovery_score - recovery_threshold) / max(1.0 - recovery_threshold, eps), 0.0, 1.0)

                    structure_alignment = torch.maximum(matrix_alignment, structure_support)
                    effective_descent_weight = descent_weight * (1.0 + 0.18 * stress_route)
                    effective_coherence_weight = coherence_weight * (1.0 + 0.22 * stable_route)
                    effective_stability_weight = stability_weight * (1.0 + 0.18 * stable_route + 0.10 * structured_route)
                    effective_oscillation_penalty = oscillation_penalty * (1.0 + 0.35 * stress_route)
                    effective_conflict_penalty = conflict_penalty * (1.0 + 0.30 * stress_route)
                    effective_stable_bonus = stable_consensus_bonus + stable_route_bonus * stable_route + 0.03 * structured_route
                    effective_matrix_bonus = matrix_consensus_bonus + structured_route_bonus * structured_route

                    candidate_route_bonus = zero_metric
                    if candidate_name == "gradient":
                        candidate_route_bonus = stress_route_bonus * stress_route * descent_alignment
                    elif candidate_name == "trusted_direction":
                        candidate_route_bonus = stable_route_bonus * stable_route * memory_coherence - stress_memory_penalty * stress_route * torch.clamp(conflict_score, 0.0, 1.0)
                    elif candidate_name == "stable_consensus":
                        candidate_route_bonus = (
                            stable_route_bonus * stable_route * consensus_strength
                            + 0.7 * structured_route_bonus * structured_route * structure_alignment
                            + 0.4 * stress_route_bonus * stress_route * descent_alignment
                        )
                    elif candidate_name == "low_rank_matrix":
                        candidate_route_bonus = structured_route_bonus * structured_route * structure_alignment - 0.25 * stress_route_bonus * stress_route * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)

                    raw_score = (
                        effective_descent_weight * descent_alignment
                        + effective_coherence_weight * memory_coherence
                        + improvement_weight * improvement_history
                        + recoverability_weight * recovery_gate * recovery_score
                        + effective_stability_weight * stability_score
                        + effective_stable_bonus * consensus_strength
                        + effective_matrix_bonus * matrix_alignment
                        + candidate_route_bonus
                        - effective_oscillation_penalty * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)
                        - effective_conflict_penalty * torch.clamp(conflict_score, 0.0, 1.0)
                        - cost_penalty_weight * V4_COST_PENALTIES[candidate_name]
                    )
                    score_columns.append(torch.where(valid, raw_score.float(), torch.full_like(raw_score.float(), -1e6)))
                    valid_columns.append(valid)
                    direction_columns.append(candidate_dirs)
                    selected_components[f"{candidate_name}__recovery"] = recovery_score.float()
                    selected_components[f"{candidate_name}__coherence"] = memory_coherence.float()
                    selected_components[f"{candidate_name}__conflict"] = conflict_score.float()
                    selected_components[f"{candidate_name}__oscillation"] = oscillation_score.float()
                    selected_components[f"{candidate_name}__consensus"] = consensus_strength.float()
                    selected_components[f"{candidate_name}__structure"] = structure_alignment.float()

                score_matrix = torch.stack(score_columns, dim=1)
                valid_matrix = torch.stack(valid_columns, dim=1)
                direction_tensor = torch.stack(direction_columns, dim=1)
                fallback_index = self._candidate_index["gradient"]

                effective_fallback_threshold = torch.full((block_count,), fallback_threshold, dtype=torch.float32, device=param.device)
                effective_fallback_threshold = torch.clamp(
                    effective_fallback_threshold
                    - route_threshold_relaxation * (0.70 * structured_route * structure_support + 0.45 * stress_route),
                    min=0.0,
                )

                best_scores, best_indices = score_matrix.max(dim=1)
                fallback_mask = (~valid_matrix.any(dim=1)) | (best_scores < effective_fallback_threshold)
                best_indices = torch.where(fallback_mask, torch.full_like(best_indices, fallback_index), best_indices)

                if selection_mode == "softmax_weighted_average":
                    weights = _softmax_masked(score_matrix, valid_matrix, selection_temperature)
                    weights = torch.where(
                        fallback_mask.unsqueeze(1),
                        torch.nn.functional.one_hot(torch.full_like(best_indices, fallback_index), num_classes=len(self.candidate_names)).to(weights.dtype),
                        weights,
                    )
                    selected_dirs, _ = _normalize_rows((weights.unsqueeze(2) * direction_tensor).sum(dim=1), eps)
                    selected_scores = (weights * score_matrix.clamp_min(-5.0)).sum(dim=1)
                    selected_index_summary = weights.argmax(dim=1)
                else:
                    selected_dirs = direction_tensor[torch.arange(block_count, device=param.device), best_indices]
                    selected_scores = best_scores
                    selected_index_summary = best_indices

                selected_scores = torch.where(fallback_mask, torch.zeros_like(selected_scores), selected_scores)
                selected_index_summary = torch.where(fallback_mask, torch.full_like(selected_index_summary, fallback_index), selected_index_summary)

                selected_recovery = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_coherence = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_conflict = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_oscillation = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_consensus = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_structure = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                for candidate_name, candidate_index in self._candidate_index.items():
                    mask = selected_index_summary == candidate_index
                    if not bool(mask.any()):
                        continue
                    selected_recovery = torch.where(mask, selected_components.get(f"{candidate_name}__recovery", torch.zeros_like(selected_recovery)), selected_recovery)
                    selected_coherence = torch.where(mask, selected_components.get(f"{candidate_name}__coherence", torch.zeros_like(selected_coherence)), selected_coherence)
                    selected_conflict = torch.where(mask, selected_components.get(f"{candidate_name}__conflict", torch.zeros_like(selected_conflict)), selected_conflict)
                    selected_oscillation = torch.where(mask, selected_components.get(f"{candidate_name}__oscillation", torch.zeros_like(selected_oscillation)), selected_oscillation)
                    selected_consensus = torch.where(mask, selected_components.get(f"{candidate_name}__consensus", torch.zeros_like(selected_consensus)), selected_consensus)
                    selected_structure = torch.where(mask, selected_components.get(f"{candidate_name}__structure", torch.zeros_like(selected_structure)), selected_structure)

                trust_ema = trust_decay * state["trust_ema"] + (1.0 - trust_decay) * selected_scores.float()
                trust_scale = 1.0 + 0.50 * (selected_scores.float() - 0.45) + 0.16 * selected_consensus - 0.08 * selected_conflict
                trust_scale = torch.clamp(trust_scale, min_scale, max_scale)
                trust_scale = torch.maximum(trust_scale, trust_ema.clamp(min_scale, max_scale))

                block_energy = descent_blocks.detach().float().pow(2).mean(dim=1)
                energy_ema = state["grad_sq_ema"]
                energy_ema = torch.where(
                    energy_ema > eps,
                    energy_decay * energy_ema + (1.0 - energy_decay) * block_energy,
                    block_energy,
                )
                state["grad_sq_ema"] = energy_ema
                dimension_scale = float(max(1, grad_blocks.shape[1])) ** dimension_power

                effective_energy_power = torch.full((block_count,), energy_power, dtype=torch.float32, device=param.device)
                conv_step_multiplier = torch.ones(block_count, dtype=torch.float32, device=param.device)
                if conv_support is not None:
                    conv_step_multiplier = conv_step_floor + (1.0 - conv_step_floor) * structure_support
                    effective_energy_power = effective_energy_power + conv_energy_power_bonus * (1.0 - structure_support)

                if magnitude_mode == "energy_normalized":
                    energy_scale = (energy_ema.sqrt() + eps).pow(effective_energy_power)
                    step_norm = lr * trust_scale * grad_norms.float() / torch.clamp(energy_scale, min=eps)
                    step_norm = step_norm / max(dimension_scale, eps)
                else:
                    step_norm = lr * trust_scale * grad_norms.float() / max(dimension_scale, eps)
                step_norm = step_norm * conv_step_multiplier

                route_step_multiplier = torch.clamp(
                    1.0
                    + route_step_gain * stable_route * selected_consensus
                    + 0.5 * route_step_gain * structured_route * selected_structure
                    - route_stress_damping * stress_route * selected_conflict,
                    0.94,
                    1.08,
                )
                step_norm = step_norm * route_step_multiplier

                if max_update_ratio > 0.0:
                    step_caps = max_update_ratio * (param_norms + eps)
                    if conv_support is not None:
                        step_caps = torch.minimum(step_caps, conv_max_update_ratio * (param_norms + eps))
                    step_norm = torch.minimum(step_norm, step_caps)

                update_blocks = selected_dirs * step_norm.unsqueeze(1).to(selected_dirs.dtype)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(self._restore_blocks(update_blocks, layout), alpha=1.0)

                trusted_next, _ = _normalize_rows(trusted_blocks * memory_decay + selected_dirs * (1.0 - memory_decay), eps)
                smoothed_next, _ = _normalize_rows(smoothed_blocks * smooth_decay + selected_dirs * (1.0 - smooth_decay), eps)
                state["trusted_direction"] = self._restore_blocks(trusted_next, layout)
                state["smoothed_direction"] = self._restore_blocks(smoothed_next, layout)
                state["prev_grad"] = self._restore_blocks(descent_blocks, layout)
                state["prev_update"] = self._restore_blocks(update_blocks, layout)
                state["grad_norm_ema"] = grad_norm_ema
                state["trust_ema"] = trust_ema
                state["recoverability_ema"] = 0.9 * state["recoverability_ema"] + 0.1 * selected_recovery
                state["last_selected_index"] = selected_index_summary
                state["last_selected_score"] = selected_scores

                name_counts = torch.bincount(selected_index_summary, minlength=len(self.candidate_names))
                dominant_index = int(name_counts.argmax().item())
                selected_candidate_names.append(self.candidate_names[dominant_index])

                route_matrix = torch.stack([stable_route, structured_route, stress_route], dim=1)
                route_index = route_matrix.argmax(dim=1)
                route_names = ("dense_stable", "conv_structured", "stress_response")
                route_name_counts = torch.bincount(route_index, minlength=3)
                selected_route_names.append(route_names[int(route_name_counts.argmax().item())])

                selected_score_values.extend(float(value) for value in selected_scores.detach().cpu())
                recoverability_values.extend(float(value) for value in selected_recovery.detach().cpu())
                coherence_values.extend(float(value) for value in selected_coherence.detach().cpu())
                conflict_values.extend(float(value) for value in selected_conflict.detach().cpu())
                oscillation_values.extend(float(value) for value in selected_oscillation.detach().cpu())
                block_step_norm_values.extend(float(value) for value in step_norm.detach().cpu())
                fallback_values.extend(float(value) for value in fallback_mask.float().detach().cpu())
                consensus_values.extend(float(value) for value in selected_consensus.detach().cpu())
                energy_values.extend(float(value) for value in energy_ema.detach().cpu())
                conv_support_values.extend(float(value) for value in structure_support.detach().cpu())
                conv_step_values.extend(float(value) for value in (conv_step_multiplier * route_step_multiplier).detach().cpu())
                route_stable_values.extend(float(value) for value in stable_route.detach().cpu())
                route_structured_values.extend(float(value) for value in structured_route.detach().cpu())
                route_stress_values.extend(float(value) for value in stress_route.detach().cpu())
                route_entropy_values.extend(float(value) for value in route_entropy.detach().cpu())
                route_structure_values.extend(float(value) for value in structure_support.detach().cpu())
                active_params += 1
                total_blocks += block_count

        runtime_ms = (time.perf_counter() - step_start) * 1000.0
        dominant_candidate = "gradient"
        if selected_candidate_names:
            dominant_candidate = max(set(selected_candidate_names), key=selected_candidate_names.count)
        dominant_route = "dense_stable"
        if selected_route_names:
            dominant_route = max(set(selected_route_names), key=selected_route_names.count)

        self._record_step(
            {
                "optimizer": self.optimizer_name,
                "selected_candidate_type": dominant_candidate,
                "selected_route_type": dominant_route,
                "selection_mode": str(self.param_groups[0]["selection_mode"]) if self.param_groups else "winner_take_all",
                "trust_score": average(selected_score_values),
                "recovery_score": average(recoverability_values),
                "coherence_score": average(coherence_values),
                "conflict_score": average(conflict_values),
                "oscillation_score": average(oscillation_values),
                "consensus_strength": average(consensus_values),
                "block_energy": average(energy_values),
                "block_step_norm": average(block_step_norm_values),
                "fallback_rate": average(fallback_values),
                "filter_support": average(conv_support_values),
                "conv_step_multiplier": average(conv_step_values),
                "route_dense_stable": average(route_stable_values),
                "route_conv_structured": average(route_structured_values),
                "route_stress_response": average(route_stress_values),
                "route_entropy": average(route_entropy_values),
                "route_structure_support": average(route_structure_values),
                "runtime_overhead_ms": runtime_ms,
                "active_params": float(active_params),
                "block_count": float(total_blocks),
                "divergence_flag": divergence_flag,
            }
        )
        return loss_tensor
