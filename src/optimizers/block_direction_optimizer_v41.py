from __future__ import annotations

import math
import time
from typing import Any

import torch

from .block_direction_optimizer_v2 import _normalize_rows, _row_cosine, _softmax_masked
from .block_direction_optimizer_v4_fast import BlockDirectionOptimizerV4Fast
from .optimizer_utils import average, clamp_scalar


V41_CANDIDATE_NAMES = (
    "gradient",
    "stable_consensus",
    "trusted_direction",
    "low_rank_matrix",
    "filter_consensus",
)

V41_COST_PENALTIES = {
    "gradient": 0.0,
    "stable_consensus": 0.01,
    "trusted_direction": 0.01,
    "low_rank_matrix": 0.04,
    "filter_consensus": 0.02,
}


class BlockDirectionOptimizerV41(BlockDirectionOptimizerV4Fast):
    """V4.1 keeps V4Fast's small candidate core and adds conv-aware structure.

    The branch target is narrow and practical:
    - keep V4Fast's stronger MLP and stress-task behavior
    - improve convolutional tasks without reintroducing V3's wide/slow hot path

    The new conv-specific rule is a filter-consensus candidate. For tensors with
    shape `[out_channels, in_channels, ...]`, it builds a candidate by combining:
    - the raw filter gradient direction
    - a channel profile consensus within each filter
    - a spatial profile consensus within each filter
    - a shared bank profile across filters

    This preserves the block-direction-selection principle: the optimizer still
    chooses among candidate directions rather than using Adam-style moments to
    define the direction.
    """

    candidate_names = V41_CANDIDATE_NAMES

    def __init__(
        self,
        params,
        *,
        use_filter_consensus_candidate: bool = False,
        filter_channel_mix: float = 0.18,
        filter_spatial_mix: float = 0.20,
        filter_bank_mix: float = 0.10,
        filter_consensus_bonus: float = 0.10,
        conv_max_update_ratio: float = 0.10,
        conv_energy_power_bonus: float = 0.15,
        conv_step_floor: float = 0.82,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("lr", 1.2e-2)
        kwargs.setdefault("coherence_weight", 0.22)
        kwargs.setdefault("stability_weight", 0.20)
        kwargs.setdefault("consensus_memory_mix", 0.44)
        kwargs.setdefault("consensus_matrix_mix", 0.12)
        kwargs.setdefault("stable_consensus_bonus", 0.08)
        kwargs.setdefault("matrix_consensus_bonus", 0.04)
        kwargs.setdefault("energy_power", 0.58)
        kwargs.setdefault("small_matrix_cutoff", 768)
        super().__init__(params, **kwargs)
        self._candidate_index = {name: index for index, name in enumerate(self.candidate_names)}
        for group in self.param_groups:
            group["use_filter_consensus_candidate"] = bool(use_filter_consensus_candidate)
            group["filter_channel_mix"] = float(filter_channel_mix)
            group["filter_spatial_mix"] = float(filter_spatial_mix)
            group["filter_bank_mix"] = float(filter_bank_mix)
            group["filter_consensus_bonus"] = float(filter_consensus_bonus)
            group["conv_max_update_ratio"] = float(conv_max_update_ratio)
            group["conv_energy_power_bonus"] = float(conv_energy_power_bonus)
            group["conv_step_floor"] = float(conv_step_floor)
        self._initialize_physical_optimizer("BlockDirectionOptimizerV4.1")

    def _filter_consensus_blocks(
        self,
        grad: torch.Tensor,
        strategy: str,
        eps: float,
        *,
        small_matrix_cutoff: int,
        channel_mix: float,
        spatial_mix: float,
        bank_mix: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if grad.ndim < 3:
            return None, None

        grad_view = -grad.detach().reshape(grad.shape[0], grad.shape[1], -1)
        base_flat = grad_view.reshape(grad.shape[0], -1)
        base_dirs, base_norms = _normalize_rows(base_flat, eps)
        if not bool((base_norms > eps).any()):
            return None, None

        channel_profile = grad_view.mean(dim=2, keepdim=True).expand_as(grad_view).reshape(grad.shape[0], -1)
        spatial_profile = grad_view.mean(dim=1, keepdim=True).expand_as(grad_view).reshape(grad.shape[0], -1)
        bank_profile = grad_view.mean(dim=0, keepdim=True).expand_as(grad_view).reshape(grad.shape[0], -1)

        channel_dirs, _ = _normalize_rows(channel_profile, eps)
        spatial_dirs, _ = _normalize_rows(spatial_profile, eps)
        bank_dirs, _ = _normalize_rows(bank_profile, eps)

        channel_alignment = torch.clamp(_row_cosine(base_dirs, channel_dirs, eps), min=0.0)
        spatial_alignment = torch.clamp(_row_cosine(base_dirs, spatial_dirs, eps), min=0.0)
        bank_alignment = torch.clamp(_row_cosine(base_dirs, bank_dirs, eps), min=0.0)

        candidate = base_dirs.clone()
        candidate = candidate + channel_mix * channel_alignment.unsqueeze(1).to(base_dirs.dtype) * channel_dirs
        candidate = candidate + spatial_mix * spatial_alignment.unsqueeze(1).to(base_dirs.dtype) * spatial_dirs
        candidate = candidate + bank_mix * bank_alignment.unsqueeze(1).to(base_dirs.dtype) * bank_dirs
        candidate_dirs, candidate_norms = _normalize_rows(candidate, eps)
        if not bool((candidate_norms > eps).any()):
            return None, None

        support = (
            0.40 * channel_alignment
            + 0.35 * spatial_alignment
            + 0.25 * bank_alignment
        ).clamp(0.0, 1.0)
        candidate_tensor = candidate_dirs.reshape_as(grad)
        blocks = self._block_view_v4(candidate_tensor, strategy, small_matrix_cutoff)[0]
        support_tensor = support.unsqueeze(1).expand_as(base_flat).reshape_as(grad)
        support_blocks = self._block_view_v4(support_tensor, strategy, small_matrix_cutoff)[0].float().mean(dim=1)
        return blocks, support_blocks

    @torch.no_grad()
    def step(self, closure=None):
        step_start = time.perf_counter()
        loss_tensor, _ = self._prepare_closure(closure)
        improvement = float(getattr(self._tracker, "loss_improvement", 0.0) or 0.0)
        loss_scale = max(abs(float(self.current_loss or 0.0)), 1e-3)
        reward = math.tanh(improvement / loss_scale) if math.isfinite(improvement) else 0.0

        selected_candidate_names: list[str] = []
        selected_score_values: list[float] = []
        recoverability_values: list[float] = []
        coherence_values: list[float] = []
        conflict_values: list[float] = []
        oscillation_values: list[float] = []
        block_step_norm_values: list[float] = []
        fallback_values: list[float] = []
        consensus_values: list[float] = []
        energy_values: list[float] = []
        filter_support_values: list[float] = []
        conv_step_values: list[float] = []
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
            use_filter_consensus_candidate = bool(group.get("use_filter_consensus_candidate", True))
            filter_channel_mix = clamp_scalar(float(group.get("filter_channel_mix", 0.18)), 0.0, 1.5)
            filter_spatial_mix = clamp_scalar(float(group.get("filter_spatial_mix", 0.20)), 0.0, 1.5)
            filter_bank_mix = clamp_scalar(float(group.get("filter_bank_mix", 0.10)), 0.0, 1.5)
            filter_consensus_bonus = float(group.get("filter_consensus_bonus", 0.10))
            conv_max_update_ratio = max(0.0, float(group.get("conv_max_update_ratio", 0.10)))
            conv_energy_power_bonus = clamp_scalar(float(group.get("conv_energy_power_bonus", 0.15)), 0.0, 1.0)
            conv_step_floor = clamp_scalar(float(group.get("conv_step_floor", 0.82)), 0.0, 1.0)

            enabled_candidates = {
                "gradient": bool(group["use_gradient_candidate"]),
                "stable_consensus": use_stable_consensus_candidate,
                "trusted_direction": bool(group["use_trusted_direction_candidate"]),
                "low_rank_matrix": bool(group["use_low_rank_candidate"]),
                "filter_consensus": use_filter_consensus_candidate,
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

                filter_dirs = None
                filter_support = None
                if enabled_candidates["filter_consensus"]:
                    filter_candidate, filter_support = self._filter_consensus_blocks(
                        grad,
                        block_strategy,
                        eps,
                        small_matrix_cutoff=small_matrix_cutoff,
                        channel_mix=filter_channel_mix,
                        spatial_mix=filter_spatial_mix,
                        bank_mix=filter_bank_mix,
                    )
                    if filter_candidate is not None:
                        filter_dirs, _ = _normalize_rows(filter_candidate, eps)
                        candidate_blocks["filter_consensus"] = filter_dirs

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
                    filter_alignment = (
                        ((_row_cosine(candidate_dirs, filter_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if filter_dirs is not None
                        else zero_metric
                    )
                    memory_coherence = torch.maximum(trusted_alignment, smoothed_alignment)
                    consensus_strength = (
                        0.45 * descent_alignment
                        + 0.25 * memory_coherence
                        + 0.15 * matrix_alignment
                        + 0.15 * filter_alignment
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

                    candidate_filter_support = filter_support if filter_support is not None else zero_metric
                    filter_bonus = torch.zeros_like(candidate_filter_support)
                    if filter_support is not None:
                        if candidate_name == "filter_consensus":
                            filter_bonus = filter_consensus_bonus * candidate_filter_support
                        elif candidate_name in {"stable_consensus", "low_rank_matrix"}:
                            filter_bonus = 0.5 * filter_consensus_bonus * candidate_filter_support

                    raw_score = (
                        descent_weight * descent_alignment
                        + coherence_weight * memory_coherence
                        + improvement_weight * improvement_history
                        + recoverability_weight * recovery_gate * recovery_score
                        + stability_weight * stability_score
                        + stable_consensus_bonus * consensus_strength
                        + matrix_consensus_bonus * matrix_alignment
                        + filter_bonus
                        - oscillation_penalty * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)
                        - conflict_penalty * torch.clamp(conflict_score, 0.0, 1.0)
                        - cost_penalty_weight * V41_COST_PENALTIES[candidate_name]
                    )
                    score_columns.append(torch.where(valid, raw_score.float(), torch.full_like(raw_score.float(), -1e6)))
                    valid_columns.append(valid)
                    direction_columns.append(candidate_dirs)
                    selected_components[f"{candidate_name}__recovery"] = recovery_score.float()
                    selected_components[f"{candidate_name}__coherence"] = memory_coherence.float()
                    selected_components[f"{candidate_name}__conflict"] = conflict_score.float()
                    selected_components[f"{candidate_name}__oscillation"] = oscillation_score.float()
                    selected_components[f"{candidate_name}__consensus"] = consensus_strength.float()
                    selected_components[f"{candidate_name}__filter_support"] = candidate_filter_support.float()

                score_matrix = torch.stack(score_columns, dim=1)
                valid_matrix = torch.stack(valid_columns, dim=1)
                direction_tensor = torch.stack(direction_columns, dim=1)
                fallback_index = self._candidate_index["gradient"]

                best_scores, best_indices = score_matrix.max(dim=1)
                fallback_mask = (~valid_matrix.any(dim=1)) | (best_scores < fallback_threshold)
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
                selected_filter_support = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                for candidate_name, candidate_index in self._candidate_index.items():
                    mask = selected_index_summary == candidate_index
                    if not bool(mask.any()):
                        continue
                    selected_recovery = torch.where(mask, selected_components.get(f"{candidate_name}__recovery", torch.zeros_like(selected_recovery)), selected_recovery)
                    selected_coherence = torch.where(mask, selected_components.get(f"{candidate_name}__coherence", torch.zeros_like(selected_coherence)), selected_coherence)
                    selected_conflict = torch.where(mask, selected_components.get(f"{candidate_name}__conflict", torch.zeros_like(selected_conflict)), selected_conflict)
                    selected_oscillation = torch.where(mask, selected_components.get(f"{candidate_name}__oscillation", torch.zeros_like(selected_oscillation)), selected_oscillation)
                    selected_consensus = torch.where(mask, selected_components.get(f"{candidate_name}__consensus", torch.zeros_like(selected_consensus)), selected_consensus)
                    selected_filter_support = torch.where(mask, selected_components.get(f"{candidate_name}__filter_support", torch.zeros_like(selected_filter_support)), selected_filter_support)

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
                if grad.ndim >= 3:
                    conv_step_multiplier = conv_step_floor + (1.0 - conv_step_floor) * selected_filter_support
                    effective_energy_power = effective_energy_power + conv_energy_power_bonus * (1.0 - selected_filter_support)
                if magnitude_mode == "energy_normalized":
                    energy_scale = (energy_ema.sqrt() + eps).pow(effective_energy_power)
                    step_norm = lr * trust_scale * grad_norms.float() / torch.clamp(energy_scale, min=eps)
                    step_norm = step_norm / max(dimension_scale, eps)
                else:
                    step_norm = lr * trust_scale * grad_norms.float() / max(dimension_scale, eps)
                step_norm = step_norm * conv_step_multiplier

                param_norms = param_blocks.detach().float().norm(dim=1)
                if max_update_ratio > 0.0:
                    step_caps = max_update_ratio * (param_norms + eps)
                    if grad.ndim >= 3:
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
                selected_score_values.extend(float(value) for value in selected_scores.detach().cpu())
                recoverability_values.extend(float(value) for value in selected_recovery.detach().cpu())
                coherence_values.extend(float(value) for value in selected_coherence.detach().cpu())
                conflict_values.extend(float(value) for value in selected_conflict.detach().cpu())
                oscillation_values.extend(float(value) for value in selected_oscillation.detach().cpu())
                block_step_norm_values.extend(float(value) for value in step_norm.detach().cpu())
                fallback_values.extend(float(value) for value in fallback_mask.float().detach().cpu())
                consensus_values.extend(float(value) for value in selected_consensus.detach().cpu())
                energy_values.extend(float(value) for value in energy_ema.detach().cpu())
                filter_support_values.extend(float(value) for value in selected_filter_support.detach().cpu())
                conv_step_values.extend(float(value) for value in conv_step_multiplier.detach().cpu())
                active_params += 1
                total_blocks += block_count

        runtime_ms = (time.perf_counter() - step_start) * 1000.0
        dominant_candidate = "gradient"
        if selected_candidate_names:
            dominant_candidate = max(set(selected_candidate_names), key=selected_candidate_names.count)

        self._record_step(
            {
                "optimizer": self.optimizer_name,
                "selected_candidate_type": dominant_candidate,
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
                "filter_support": average(filter_support_values),
                "conv_step_multiplier": average(conv_step_values),
                "runtime_overhead_ms": runtime_ms,
                "active_params": float(active_params),
                "block_count": float(total_blocks),
                "divergence_flag": divergence_flag,
            }
        )
        return loss_tensor
