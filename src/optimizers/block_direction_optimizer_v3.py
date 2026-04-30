from __future__ import annotations

import math
import time
from typing import Any

import torch

from .block_direction_optimizer_v2 import (
    BlockDirectionOptimizerV2,
    _normalize_rows,
    _row_cosine,
    _row_sign_flip_ratio,
    _softmax_masked,
)
from .optimizer_utils import average, clamp_scalar


V3_CANDIDATE_NAMES = (
    "gradient",
    "normalized_gradient",
    "stable_consensus",
    "trusted_direction",
    "smoothed_direction",
    "projection_corrected",
    "orthogonal_escape",
    "sparse_topk",
    "low_rank_matrix",
    "sign_direction",
    "muon_like_orthogonal",
)

V3_COST_PENALTIES = {
    "gradient": 0.0,
    "normalized_gradient": 0.0,
    "stable_consensus": 0.01,
    "trusted_direction": 0.01,
    "smoothed_direction": 0.01,
    "projection_corrected": 0.03,
    "orthogonal_escape": 0.05,
    "sparse_topk": 0.03,
    "low_rank_matrix": 0.07,
    "sign_direction": 0.02,
    "muon_like_orthogonal": 0.12,
}

V3_STRESS_SPECIALISTS = {"projection_corrected", "orthogonal_escape", "sign_direction"}


class BlockDirectionOptimizerV3(BlockDirectionOptimizerV2):
    """A sharper block-direction branch with explicit matrix consensus.

    V3 keeps the V2 trust-scored candidate-selection framework, but narrows the
    default path toward the rules that survived the V2 study:

    - winner-take-all selection by default
    - block structure kept on by default
    - recoverability retained only as a trust gate
    - projection and orthogonal escape moved off the default path
    - a row/column matrix-consensus candidate replaces the weaker generic
      low-rank default for 2D tensors
    - a stable-consensus candidate blends gradient, memory, and matrix agreement
      when the block is in a low-stress regime
    - stress-specialist candidates are gated so they do not dominate ordinary
      classification and regression tasks
    """

    candidate_names = V3_CANDIDATE_NAMES

    def __init__(
        self,
        params,
        *,
        matrix_candidate_mode: str = "row_column_consensus",
        row_column_mix: float = 0.6,
        row_column_energy_weight: float = 0.55,
        use_stable_consensus_candidate: bool = True,
        consensus_memory_mix: float = 0.42,
        consensus_matrix_mix: float = 0.18,
        stable_consensus_bonus: float = 0.12,
        matrix_consensus_bonus: float = 0.06,
        stress_gate_threshold: float = 0.28,
        stress_gate_power: float = 1.0,
        stress_candidate_penalty: float = 0.16,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("lr", 1.8e-2)
        kwargs.setdefault("weight_decay", 0.0)
        kwargs.setdefault("block_strategy", "smart_v3")
        kwargs.setdefault("selection_mode", "winner_take_all")
        kwargs.setdefault("selection_temperature", 0.25)
        kwargs.setdefault("memory_decay", 0.88)
        kwargs.setdefault("smooth_decay", 0.96)
        kwargs.setdefault("grad_decay", 0.97)
        kwargs.setdefault("trust_decay", 0.90)
        kwargs.setdefault("descent_weight", 0.48)
        kwargs.setdefault("coherence_weight", 0.22)
        kwargs.setdefault("improvement_weight", 0.08)
        kwargs.setdefault("recoverability_weight", 0.05)
        kwargs.setdefault("stability_weight", 0.14)
        kwargs.setdefault("oscillation_penalty", 0.16)
        kwargs.setdefault("conflict_penalty", 0.16)
        kwargs.setdefault("cost_penalty_weight", 0.03)
        kwargs.setdefault("recovery_threshold", 0.52)
        kwargs.setdefault("recoverability_keep_ratio", 0.50)
        kwargs.setdefault("recoverability_noise_scale", 0.03)
        kwargs.setdefault("recoverability_drop_fraction", 0.10)
        kwargs.setdefault("recoverability_samples", 1)
        kwargs.setdefault("topk_fraction", 0.18)
        kwargs.setdefault("projection_strength", 0.0)
        kwargs.setdefault("orthogonal_strength", 0.0)
        kwargs.setdefault("dimension_power", 0.08)
        kwargs.setdefault("max_update_ratio", 0.18)
        kwargs.setdefault("min_scale", 0.72)
        kwargs.setdefault("max_scale", 1.75)
        kwargs.setdefault("fallback_threshold", 0.10)
        kwargs.setdefault("magnitude_mode", "block_norm")
        kwargs.setdefault("rmsprop_decay", 0.97)
        kwargs.setdefault("use_gradient_candidate", True)
        kwargs.setdefault("use_normalized_gradient_candidate", True)
        kwargs.setdefault("use_trusted_direction_candidate", True)
        kwargs.setdefault("use_smoothed_direction_candidate", True)
        kwargs.setdefault("use_projection_candidate", False)
        kwargs.setdefault("use_orthogonal_escape_candidate", False)
        kwargs.setdefault("use_sparse_topk_candidate", True)
        kwargs.setdefault("use_low_rank_candidate", True)
        kwargs.setdefault("use_sign_candidate", True)
        kwargs.setdefault("use_muon_like_candidate", False)
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group["matrix_candidate_mode"] = matrix_candidate_mode
            group["row_column_mix"] = row_column_mix
            group["row_column_energy_weight"] = row_column_energy_weight
            group["use_stable_consensus_candidate"] = use_stable_consensus_candidate
            group["consensus_memory_mix"] = consensus_memory_mix
            group["consensus_matrix_mix"] = consensus_matrix_mix
            group["stable_consensus_bonus"] = stable_consensus_bonus
            group["matrix_consensus_bonus"] = matrix_consensus_bonus
            group["stress_gate_threshold"] = stress_gate_threshold
            group["stress_gate_power"] = stress_gate_power
            group["stress_candidate_penalty"] = stress_candidate_penalty
        self._initialize_physical_optimizer("BlockDirectionOptimizerV3")

    @staticmethod
    def _resolve_block_strategy(param: torch.Tensor, strategy: str) -> str:
        if strategy == "smart_v3":
            if param.ndim == 2:
                return "row"
            if param.ndim <= 1:
                return "scalar"
            return "row"
        return BlockDirectionOptimizerV2._resolve_block_strategy(param, strategy)

    def _matrix_candidate_blocks(
        self,
        grad: torch.Tensor,
        strategy: str,
        eps: float,
        *,
        use_low_rank: bool,
        use_muon_like: bool,
    ) -> dict[str, torch.Tensor]:
        if grad.ndim != 2 or (not use_low_rank and not use_muon_like):
            return {}
        candidates: dict[str, torch.Tensor] = {}
        grad_matrix = -grad.detach()
        group = self.param_groups[0] if self.param_groups else {}
        matrix_candidate_mode = str(group.get("matrix_candidate_mode", "row_column_consensus"))
        row_column_mix = clamp_scalar(float(group.get("row_column_mix", 0.6)), 0.0, 1.0)
        row_column_energy_weight = clamp_scalar(float(group.get("row_column_energy_weight", 0.55)), 0.0, 1.0)

        try:
            if use_low_rank:
                if matrix_candidate_mode == "rank1_svd":
                    u, s, vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
                    rank_one = (u[:, :1] * s[:1]) @ vh[:1, :]
                    matrix_candidate = rank_one.to(dtype=grad.dtype, device=grad.device)
                else:
                    row_dirs, row_norms = _normalize_rows(grad_matrix, eps)
                    column_dirs_t, column_norms = _normalize_rows(grad_matrix.transpose(0, 1).contiguous(), eps)
                    column_dirs = column_dirs_t.transpose(0, 1).contiguous()
                    row_energy = row_norms.unsqueeze(1).to(dtype=grad.dtype, device=grad.device)
                    column_energy = column_norms.unsqueeze(0).to(dtype=grad.dtype, device=grad.device)
                    row_component = row_dirs * row_energy.pow(row_column_energy_weight)
                    column_component = column_dirs * column_energy.pow(max(0.0, 1.0 - row_column_energy_weight))
                    matrix_candidate = row_column_mix * row_component + (1.0 - row_column_mix) * column_component
                candidates["low_rank_matrix"] = self._block_view(matrix_candidate, strategy)[0]

            if use_muon_like:
                u, _, vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
                orthogonal = u @ vh
                candidates["muon_like_orthogonal"] = self._block_view(
                    orthogonal.to(dtype=grad.dtype, device=grad.device),
                    strategy,
                )[0]
        except RuntimeError:
            return candidates
        return candidates

    def _build_stable_consensus_candidate(
        self,
        *,
        descent_dirs: torch.Tensor,
        trusted_dirs: torch.Tensor,
        smoothed_dirs: torch.Tensor,
        trusted_exists: torch.Tensor,
        smoothed_exists: torch.Tensor,
        matrix_dirs: torch.Tensor | None,
        memory_mix: float,
        matrix_mix: float,
        eps: float,
    ) -> torch.Tensor | None:
        if not bool(trusted_exists.any() or smoothed_exists.any() or matrix_dirs is not None):
            return None

        contributions = [descent_dirs]
        if bool(trusted_exists.any()):
            trusted_alignment = torch.clamp(_row_cosine(descent_dirs, trusted_dirs, eps), min=0.0)
            contributions.append(memory_mix * trusted_alignment.unsqueeze(1).to(descent_dirs.dtype) * trusted_dirs)
        if bool(smoothed_exists.any()):
            smoothed_alignment = torch.clamp(_row_cosine(descent_dirs, smoothed_dirs, eps), min=0.0)
            contributions.append(0.6 * memory_mix * smoothed_alignment.unsqueeze(1).to(descent_dirs.dtype) * smoothed_dirs)
        if matrix_dirs is not None:
            matrix_alignment = torch.clamp(_row_cosine(descent_dirs, matrix_dirs, eps), min=0.0)
            contributions.append(matrix_mix * matrix_alignment.unsqueeze(1).to(descent_dirs.dtype) * matrix_dirs)

        candidate = contributions[0]
        for contribution in contributions[1:]:
            candidate = candidate + contribution
        candidate_dirs, candidate_norms = _normalize_rows(candidate, eps)
        if not bool((candidate_norms > eps).any()):
            return None
        return candidate_dirs

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
        stress_values: list[float] = []
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
            recoverability_samples = int(group["recoverability_samples"])
            topk_fraction = float(group["topk_fraction"])
            projection_strength = float(group["projection_strength"])
            orthogonal_strength = float(group["orthogonal_strength"])
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            fallback_threshold = float(group["fallback_threshold"])
            magnitude_mode = str(group["magnitude_mode"])
            rmsprop_decay = clamp_scalar(float(group["rmsprop_decay"]), 0.0, 0.9999)
            maximize = bool(group["maximize"])
            use_stable_consensus_candidate = bool(group.get("use_stable_consensus_candidate", True))
            consensus_memory_mix = clamp_scalar(float(group.get("consensus_memory_mix", 0.42)), 0.0, 1.5)
            consensus_matrix_mix = clamp_scalar(float(group.get("consensus_matrix_mix", 0.18)), 0.0, 1.0)
            stable_consensus_bonus = float(group.get("stable_consensus_bonus", 0.12))
            matrix_consensus_bonus = float(group.get("matrix_consensus_bonus", 0.06))
            stress_gate_threshold = clamp_scalar(float(group.get("stress_gate_threshold", 0.28)), 0.0, 0.95)
            stress_gate_power = max(0.5, float(group.get("stress_gate_power", 1.0)))
            stress_candidate_penalty = float(group.get("stress_candidate_penalty", 0.16))

            enabled_candidates = {
                "gradient": bool(group["use_gradient_candidate"]),
                "normalized_gradient": bool(group["use_normalized_gradient_candidate"]),
                "stable_consensus": use_stable_consensus_candidate,
                "trusted_direction": bool(group["use_trusted_direction_candidate"]),
                "smoothed_direction": bool(group["use_smoothed_direction_candidate"]),
                "projection_corrected": bool(group["use_projection_candidate"]),
                "orthogonal_escape": bool(group["use_orthogonal_escape_candidate"]),
                "sparse_topk": bool(group["use_sparse_topk_candidate"]),
                "low_rank_matrix": bool(group["use_low_rank_candidate"]),
                "sign_direction": bool(group["use_sign_candidate"]),
                "muon_like_orthogonal": bool(group["use_muon_like_candidate"]),
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
                layout, block_count = self._init_state(state, param, block_strategy)
                self._update_quality_memory(state, reward)
                state["step"] = int(state["step"]) + 1

                grad_blocks, _ = self._block_view(grad, block_strategy)
                param_blocks, _ = self._block_view(param.detach(), block_strategy)
                descent_blocks = -grad_blocks
                descent_dirs, grad_norms = _normalize_rows(descent_blocks, eps)
                if not bool((grad_norms > eps).any()):
                    continue

                trusted_blocks = state["trusted_direction"]
                smoothed_blocks = state["smoothed_direction"]
                prev_grad_blocks = state["prev_grad"]
                prev_update_blocks = state["prev_update"]

                trusted_dirs, trusted_norms = _normalize_rows(trusted_blocks, eps)
                smoothed_dirs, smoothed_norms = _normalize_rows(smoothed_blocks, eps)
                prev_grad_dirs, _ = _normalize_rows(prev_grad_blocks, eps)
                prev_update_dirs, _ = _normalize_rows(prev_update_blocks, eps)

                trusted_exists = trusted_norms > eps
                smoothed_exists = smoothed_norms > eps
                anchor_dirs = torch.where(trusted_exists.unsqueeze(1), trusted_dirs, prev_update_dirs)
                anchor_norms = torch.where(trusted_exists, trusted_norms, prev_update_blocks.detach().float().norm(dim=1))
                anchor_exists = anchor_norms > eps

                anchor_alignment = _row_cosine(descent_dirs, anchor_dirs, eps)
                gate = torch.clamp(-anchor_alignment, min=0.0)
                base_oscillation = (
                    0.5 * (1.0 - _row_cosine(descent_dirs, prev_grad_dirs, eps))
                    + 0.5 * _row_sign_flip_ratio(descent_dirs, prev_update_dirs)
                ).clamp(0.0, 1.5)
                prev_grad_norm_ema = state["grad_norm_ema"]
                grad_norm_ratio = torch.abs(torch.log((grad_norms.float() + eps) / (prev_grad_norm_ema + eps)))
                grad_norm_shock = torch.clamp(grad_norm_ratio / 2.0, 0.0, 1.0)
                stress_score = (
                    0.42 * torch.clamp(gate, 0.0, 1.0)
                    + 0.38 * torch.clamp(base_oscillation / 1.5, 0.0, 1.0)
                    + 0.20 * grad_norm_shock
                ).clamp(0.0, 1.0)
                stress_gate = torch.clamp(
                    (stress_score - stress_gate_threshold) / max(1.0 - stress_gate_threshold, eps),
                    0.0,
                    1.0,
                ).pow(stress_gate_power)
                stable_gate = 1.0 - stress_gate

                projection_raw = descent_dirs.clone()
                if bool(anchor_exists.any()):
                    dots = (descent_dirs.detach().float() * anchor_dirs.detach().float()).sum(dim=1, keepdim=True)
                    projection_raw = descent_dirs - projection_strength * gate.unsqueeze(1).to(descent_dirs.dtype) * dots.to(descent_dirs.dtype) * anchor_dirs

                projected_dirs, projected_norms = _normalize_rows(projection_raw, eps)
                orthogonal_component = torch.zeros_like(descent_dirs)
                if bool(anchor_exists.any()):
                    update_dots = (prev_update_dirs.detach().float() * descent_dirs.detach().float()).sum(dim=1, keepdim=True)
                    orthogonal_component = prev_update_dirs - update_dots.to(prev_update_dirs.dtype) * descent_dirs
                escape_raw = descent_dirs + orthogonal_strength * gate.unsqueeze(1).to(descent_dirs.dtype) * orthogonal_component
                escape_dirs, escape_norms = _normalize_rows(escape_raw, eps)
                sign_dirs, sign_norms = _normalize_rows(torch.sign(descent_blocks), eps)
                sparse_dirs, sparse_norms = _normalize_rows(self._topk_candidate(descent_blocks, topk_fraction, eps), eps)

                candidate_blocks: dict[str, torch.Tensor] = {}
                if enabled_candidates["gradient"]:
                    candidate_blocks["gradient"] = descent_blocks
                if enabled_candidates["normalized_gradient"]:
                    candidate_blocks["normalized_gradient"] = descent_dirs
                if enabled_candidates["trusted_direction"] and bool(trusted_exists.any()):
                    candidate_blocks["trusted_direction"] = trusted_dirs
                if enabled_candidates["smoothed_direction"] and bool(smoothed_exists.any()):
                    candidate_blocks["smoothed_direction"] = smoothed_dirs
                if enabled_candidates["projection_corrected"] and bool((projected_norms > eps).any()):
                    candidate_blocks["projection_corrected"] = projected_dirs
                if enabled_candidates["orthogonal_escape"] and bool((escape_norms > eps).any()):
                    candidate_blocks["orthogonal_escape"] = escape_dirs
                if enabled_candidates["sparse_topk"] and bool((sparse_norms > eps).any()):
                    candidate_blocks["sparse_topk"] = sparse_dirs
                if enabled_candidates["sign_direction"] and bool((sign_norms > eps).any()):
                    candidate_blocks["sign_direction"] = sign_dirs
                candidate_blocks.update(
                    self._matrix_candidate_blocks(
                        grad,
                        block_strategy,
                        eps,
                        use_low_rank=enabled_candidates["low_rank_matrix"],
                        use_muon_like=enabled_candidates["muon_like_orthogonal"],
                    )
                )

                matrix_dirs = None
                if "low_rank_matrix" in candidate_blocks:
                    matrix_dirs, _ = _normalize_rows(candidate_blocks["low_rank_matrix"], eps)
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
                grad_norm_ema = torch.where(
                    prev_grad_norm_ema > eps,
                    grad_decay * prev_grad_norm_ema + (1.0 - grad_decay) * grad_norms.float(),
                    grad_norms.float(),
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
                    if candidate_name in V3_STRESS_SPECIALISTS:
                        valid = valid & ((stress_gate > 0.06) | (gate > 0.15))

                    descent_alignment = ((_row_cosine(candidate_dirs, descent_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    trusted_alignment = (
                        ((_row_cosine(candidate_dirs, trusted_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if bool(trusted_exists.any())
                        else torch.zeros_like(descent_alignment)
                    )
                    smoothed_alignment = (
                        ((_row_cosine(candidate_dirs, smoothed_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if bool(smoothed_exists.any())
                        else torch.zeros_like(descent_alignment)
                    )
                    matrix_alignment = (
                        ((_row_cosine(candidate_dirs, matrix_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                        if matrix_dirs is not None
                        else torch.zeros_like(descent_alignment)
                    )
                    memory_coherence = torch.where(trusted_exists, trusted_alignment, smoothed_alignment)
                    improvement_history = ((quality_matrix[:, self._candidate_index[candidate_name]].float() + 1.0) * 0.5).clamp(0.0, 1.0)
                    oscillation_score = (
                        0.5 * (1.0 - _row_cosine(candidate_dirs, prev_grad_dirs, eps))
                        + 0.5 * _row_sign_flip_ratio(candidate_dirs, prev_update_dirs)
                    ).clamp(0.0, 1.5)
                    trusted_conflict = (
                        torch.clamp(-_row_cosine(candidate_dirs, trusted_dirs, eps), min=0.0)
                        if bool(trusted_exists.any())
                        else torch.zeros_like(descent_alignment)
                    )
                    update_conflict = torch.clamp(-_row_cosine(candidate_dirs, prev_update_dirs, eps), min=0.0)
                    conflict_score = 0.5 * (trusted_conflict + update_conflict)
                    support_strength = descent_alignment
                    stability_score = 1.0 / (
                        1.0
                        + torch.abs(
                            torch.log(
                                (grad_norms.float() * support_strength.clamp_min(0.05) + eps)
                                / (grad_norm_ema + eps)
                            )
                        )
                    )
                    recovery_score = self._recoverability_score(
                        candidate,
                        candidate_dirs,
                        keep_ratio=recoverability_keep_ratio,
                        noise_scale=recoverability_noise_scale,
                        drop_fraction=recoverability_drop_fraction,
                        samples=recoverability_samples,
                        eps=eps,
                    )
                    recovery_gate = torch.clamp((recovery_score - recovery_threshold) / max(1.0 - recovery_threshold, eps), 0.0, 1.0)
                    consensus_strength = (
                        0.45 * descent_alignment
                        + 0.25 * torch.maximum(trusted_alignment, smoothed_alignment)
                        + 0.30 * matrix_alignment
                    ).clamp(0.0, 1.0)
                    stable_bonus = stable_gate * stable_consensus_bonus * consensus_strength
                    if candidate_name not in {"stable_consensus", "trusted_direction", "smoothed_direction", "low_rank_matrix"}:
                        stable_bonus = 0.45 * stable_bonus
                    matrix_bonus = stable_gate * matrix_consensus_bonus * matrix_alignment
                    if candidate_name not in {"stable_consensus", "low_rank_matrix"}:
                        matrix_bonus = 0.35 * matrix_bonus
                    specialist_penalty = 0.0
                    if candidate_name in V3_STRESS_SPECIALISTS:
                        specialist_penalty = stress_candidate_penalty * stable_gate

                    raw_score = (
                        descent_weight * descent_alignment
                        + coherence_weight * memory_coherence
                        + improvement_weight * improvement_history
                        + recoverability_weight * recovery_gate * recovery_score
                        + stability_weight * stability_score
                        + stable_bonus
                        + matrix_bonus
                        - oscillation_penalty * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)
                        - conflict_penalty * torch.clamp(conflict_score, 0.0, 1.0)
                        - cost_penalty_weight * V3_COST_PENALTIES[candidate_name]
                        - specialist_penalty
                    )
                    score_columns.append(torch.where(valid, raw_score.float(), torch.full_like(raw_score.float(), -1e6)))
                    valid_columns.append(valid)
                    direction_columns.append(candidate_dirs)
                    selected_components[f"{candidate_name}__recovery"] = recovery_score.float()
                    selected_components[f"{candidate_name}__coherence"] = memory_coherence.float()
                    selected_components[f"{candidate_name}__conflict"] = conflict_score.float()
                    selected_components[f"{candidate_name}__oscillation"] = oscillation_score.float()
                    selected_components[f"{candidate_name}__consensus"] = consensus_strength.float()

                score_matrix = torch.stack(score_columns, dim=1)
                valid_matrix = torch.stack(valid_columns, dim=1)
                direction_tensor = torch.stack(direction_columns, dim=1)
                gradient_index = self._candidate_index["gradient"]
                fallback_index = gradient_index
                if not enabled_candidates["gradient"] and enabled_candidates["normalized_gradient"]:
                    fallback_index = self._candidate_index["normalized_gradient"]
                elif not enabled_candidates["gradient"] and not enabled_candidates["normalized_gradient"] and enabled_candidates["sign_direction"]:
                    fallback_index = self._candidate_index["sign_direction"]

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
                elif selection_mode == "top2_blend":
                    _, top_indices = torch.topk(score_matrix, k=min(2, score_matrix.shape[1]), dim=1)
                    top_mask = torch.zeros_like(score_matrix, dtype=torch.bool)
                    top_mask.scatter_(1, top_indices, True)
                    weights = _softmax_masked(score_matrix, top_mask & valid_matrix, selection_temperature)
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
                for candidate_name, candidate_index in self._candidate_index.items():
                    mask = selected_index_summary == candidate_index
                    if not bool(mask.any()):
                        continue
                    selected_recovery = torch.where(mask, selected_components.get(f"{candidate_name}__recovery", torch.zeros_like(selected_recovery)), selected_recovery)
                    selected_coherence = torch.where(mask, selected_components.get(f"{candidate_name}__coherence", torch.zeros_like(selected_coherence)), selected_coherence)
                    selected_conflict = torch.where(mask, selected_components.get(f"{candidate_name}__conflict", torch.zeros_like(selected_conflict)), selected_conflict)
                    selected_oscillation = torch.where(mask, selected_components.get(f"{candidate_name}__oscillation", torch.zeros_like(selected_oscillation)), selected_oscillation)
                    selected_consensus = torch.where(mask, selected_components.get(f"{candidate_name}__consensus", torch.zeros_like(selected_consensus)), selected_consensus)

                trust_ema = trust_decay * state["trust_ema"] + (1.0 - trust_decay) * selected_scores.float()
                trust_scale = (
                    1.0
                    + 0.52 * (selected_scores.float() - 0.45)
                    + 0.16 * (state["recoverability_ema"] - 0.5)
                    + 0.12 * selected_consensus * stable_gate
                    - 0.08 * selected_conflict * stress_gate
                )
                trust_scale = torch.clamp(trust_scale, min_scale, max_scale)
                trust_scale = torch.maximum(trust_scale, trust_ema.clamp(min_scale, max_scale))

                dimension_scale = float(max(1, grad_blocks.shape[1])) ** dimension_power
                if magnitude_mode == "rmsprop_like":
                    grad_sq_ema = state["grad_sq_ema"]
                    grad_sq_ema = torch.where(
                        grad_sq_ema > eps,
                        rmsprop_decay * grad_sq_ema + (1.0 - rmsprop_decay) * grad_blocks.detach().float().pow(2).mean(dim=1),
                        grad_blocks.detach().float().pow(2).mean(dim=1),
                    )
                    step_norm = lr * trust_scale * selected_scores.float().clamp_min(0.1) / (grad_sq_ema.sqrt() + eps)
                    state["grad_sq_ema"] = grad_sq_ema
                else:
                    support = ((_row_cosine(selected_dirs, descent_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0)
                    step_norm = lr * trust_scale * grad_norms.float() * (0.45 + 0.55 * support) / max(dimension_scale, eps)

                param_norms = param_blocks.detach().float().norm(dim=1)
                if max_update_ratio > 0.0:
                    step_caps = max_update_ratio * (param_norms + eps)
                    step_norm = torch.minimum(step_norm, step_caps)

                update_blocks = selected_dirs * step_norm.unsqueeze(1).to(selected_dirs.dtype)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(self._restore_blocks(update_blocks, layout), alpha=1.0)

                trusted_next, _ = _normalize_rows(trusted_blocks * memory_decay + selected_dirs * (1.0 - memory_decay), eps)
                smoothed_next, _ = _normalize_rows(smoothed_blocks * smooth_decay + selected_dirs * (1.0 - smooth_decay), eps)
                trusted_blocks.copy_(trusted_next)
                smoothed_blocks.copy_(smoothed_next)
                prev_grad_blocks.copy_(descent_blocks)
                prev_update_blocks.copy_(update_blocks)
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
                stress_values.extend(float(value) for value in stress_gate.detach().cpu())
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
                "stress_gate": average(stress_values),
                "block_step_norm": average(block_step_norm_values),
                "fallback_rate": average(fallback_values),
                "runtime_overhead_ms": runtime_ms,
                "active_params": float(active_params),
                "block_count": float(total_blocks),
                "divergence_flag": divergence_flag,
            }
        )
        return loss_tensor
