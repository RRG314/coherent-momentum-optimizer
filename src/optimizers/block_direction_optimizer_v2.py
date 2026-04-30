from __future__ import annotations

import math
import time
from typing import Any

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import average, bounded_scale, clamp_scalar, safe_float


DEFAULT_CANDIDATE_NAMES = (
    "gradient",
    "normalized_gradient",
    "trusted_direction",
    "smoothed_direction",
    "projection_corrected",
    "orthogonal_escape",
    "sparse_topk",
    "low_rank_matrix",
    "sign_direction",
    "muon_like_orthogonal",
)

_COST_PENALTIES = {
    "gradient": 0.0,
    "normalized_gradient": 0.0,
    "trusted_direction": 0.01,
    "smoothed_direction": 0.01,
    "projection_corrected": 0.03,
    "orthogonal_escape": 0.05,
    "sparse_topk": 0.04,
    "low_rank_matrix": 0.10,
    "sign_direction": 0.01,
    "muon_like_orthogonal": 0.12,
}


def _normalize_rows(blocks: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    norms = blocks.detach().float().norm(dim=1, keepdim=True)
    valid = torch.isfinite(norms) & (norms > eps)
    safe_norms = torch.where(valid, norms + eps, torch.ones_like(norms))
    normalized = blocks / safe_norms
    normalized = torch.where(valid, normalized, torch.zeros_like(blocks))
    return normalized, norms.squeeze(1)


def _row_cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    a_float = a.detach().float()
    b_float = b.detach().float()
    denom = a_float.norm(dim=1) * b_float.norm(dim=1)
    safe_denom = torch.where(denom > eps, denom + eps, torch.ones_like(denom))
    dots = (a_float * b_float).sum(dim=1)
    cosine = dots / safe_denom
    cosine = torch.where(denom > eps, cosine, torch.zeros_like(cosine))
    return cosine.clamp(-1.0, 1.0)


def _row_sign_flip_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=a.device)
    a_sign = torch.sign(a.detach())
    b_sign = torch.sign(b.detach())
    flips = (a_sign != b_sign).float()
    return flips.mean(dim=1)


def _softmax_masked(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-4)
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(masked_scores / safe_temperature, dim=1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weight_sums = weights.sum(dim=1, keepdim=True)
    safe_weight_sums = torch.where(weight_sums > 0.0, weight_sums, torch.ones_like(weight_sums))
    weights = torch.where(weight_sums > 0.0, weights / safe_weight_sums, torch.zeros_like(weights))
    return weights


class BlockDirectionOptimizerV2(PhysicalOptimizerBase, torch.optim.Optimizer):
    """Blockwise direction-selection optimizer with trust-scored candidate updates.

    The core rule is not Adam-like: for each parameter block, the optimizer builds a
    set of candidate update directions, scores them with a trust function, and then
    selects or blends directions before applying a bounded block step. Recoverability
    is used as a gate on candidate trust instead of generating the update itself.
    """

    candidate_names = DEFAULT_CANDIDATE_NAMES

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        block_strategy: str = "smart",
        selection_mode: str = "top2_blend",
        selection_temperature: float = 0.35,
        memory_decay: float = 0.90,
        smooth_decay: float = 0.94,
        grad_decay: float = 0.96,
        trust_decay: float = 0.88,
        descent_weight: float = 0.42,
        coherence_weight: float = 0.18,
        improvement_weight: float = 0.10,
        recoverability_weight: float = 0.12,
        stability_weight: float = 0.10,
        oscillation_penalty: float = 0.14,
        conflict_penalty: float = 0.12,
        cost_penalty_weight: float = 0.04,
        recovery_threshold: float = 0.48,
        recoverability_keep_ratio: float = 0.55,
        recoverability_noise_scale: float = 0.04,
        recoverability_drop_fraction: float = 0.15,
        recoverability_samples: int = 2,
        topk_fraction: float = 0.20,
        projection_strength: float = 0.45,
        orthogonal_strength: float = 0.30,
        dimension_power: float = 0.10,
        max_update_ratio: float = 0.16,
        min_scale: float = 0.55,
        max_scale: float = 1.85,
        fallback_threshold: float = 0.08,
        magnitude_mode: str = "block_norm",
        rmsprop_decay: float = 0.97,
        use_gradient_candidate: bool = True,
        use_normalized_gradient_candidate: bool = True,
        use_trusted_direction_candidate: bool = True,
        use_smoothed_direction_candidate: bool = True,
        use_projection_candidate: bool = True,
        use_orthogonal_escape_candidate: bool = True,
        use_sparse_topk_candidate: bool = True,
        use_low_rank_candidate: bool = True,
        use_sign_candidate: bool = True,
        use_muon_like_candidate: bool = False,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            block_strategy=block_strategy,
            selection_mode=selection_mode,
            selection_temperature=selection_temperature,
            memory_decay=memory_decay,
            smooth_decay=smooth_decay,
            grad_decay=grad_decay,
            trust_decay=trust_decay,
            descent_weight=descent_weight,
            coherence_weight=coherence_weight,
            improvement_weight=improvement_weight,
            recoverability_weight=recoverability_weight,
            stability_weight=stability_weight,
            oscillation_penalty=oscillation_penalty,
            conflict_penalty=conflict_penalty,
            cost_penalty_weight=cost_penalty_weight,
            recovery_threshold=recovery_threshold,
            recoverability_keep_ratio=recoverability_keep_ratio,
            recoverability_noise_scale=recoverability_noise_scale,
            recoverability_drop_fraction=recoverability_drop_fraction,
            recoverability_samples=recoverability_samples,
            topk_fraction=topk_fraction,
            projection_strength=projection_strength,
            orthogonal_strength=orthogonal_strength,
            dimension_power=dimension_power,
            max_update_ratio=max_update_ratio,
            min_scale=min_scale,
            max_scale=max_scale,
            fallback_threshold=fallback_threshold,
            magnitude_mode=magnitude_mode,
            rmsprop_decay=rmsprop_decay,
            use_gradient_candidate=use_gradient_candidate,
            use_normalized_gradient_candidate=use_normalized_gradient_candidate,
            use_trusted_direction_candidate=use_trusted_direction_candidate,
            use_smoothed_direction_candidate=use_smoothed_direction_candidate,
            use_projection_candidate=use_projection_candidate,
            use_orthogonal_escape_candidate=use_orthogonal_escape_candidate,
            use_sparse_topk_candidate=use_sparse_topk_candidate,
            use_low_rank_candidate=use_low_rank_candidate,
            use_sign_candidate=use_sign_candidate,
            use_muon_like_candidate=use_muon_like_candidate,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._candidate_index = {name: index for index, name in enumerate(self.candidate_names)}
        self._initialize_physical_optimizer("BlockDirectionOptimizerV2")

    @staticmethod
    def _resolve_block_strategy(param: torch.Tensor, strategy: str) -> str:
        if strategy == "smart":
            if param.ndim == 2:
                return "row"
            if param.ndim <= 1:
                return "tensor"
            return "row"
        if strategy == "layer":
            return "tensor"
        if strategy == "matrix":
            return "tensor" if param.ndim == 2 else "tensor"
        return strategy

    @classmethod
    def _block_view(cls, tensor: torch.Tensor, strategy: str) -> tuple[torch.Tensor, tuple[str, tuple[int, ...]]]:
        resolved = cls._resolve_block_strategy(tensor, strategy)
        shape = tuple(tensor.shape)
        if resolved == "scalar":
            return tensor.reshape(-1, 1), (resolved, shape)
        if resolved == "column" and tensor.ndim == 2:
            return tensor.transpose(0, 1).contiguous().reshape(tensor.shape[1], tensor.shape[0]), (resolved, shape)
        if resolved == "row" and tensor.ndim >= 2:
            return tensor.reshape(tensor.shape[0], -1), (resolved, shape)
        return tensor.reshape(1, -1), (resolved, shape)

    @staticmethod
    def _restore_blocks(blocks: torch.Tensor, layout: tuple[str, tuple[int, ...]]) -> torch.Tensor:
        strategy, shape = layout
        if strategy == "column" and len(shape) == 2:
            return blocks.reshape(shape[1], shape[0]).transpose(0, 1).reshape(shape)
        return blocks.reshape(shape)

    def _init_state(self, state: dict[str, Any], param: torch.Tensor, strategy: str) -> tuple[tuple[str, tuple[int, ...]], int]:
        block_view, layout = self._block_view(param, strategy)
        block_shape = tuple(block_view.shape)
        if len(state) != 0 and tuple(state.get("trusted_direction", torch.empty(0)).shape) == block_shape:
            return layout, block_shape[0]
        block_count = block_shape[0]
        device = param.device
        dtype = param.dtype
        state.clear()
        state["step"] = 0
        state["trusted_direction"] = torch.zeros(block_shape, dtype=dtype, device=device)
        state["smoothed_direction"] = torch.zeros(block_shape, dtype=dtype, device=device)
        state["prev_grad"] = torch.zeros(block_shape, dtype=dtype, device=device)
        state["prev_update"] = torch.zeros(block_shape, dtype=dtype, device=device)
        state["grad_norm_ema"] = torch.zeros(block_count, dtype=torch.float32, device=device)
        state["trust_ema"] = torch.ones(block_count, dtype=torch.float32, device=device)
        state["recoverability_ema"] = torch.ones(block_count, dtype=torch.float32, device=device)
        state["candidate_quality"] = torch.zeros(block_count, len(self.candidate_names), dtype=torch.float32, device=device)
        state["last_selected_index"] = torch.full((block_count,), self._candidate_index["gradient"], dtype=torch.long, device=device)
        state["last_selected_score"] = torch.zeros(block_count, dtype=torch.float32, device=device)
        state["grad_sq_ema"] = torch.zeros(block_count, dtype=torch.float32, device=device)
        return layout, block_count

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
        try:
            if use_low_rank:
                u, s, vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
                rank_one = (u[:, :1] * s[:1]) @ vh[:1, :]
                candidates["low_rank_matrix"] = self._block_view(rank_one.to(dtype=grad.dtype, device=grad.device), strategy)[0]
            if use_muon_like:
                u, _, vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
                orthogonal = u @ vh
                candidates["muon_like_orthogonal"] = self._block_view(orthogonal.to(dtype=grad.dtype, device=grad.device), strategy)[0]
        except RuntimeError:
            return {}
        return candidates

    def _topk_candidate(self, blocks: torch.Tensor, fraction: float, eps: float) -> torch.Tensor:
        block_width = blocks.shape[1]
        if block_width <= 1:
            return blocks
        keep = max(1, min(block_width, int(math.ceil(block_width * clamp_scalar(fraction, 0.05, 1.0)))))
        magnitudes = blocks.detach().abs()
        topk_indices = magnitudes.topk(keep, dim=1).indices
        mask = torch.zeros_like(blocks, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        sparse = torch.where(mask, blocks, torch.zeros_like(blocks))
        sparse_norm = sparse.detach().float().norm(dim=1)
        fallback = sparse_norm <= eps
        if bool(fallback.any()):
            sparse = torch.where(fallback.unsqueeze(1), blocks, sparse)
        return sparse

    def _recoverability_score(
        self,
        candidate_blocks: torch.Tensor,
        candidate_dirs: torch.Tensor,
        *,
        keep_ratio: float,
        noise_scale: float,
        drop_fraction: float,
        samples: int,
        eps: float,
    ) -> torch.Tensor:
        block_count, block_width = candidate_blocks.shape
        if samples <= 0 or block_width <= 1:
            return torch.ones(block_count, dtype=torch.float32, device=candidate_blocks.device)
        keep_ratio = clamp_scalar(float(keep_ratio), 0.2, 0.9)
        drop_fraction = clamp_scalar(float(drop_fraction), 0.0, 0.5)
        scores: list[torch.Tensor] = []
        rms = candidate_blocks.detach().float().pow(2).mean(dim=1, keepdim=True).sqrt()
        for sample_index in range(samples):
            if sample_index % 3 == 0:
                mask = (torch.rand_like(candidate_blocks) < keep_ratio).to(candidate_blocks.dtype)
                perturbed = candidate_blocks * mask / max(keep_ratio, eps)
            elif sample_index % 3 == 1:
                perturbed = candidate_blocks + torch.randn_like(candidate_blocks) * (noise_scale * rms.to(candidate_blocks.dtype))
            else:
                drop = max(1, int(math.ceil(block_width * max(drop_fraction, 1.0 / max(2, block_width)))))
                magnitudes = candidate_blocks.detach().abs()
                drop_indices = magnitudes.topk(drop, dim=1).indices
                mask = torch.ones_like(candidate_blocks, dtype=torch.bool)
                mask.scatter_(1, drop_indices, False)
                perturbed = torch.where(mask, candidate_blocks, torch.zeros_like(candidate_blocks))
            perturbed_dirs, _ = _normalize_rows(perturbed, eps)
            scores.append((_row_cosine(perturbed_dirs, candidate_dirs, eps) + 1.0) * 0.5)
        return torch.stack(scores, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def _update_quality_memory(self, state: dict[str, Any], reward: float) -> None:
        qualities = state["candidate_quality"]
        qualities.mul_(0.995)
        reward_tensor = torch.full((qualities.shape[0],), float(max(-1.0, min(1.0, reward))), dtype=qualities.dtype, device=qualities.device)
        selected = state["last_selected_index"]
        row_index = torch.arange(selected.shape[0], device=selected.device)
        old_values = qualities[row_index, selected]
        qualities[row_index, selected] = 0.92 * old_values + 0.08 * reward_tensor

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

            enabled_candidates = {
                "gradient": bool(group["use_gradient_candidate"]),
                "normalized_gradient": bool(group["use_normalized_gradient_candidate"]),
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

                score_columns: list[torch.Tensor] = []
                valid_columns: list[torch.Tensor] = []
                direction_columns: list[torch.Tensor] = []
                selected_components: dict[str, torch.Tensor] = {}
                quality_matrix = state["candidate_quality"]
                grad_norm_ema = state["grad_norm_ema"]
                grad_norm_ema = torch.where(grad_norm_ema > eps, grad_decay * grad_norm_ema + (1.0 - grad_decay) * grad_norms.float(), grad_norms.float())

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
                    memory_coherence = torch.where(
                        trusted_exists,
                        ((_row_cosine(candidate_dirs, trusted_dirs, eps) + 1.0) * 0.5).clamp(0.0, 1.0),
                        torch.zeros_like(descent_alignment),
                    )
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
                    raw_score = (
                        descent_weight * descent_alignment
                        + coherence_weight * memory_coherence
                        + improvement_weight * improvement_history
                        + recoverability_weight * recovery_gate * recovery_score
                        + stability_weight * stability_score
                        - oscillation_penalty * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)
                        - conflict_penalty * torch.clamp(conflict_score, 0.0, 1.0)
                        - cost_penalty_weight * _COST_PENALTIES[candidate_name]
                    )
                    score_columns.append(torch.where(valid, raw_score.float(), torch.full_like(raw_score.float(), -1e6)))
                    valid_columns.append(valid)
                    direction_columns.append(candidate_dirs)
                    selected_components[f"{candidate_name}__recovery"] = recovery_score.float()
                    selected_components[f"{candidate_name}__coherence"] = memory_coherence.float()
                    selected_components[f"{candidate_name}__conflict"] = conflict_score.float()
                    selected_components[f"{candidate_name}__oscillation"] = oscillation_score.float()

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
                best_indices = torch.where(
                    fallback_mask,
                    torch.full_like(best_indices, fallback_index),
                    best_indices,
                )

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
                for candidate_name, candidate_index in self._candidate_index.items():
                    mask = selected_index_summary == candidate_index
                    if not bool(mask.any()):
                        continue
                    selected_recovery = torch.where(mask, selected_components.get(f"{candidate_name}__recovery", torch.zeros_like(selected_recovery)), selected_recovery)
                    selected_coherence = torch.where(mask, selected_components.get(f"{candidate_name}__coherence", torch.zeros_like(selected_coherence)), selected_coherence)
                    selected_conflict = torch.where(mask, selected_components.get(f"{candidate_name}__conflict", torch.zeros_like(selected_conflict)), selected_conflict)
                    selected_oscillation = torch.where(mask, selected_components.get(f"{candidate_name}__oscillation", torch.zeros_like(selected_oscillation)), selected_oscillation)

                trust_ema = trust_decay * state["trust_ema"] + (1.0 - trust_decay) * selected_scores.float()
                trust_scale = (
                    1.0
                    + 0.55 * (selected_scores.float() - 0.45)
                    + 0.20 * (state["recoverability_ema"] - 0.5)
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
                    step_norm = lr * trust_scale * grad_norms.float() * (0.4 + 0.6 * support) / max(dimension_scale, eps)

                param_norms = param_blocks.detach().float().norm(dim=1)
                if max_update_ratio > 0.0:
                    step_caps = max_update_ratio * (param_norms + eps)
                    step_norm = torch.minimum(step_norm, step_caps)

                update_blocks = selected_dirs * step_norm.unsqueeze(1).to(selected_dirs.dtype)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(self._restore_blocks(update_blocks, layout), alpha=1.0)

                trusted_next, _ = _normalize_rows(
                    trusted_blocks * memory_decay + selected_dirs * (1.0 - memory_decay),
                    eps,
                )
                smoothed_next, _ = _normalize_rows(
                    smoothed_blocks * smooth_decay + selected_dirs * (1.0 - smooth_decay),
                    eps,
                )
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
                "block_step_norm": average(block_step_norm_values),
                "fallback_rate": average(fallback_values),
                "runtime_overhead_ms": runtime_ms,
                "active_params": float(active_params),
                "block_count": float(total_blocks),
                "divergence_flag": divergence_flag,
            }
        )
        return loss_tensor
