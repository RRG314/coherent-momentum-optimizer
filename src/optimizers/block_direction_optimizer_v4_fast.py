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
from .block_direction_optimizer_v3 import BlockDirectionOptimizerV3
from .optimizer_utils import average, clamp_scalar


V4_CANDIDATE_NAMES = (
    "gradient",
    "stable_consensus",
    "trusted_direction",
    "low_rank_matrix",
)

V4_COST_PENALTIES = {
    "gradient": 0.0,
    "stable_consensus": 0.01,
    "trusted_direction": 0.01,
    "low_rank_matrix": 0.04,
}


class BlockDirectionOptimizerV4Fast(BlockDirectionOptimizerV3):
    """Fast block-direction optimizer with a reduced, structured candidate set.

    V4 keeps the novel part of the block branch: per-block candidate-direction
    selection with memory and structured matrix consensus. The main change is
    aggressive simplification of the hot path so the branch becomes practical:

    - only four candidate directions in the default path
    - no per-candidate recoverability computation by default
    - tensor blocks for vectors and small matrices, row blocks only when the
      matrix is large enough to justify the extra structure
    - a lightweight block-energy normalization for step magnitude
    """

    candidate_names = V4_CANDIDATE_NAMES

    def __init__(
        self,
        params,
        *,
        use_typed_profiles: bool = True,
        small_matrix_cutoff: int = 1024,
        energy_decay: float = 0.97,
        energy_power: float = 0.5,
        use_recoverability_gate: bool = False,
        recoverability_interval: int = 8,
        use_conv_structure_support: bool = True,
        conv_lr_scale: float = 0.88,
        conv_coherence_weight: float = 0.22,
        conv_stability_weight: float = 0.20,
        conv_consensus_memory_mix: float = 0.44,
        conv_consensus_matrix_mix: float = 0.12,
        conv_stable_consensus_bonus: float = 0.08,
        conv_matrix_consensus_bonus: float = 0.04,
        conv_small_matrix_cutoff: int = 768,
        conv_energy_power: float = 0.58,
        conv_consensus_bonus: float = 0.0,
        conv_memory_bonus: float = 0.0,
        conv_fallback_relaxation: float = 0.0,
        conv_max_update_ratio: float = 0.10,
        conv_energy_power_bonus: float = 0.15,
        conv_step_floor: float = 0.82,
        conv_support_channel_weight: float = 0.40,
        conv_support_spatial_weight: float = 0.35,
        conv_support_bank_weight: float = 0.25,
        conv_support_power: float = 1.0,
        **kwargs: Any,
    ) -> None:
        param_list = list(params)
        if use_typed_profiles and param_list and not isinstance(param_list[0], dict):
            dense_params = [param for param in param_list if getattr(param, "ndim", 0) < 3]
            conv_params = [param for param in param_list if getattr(param, "ndim", 0) >= 3]
            grouped_params: list[dict[str, Any]] = []
            if dense_params:
                grouped_params.append({"params": dense_params, "block_profile": "dense"})
            if conv_params:
                grouped_params.append({"params": conv_params, "block_profile": "conv"})
            params = grouped_params
        else:
            params = param_list

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
        kwargs.setdefault("projection_strength", 0.0)
        kwargs.setdefault("orthogonal_strength", 0.0)
        kwargs.setdefault("topk_fraction", 0.0)
        kwargs.setdefault("dimension_power", 0.06)
        kwargs.setdefault("max_update_ratio", 0.16)
        kwargs.setdefault("min_scale", 0.82)
        kwargs.setdefault("max_scale", 1.45)
        kwargs.setdefault("fallback_threshold", 0.06)
        kwargs.setdefault("magnitude_mode", "energy_normalized")
        kwargs.setdefault("rmsprop_decay", 0.97)
        kwargs.setdefault("use_gradient_candidate", True)
        kwargs.setdefault("use_normalized_gradient_candidate", False)
        kwargs.setdefault("use_trusted_direction_candidate", True)
        kwargs.setdefault("use_smoothed_direction_candidate", True)
        kwargs.setdefault("use_projection_candidate", False)
        kwargs.setdefault("use_orthogonal_escape_candidate", False)
        kwargs.setdefault("use_sparse_topk_candidate", False)
        kwargs.setdefault("use_low_rank_candidate", True)
        kwargs.setdefault("use_sign_candidate", False)
        kwargs.setdefault("use_muon_like_candidate", False)
        kwargs.setdefault("use_stable_consensus_candidate", True)
        kwargs.setdefault("consensus_memory_mix", 0.38)
        kwargs.setdefault("consensus_matrix_mix", 0.14)
        kwargs.setdefault("stable_consensus_bonus", 0.10)
        kwargs.setdefault("matrix_consensus_bonus", 0.05)
        kwargs.setdefault("stress_gate_threshold", 0.32)
        kwargs.setdefault("stress_gate_power", 1.0)
        kwargs.setdefault("stress_candidate_penalty", 0.0)
        kwargs.setdefault("matrix_candidate_mode", "row_column_consensus")
        kwargs.setdefault("row_column_mix", 0.6)
        kwargs.setdefault("row_column_energy_weight", 0.55)
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group.setdefault("block_profile", "dense" if bool(group.get("block_profile", "dense") != "conv") else "conv")
            group["small_matrix_cutoff"] = int(small_matrix_cutoff)
            group["energy_decay"] = float(energy_decay)
            group["energy_power"] = float(energy_power)
            group["use_recoverability_gate"] = bool(use_recoverability_gate)
            group["recoverability_interval"] = int(recoverability_interval)
            group["use_typed_profiles"] = bool(use_typed_profiles)
            group["use_conv_structure_support"] = bool(use_conv_structure_support)
            group["conv_lr_scale"] = float(conv_lr_scale)
            group["conv_coherence_weight"] = float(conv_coherence_weight)
            group["conv_stability_weight"] = float(conv_stability_weight)
            group["conv_consensus_memory_mix"] = float(conv_consensus_memory_mix)
            group["conv_consensus_matrix_mix"] = float(conv_consensus_matrix_mix)
            group["conv_stable_consensus_bonus"] = float(conv_stable_consensus_bonus)
            group["conv_matrix_consensus_bonus"] = float(conv_matrix_consensus_bonus)
            group["conv_small_matrix_cutoff"] = int(conv_small_matrix_cutoff)
            group["conv_energy_power"] = float(conv_energy_power)
            group["conv_consensus_bonus"] = float(conv_consensus_bonus)
            group["conv_memory_bonus"] = float(conv_memory_bonus)
            group["conv_fallback_relaxation"] = float(conv_fallback_relaxation)
            group["conv_max_update_ratio"] = float(conv_max_update_ratio)
            group["conv_energy_power_bonus"] = float(conv_energy_power_bonus)
            group["conv_step_floor"] = float(conv_step_floor)
            group["conv_support_channel_weight"] = float(conv_support_channel_weight)
            group["conv_support_spatial_weight"] = float(conv_support_spatial_weight)
            group["conv_support_bank_weight"] = float(conv_support_bank_weight)
            group["conv_support_power"] = float(conv_support_power)
            if str(group.get("block_profile", "dense")) == "conv":
                group["lr"] = float(group["lr"]) * float(conv_lr_scale)
                group["coherence_weight"] = float(conv_coherence_weight)
                group["stability_weight"] = float(conv_stability_weight)
                group["consensus_memory_mix"] = float(conv_consensus_memory_mix)
                group["consensus_matrix_mix"] = float(conv_consensus_matrix_mix)
                group["stable_consensus_bonus"] = float(conv_stable_consensus_bonus)
                group["matrix_consensus_bonus"] = float(conv_matrix_consensus_bonus)
                group["small_matrix_cutoff"] = int(conv_small_matrix_cutoff)
                group["energy_power"] = float(conv_energy_power)
        self._initialize_physical_optimizer("BlockDirectionOptimizerV4Fast")

    @staticmethod
    def _resolve_block_strategy(param: torch.Tensor, strategy: str) -> str:
        if strategy == "smart_v4":
            if param.ndim <= 1:
                return "tensor"
            if param.ndim == 2 and param.numel() < 1024:
                return "tensor"
            if param.ndim >= 2:
                return "row"
            return "tensor"
        return BlockDirectionOptimizerV3._resolve_block_strategy(param, strategy)

    def _block_view_v4(self, tensor: torch.Tensor, strategy: str, small_matrix_cutoff: int) -> tuple[torch.Tensor, tuple[str, tuple[int, ...]]]:
        resolved = strategy
        if strategy == "smart_v4":
            if tensor.ndim <= 1:
                resolved = "tensor"
            elif tensor.ndim == 2 and tensor.numel() < int(small_matrix_cutoff):
                resolved = "tensor"
            elif tensor.ndim >= 2:
                resolved = "row"
            else:
                resolved = "tensor"
        return BlockDirectionOptimizerV2._block_view(tensor, resolved)

    def _init_state_v4(
        self,
        state: dict[str, Any],
        param: torch.Tensor,
        strategy: str,
        small_matrix_cutoff: int,
    ) -> tuple[tuple[str, tuple[int, ...]], int]:
        block_view, layout = self._block_view_v4(param, strategy, small_matrix_cutoff)
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

    def _conv_structure_support(
        self,
        grad: torch.Tensor,
        strategy: str,
        eps: float,
        *,
        small_matrix_cutoff: int,
        channel_weight: float,
        spatial_weight: float,
        bank_weight: float,
        support_power: float,
    ) -> torch.Tensor | None:
        if grad.ndim < 3:
            return None

        grad_view = -grad.detach().reshape(grad.shape[0], grad.shape[1], -1)
        filter_flat = grad_view.reshape(grad.shape[0], -1)
        base_dirs, base_norms = _normalize_rows(filter_flat, eps)
        if not bool((base_norms > eps).any()):
            return None

        channel_slices = grad_view.reshape(-1, grad_view.shape[-1])
        channel_dirs, channel_norms = _normalize_rows(channel_slices, eps)
        channel_dirs = channel_dirs.reshape(grad_view.shape[0], grad_view.shape[1], -1)
        channel_mask = (channel_norms.reshape(grad_view.shape[0], grad_view.shape[1]) > eps).float().unsqueeze(2)
        channel_count = channel_mask.sum(dim=1).clamp_min(1.0)
        channel_support = (channel_dirs * channel_mask).sum(dim=1) / channel_count
        channel_support = torch.clamp(channel_support.norm(dim=1), 0.0, 1.0)

        spatial_slices = grad_view.transpose(1, 2).contiguous().reshape(-1, grad_view.shape[1])
        spatial_dirs, spatial_norms = _normalize_rows(spatial_slices, eps)
        spatial_dirs = spatial_dirs.reshape(grad_view.shape[0], grad_view.shape[2], grad_view.shape[1])
        spatial_mask = (spatial_norms.reshape(grad_view.shape[0], grad_view.shape[2]) > eps).float().unsqueeze(2)
        spatial_count = spatial_mask.sum(dim=1).clamp_min(1.0)
        spatial_support = (spatial_dirs * spatial_mask).sum(dim=1) / spatial_count
        spatial_support = torch.clamp(spatial_support.norm(dim=1), 0.0, 1.0)

        bank_mean = base_dirs.mean(dim=0, keepdim=True)
        bank_mean_dirs, bank_mean_norms = _normalize_rows(bank_mean, eps)
        if bool((bank_mean_norms > eps).any()):
            bank_support = torch.clamp(_row_cosine(base_dirs, bank_mean_dirs.expand_as(base_dirs), eps), min=0.0, max=1.0)
        else:
            bank_support = torch.zeros_like(channel_support)

        total_weight = max(channel_weight + spatial_weight + bank_weight, eps)
        support = (
            channel_weight * channel_support
            + spatial_weight * spatial_support
            + bank_weight * bank_support
        ) / total_weight
        if support_power != 1.0:
            support = support.pow(support_power)

        support_tensor = support.unsqueeze(1).expand_as(filter_flat).reshape_as(grad)
        return self._block_view_v4(support_tensor, strategy, small_matrix_cutoff)[0].float().mean(dim=1)

    def _matrix_candidate_blocks(
        self,
        grad: torch.Tensor,
        strategy: str,
        eps: float,
        *,
        use_low_rank: bool,
        use_muon_like: bool,
        small_matrix_cutoff: int | None = None,
        matrix_candidate_mode: str | None = None,
        row_column_mix: float | None = None,
        row_column_energy_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if grad.ndim < 2 or (not use_low_rank and not use_muon_like):
            return {}

        candidates: dict[str, torch.Tensor] = {}
        grad_matrix = -grad.detach().reshape(grad.shape[0], -1)
        group = self.param_groups[0] if self.param_groups else {}
        matrix_candidate_mode = str(matrix_candidate_mode or group.get("matrix_candidate_mode", "row_column_consensus"))
        row_column_mix = clamp_scalar(float(row_column_mix if row_column_mix is not None else group.get("row_column_mix", 0.6)), 0.0, 1.0)
        row_column_energy_weight = clamp_scalar(
            float(row_column_energy_weight if row_column_energy_weight is not None else group.get("row_column_energy_weight", 0.55)),
            0.0,
            1.0,
        )
        resolved_cutoff = int(small_matrix_cutoff if small_matrix_cutoff is not None else group.get("small_matrix_cutoff", 1024))

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
                matrix_candidate = matrix_candidate.reshape_as(grad)
                candidates["low_rank_matrix"] = self._block_view_v4(matrix_candidate, strategy, resolved_cutoff)[0]

            if use_muon_like:
                u, _, vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
                orthogonal = (u @ vh).to(dtype=grad.dtype, device=grad.device).reshape_as(grad)
                candidates["muon_like_orthogonal"] = self._block_view_v4(
                    orthogonal,
                    strategy,
                    resolved_cutoff,
                )[0]
        except RuntimeError:
            return candidates
        return candidates

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
        conv_support_values: list[float] = []
        conv_bonus_values: list[float] = []
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
            use_conv_structure_support = bool(group.get("use_conv_structure_support", False))
            conv_consensus_bonus = float(group.get("conv_consensus_bonus", 0.0))
            conv_memory_bonus = float(group.get("conv_memory_bonus", 0.0))
            conv_fallback_relaxation = max(0.0, float(group.get("conv_fallback_relaxation", 0.0)))
            conv_max_update_ratio = max(0.0, float(group.get("conv_max_update_ratio", max_update_ratio)))
            conv_energy_power_bonus = clamp_scalar(float(group.get("conv_energy_power_bonus", 0.0)), 0.0, 1.0)
            conv_step_floor = clamp_scalar(float(group.get("conv_step_floor", 1.0)), 0.0, 1.0)
            conv_cap_floor = clamp_scalar(float(group.get("conv_cap_floor", 1.0)), 0.0, 1.0)
            conv_support_channel_weight = max(0.0, float(group.get("conv_support_channel_weight", 0.40)))
            conv_support_spatial_weight = max(0.0, float(group.get("conv_support_spatial_weight", 0.35)))
            conv_support_bank_weight = max(0.0, float(group.get("conv_support_bank_weight", 0.25)))
            conv_support_power = clamp_scalar(float(group.get("conv_support_power", 1.0)), 0.25, 4.0)
            use_stable_consensus_candidate = bool(group.get("use_stable_consensus_candidate", True))
            consensus_memory_mix = clamp_scalar(float(group.get("consensus_memory_mix", 0.38)), 0.0, 1.5)
            consensus_matrix_mix = clamp_scalar(float(group.get("consensus_matrix_mix", 0.14)), 0.0, 1.0)
            stable_consensus_bonus = float(group.get("stable_consensus_bonus", 0.10))
            matrix_consensus_bonus = float(group.get("matrix_consensus_bonus", 0.05))

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

                conv_support = None
                if use_conv_structure_support and grad.ndim >= 3:
                    conv_support = self._conv_structure_support(
                        grad,
                        block_strategy,
                        eps,
                        small_matrix_cutoff=small_matrix_cutoff,
                        channel_weight=conv_support_channel_weight,
                        spatial_weight=conv_support_spatial_weight,
                        bank_weight=conv_support_bank_weight,
                        support_power=conv_support_power,
                    )

                matrix_dirs = None
                matrix_candidates = self._matrix_candidate_blocks(
                    grad,
                    "tensor" if block_strategy == "smart_v4" and grad.ndim == 2 and grad.numel() < small_matrix_cutoff else block_strategy,
                    eps,
                    use_low_rank=enabled_candidates["low_rank_matrix"],
                    use_muon_like=False,
                    small_matrix_cutoff=small_matrix_cutoff,
                    matrix_candidate_mode=str(group.get("matrix_candidate_mode", "row_column_consensus")),
                    row_column_mix=float(group.get("row_column_mix", 0.6)),
                    row_column_energy_weight=float(group.get("row_column_energy_weight", 0.55)),
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
                    memory_coherence = torch.maximum(trusted_alignment, smoothed_alignment)
                    consensus_strength = (
                        0.50 * descent_alignment
                        + 0.30 * memory_coherence
                        + 0.20 * matrix_alignment
                    ).clamp(0.0, 1.0)
                    improvement_history = ((quality_matrix[:, self._candidate_index[candidate_name]].float() + 1.0) * 0.5).clamp(0.0, 1.0)
                    oscillation_score = (
                        0.5 * (1.0 - _row_cosine(candidate_dirs, prev_grad_dirs, eps))
                        + 0.5 * _row_sign_flip_ratio(candidate_dirs, prev_update_dirs)
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

                    candidate_conv_bonus = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                    if conv_support is not None:
                        if candidate_name == "stable_consensus":
                            candidate_conv_bonus = conv_consensus_bonus * conv_support * consensus_strength
                        elif candidate_name == "trusted_direction":
                            candidate_conv_bonus = conv_memory_bonus * conv_support * memory_coherence
                        elif candidate_name == "low_rank_matrix":
                            candidate_conv_bonus = 0.5 * conv_consensus_bonus * conv_support * matrix_alignment

                    raw_score = (
                        descent_weight * descent_alignment
                        + coherence_weight * memory_coherence
                        + improvement_weight * improvement_history
                        + recoverability_weight * recovery_gate * recovery_score
                        + stability_weight * stability_score
                        + stable_consensus_bonus * consensus_strength
                        + matrix_consensus_bonus * matrix_alignment
                        + candidate_conv_bonus
                        - oscillation_penalty * torch.clamp(oscillation_score / 1.5, 0.0, 1.0)
                        - conflict_penalty * torch.clamp(conflict_score, 0.0, 1.0)
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
                    selected_components[f"{candidate_name}__conv_support"] = conv_support.float() if conv_support is not None else torch.zeros(block_count, dtype=torch.float32, device=param.device)
                    selected_components[f"{candidate_name}__conv_bonus"] = candidate_conv_bonus.float()

                score_matrix = torch.stack(score_columns, dim=1)
                valid_matrix = torch.stack(valid_columns, dim=1)
                direction_tensor = torch.stack(direction_columns, dim=1)
                fallback_index = self._candidate_index["gradient"]

                effective_fallback_threshold = torch.full((block_count,), fallback_threshold, dtype=torch.float32, device=param.device)
                if conv_support is not None and conv_fallback_relaxation > 0.0:
                    effective_fallback_threshold = torch.clamp(
                        effective_fallback_threshold - conv_fallback_relaxation * conv_support,
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
                selected_conv_support = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                selected_conv_bonus = torch.zeros(block_count, dtype=torch.float32, device=param.device)
                for candidate_name, candidate_index in self._candidate_index.items():
                    mask = selected_index_summary == candidate_index
                    if not bool(mask.any()):
                        continue
                    selected_recovery = torch.where(mask, selected_components.get(f"{candidate_name}__recovery", torch.zeros_like(selected_recovery)), selected_recovery)
                    selected_coherence = torch.where(mask, selected_components.get(f"{candidate_name}__coherence", torch.zeros_like(selected_coherence)), selected_coherence)
                    selected_conflict = torch.where(mask, selected_components.get(f"{candidate_name}__conflict", torch.zeros_like(selected_conflict)), selected_conflict)
                    selected_oscillation = torch.where(mask, selected_components.get(f"{candidate_name}__oscillation", torch.zeros_like(selected_oscillation)), selected_oscillation)
                    selected_consensus = torch.where(mask, selected_components.get(f"{candidate_name}__consensus", torch.zeros_like(selected_consensus)), selected_consensus)
                    selected_conv_support = torch.where(mask, selected_components.get(f"{candidate_name}__conv_support", torch.zeros_like(selected_conv_support)), selected_conv_support)
                    selected_conv_bonus = torch.where(mask, selected_components.get(f"{candidate_name}__conv_bonus", torch.zeros_like(selected_conv_bonus)), selected_conv_bonus)

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
                    conv_step_multiplier = conv_step_floor + (1.0 - conv_step_floor) * selected_conv_support
                    effective_energy_power = effective_energy_power + conv_energy_power_bonus * (1.0 - selected_conv_support)
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
                    if conv_support is not None:
                        step_caps = torch.minimum(step_caps, conv_max_update_ratio * (param_norms + eps))
                    step_norm = torch.minimum(step_norm, step_caps)

                update_blocks = selected_dirs * step_norm.unsqueeze(1).to(selected_dirs.dtype)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(BlockDirectionOptimizerV2._restore_blocks(update_blocks, layout), alpha=1.0)

                trusted_next, _ = _normalize_rows(trusted_blocks * memory_decay + selected_dirs * (1.0 - memory_decay), eps)
                smoothed_next, _ = _normalize_rows(smoothed_blocks * smooth_decay + selected_dirs * (1.0 - smooth_decay), eps)
                state["trusted_direction"] = BlockDirectionOptimizerV2._restore_blocks(trusted_next, layout)
                state["smoothed_direction"] = BlockDirectionOptimizerV2._restore_blocks(smoothed_next, layout)
                state["prev_grad"] = BlockDirectionOptimizerV2._restore_blocks(descent_blocks, layout)
                state["prev_update"] = BlockDirectionOptimizerV2._restore_blocks(update_blocks, layout)
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
                conv_support_values.extend(float(value) for value in selected_conv_support.detach().cpu())
                conv_bonus_values.extend(float(value) for value in selected_conv_bonus.detach().cpu())
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
                "filter_support": average(conv_support_values),
                "conv_trust_bonus": average(conv_bonus_values),
                "conv_step_multiplier": average(conv_step_values),
                "runtime_overhead_ms": runtime_ms,
                "active_params": float(active_params),
                "block_count": float(total_blocks),
                "divergence_flag": divergence_flag,
            }
        )
        return loss_tensor
