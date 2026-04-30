from __future__ import annotations

from typing import Any

import torch

from .block_direction_optimizer_v42 import BlockDirectionOptimizerV42
from .optimizer_utils import clamp_scalar


class BlockDirectionOptimizerV43(BlockDirectionOptimizerV42):
    """V4.3 keeps V4Fast's dense defaults and localizes structure trust to conv blocks.

    The V4.2 branch improved CNN behavior, but some of that gain came bundled with
    broader default changes that softened dense-task performance. V4.3 keeps the
    conv-safe step logic and conv-structure trust, but restores the global dense
    path to the stronger V4Fast regime:

    - dense/vector layers stay on V4Fast-like defaults
    - only convolutional tensors receive structure-aware trust bonuses
    - conv structure support is thresholded so weak/noisy support does not
      perturb the default direction selector
    """

    def __init__(
        self,
        params,
        *,
        conv_support_threshold: float = 0.0,
        conv_support_power: float = 1.0,
        conv_consensus_bonus: float = 0.0,
        conv_memory_bonus: float = 0.0,
        conv_fallback_relaxation: float = 0.0,
        conv_max_update_ratio: float = 0.10,
        conv_energy_power_bonus: float = 0.18,
        conv_step_floor: float = 0.80,
        **kwargs: Any,
    ) -> None:
        param_list = list(params)
        if param_list and not isinstance(param_list[0], dict):
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
        super().__init__(
            params,
            conv_consensus_bonus=conv_consensus_bonus,
            conv_memory_bonus=conv_memory_bonus,
            conv_fallback_relaxation=conv_fallback_relaxation,
            conv_max_update_ratio=conv_max_update_ratio,
            conv_energy_power_bonus=conv_energy_power_bonus,
            conv_step_floor=conv_step_floor,
            **kwargs,
        )
        for group in self.param_groups:
            group["conv_support_threshold"] = float(conv_support_threshold)
            group["conv_support_power"] = float(conv_support_power)
            profile = str(group.get("block_profile", "dense"))
            if profile == "conv":
                group["coherence_weight"] = max(float(group["coherence_weight"]), 0.22)
                group["stability_weight"] = max(float(group["stability_weight"]), 0.20)
                group["consensus_memory_mix"] = max(float(group["consensus_memory_mix"]), 0.44)
                group["consensus_matrix_mix"] = min(float(group["consensus_matrix_mix"]), 0.12)
                group["stable_consensus_bonus"] = min(float(group["stable_consensus_bonus"]), 0.08)
                group["matrix_consensus_bonus"] = min(float(group["matrix_consensus_bonus"]), 0.04)
                group["energy_power"] = max(float(group["energy_power"]), 0.58)
        self._initialize_physical_optimizer("BlockDirectionOptimizerV4.3")

    def _conv_structure_support(
        self,
        grad: torch.Tensor,
        strategy: str,
        eps: float,
        *,
        small_matrix_cutoff: int,
    ) -> torch.Tensor | None:
        support = super()._conv_structure_support(
            grad,
            strategy,
            eps,
            small_matrix_cutoff=small_matrix_cutoff,
        )
        if support is None:
            return None
        group = self.param_groups[0] if self.param_groups else {}
        threshold = clamp_scalar(float(group.get("conv_support_threshold", 0.12)), 0.0, 0.95)
        power = clamp_scalar(float(group.get("conv_support_power", 1.5)), 0.25, 4.0)
        gated = torch.clamp((support - threshold) / max(1.0 - threshold, eps), 0.0, 1.0)
        if power != 1.0:
            gated = gated.pow(power)
        return gated
