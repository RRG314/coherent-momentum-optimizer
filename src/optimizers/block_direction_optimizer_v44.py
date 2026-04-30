from __future__ import annotations

from typing import Any

from .block_direction_optimizer_v42 import BlockDirectionOptimizerV42


class BlockDirectionOptimizerV44(BlockDirectionOptimizerV42):
    """Typed-profile block optimizer: V4Fast dense core + V4.2 conv safety.

    V4.4 is the evidence-backed synthesis branch:

    - dense/vector tensors keep the stronger V4Fast-style defaults
    - convolutional tensors use a separate conv profile
    - conv structure support is still measured for diagnostics and conv-safe
      scaling, but it does not add default trust bonuses
    - the core update is still blockwise candidate-direction selection rather
      than Adam-style moment transformation
    """

    def __init__(
        self,
        params,
        *,
        conv_lr_scale: float = 0.9,
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
            profile = str(group.get("block_profile", "dense"))
            if profile == "conv":
                group["lr"] = float(group["lr"]) * float(conv_lr_scale)
                group["coherence_weight"] = float(conv_coherence_weight)
                group["stability_weight"] = float(conv_stability_weight)
                group["consensus_memory_mix"] = float(conv_consensus_memory_mix)
                group["consensus_matrix_mix"] = float(conv_consensus_matrix_mix)
                group["stable_consensus_bonus"] = float(conv_stable_consensus_bonus)
                group["matrix_consensus_bonus"] = float(conv_matrix_consensus_bonus)
                group["small_matrix_cutoff"] = int(conv_small_matrix_cutoff)
                group["energy_power"] = float(conv_energy_power)
                group["conv_lr_scale"] = float(conv_lr_scale)
            else:
                group["conv_lr_scale"] = 1.0

        self._initialize_physical_optimizer("BlockDirectionOptimizerV4.4")
