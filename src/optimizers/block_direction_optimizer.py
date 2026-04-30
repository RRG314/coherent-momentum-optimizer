from __future__ import annotations

import math
from typing import Any

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import average, clamp_scalar, cosine_similarity, safe_float


def _normalize_rows(blocks: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    norms = blocks.detach().float().norm(dim=1, keepdim=True)
    valid = torch.isfinite(norms) & (norms > eps)
    safe_denominator = torch.where(valid, norms + eps, torch.ones_like(norms))
    normalized = blocks / safe_denominator
    normalized = torch.where(valid, normalized, torch.zeros_like(blocks))
    return normalized, norms.squeeze(1)


class BlockDirectionOptimizer(PhysicalOptimizerBase, torch.optim.Optimizer):
    """Structure-aware block direction optimizer.

    This optimizer keeps a blockwise direction memory instead of Adam moments.
    Matrix-like parameters are split row-wise so each output block gets its own
    coherence, recoverability, and trust state. Recoverability acts only as a
    trust gate on candidate directions rather than generating the direction.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        memory_decay: float = 0.85,
        grad_decay: float = 0.95,
        coherence_strength: float = 0.35,
        norm_ratio_strength: float = 0.20,
        conflict_penalty: float = 0.18,
        blend_strength: float = 0.10,
        recoverability_strength: float = 0.08,
        recoverability_keep_ratio: float = 0.55,
        recoverability_samples: int = 2,
        dimension_power: float = 0.0,
        max_update_ratio: float = 0.12,
        min_scale: float = 0.60,
        max_scale: float = 1.65,
        use_signed_consensus: bool = True,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            memory_decay=memory_decay,
            grad_decay=grad_decay,
            coherence_strength=coherence_strength,
            norm_ratio_strength=norm_ratio_strength,
            conflict_penalty=conflict_penalty,
            blend_strength=blend_strength,
            recoverability_strength=recoverability_strength,
            recoverability_keep_ratio=recoverability_keep_ratio,
            recoverability_samples=recoverability_samples,
            dimension_power=dimension_power,
            max_update_ratio=max_update_ratio,
            min_scale=min_scale,
            max_scale=max_scale,
            use_signed_consensus=use_signed_consensus,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("BlockDirectionOptimizer")

    @staticmethod
    def _block_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim <= 1:
            return tensor.reshape(1, -1)
        return tensor.reshape(tensor.shape[0], -1)

    def _init_state(self, state: dict[str, Any], param: torch.Tensor) -> None:
        block_shape = tuple(self._block_view(param).shape)
        if len(state) != 0 and tuple(state["direction_memory"].shape) == block_shape:
            return
        block_count = block_shape[0]
        device = param.device
        state["step"] = 0
        state["direction_memory"] = torch.zeros(block_shape, dtype=param.dtype, device=device)
        state["prev_update"] = torch.zeros(block_shape, dtype=param.dtype, device=device)
        state["grad_norm_ema"] = torch.zeros(block_count, dtype=torch.float32, device=device)
        state["trust_ema"] = torch.ones(block_count, dtype=torch.float32, device=device)
        state["recoverability_ema"] = torch.ones(block_count, dtype=torch.float32, device=device)

    def _recoverability_score(
        self,
        grad_blocks: torch.Tensor,
        grad_dirs: torch.Tensor,
        *,
        keep_ratio: float,
        samples: int,
        eps: float,
    ) -> torch.Tensor:
        block_count, block_width = grad_blocks.shape
        if samples <= 0 or block_width <= 1:
            return torch.ones(block_count, dtype=grad_blocks.dtype, device=grad_blocks.device)
        keep_ratio = clamp_scalar(float(keep_ratio), 0.15, 0.85)
        scores: list[torch.Tensor] = []
        for _ in range(samples):
            mask = (torch.rand_like(grad_blocks) < keep_ratio).to(grad_blocks.dtype)
            complement = 1.0 - mask
            dir_a, _ = _normalize_rows(grad_blocks * mask, eps)
            dir_b, _ = _normalize_rows(grad_blocks * complement, eps)
            score_a = (dir_a * grad_dirs).sum(dim=1).clamp(-1.0, 1.0)
            score_b = (dir_b * grad_dirs).sum(dim=1).clamp(-1.0, 1.0)
            agreement = (dir_a * dir_b).sum(dim=1).clamp(-1.0, 1.0)
            valid_a = mask.sum(dim=1) > 0
            valid_b = complement.sum(dim=1) > 0
            support = valid_a.float() + valid_b.float()
            score = torch.where(valid_a, score_a, torch.zeros_like(score_a))
            score = score + torch.where(valid_b, score_b, torch.zeros_like(score_b))
            score = torch.where(support > 0.0, score / support.clamp_min(1.0), torch.zeros_like(score))
            score = 0.7 * score + 0.3 * torch.where(valid_a & valid_b, agreement, score)
            scores.append(score)
        stacked = torch.stack(scores, dim=0)
        return ((stacked.mean(dim=0) + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, _ = self._prepare_closure(closure)

        coherence_values: list[float] = []
        trust_values: list[float] = []
        norm_ratio_values: list[float] = []
        memory_support_values: list[float] = []
        recoverability_values: list[float] = []
        update_ratio_values: list[float] = []
        conflict_values: list[float] = []
        active_params = 0
        total_blocks = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            memory_decay = clamp_scalar(float(group["memory_decay"]), 0.0, 0.9999)
            grad_decay = clamp_scalar(float(group["grad_decay"]), 0.0, 0.9999)
            coherence_strength = float(group["coherence_strength"])
            norm_ratio_strength = float(group["norm_ratio_strength"])
            conflict_penalty = float(group["conflict_penalty"])
            blend_strength = float(group["blend_strength"])
            recoverability_strength = float(group["recoverability_strength"])
            recoverability_keep_ratio = float(group["recoverability_keep_ratio"])
            recoverability_samples = int(group["recoverability_samples"])
            dimension_power = clamp_scalar(float(group["dimension_power"]), 0.0, 1.0)
            max_update_ratio = max(0.0, float(group["max_update_ratio"]))
            min_scale = float(group["min_scale"])
            max_scale = float(group["max_scale"])
            use_signed_consensus = bool(group["use_signed_consensus"])
            maximize = bool(group["maximize"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    continue

                grad_blocks = self._block_view(grad)
                grad_dirs, grad_norms = _normalize_rows(grad_blocks, eps)
                if not bool((grad_norms > eps).any()):
                    continue

                state = self.state[param]
                self._init_state(state, param)
                state["step"] = int(state["step"]) + 1

                memory_blocks = state["direction_memory"]
                prev_update_blocks = state["prev_update"]
                memory_dirs, memory_norms = _normalize_rows(memory_blocks, eps)
                memory_exists = memory_norms > eps

                coherence = torch.zeros_like(grad_norms)
                if bool(memory_exists.any()):
                    coherence = (grad_dirs * memory_dirs).sum(dim=1).clamp(-1.0, 1.0)
                conflict = torch.clamp(-coherence, min=0.0)

                same_direction_blend = blend_strength * torch.clamp(coherence, min=0.0)
                blended_consensus, _ = _normalize_rows(
                    (1.0 - same_direction_blend.unsqueeze(1)) * grad_dirs + same_direction_blend.unsqueeze(1) * memory_dirs,
                    eps,
                )
                if use_signed_consensus:
                    opposite_consensus, _ = _normalize_rows(
                        (1.0 + blend_strength) * grad_dirs - blend_strength * memory_dirs,
                        eps,
                    )
                    consensus = torch.where(
                        (memory_exists & (coherence < 0.0)).unsqueeze(1),
                        opposite_consensus,
                        blended_consensus,
                    )
                else:
                    consensus = blended_consensus
                consensus = torch.where(memory_exists.unsqueeze(1), consensus, grad_dirs)

                grad_norm_ema = state["grad_norm_ema"]
                grad_norm_ema = torch.where(grad_norm_ema > eps, grad_decay * grad_norm_ema + (1.0 - grad_decay) * grad_norms, grad_norms)
                norm_ratio = grad_norms / (grad_norm_ema + eps)

                recoverability = self._recoverability_score(
                    grad_blocks,
                    grad_dirs,
                    keep_ratio=recoverability_keep_ratio,
                    samples=recoverability_samples,
                    eps=eps,
                )
                recoverability_ema = state["recoverability_ema"]
                recoverability_ema = 0.85 * recoverability_ema + 0.15 * recoverability.float()

                candidate_trust = (
                    1.0
                    + coherence_strength * coherence.float()
                    + norm_ratio_strength * torch.log1p(torch.clamp(norm_ratio.float() - 1.0, min=0.0))
                    + recoverability_strength * (recoverability_ema - 0.5) * 2.0
                    - conflict_penalty * conflict.float()
                ).clamp(min_scale, max_scale)
                trust_ema = 0.85 * state["trust_ema"] + 0.15 * candidate_trust
                trust_scale = trust_ema.clamp(min_scale, max_scale)

                block_width = max(1, grad_blocks.shape[1])
                dimension_scale = float(block_width) ** dimension_power
                step_norm = lr * trust_scale * grad_norms.float() / max(dimension_scale, eps)
                param_blocks = self._block_view(param.detach())
                param_norms = param_blocks.float().norm(dim=1)
                if max_update_ratio > 0.0:
                    step_cap = max_update_ratio * (param_norms + eps)
                    step_norm = torch.minimum(step_norm, step_cap)

                update_blocks = consensus * step_norm.unsqueeze(1).to(consensus.dtype)
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update_blocks.reshape_as(param), alpha=-1.0)

                new_memory, _ = _normalize_rows(memory_blocks * memory_decay + consensus * (1.0 - memory_decay), eps)
                memory_blocks.copy_(new_memory)
                prev_update_blocks.copy_(update_blocks)
                state["grad_norm_ema"] = grad_norm_ema
                state["trust_ema"] = trust_ema
                state["recoverability_ema"] = recoverability_ema

                memory_support = torch.where(
                    memory_exists,
                    ((consensus * memory_dirs).sum(dim=1).clamp(-1.0, 1.0) + 1.0) * 0.5,
                    torch.zeros_like(coherence),
                )
                coherence_values.extend(float(value) for value in coherence.detach().cpu())
                trust_values.extend(float(value) for value in trust_scale.detach().cpu())
                norm_ratio_values.extend(float(value) for value in norm_ratio.detach().cpu())
                memory_support_values.extend(float(value) for value in memory_support.detach().cpu())
                recoverability_values.extend(float(value) for value in recoverability_ema.detach().cpu())
                conflict_values.extend(float(value) for value in conflict.detach().cpu())
                update_ratio_values.extend(float(value) for value in (step_norm / (param_norms + eps)).detach().cpu())
                active_params += 1
                total_blocks += int(grad_blocks.shape[0])

        self._record_step(
            {
                "optimizer": self.optimizer_name,
                "block_coherence": average(coherence_values),
                "block_trust_scale": average(trust_values),
                "block_norm_ratio": average(norm_ratio_values),
                "memory_ratio": average(memory_support_values),
                "memory_support": average(memory_support_values),
                "recoverability_score": average(recoverability_values),
                "block_conflict": average(conflict_values),
                "update_ratio": average(update_ratio_values),
                "active_params": float(active_params),
                "block_count": float(total_blocks),
            }
        )
        return loss_tensor
