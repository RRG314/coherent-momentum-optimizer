from __future__ import annotations

import torch

from .base import PhysicalOptimizerBase
from .optimizer_utils import (
    DEFAULT_EPS,
    average,
    bounded_scale,
    clamp_scalar,
    cosine_similarity,
    flatten_tensors,
    safe_float,
    sign_flip_ratio,
)


class CoherentDirectionReferenceOptimizer(PhysicalOptimizerBase, torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        alignment_strength: float = 0.15,
        damping_strength: float = 0.2,
        coherence_strength: float = 0.15,
        rotation_penalty: float = 0.25,
        field_clip: float = 2.0,
        min_scale: float = 0.6,
        max_scale: float = 1.4,
        layerwise_mode: bool = True,
        global_mode: bool = True,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            alignment_strength=alignment_strength,
            damping_strength=damping_strength,
            coherence_strength=coherence_strength,
            rotation_penalty=rotation_penalty,
            field_clip=field_clip,
            min_scale=min_scale,
            max_scale=max_scale,
            layerwise_mode=layerwise_mode,
            global_mode=global_mode,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self._initialize_physical_optimizer("CoherentDirectionReferenceOptimizer")

    def _group_global_metrics(self, group) -> dict[str, float]:
        grads: list[torch.Tensor] = []
        momenta: list[torch.Tensor] = []
        prev_grads: list[torch.Tensor] = []
        prev_updates: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        eps = float(group["eps"])
        beta1, beta2 = group["betas"]
        maximize = bool(group["maximize"])
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if maximize:
                grad = -grad
            state = self.state.get(param, {})
            exp_avg = state.get("exp_avg", torch.zeros_like(param))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(param))
            step = int(state.get("step", 0)) + 1
            next_exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
            next_exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            adam_direction = (next_exp_avg / max(bias_correction1, DEFAULT_EPS)) / (
                (next_exp_avg_sq / max(bias_correction2, DEFAULT_EPS)).sqrt() + eps
            )
            grads.append(grad)
            momenta.append(next_exp_avg)
            prev_grads.append(state.get("prev_grad", torch.zeros_like(param)))
            prev_updates.append(state.get("prev_update", torch.zeros_like(param)))
            updates.append(adam_direction)

        grad_cat = flatten_tensors(grads)
        momentum_cat = flatten_tensors(momenta)
        prev_grad_cat = flatten_tensors(prev_grads)
        prev_update_cat = flatten_tensors(prev_updates)
        update_cat = flatten_tensors(updates)

        grad_momentum_cos = cosine_similarity(grad_cat, momentum_cat)
        grad_prev_grad_cos = cosine_similarity(grad_cat, prev_grad_cat)
        update_prev_update_cos = cosine_similarity(update_cat, prev_update_cat)
        rotation_score = 0.5 * (1.0 - grad_prev_grad_cos) + 0.5 * max(0.0, -update_prev_update_cos)
        coherence_score = average([max(0.0, grad_momentum_cos), max(0.0, update_prev_update_cos)])
        return {
            "grad_momentum_cos": grad_momentum_cos,
            "grad_prev_grad_cos": grad_prev_grad_cos,
            "update_prev_update_cos": update_prev_update_cos,
            "rotation_score": rotation_score,
            "coherence_score": coherence_score,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor, current_loss = self._prepare_closure(closure)

        grad_momentum_cos_values: list[float] = []
        grad_prev_grad_cos_values: list[float] = []
        update_prev_update_cos_values: list[float] = []
        rotation_scores: list[float] = []
        coherence_scores: list[float] = []
        field_strengths: list[float] = []
        lr_scales: list[float] = []
        momentum_scales: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            alignment_strength = group["alignment_strength"]
            damping_strength = group["damping_strength"]
            coherence_strength = group["coherence_strength"]
            rotation_penalty = group["rotation_penalty"]
            field_clip = group["field_clip"]
            min_scale = group["min_scale"]
            max_scale = group["max_scale"]
            layerwise_mode = bool(group["layerwise_mode"])
            global_mode = bool(group["global_mode"])
            maximize = bool(group["maximize"])

            global_metrics = self._group_global_metrics(group) if global_mode else None

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if maximize:
                    grad = -grad
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)
                    state["prev_update"] = torch.zeros_like(param)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]
                prev_update = state["prev_update"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                exp_avg_hat = exp_avg / max(bias_correction1, DEFAULT_EPS)
                exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, DEFAULT_EPS)
                adam_direction = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)

                local_grad_momentum_cos = cosine_similarity(grad, exp_avg)
                local_grad_prev_grad_cos = cosine_similarity(grad, prev_grad)
                local_update_prev_update_cos = cosine_similarity(adam_direction, prev_update)
                local_rotation_score = 0.5 * (1.0 - local_grad_prev_grad_cos) + 0.5 * sign_flip_ratio(adam_direction, prev_update)
                local_rotation_score = clamp_scalar(local_rotation_score, 0.0, 1.5)
                local_coherence_score = average([max(0.0, local_grad_momentum_cos), max(0.0, local_update_prev_update_cos)])

                grad_momentum_cos = local_grad_momentum_cos
                grad_prev_grad_cos = local_grad_prev_grad_cos
                update_prev_update_cos = local_update_prev_update_cos
                rotation_score = local_rotation_score
                coherence_score = local_coherence_score
                if global_metrics is not None and layerwise_mode:
                    grad_momentum_cos = average([grad_momentum_cos, global_metrics["grad_momentum_cos"]])
                    grad_prev_grad_cos = average([grad_prev_grad_cos, global_metrics["grad_prev_grad_cos"]])
                    update_prev_update_cos = average([update_prev_update_cos, global_metrics["update_prev_update_cos"]])
                    rotation_score = average([rotation_score, global_metrics["rotation_score"]])
                    coherence_score = average([coherence_score, global_metrics["coherence_score"]])
                elif global_metrics is not None and not layerwise_mode:
                    grad_momentum_cos = global_metrics["grad_momentum_cos"]
                    grad_prev_grad_cos = global_metrics["grad_prev_grad_cos"]
                    update_prev_update_cos = global_metrics["update_prev_update_cos"]
                    rotation_score = global_metrics["rotation_score"]
                    coherence_score = global_metrics["coherence_score"]

                field_strength = clamp_scalar(abs(grad_momentum_cos) + coherence_score, 0.0, field_clip)
                momentum_scale = bounded_scale(
                    1.0
                    + alignment_strength * max(0.0, grad_momentum_cos)
                    - damping_strength * max(0.0, -grad_momentum_cos)
                    - rotation_penalty * rotation_score,
                    min_scale,
                    max_scale,
                )
                lr_scale = bounded_scale(
                    1.0
                    + coherence_strength * coherence_score
                    - damping_strength * max(0.0, -grad_prev_grad_cos)
                    - rotation_penalty * rotation_score,
                    min_scale,
                    max_scale,
                )

                controlled_direction = adam_direction * momentum_scale
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(controlled_direction, alpha=-(lr * lr_scale))

                prev_grad.copy_(grad)
                prev_update.copy_(controlled_direction.detach())

                grad_momentum_cos_values.append(grad_momentum_cos)
                grad_prev_grad_cos_values.append(grad_prev_grad_cos)
                update_prev_update_cos_values.append(update_prev_update_cos)
                rotation_scores.append(rotation_score)
                coherence_scores.append(coherence_score)
                field_strengths.append(field_strength)
                lr_scales.append(lr_scale)
                momentum_scales.append(momentum_scale)

        self._record_step(
            {
                "loss": current_loss,
                "grad_momentum_cosine": average(grad_momentum_cos_values),
                "grad_previous_grad_cosine": average(grad_prev_grad_cos_values),
                "update_previous_update_cosine": average(update_prev_update_cos_values),
                "rotation_score": average(rotation_scores),
                "coherence_score": average(coherence_scores),
                "field_strength": average(field_strengths),
                "effective_lr_scale": average(lr_scales),
                "effective_momentum_scale": average(momentum_scales),
            }
        )
        return loss_tensor
