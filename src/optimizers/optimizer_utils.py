from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


DEFAULT_EPS = 1e-12


def safe_float(value: float | int | torch.Tensor | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        item = float(value.detach().float().mean().item())
        if math.isfinite(item):
            return item
    return default


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_entropy(tensor: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    values = tensor.detach().abs().reshape(-1).float()
    if values.numel() <= 1:
        return 0.0
    total = values.sum()
    if not torch.isfinite(total) or total <= eps:
        return 0.0
    probs = values / (total + eps)
    entropy = -(probs * (probs + eps).log()).sum()
    normalized = entropy / math.log(values.numel() + 1.0)
    return max(0.0, min(1.0, safe_float(normalized)))


def clamp_scalar(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_sigmoid(value: float, sharpness: float = 8.0) -> float:
    clipped = max(-60.0, min(60.0, sharpness * value))
    return 1.0 / (1.0 + math.exp(-clipped))


def bounded_scale(value: float, min_scale: float, max_scale: float) -> float:
    return clamp_scalar(value, min_scale, max_scale)


def tensor_rms(tensor: torch.Tensor) -> float:
    return safe_float(tensor.detach().float().pow(2).mean().sqrt())


def tensor_energy(tensor: torch.Tensor) -> float:
    return safe_float(tensor.detach().float().pow(2).mean())


def norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    return safe_float(numerator.norm()) / (safe_float(denominator.norm()) + eps)


def sign_flip_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().reshape(-1)
    b_flat = b.detach().reshape(-1)
    if a_flat.numel() == 0:
        return 0.0
    flips = torch.sign(a_flat) != torch.sign(b_flat)
    return safe_float(flips.float().mean())


def sign_flip_ratio_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.detach().reshape(-1)
    b_flat = b.detach().reshape(-1)
    if a_flat.numel() == 0:
        return torch.zeros((), device=a.device, dtype=torch.float32)
    flips = torch.sign(a_flat) != torch.sign(b_flat)
    return flips.float().mean()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    a_flat = a.detach().reshape(-1).float()
    b_flat = b.detach().reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    if not torch.isfinite(denom) or denom <= eps:
        return 0.0
    value = float(torch.dot(a_flat, b_flat).item() / (float(denom.item()) + eps))
    return max(-1.0, min(1.0, value))


def cosine_similarity_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    a_flat = a.detach().reshape(-1).float()
    b_flat = b.detach().reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    valid = torch.isfinite(denom) & (denom > eps)
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    value = torch.dot(a_flat, b_flat) / (safe_denom + eps)
    value = value.clamp(-1.0, 1.0)
    return torch.where(valid, value, torch.zeros_like(value))


def average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return float("inf")
        total += float(grad.pow(2).sum().item())
    return math.sqrt(max(total, 0.0))


def parameter_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        data = param.detach()
        if not torch.isfinite(data).all():
            return float("inf")
        total += float(data.pow(2).sum().item())
    return math.sqrt(max(total, 0.0))


def layerwise_entropy(parameters: Iterable[torch.nn.Parameter]) -> float:
    entropies: list[float] = []
    for param in parameters:
        if param.grad is None:
            continue
        entropies.append(tensor_entropy(param.grad))
    if not entropies:
        return 0.0
    return float(sum(entropies) / len(entropies))


def compute_update_energy(update: torch.Tensor) -> float:
    return safe_float(update.detach().float().pow(2).mean())


def update_ratio(update: torch.Tensor, parameter: torch.Tensor, lr: float = 1.0, eps: float = DEFAULT_EPS) -> float:
    numerator = safe_float((update.detach() * lr).norm())
    denominator = safe_float(parameter.detach().norm()) + eps
    return numerator / denominator


def clip_by_update_energy(
    update: torch.Tensor,
    reference_energy: float,
    max_ratio: float,
    eps: float = DEFAULT_EPS,
) -> tuple[torch.Tensor, float]:
    if max_ratio <= 0:
        return torch.zeros_like(update), 0.0
    energy = compute_update_energy(update)
    if not math.isfinite(energy) or energy <= eps:
        return torch.zeros_like(update), 0.0
    allowed = max(reference_energy * max_ratio, eps)
    if energy <= allowed:
        return update, energy
    scale = math.sqrt(allowed / (energy + eps))
    clipped = update * scale
    return clipped, compute_update_energy(clipped)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loss_is_finite(loss_value: float | None, threshold: float = 1e6) -> bool:
    if loss_value is None:
        return True
    return math.isfinite(loss_value) and abs(loss_value) <= threshold


@dataclass(slots=True)
class ImprovementTracker:
    best_loss: float | None = None
    last_loss: float | None = None
    loss_improvement: float = 0.0
    stagnation_counter: int = 0

    def update(self, loss_value: float | None, improvement_tolerance: float = 1e-8) -> None:
        if loss_value is None:
            self.loss_improvement = 0.0
            return
        if self.last_loss is None:
            self.loss_improvement = 0.0
        else:
            self.loss_improvement = self.last_loss - loss_value
        self.last_loss = loss_value
        if self.best_loss is None or loss_value < self.best_loss - improvement_tolerance:
            self.best_loss = loss_value
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1


def collect_layer_statistics(parameters: Iterable[torch.nn.Parameter]) -> dict[str, float]:
    gradient_energies: list[float] = []
    entropies: list[float] = []
    gradient_norms: list[float] = []
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        gradient_energies.append(compute_update_energy(grad))
        entropies.append(tensor_entropy(grad))
        gradient_norms.append(safe_float(grad.norm()))
    if not gradient_energies:
        return {
            "gradient_energy": 0.0,
            "gradient_entropy": 0.0,
            "layerwise_entropy": 0.0,
            "gradient_norm": 0.0,
        }
    return {
        "gradient_energy": float(sum(gradient_energies) / len(gradient_energies)),
        "gradient_entropy": float(sum(entropies) / len(entropies)),
        "layerwise_entropy": float(sum(entropies) / len(entropies)),
        "gradient_norm": float(sum(gradient_norms) / len(gradient_norms)),
    }


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flat_parts = [tensor.detach().reshape(-1).float() for tensor in tensors if tensor is not None]
    if not flat_parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(flat_parts)
