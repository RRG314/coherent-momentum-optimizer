from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


Batch = tuple[torch.Tensor, torch.Tensor]
TrainingStepFn = Callable[[nn.Module, Any, int, int, "TaskContext"], torch.Tensor]
EvaluationFn = Callable[[nn.Module, "TaskContext"], dict[str, float]]


@dataclass(slots=True)
class TrainingPhase:
    start_fraction: float
    loader: Iterable[Any]
    label: str = "train"


@dataclass(slots=True)
class TaskContext:
    name: str
    family: str
    problem_type: str
    model: nn.Module
    epochs: int
    train_phases: list[TrainingPhase]
    training_step: TrainingStepFn
    evaluate: EvaluationFn
    target_loss: float | None
    target_accuracy: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_train_steps(self) -> int:
        return int(sum(len(phase.loader) for phase in self.train_phases) * self.epochs)


class LinearModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LargeMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TanhMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitsCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 2 * 2, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VisionSmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VisionDeeperCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TinyResNetLikeCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block1 = ResidualBlock(24)
        self.down = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.block2 = ResidualBlock(48)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        return self.head(x)


class DigitsAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 12), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class DirectParameterModel(nn.Module):
    def __init__(self, size: int = 2, init_scale: float = 0.5) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(size) * init_scale)

    def forward(self) -> torch.Tensor:
        return self.position


class DirectMatrixModel(nn.Module):
    def __init__(self, rows: int = 12, cols: int = 12, init_scale: float = 0.2) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(rows, cols) * init_scale)

    def forward(self) -> torch.Tensor:
        return self.matrix


class HarmonicPINN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, hidden: int = 48, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, hidden), nn.Tanh()]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HarmonicPINN(PINNMLP):
    def __init__(self) -> None:
        super().__init__(in_features=1, out_features=1, hidden=32, depth=2)


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    device: torch.device,
    target_dtype: torch.dtype,
    reshape: tuple[int, ...] | None = None,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if reshape is not None:
        x_tensor = x_tensor.reshape(x_tensor.shape[0], *reshape)
    y_tensor = torch.tensor(y, dtype=target_dtype)
    dataset = TensorDataset(x_tensor.to(device), y_tensor.to(device))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def _repo_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".datasets"


def _torchvision_is_available() -> bool:
    return importlib.util.find_spec("torchvision") is not None


def _load_torchvision_image_arrays(
    dataset_name: str,
    seed: int,
    train_limit: int,
    val_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _torchvision_is_available():
        raise RuntimeError("torchvision is not installed.")
    from torchvision import datasets  # type: ignore

    root = _repo_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root=str(root), train=True, download=True)
        val_ds = datasets.MNIST(root=str(root), train=False, download=True)
        x_train = train_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        y_train = train_ds.targets.numpy().astype(np.int64)
        x_val = val_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        y_val = val_ds.targets.numpy().astype(np.int64)
    elif dataset_name == "fashion_mnist":
        train_ds = datasets.FashionMNIST(root=str(root), train=True, download=True)
        val_ds = datasets.FashionMNIST(root=str(root), train=False, download=True)
        x_train = train_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        y_train = train_ds.targets.numpy().astype(np.int64)
        x_val = val_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        y_val = val_ds.targets.numpy().astype(np.int64)
    elif dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=str(root), train=True, download=True)
        val_ds = datasets.CIFAR10(root=str(root), train=False, download=True)
        x_train = train_ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_train = np.asarray(train_ds.targets, dtype=np.int64)
        x_val = val_ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        y_val = np.asarray(val_ds.targets, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(x_train))[: min(train_limit, len(x_train))]
    val_idx = rng.permutation(len(x_val))[: min(val_limit, len(x_val))]
    return x_train[train_idx], x_val[val_idx], y_train[train_idx], y_val[val_idx]


def _train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify_y = y if stratify else None
    return train_test_split(x, y, test_size=0.25, random_state=seed, stratify=stratify_y)


def _scaled_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_val, y_train, y_val = _train_val_split(x, y, seed, stratify=stratify)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    return x_train, x_val, y_train, y_val


def _supervised_step(model: nn.Module, batch: Batch, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x, y = batch
    criterion: nn.Module = context.metadata["criterion"]
    outputs = model(x)
    if context.problem_type == "binary_classification":
        y = y.float().view(-1, 1)
        return criterion(outputs, y)
    if context.problem_type == "classification":
        return criterion(outputs, y.long())
    if context.problem_type == "autoencoder":
        return criterion(outputs, x.view(x.shape[0], -1))
    return criterion(outputs.view_as(y), y.float())


def _conflicting_supervised_step(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x, y_primary, y_secondary = batch
    criterion: nn.Module = context.metadata["criterion"]
    y = y_primary if step % 2 == 1 else y_secondary
    outputs = model(x)
    y = y.float().view(-1, 1)
    return criterion(outputs, y)


def _evaluate_supervised(model: nn.Module, context: TaskContext) -> dict[str, float]:
    model.eval()
    criterion: nn.Module = context.metadata["criterion"]
    val_loader: DataLoader = context.metadata["val_loader"]
    losses: list[float] = []
    accuracies: list[float] = []
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            if context.problem_type == "binary_classification":
                y = y.float().view(-1, 1)
                loss = criterion(outputs, y)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                accuracies.append(float((preds == y).float().mean().item()))
            elif context.problem_type == "classification":
                loss = criterion(outputs, y.long())
                preds = outputs.argmax(dim=1)
                accuracies.append(float((preds == y).float().mean().item()))
            elif context.problem_type == "autoencoder":
                flat_x = x.view(x.shape[0], -1)
                loss = criterion(outputs, flat_x)
            else:
                loss = criterion(outputs.view_as(y), y.float())
            losses.append(float(loss.item()))
    results = {"val_loss": float(np.mean(losses) if losses else 0.0)}
    if accuracies:
        results["val_accuracy"] = float(np.mean(accuracies))
    return results


def _direct_saddle_step(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    coords = model()
    x = coords[0]
    y = coords[1]
    return x.pow(2) - y.pow(2) + 0.05 * y.pow(4) + 0.1 * (x - y).pow(2)


def _evaluate_direct(model: nn.Module, context: TaskContext) -> dict[str, float]:
    with torch.no_grad():
        loss = float(_direct_saddle_step(model, None, 0, 0, context).item())
    return {"val_loss": loss}


def _direct_objective_step(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    coords = model()
    objective = context.metadata["objective_fn"]
    return objective(coords, epoch, step, context)


def _evaluate_direct_objective(model: nn.Module, context: TaskContext) -> dict[str, float]:
    with torch.no_grad():
        coords = model()
        objective = context.metadata.get("eval_objective_fn", context.metadata["objective_fn"])
        loss = float(objective(coords, context.epochs, context.total_train_steps(), context).item())
    return {"val_loss": loss}


def _pinn_training_step(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    component_fn: Callable[[nn.Module, Any, int, int, TaskContext], dict[str, torch.Tensor]] = context.metadata["pinn_component_fn"]
    component_weights: dict[str, float] = context.metadata["pinn_component_weights"]
    training_components: list[str] = context.metadata["pinn_training_components"]
    components = component_fn(model, batch, epoch, step, context)
    total = torch.zeros((), device=next(model.parameters()).device)
    for name in training_components:
        total = total + float(component_weights.get(name, 1.0)) * components[name]
    return total


def _pinn_component_closure_builder(
    model: nn.Module,
    batch: Any,
    epoch: int,
    step: int,
    context: TaskContext,
) -> dict[str, Any]:
    component_fn: Callable[[nn.Module, Any, int, int, TaskContext], dict[str, torch.Tensor]] = context.metadata["pinn_component_fn"]
    optimizer_components: list[str] = context.metadata["pinn_optimizer_components"]
    closures: dict[str, Callable[[], torch.Tensor]] = {}
    for name in optimizer_components:
        closures[name] = lambda component_name=name: component_fn(model, batch, epoch, step, context)[component_name]
    return {
        "closures": closures,
        "metadata": {
            "training_components": list(context.metadata["pinn_training_components"]),
            "residual_name": str(context.metadata.get("pinn_residual_name", "residual")),
            "perturbed_residual_name": str(context.metadata.get("pinn_perturbed_name", "residual_perturbed")),
        },
    }


def _deterministic_perturb_1d(x: torch.Tensor, step: int, scale: float, low: float, high: float) -> torch.Tensor:
    phase = float(step + 1)
    perturbed = x + scale * torch.sin(3.0 * x + phase * 0.173)
    return perturbed.clamp(low, high)


def _deterministic_perturb_2d(points: torch.Tensor, step: int, scale: float) -> torch.Tensor:
    phase = float(step + 1)
    delta_x = scale * torch.sin(2.0 * points[:, :1] + phase * 0.173)
    delta_t = scale * torch.cos(5.0 * points[:, 1:2] + phase * 0.137)
    perturbed = torch.cat([points[:, :1] + delta_x, points[:, 1:2] + delta_t], dim=1)
    perturbed[:, :1] = perturbed[:, :1].clamp(0.0, 1.0)
    perturbed[:, 1:2] = perturbed[:, 1:2].clamp(0.0, 1.0)
    return perturbed


def _harmonic_pinn_components(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> dict[str, torch.Tensor]:
    t = batch[0].clone().detach().requires_grad_(True)
    y = model(t)
    dy = torch.autograd.grad(y.sum(), t, create_graph=True)[0]
    d2y = torch.autograd.grad(dy.sum(), t, create_graph=True)[0]
    residual = (d2y + y).pow(2).mean()

    t_perturbed = _deterministic_perturb_1d(t, step, scale=0.08, low=0.0, high=2.0 * np.pi)
    t_perturbed = t_perturbed.detach().requires_grad_(True)
    y_perturbed = model(t_perturbed)
    dy_perturbed = torch.autograd.grad(y_perturbed.sum(), t_perturbed, create_graph=True)[0]
    d2y_perturbed = torch.autograd.grad(dy_perturbed.sum(), t_perturbed, create_graph=True)[0]
    residual_perturbed = (d2y_perturbed + y_perturbed).pow(2).mean()

    bc_t = torch.zeros((8, 1), device=t.device, requires_grad=True)
    bc_y = model(bc_t)
    bc_dy = torch.autograd.grad(bc_y.sum(), bc_t, create_graph=True)[0]
    return {
        "residual": residual,
        "residual_perturbed": residual_perturbed,
        "boundary_value": (bc_y - 1.0).pow(2).mean(),
        "boundary_derivative": bc_dy.pow(2).mean(),
    }


def _poisson_pinn_components(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> dict[str, torch.Tensor]:
    x = batch[0].clone().detach().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    forcing = (np.pi ** 2) * torch.sin(np.pi * x)
    residual = (d2u + forcing).pow(2).mean()

    x_perturbed = _deterministic_perturb_1d(x, step, scale=0.05, low=0.0, high=1.0)
    x_perturbed = x_perturbed.detach().requires_grad_(True)
    u_perturbed = model(x_perturbed)
    du_perturbed = torch.autograd.grad(u_perturbed.sum(), x_perturbed, create_graph=True)[0]
    d2u_perturbed = torch.autograd.grad(du_perturbed.sum(), x_perturbed, create_graph=True)[0]
    forcing_perturbed = (np.pi ** 2) * torch.sin(np.pi * x_perturbed)
    residual_perturbed = (d2u_perturbed + forcing_perturbed).pow(2).mean()

    left = torch.zeros((16, 1), device=x.device)
    right = torch.ones((16, 1), device=x.device)
    return {
        "residual": residual,
        "residual_perturbed": residual_perturbed,
        "boundary_left": model(left).pow(2).mean(),
        "boundary_right": model(right).pow(2).mean(),
    }


def _heat_pinn_components(model: nn.Module, batch: Any, epoch: int, step: int, context: TaskContext) -> dict[str, torch.Tensor]:
    alpha = float(context.metadata.get("diffusivity", 0.2))
    points = batch[0].clone().detach().requires_grad_(True)
    x = points[:, :1]
    t = points[:, 1:2]
    u = model(points)
    grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    du_dx = grads[:, :1]
    du_dt = grads[:, 1:2]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), points, create_graph=True)[0][:, :1]
    residual = (du_dt - alpha * d2u_dx2).pow(2).mean()

    perturbed_points = _deterministic_perturb_2d(points, step, scale=0.04)
    perturbed_points = perturbed_points.detach().requires_grad_(True)
    u_perturbed = model(perturbed_points)
    perturbed_grads = torch.autograd.grad(u_perturbed.sum(), perturbed_points, create_graph=True)[0]
    du_dx_perturbed = perturbed_grads[:, :1]
    du_dt_perturbed = perturbed_grads[:, 1:2]
    d2u_dx2_perturbed = torch.autograd.grad(du_dx_perturbed.sum(), perturbed_points, create_graph=True)[0][:, :1]
    residual_perturbed = (du_dt_perturbed - alpha * d2u_dx2_perturbed).pow(2).mean()

    time_boundary = torch.linspace(0.0, 1.0, 16, device=points.device).view(-1, 1)
    left_points = torch.cat([torch.zeros_like(time_boundary), time_boundary], dim=1)
    right_points = torch.cat([torch.ones_like(time_boundary), time_boundary], dim=1)

    x_initial = torch.linspace(0.0, 1.0, 24, device=points.device).view(-1, 1)
    initial_points = torch.cat([x_initial, torch.zeros_like(x_initial)], dim=1)
    initial_target = torch.sin(np.pi * x_initial)
    initial_noise = float(context.metadata.get("initial_noise", 0.0))
    if initial_noise > 0.0:
        initial_target = initial_target + initial_noise * torch.sin((step + 1) * 0.173 + 6.0 * x_initial)

    return {
        "residual": residual,
        "residual_perturbed": residual_perturbed,
        "boundary": model(left_points).pow(2).mean() + model(right_points).pow(2).mean(),
        "initial": (model(initial_points) - initial_target).pow(2).mean(),
    }


def _evaluate_harmonic_pinn(model: nn.Module, context: TaskContext) -> dict[str, float]:
    model.eval()
    device = context.metadata["device"]
    t = torch.linspace(0.0, 2.0 * np.pi, 192, device=device).view(-1, 1)
    with torch.no_grad():
        pred = model(t)
        target = torch.cos(t)
        loss = torch.mean((pred - target) ** 2)
    return {"val_loss": float(loss.item())}


def _evaluate_poisson_pinn(model: nn.Module, context: TaskContext) -> dict[str, float]:
    model.eval()
    device = context.metadata["device"]
    x = torch.linspace(0.0, 1.0, 192, device=device).view(-1, 1)
    with torch.no_grad():
        pred = model(x)
        target = torch.sin(np.pi * x)
        loss = torch.mean((pred - target) ** 2)
    return {"val_loss": float(loss.item())}


def _evaluate_heat_pinn(model: nn.Module, context: TaskContext) -> dict[str, float]:
    model.eval()
    device = context.metadata["device"]
    alpha = float(context.metadata.get("diffusivity", 0.2))
    x = torch.linspace(0.0, 1.0, 48, device=device)
    t = torch.linspace(0.0, 1.0, 48, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    points = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(points).view_as(xx)
        target = torch.exp(-(np.pi ** 2) * alpha * tt) * torch.sin(np.pi * xx)
        loss = torch.mean((pred - target) ** 2)
    return {"val_loss": float(loss.item())}


def _phase(loader: DataLoader, start_fraction: float = 0.0, label: str = "train") -> TrainingPhase:
    return TrainingPhase(start_fraction=start_fraction, loader=loader, label=label)


def _context_from_arrays(
    *,
    name: str,
    family: str,
    problem_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    target_loss: float | None,
    target_accuracy: float | None,
) -> TaskContext:
    return TaskContext(
        name=name,
        family=family,
        problem_type=problem_type,
        model=model,
        epochs=epochs,
        train_phases=[_phase(train_loader)],
        training_step=_supervised_step,
        evaluate=_evaluate_supervised,
        target_loss=target_loss,
        target_accuracy=target_accuracy,
        metadata={"criterion": criterion, "val_loader": val_loader},
    )


def _make_direct_context(
    *,
    name: str,
    family: str,
    objective_fn: Callable[[torch.Tensor, int, int, TaskContext], torch.Tensor],
    device: torch.device,
    epochs: int,
    target_loss: float | None,
    init_scale: float = 0.5,
    num_steps: int = 48,
    eval_objective_fn: Callable[[torch.Tensor, int, int, TaskContext], torch.Tensor] | None = None,
) -> TaskContext:
    loader = [(torch.zeros(1, device=device), torch.zeros(1, device=device)) for _ in range(num_steps)]
    return TaskContext(
        name=name,
        family=family,
        problem_type="regression",
        model=DirectParameterModel(size=2, init_scale=init_scale).to(device),
        epochs=epochs,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_direct_objective_step,
        evaluate=_evaluate_direct_objective,
        target_loss=target_loss,
        target_accuracy=None,
        metadata={
            "device": device,
            "objective_fn": objective_fn,
            "eval_objective_fn": eval_objective_fn or objective_fn,
        },
    )


def _make_matrix_direct_context(
    *,
    name: str,
    family: str,
    objective_fn: Callable[[torch.Tensor, int, int, TaskContext], torch.Tensor],
    device: torch.device,
    epochs: int,
    target_loss: float | None,
    rows: int = 12,
    cols: int = 12,
    init_scale: float = 0.2,
    num_steps: int = 48,
    eval_objective_fn: Callable[[torch.Tensor, int, int, TaskContext], torch.Tensor] | None = None,
) -> TaskContext:
    loader = [(torch.zeros(1, device=device), torch.zeros(1, device=device)) for _ in range(num_steps)]
    return TaskContext(
        name=name,
        family=family,
        problem_type="regression",
        model=DirectMatrixModel(rows=rows, cols=cols, init_scale=init_scale).to(device),
        epochs=epochs,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_direct_objective_step,
        evaluate=_evaluate_direct_objective,
        target_loss=target_loss,
        target_accuracy=None,
        metadata={
            "device": device,
            "objective_fn": objective_fn,
            "eval_objective_fn": eval_objective_fn or objective_fn,
        },
    )


def _make_linear_regression(seed: int, device: torch.device) -> TaskContext:
    x, y = make_regression(n_samples=384, n_features=20, noise=8.0, random_state=seed)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed, stratify=False)
    return _context_from_arrays(
        name="linear_regression",
        family="convex",
        problem_type="regression",
        model=LinearModel(20, 1).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.MSELoss(),
        epochs=20,
        target_loss=0.12,
        target_accuracy=None,
    )


def _make_noisy_regression(seed: int, device: torch.device) -> TaskContext:
    x, y = make_regression(n_samples=420, n_features=24, noise=18.0, random_state=seed)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0.0, 35.0, size=y.shape[0])
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed, stratify=False)
    return _context_from_arrays(
        name="noisy_regression",
        family="convex",
        problem_type="regression",
        model=LinearModel(24, 1).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.MSELoss(),
        epochs=24,
        target_loss=0.55,
        target_accuracy=None,
    )


def _make_logistic_regression(seed: int, device: torch.device) -> TaskContext:
    x, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        flip_y=0.02,
        class_sep=1.2,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="logistic_regression",
        family="convex",
        problem_type="binary_classification",
        model=LinearModel(20, 1).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=18,
        target_loss=0.32,
        target_accuracy=0.88,
    )


def _make_synthetic_classification(seed: int, device: torch.device) -> TaskContext:
    x, y = make_classification(
        n_samples=640,
        n_features=24,
        n_informative=14,
        n_redundant=6,
        n_classes=3,
        class_sep=1.0,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="synthetic_classification",
        family="convex",
        problem_type="classification",
        model=SmallMLP(24, 3, hidden=48).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=22,
        target_loss=0.58,
        target_accuracy=0.78,
    )


def _make_moons(seed: int, device: torch.device) -> TaskContext:
    x, y = make_moons(n_samples=480, noise=0.18, random_state=seed)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="moons_mlp",
        family="neural",
        problem_type="binary_classification",
        model=SmallMLP(2, 1, hidden=32).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=22,
        target_loss=0.26,
        target_accuracy=0.92,
    )


def _make_circles(seed: int, device: torch.device) -> TaskContext:
    x, y = make_circles(n_samples=480, noise=0.12, factor=0.5, random_state=seed)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="circles_mlp",
        family="neural",
        problem_type="binary_classification",
        model=SmallMLP(2, 1, hidden=40).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=24,
        target_loss=0.30,
        target_accuracy=0.85,
    )


def _make_digits_logistic(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.data / 16.0
    y = digits.target
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="digits_logistic",
        family="neural",
        problem_type="classification",
        model=LinearModel(64, 10).to(device),
        train_loader=_make_loader(x_train, y_train, 48, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=18,
        target_loss=0.35,
        target_accuracy=0.91,
    )


def _make_digits_cnn(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    train_loader = _make_loader(x_train, y_train, 64, seed, device, torch.long, reshape=(1, 8, 8))
    val_loader = _make_loader(x_val, y_val, 128, seed + 1, device, torch.long, reshape=(1, 8, 8))
    return _context_from_arrays(
        name="digits_cnn",
        family="neural",
        problem_type="classification",
        model=DigitsCNN().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=18,
        target_loss=0.28,
        target_accuracy=0.93,
    )


def _make_torchvision_cnn_context(
    *,
    name: str,
    dataset_name: str,
    device: torch.device,
    seed: int,
    model: nn.Module,
    train_limit: int,
    val_limit: int,
    batch_size: int,
    epochs: int,
    target_loss: float | None,
    target_accuracy: float | None,
) -> TaskContext:
    x_train, x_val, y_train, y_val = _load_torchvision_image_arrays(
        dataset_name=dataset_name,
        seed=seed,
        train_limit=train_limit,
        val_limit=val_limit,
    )
    train_loader = _make_loader(x_train, y_train, batch_size, seed, device, torch.long)
    val_loader = _make_loader(x_val, y_val, max(128, batch_size * 2), seed + 1, device, torch.long)
    return _context_from_arrays(
        name=name,
        family="vision",
        problem_type="classification",
        model=model.to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=epochs,
        target_loss=target_loss,
        target_accuracy=target_accuracy,
    )


def _make_mnist_small_cnn(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="mnist_small_cnn",
        dataset_name="mnist",
        device=device,
        seed=seed,
        model=VisionSmallCNN(in_channels=1, num_classes=10),
        train_limit=4096,
        val_limit=1024,
        batch_size=64,
        epochs=10,
        target_loss=0.12,
        target_accuracy=0.96,
    )


def _make_mnist_deeper_cnn(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="mnist_deeper_cnn",
        dataset_name="mnist",
        device=device,
        seed=seed,
        model=VisionDeeperCNN(in_channels=1, num_classes=10),
        train_limit=4096,
        val_limit=1024,
        batch_size=64,
        epochs=10,
        target_loss=0.10,
        target_accuracy=0.97,
    )


def _make_fashion_mnist_small_cnn(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="fashion_mnist_small_cnn",
        dataset_name="fashion_mnist",
        device=device,
        seed=seed,
        model=VisionSmallCNN(in_channels=1, num_classes=10),
        train_limit=4096,
        val_limit=1024,
        batch_size=64,
        epochs=12,
        target_loss=0.42,
        target_accuracy=0.84,
    )


def _make_fashion_mnist_deeper_cnn(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="fashion_mnist_deeper_cnn",
        dataset_name="fashion_mnist",
        device=device,
        seed=seed,
        model=VisionDeeperCNN(in_channels=1, num_classes=10),
        train_limit=4096,
        val_limit=1024,
        batch_size=64,
        epochs=12,
        target_loss=0.38,
        target_accuracy=0.86,
    )


def _make_cifar10_subset_small_cnn(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="cifar10_subset_small_cnn",
        dataset_name="cifar10",
        device=device,
        seed=seed,
        model=VisionSmallCNN(in_channels=3, num_classes=10),
        train_limit=5000,
        val_limit=1000,
        batch_size=96,
        epochs=12,
        target_loss=1.45,
        target_accuracy=0.48,
    )


def _make_cifar10_subset_resnetlike(seed: int, device: torch.device) -> TaskContext:
    return _make_torchvision_cnn_context(
        name="cifar10_subset_resnetlike",
        dataset_name="cifar10",
        device=device,
        seed=seed,
        model=TinyResNetLikeCNN(in_channels=3, num_classes=10),
        train_limit=5000,
        val_limit=1000,
        batch_size=96,
        epochs=12,
        target_loss=1.35,
        target_accuracy=0.52,
    )


def _make_digits_cnn_label_noise(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target.copy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(y_train.shape[0]) < 0.18
    noisy_labels = y_train.copy()
    noisy_labels[flip_mask] = rng.integers(0, 10, size=int(flip_mask.sum()))
    train_loader = _make_loader(x_train, noisy_labels, 64, seed, device, torch.long, reshape=(1, 8, 8))
    val_loader = _make_loader(x_val, y_val, 128, seed + 1, device, torch.long, reshape=(1, 8, 8))
    return _context_from_arrays(
        name="digits_cnn_label_noise",
        family="stress",
        problem_type="classification",
        model=DigitsCNN().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=18,
        target_loss=0.75,
        target_accuracy=0.72,
    )


def _make_digits_cnn_input_noise(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.images.astype(np.float32) / 16.0
    y = digits.target
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    x_train = np.clip(x_train + rng.normal(0.0, 0.18, size=x_train.shape).astype(np.float32), 0.0, 1.0)
    x_val = np.clip(x_val + rng.normal(0.0, 0.18, size=x_val.shape).astype(np.float32), 0.0, 1.0)
    train_loader = _make_loader(x_train, y_train, 64, seed, device, torch.long, reshape=(1, 8, 8))
    val_loader = _make_loader(x_val, y_val, 128, seed + 1, device, torch.long, reshape=(1, 8, 8))
    return _context_from_arrays(
        name="digits_cnn_input_noise",
        family="stress",
        problem_type="classification",
        model=DigitsCNN().to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=18,
        target_loss=0.65,
        target_accuracy=0.78,
    )


def _make_digits_mlp(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="digits_mlp",
        family="neural",
        problem_type="classification",
        model=SmallMLP(64, 10, hidden=64).to(device),
        train_loader=_make_loader(x_train, y_train, 48, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=20,
        target_loss=0.24,
        target_accuracy=0.94,
    )


def _make_digits_autoencoder(seed: int, device: torch.device) -> TaskContext:
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = np.zeros(x.shape[0], dtype=np.float32)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=seed)
    return _context_from_arrays(
        name="digits_autoencoder",
        family="neural",
        problem_type="autoencoder",
        model=DigitsAutoencoder().to(device),
        train_loader=_make_loader(x_train, y_train, 64, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.float32),
        criterion=nn.MSELoss(),
        epochs=18,
        target_loss=0.035,
        target_accuracy=None,
    )


def _make_breast_cancer(seed: int, device: torch.device) -> TaskContext:
    dataset = load_breast_cancer()
    x, y = dataset.data, dataset.target
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="breast_cancer_mlp",
        family="neural",
        problem_type="binary_classification",
        model=SmallMLP(x.shape[1], 1, hidden=48).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=20,
        target_loss=0.15,
        target_accuracy=0.96,
    )


def _make_wine(seed: int, device: torch.device) -> TaskContext:
    dataset = load_wine()
    x, y = dataset.data, dataset.target
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="wine_mlp",
        family="neural",
        problem_type="classification",
        model=SmallMLP(x.shape[1], 3, hidden=48).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=20,
        target_loss=0.32,
        target_accuracy=0.90,
    )


def _make_noisy_gradients(seed: int, device: torch.device) -> TaskContext:
    context = _make_synthetic_classification(seed, device)
    context.name = "noisy_gradients_classification"
    context.family = "stability"
    context.metadata["gradient_noise_std"] = 0.03
    context.metadata["criterion"] = nn.CrossEntropyLoss(label_smoothing=0.05)
    context.target_accuracy = 0.74
    return context


def _make_sparse_gradients(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(640, 256)).astype(np.float32)
    mask = rng.binomial(1, 0.08, size=x.shape).astype(np.float32)
    x *= mask
    weights = rng.normal(size=(256, 1)).astype(np.float32)
    logits = x @ weights + 0.25 * rng.normal(size=(640, 1)).astype(np.float32)
    y = (logits.reshape(-1) > np.median(logits)).astype(np.float32)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="sparse_gradients_linear",
        family="stability",
        problem_type="binary_classification",
        model=LinearModel(256, 1).to(device),
        train_loader=_make_loader(x_train, y_train, 32, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=18,
        target_loss=0.45,
        target_accuracy=0.84,
    )


def _make_high_curvature(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(512, 20)).astype(np.float32)
    scales = np.logspace(-2, 2, 20).astype(np.float32)
    x = x * scales
    y = x @ rng.normal(size=(20, 1)).astype(np.float32)
    y = y.reshape(-1) + rng.normal(0.0, 2.5, size=512).astype(np.float32)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed, stratify=False)
    return _context_from_arrays(
        name="high_curvature_regression",
        family="stability",
        problem_type="regression",
        model=LinearModel(20, 1).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.float32),
        criterion=nn.MSELoss(),
        epochs=22,
        target_loss=0.25,
        target_accuracy=None,
    )


def _make_saddle(seed: int, device: torch.device) -> TaskContext:
    loader = [(torch.zeros(1, device=device), torch.zeros(1, device=device)) for _ in range(48)]
    return TaskContext(
        name="saddle_objective",
        family="stability",
        problem_type="regression",
        model=DirectParameterModel().to(device),
        epochs=24,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_direct_saddle_step,
        evaluate=_evaluate_direct,
        target_loss=-0.8,
        target_accuracy=None,
        metadata={"device": device},
    )


def _make_nonstationary(seed: int, device: torch.device) -> TaskContext:
    x_a, y_a = make_moons(n_samples=480, noise=0.15, random_state=seed)
    x_b, y_b = make_moons(n_samples=480, noise=0.25, random_state=seed + 101)
    x_b = x_b[:, ::-1]
    x_a_train, x_a_val, y_a_train, y_a_val = _scaled_split(x_a, y_a, seed)
    x_b_train, x_b_val, y_b_train, y_b_val = _scaled_split(x_b, y_b, seed + 1)
    train_loader_a = _make_loader(x_a_train, y_a_train, 24, seed, device, torch.float32)
    train_loader_b = _make_loader(x_b_train, y_b_train, 24, seed + 1, device, torch.float32)
    val_loader_b = _make_loader(x_b_val, y_b_val, 64, seed + 2, device, torch.float32)
    context = TaskContext(
        name="nonstationary_moons",
        family="stability",
        problem_type="binary_classification",
        model=SmallMLP(2, 1, hidden=40).to(device),
        epochs=24,
        train_phases=[TrainingPhase(0.0, train_loader_a, "phase_a"), TrainingPhase(0.5, train_loader_b, "phase_b")],
        training_step=_supervised_step,
        evaluate=_evaluate_supervised,
        target_loss=0.34,
        target_accuracy=0.84,
        metadata={"criterion": nn.BCEWithLogitsLoss(), "val_loader": val_loader_b},
    )
    return context


def _make_small_batch_instability(seed: int, device: torch.device) -> TaskContext:
    x, y = make_circles(n_samples=420, noise=0.18, factor=0.45, random_state=seed)
    flip_rng = np.random.default_rng(seed)
    flip_mask = flip_rng.random(size=y.shape[0]) < 0.08
    y = y.copy()
    y[flip_mask] = 1 - y[flip_mask]
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    context = _context_from_arrays(
        name="small_batch_instability",
        family="stability",
        problem_type="binary_classification",
        model=SmallMLP(2, 1, hidden=48).to(device),
        train_loader=_make_loader(x_train, y_train, 8, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=24,
        target_loss=0.44,
        target_accuracy=0.79,
    )
    context.metadata["gradient_noise_std"] = 0.015
    return context


def _make_overfit_small_wine(seed: int, device: torch.device) -> TaskContext:
    dataset = load_wine()
    x, y = dataset.data, dataset.target
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    x_train = x_train[:48]
    y_train = y_train[:48]
    return _context_from_arrays(
        name="overfit_small_wine",
        family="stability",
        problem_type="classification",
        model=LargeMLP(x.shape[1], 3).to(device),
        train_loader=_make_loader(x_train, y_train, 12, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 64, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=28,
        target_loss=0.62,
        target_accuracy=0.78,
    )


def _make_pinn(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    collocation = rng.uniform(0.0, 2.0 * np.pi, size=(256, 1)).astype(np.float32)
    zeros = np.zeros((collocation.shape[0],), dtype=np.float32)
    loader = _make_loader(collocation, zeros, 32, seed, device, torch.float32)
    return TaskContext(
        name="pinn_harmonic_oscillator",
        family="physics",
        problem_type="regression",
        model=HarmonicPINN().to(device),
        epochs=22,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_pinn_training_step,
        evaluate=_evaluate_harmonic_pinn,
        target_loss=0.03,
        target_accuracy=None,
        metadata={
            "device": device,
            "pinn_component_fn": _harmonic_pinn_components,
            "pinn_component_weights": {"residual": 1.0, "boundary_value": 10.0, "boundary_derivative": 10.0},
            "pinn_training_components": ["residual", "boundary_value", "boundary_derivative"],
            "pinn_optimizer_components": ["residual", "residual_perturbed", "boundary_value", "boundary_derivative"],
            "pinn_residual_name": "residual",
            "pinn_perturbed_name": "residual_perturbed",
            "component_closure_builder": _pinn_component_closure_builder,
        },
    )


def _make_pinn_poisson(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    collocation = rng.uniform(0.0, 1.0, size=(320, 1)).astype(np.float32)
    zeros = np.zeros((collocation.shape[0],), dtype=np.float32)
    loader = _make_loader(collocation, zeros, 40, seed, device, torch.float32)
    return TaskContext(
        name="pinn_poisson_1d",
        family="physics",
        problem_type="regression",
        model=PINNMLP(in_features=1, hidden=48, depth=3).to(device),
        epochs=24,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_pinn_training_step,
        evaluate=_evaluate_poisson_pinn,
        target_loss=0.01,
        target_accuracy=None,
        metadata={
            "device": device,
            "pinn_component_fn": _poisson_pinn_components,
            "pinn_component_weights": {"residual": 1.0, "boundary_left": 8.0, "boundary_right": 8.0},
            "pinn_training_components": ["residual", "boundary_left", "boundary_right"],
            "pinn_optimizer_components": ["residual", "residual_perturbed", "boundary_left", "boundary_right"],
            "pinn_residual_name": "residual",
            "pinn_perturbed_name": "residual_perturbed",
            "component_closure_builder": _pinn_component_closure_builder,
        },
    )


def _make_pinn_poisson_small_batch(seed: int, device: torch.device) -> TaskContext:
    context = _make_pinn_poisson(seed, device)
    rng = np.random.default_rng(seed + 17)
    collocation = rng.uniform(0.0, 1.0, size=(192, 1)).astype(np.float32)
    zeros = np.zeros((collocation.shape[0],), dtype=np.float32)
    context.name = "pinn_poisson_1d_small_batch"
    context.epochs = 26
    context.target_loss = 0.014
    context.train_phases = [TrainingPhase(0.0, _make_loader(collocation, zeros, 10, seed, device, torch.float32))]
    return context


def _make_pinn_heat(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    collocation = rng.uniform(0.0, 1.0, size=(512, 2)).astype(np.float32)
    zeros = np.zeros((collocation.shape[0],), dtype=np.float32)
    loader = _make_loader(collocation, zeros, 48, seed, device, torch.float32)
    return TaskContext(
        name="pinn_heat_equation",
        family="physics",
        problem_type="regression",
        model=PINNMLP(in_features=2, hidden=56, depth=3).to(device),
        epochs=28,
        train_phases=[TrainingPhase(0.0, loader)],
        training_step=_pinn_training_step,
        evaluate=_evaluate_heat_pinn,
        target_loss=0.02,
        target_accuracy=None,
        metadata={
            "device": device,
            "diffusivity": 0.2,
            "initial_noise": 0.0,
            "pinn_component_fn": _heat_pinn_components,
            "pinn_component_weights": {"residual": 1.0, "boundary": 5.0, "initial": 5.0},
            "pinn_training_components": ["residual", "boundary", "initial"],
            "pinn_optimizer_components": ["residual", "residual_perturbed", "boundary", "initial"],
            "pinn_residual_name": "residual",
            "pinn_perturbed_name": "residual_perturbed",
            "component_closure_builder": _pinn_component_closure_builder,
        },
    )


def _make_pinn_heat_noisy_initial(seed: int, device: torch.device) -> TaskContext:
    context = _make_pinn_heat(seed, device)
    context.name = "pinn_heat_equation_noisy_initial"
    context.target_loss = 0.03
    context.metadata["initial_noise"] = 0.03
    return context


def _make_stagnating_regression(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 5.0, size=(420, 12)).astype(np.float32)
    weights = rng.normal(size=(12, 1)).astype(np.float32)
    y = np.tanh(x @ weights).reshape(-1) + 0.05 * rng.normal(size=420).astype(np.float32)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed, stratify=False)
    context = _context_from_arrays(
        name="stagnating_regression",
        family="stress",
        problem_type="regression",
        model=TanhMLP(12, 1, hidden=40).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 96, seed + 1, device, torch.float32),
        criterion=nn.MSELoss(),
        epochs=24,
        target_loss=0.18,
        target_accuracy=None,
    )
    return context


def _make_unstable_deep_mlp(seed: int, device: torch.device) -> TaskContext:
    x, y = make_classification(
        n_samples=620,
        n_features=32,
        n_informative=18,
        n_redundant=8,
        n_classes=3,
        class_sep=0.85,
        flip_y=0.05,
        random_state=seed,
    )
    x = x.astype(np.float32) * 3.0
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    context = _context_from_arrays(
        name="unstable_deep_mlp",
        family="stress",
        problem_type="classification",
        model=LargeMLP(32, 3).to(device),
        train_loader=_make_loader(x_train, y_train, 10, seed, device, torch.long),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.long),
        criterion=nn.CrossEntropyLoss(),
        epochs=24,
        target_loss=0.78,
        target_accuracy=0.70,
    )
    context.metadata["gradient_noise_std"] = 0.02
    return context


def _make_loss_shock_classification(seed: int, device: torch.device) -> TaskContext:
    x, y = make_classification(
        n_samples=540,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_classes=2,
        class_sep=1.1,
        flip_y=0.01,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    y_train_shock = y_train.copy()
    rng = np.random.default_rng(seed + 17)
    shock_mask = rng.random(size=y_train_shock.shape[0]) < 0.25
    y_train_shock[shock_mask] = 1 - y_train_shock[shock_mask]
    train_loader_a = _make_loader(x_train, y_train, 24, seed, device, torch.float32)
    train_loader_b = _make_loader(x_train, y_train_shock, 24, seed + 1, device, torch.float32)
    val_loader = _make_loader(x_val, y_val, 96, seed + 2, device, torch.float32)
    return TaskContext(
        name="loss_shock_classification",
        family="stress",
        problem_type="binary_classification",
        model=SmallMLP(x.shape[1], 1, hidden=48).to(device),
        epochs=24,
        train_phases=[TrainingPhase(0.0, train_loader_a, "clean"), TrainingPhase(0.55, train_loader_b, "shock")],
        training_step=_supervised_step,
        evaluate=_evaluate_supervised,
        target_loss=0.46,
        target_accuracy=0.84,
        metadata={"criterion": nn.BCEWithLogitsLoss(), "val_loader": val_loader},
    )


def _make_label_noise_breast_cancer(seed: int, device: torch.device) -> TaskContext:
    dataset = load_breast_cancer()
    x, y = dataset.data, dataset.target.copy()
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    rng = np.random.default_rng(seed + 5)
    flip_mask = rng.random(size=y_train.shape[0]) < 0.15
    y_train = y_train.copy()
    y_train[flip_mask] = 1 - y_train[flip_mask]
    x_train = x_train[:128]
    y_train = y_train[:128]
    return _context_from_arrays(
        name="label_noise_breast_cancer",
        family="stress",
        problem_type="binary_classification",
        model=SmallMLP(x.shape[1], 1, hidden=40).to(device),
        train_loader=_make_loader(x_train, y_train, 16, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 128, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=26,
        target_loss=0.34,
        target_accuracy=0.90,
    )


def _make_conflicting_batches(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(640, 16)).astype(np.float32)
    primary_w = rng.normal(size=(16, 1)).astype(np.float32)
    secondary_w = (-0.6 * primary_w + 0.8 * rng.normal(size=(16, 1))).astype(np.float32)
    primary_logits = x @ primary_w + 0.2 * rng.normal(size=(640, 1)).astype(np.float32)
    secondary_logits = x @ secondary_w + 0.2 * rng.normal(size=(640, 1)).astype(np.float32)
    y_primary = (primary_logits.reshape(-1) > np.median(primary_logits)).astype(np.float32)
    y_secondary = (secondary_logits.reshape(-1) > np.median(secondary_logits)).astype(np.float32)

    x_train, x_val, y_primary_train, y_primary_val = _scaled_split(x, y_primary, seed)
    _, _, y_secondary_train, _ = _scaled_split(x, y_secondary, seed)
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32, device=device),
        torch.tensor(y_primary_train, dtype=torch.float32, device=device),
        torch.tensor(y_secondary_train, dtype=torch.float32, device=device),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    val_loader = _make_loader(x_val, y_primary_val, 128, seed + 1, device, torch.float32)
    return TaskContext(
        name="conflicting_batches_classification",
        family="stress",
        problem_type="binary_classification",
        model=SmallMLP(16, 1, hidden=40).to(device),
        epochs=26,
        train_phases=[TrainingPhase(0.0, train_loader)],
        training_step=_conflicting_supervised_step,
        evaluate=_evaluate_supervised,
        target_loss=0.52,
        target_accuracy=0.78,
        metadata={"criterion": nn.BCEWithLogitsLoss(), "val_loader": val_loader},
    )


def _make_block_structure_classification(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    samples = 640
    blocks = 6
    block_width = 4
    features = blocks * block_width
    x = rng.normal(size=(samples, features)).astype(np.float32)
    block_strengths = np.array([1.8, 1.5, 0.0, 0.0, -0.8, 0.2], dtype=np.float32)
    score_terms = []
    for block_index, strength in enumerate(block_strengths):
        block = x[:, block_index * block_width : (block_index + 1) * block_width]
        score_terms.append(strength * block.mean(axis=1))
    score = np.sum(np.stack(score_terms, axis=1), axis=1) + 0.7 * rng.normal(size=samples).astype(np.float32)
    y = (score > np.median(score)).astype(np.float32)
    x_train, x_val, y_train, y_val = _scaled_split(x, y, seed)
    return _context_from_arrays(
        name="block_structure_classification",
        family="structure",
        problem_type="binary_classification",
        model=SmallMLP(features, 1, hidden=48).to(device),
        train_loader=_make_loader(x_train, y_train, 24, seed, device, torch.float32),
        val_loader=_make_loader(x_val, y_val, 96, seed + 1, device, torch.float32),
        criterion=nn.BCEWithLogitsLoss(),
        epochs=24,
        target_loss=0.42,
        target_accuracy=0.84,
    )


def _low_rank_matrix_objective(matrix: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    target: torch.Tensor = context.metadata["target_matrix"]
    residual = matrix - target
    structure_penalty = residual.mean(dim=0).pow(2).mean() + residual.mean(dim=1).pow(2).mean()
    oscillation = 0.08 * math.sin(0.05 * step)
    return residual.pow(2).mean() + (0.12 + oscillation) * structure_penalty


def _make_low_rank_matrix_objective(seed: int, device: torch.device) -> TaskContext:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(12, 2)).astype(np.float32)
    right = rng.normal(size=(2, 12)).astype(np.float32)
    target = torch.tensor(left @ right, dtype=torch.float32, device=device)
    context = _make_matrix_direct_context(
        name="low_rank_matrix_objective",
        family="structure",
        objective_fn=_low_rank_matrix_objective,
        device=device,
        epochs=22,
        target_loss=0.02,
        rows=12,
        cols=12,
        init_scale=0.35,
    )
    context.metadata["target_matrix"] = target
    return context


def _plateau_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    basin = 0.03 * (x.pow(2) + 0.5 * y.pow(2))
    well = 1.15 * torch.exp(-0.8 * ((x - 1.8).pow(2) + (y + 0.6).pow(2)))
    ridge = 0.06 * torch.exp(-0.4 * ((x + 1.3).pow(2) + (y - 1.0).pow(2)))
    return basin + ridge - well


def _harmonic_oscillator_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    return 0.5 * (1.0 * x.pow(2) + 2.25 * y.pow(2))


def _quadratic_bowl_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    return 1.4 * (x - 0.7).pow(2) + 0.45 * (y + 0.4).pow(2) + 0.18 * (x + 0.6 * y).pow(2)


def _rosenbrock_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    return 18.0 * (y - x.pow(2)).pow(2) + (1.0 - x).pow(2)


def _narrow_valley_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    valley_axis = y - 0.14 * x
    return 16.0 * valley_axis.pow(2) + 0.06 * (x - 0.9).pow(2)


def _oscillatory_valley_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    valley = 8.0 * (y - 0.35 * x).pow(2) + 0.35 * x.pow(2)
    ripple = 0.28 * torch.sin(6.0 * x) * torch.cos(4.0 * y)
    return valley + ripple


def _direction_reversal_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    switch_epoch = int(context.metadata.get("switch_epoch", max(1, context.epochs // 2)))
    if epoch < switch_epoch:
        target = torch.tensor([1.4, -0.9], device=coords.device, dtype=coords.dtype)
    else:
        target = torch.tensor([-1.2, 1.1], device=coords.device, dtype=coords.dtype)
    return ((coords - target) ** 2).sum() + 0.25 * (coords[0] + coords[1]).pow(2)


def _noisy_quadratic_objective(coords: torch.Tensor, epoch: int, step: int, context: TaskContext) -> torch.Tensor:
    x = coords[0]
    y = coords[1]
    quadratic = 0.9 * (x - 0.85).pow(2) + 0.35 * (y + 0.55).pow(2)
    ripple = 0.05 * torch.sin(7.0 * x + 0.2) + 0.04 * torch.cos(5.0 * y - 0.15)
    coupling = 0.08 * (x + 0.4 * y).pow(2)
    return quadratic + coupling + ripple


def _make_plateau_escape(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="plateau_escape_objective",
        family="stress",
        objective_fn=_plateau_objective,
        device=device,
        epochs=28,
        target_loss=-0.6,
        init_scale=0.15,
        num_steps=56,
    )


def _make_harmonic_oscillator(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="harmonic_oscillator_objective",
        family="hamiltonian",
        objective_fn=_harmonic_oscillator_objective,
        device=device,
        epochs=24,
        target_loss=0.02,
        init_scale=0.4,
        num_steps=48,
    )


def _make_quadratic_bowl(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="quadratic_bowl_objective",
        family="hamiltonian",
        objective_fn=_quadratic_bowl_objective,
        device=device,
        epochs=24,
        target_loss=0.03,
        init_scale=0.45,
        num_steps=48,
    )


def _make_rosenbrock(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="rosenbrock_valley",
        family="stress",
        objective_fn=_rosenbrock_objective,
        device=device,
        epochs=28,
        target_loss=0.06,
        init_scale=0.45,
        num_steps=56,
    )


def _make_narrow_valley(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="narrow_valley_objective",
        family="hamiltonian",
        objective_fn=_narrow_valley_objective,
        device=device,
        epochs=28,
        target_loss=0.04,
        init_scale=0.45,
        num_steps=56,
    )


def _make_oscillatory_valley(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="oscillatory_valley",
        family="stress",
        objective_fn=_oscillatory_valley_objective,
        device=device,
        epochs=28,
        target_loss=0.08,
        init_scale=0.5,
        num_steps=56,
    )


def _make_direction_reversal(seed: int, device: torch.device) -> TaskContext:
    context = _make_direct_context(
        name="direction_reversal_objective",
        family="stress",
        objective_fn=_direction_reversal_objective,
        device=device,
        epochs=26,
        target_loss=0.08,
        init_scale=0.35,
        num_steps=52,
    )
    context.metadata["switch_epoch"] = 13
    return context


def _make_noisy_quadratic(seed: int, device: torch.device) -> TaskContext:
    return _make_direct_context(
        name="noisy_quadratic_objective",
        family="stress",
        objective_fn=_noisy_quadratic_objective,
        device=device,
        epochs=26,
        target_loss=0.08,
        init_scale=0.4,
        num_steps=52,
    )


def build_task_registry() -> dict[str, Callable[[int, torch.device], TaskContext]]:
    registry = {
        "linear_regression": _make_linear_regression,
        "noisy_regression": _make_noisy_regression,
        "logistic_regression": _make_logistic_regression,
        "synthetic_classification": _make_synthetic_classification,
        "moons_mlp": _make_moons,
        "circles_mlp": _make_circles,
        "digits_logistic": _make_digits_logistic,
        "digits_mlp": _make_digits_mlp,
        "digits_cnn": _make_digits_cnn,
        "digits_cnn_label_noise": _make_digits_cnn_label_noise,
        "digits_cnn_input_noise": _make_digits_cnn_input_noise,
        "digits_autoencoder": _make_digits_autoencoder,
        "breast_cancer_mlp": _make_breast_cancer,
        "wine_mlp": _make_wine,
        "pinn_harmonic_oscillator": _make_pinn,
        "pinn_poisson_1d": _make_pinn_poisson,
        "pinn_poisson_1d_small_batch": _make_pinn_poisson_small_batch,
        "pinn_heat_equation": _make_pinn_heat,
        "pinn_heat_equation_noisy_initial": _make_pinn_heat_noisy_initial,
        "noisy_gradients_classification": _make_noisy_gradients,
        "sparse_gradients_linear": _make_sparse_gradients,
        "high_curvature_regression": _make_high_curvature,
        "saddle_objective": _make_saddle,
        "nonstationary_moons": _make_nonstationary,
        "small_batch_instability": _make_small_batch_instability,
        "overfit_small_wine": _make_overfit_small_wine,
        "stagnating_regression": _make_stagnating_regression,
        "unstable_deep_mlp": _make_unstable_deep_mlp,
        "loss_shock_classification": _make_loss_shock_classification,
        "label_noise_breast_cancer": _make_label_noise_breast_cancer,
        "conflicting_batches_classification": _make_conflicting_batches,
        "conflicting_gradient_toy": _make_conflicting_batches,
        "block_structure_classification": _make_block_structure_classification,
        "low_rank_matrix_objective": _make_low_rank_matrix_objective,
        "plateau_escape_objective": _make_plateau_escape,
        "harmonic_oscillator_objective": _make_harmonic_oscillator,
        "quadratic_bowl_objective": _make_quadratic_bowl,
        "rosenbrock_valley": _make_rosenbrock,
        "narrow_valley_objective": _make_narrow_valley,
        "oscillatory_valley": _make_oscillatory_valley,
        "direction_reversal_objective": _make_direction_reversal,
        "noisy_quadratic_objective": _make_noisy_quadratic,
    }
    if _torchvision_is_available():
        registry.update(
            {
                "mnist_small_cnn": _make_mnist_small_cnn,
                "mnist_deeper_cnn": _make_mnist_deeper_cnn,
                "fashion_mnist_small_cnn": _make_fashion_mnist_small_cnn,
                "fashion_mnist_deeper_cnn": _make_fashion_mnist_deeper_cnn,
                "cifar10_subset_small_cnn": _make_cifar10_subset_small_cnn,
                "cifar10_subset_resnetlike": _make_cifar10_subset_resnetlike,
            }
        )
    return registry
