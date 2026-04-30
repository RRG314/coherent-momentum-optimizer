from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizers import CoherentMomentumOptimizerImproved  # noqa: E402
from optimizers.optimizer_utils import resolve_device, set_global_seed  # noqa: E402


class TinyRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(24, 4, device=device)
    y = 0.5 * x[:, :1] - 0.25 * x[:, 1:2] + 0.15
    return x, y


def _gpu_like_device() -> torch.device:
    device = resolve_device("auto")
    if device.type == "cpu":
        pytest.skip("No GPU-like device available for compatibility test.")
    return device


def test_gpu_step_changes_parameters_and_state_stays_on_device() -> None:
    device = _gpu_like_device()
    set_global_seed(3)
    model = TinyRegressor().to(device)
    optimizer = CoherentMomentumOptimizerImproved(model.parameters(), lr=0.02, diagnostics_every_n_steps=4, preset="standard_safe")
    criterion = torch.nn.MSELoss()
    x, y = _make_batch(device)
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    after = [param.detach().clone() for param in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == device.type


def test_cpu_to_gpu_state_dict_load_works() -> None:
    device = _gpu_like_device()
    set_global_seed(5)
    cpu_model = TinyRegressor()
    cpu_optimizer = CoherentMomentumOptimizerImproved(cpu_model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    x_cpu, y_cpu = _make_batch(torch.device("cpu"))
    cpu_optimizer.zero_grad(set_to_none=True)
    cpu_loss = criterion(cpu_model(x_cpu), y_cpu)
    cpu_optimizer.set_current_loss(cpu_loss.item())
    cpu_loss.backward()
    cpu_optimizer.step()
    state_dict = cpu_optimizer.state_dict()

    gpu_model = TinyRegressor().to(device)
    gpu_optimizer = CoherentMomentumOptimizerImproved(gpu_model.parameters(), lr=0.02)
    gpu_optimizer.load_state_dict(state_dict)
    for state in gpu_optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == device.type


def test_gpu_to_cpu_state_dict_load_works() -> None:
    device = _gpu_like_device()
    set_global_seed(7)
    gpu_model = TinyRegressor().to(device)
    gpu_optimizer = CoherentMomentumOptimizerImproved(gpu_model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    x_gpu, y_gpu = _make_batch(device)
    gpu_optimizer.zero_grad(set_to_none=True)
    gpu_loss = criterion(gpu_model(x_gpu), y_gpu)
    gpu_optimizer.set_current_loss(gpu_loss.item())
    gpu_loss.backward()
    gpu_optimizer.step()
    state_dict = gpu_optimizer.state_dict()

    cpu_model = TinyRegressor()
    cpu_optimizer = CoherentMomentumOptimizerImproved(cpu_model.parameters(), lr=0.02)
    cpu_optimizer.load_state_dict(state_dict)
    for state in cpu_optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_gpu_smoke_has_no_nans() -> None:
    device = _gpu_like_device()
    set_global_seed(9)
    model = TinyRegressor().to(device)
    optimizer = CoherentMomentumOptimizerImproved(model.parameters(), lr=0.02, preset="stress_specialist")
    criterion = torch.nn.MSELoss()
    for _ in range(4):
        x, y = _make_batch(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()
    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_diagnostics_can_be_disabled_and_throttled() -> None:
    set_global_seed(13)
    model = TinyRegressor()
    optimizer = CoherentMomentumOptimizerImproved(
        model.parameters(),
        lr=0.02,
        enable_step_diagnostics=False,
        diagnostics_every_n_steps=8,
    )
    criterion = torch.nn.MSELoss()
    for _ in range(3):
        x, y = _make_batch(torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()
    assert optimizer.diagnostics_dataframe().empty
    latest = optimizer.latest_diagnostics()
    assert latest["step"] == 3
