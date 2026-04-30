from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.benchmarking import _train_single_run  # noqa: E402
from optimizer_research.real_hamiltonian_suite import (  # noqa: E402
    export_real_hamiltonian_report,
    run_real_hamiltonian_ablation,
)
from optimizer_research.reporting import aggregate_results, compute_meaningful_wins  # noqa: E402
from optimizers import HamiltonianAdamReal  # noqa: E402
from optimizers.optimizer_utils import set_global_seed  # noqa: E402


class TinyRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(16, 4)
    y = 0.4 * x[:, :1] - 0.2 * x[:, 1:2] + 0.1
    return x, y


def test_real_hamiltonian_adam_initializes_and_steps() -> None:
    set_global_seed(5)
    model = TinyRegressor()
    optimizer = HamiltonianAdamReal(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    before = [param.detach().clone() for param in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    after = [param.detach().clone() for param in model.parameters()]
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_real_hamiltonian_leapfrog_closure_recomputes_gradient() -> None:
    set_global_seed(11)
    model = TinyRegressor()
    optimizer = HamiltonianAdamReal(model.parameters(), lr=0.03, mode="leapfrog_with_closure")
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        closure_loss = criterion(model(x), y)
        optimizer.set_current_loss(closure_loss.item())
        closure_loss.backward()
        return closure_loss

    optimizer.step(closure)
    latest = optimizer.latest_diagnostics()
    assert latest["closure_recomputed_gradient"] == 1.0
    assert latest["leapfrog_enabled"] == 1.0


def test_real_hamiltonian_leapfrog_without_closure_falls_back_cleanly() -> None:
    set_global_seed(13)
    model = TinyRegressor()
    optimizer = HamiltonianAdamReal(model.parameters(), lr=0.03, mode="leapfrog_with_closure")
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    latest = optimizer.latest_diagnostics()
    assert latest["closure_recomputed_gradient"] == 0.0
    assert latest["leapfrog_enabled"] == 0.0
    assert latest["symplectic_euler_approximation"] == 1.0


def test_real_hamiltonian_no_nans_and_state_dict_round_trip() -> None:
    set_global_seed(17)
    model = TinyRegressor()
    optimizer = HamiltonianAdamReal(model.parameters(), lr=0.03, mode="dissipative_hamiltonian")
    criterion = torch.nn.MSELoss()
    for _ in range(4):
        x, y = _make_batch()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()
    for param in model.parameters():
        assert torch.isfinite(param).all()
    state = optimizer.state_dict()
    reloaded_model = TinyRegressor()
    reloaded_optimizer = HamiltonianAdamReal(reloaded_model.parameters(), lr=0.03, mode="dissipative_hamiltonian")
    reloaded_optimizer.load_state_dict(state)
    assert "physical_global_state" in reloaded_optimizer.state_dict()


def test_real_hamiltonian_diagnostics_exist() -> None:
    set_global_seed(23)
    model = TinyRegressor()
    optimizer = HamiltonianAdamReal(model.parameters(), lr=0.03)
    criterion = torch.nn.MSELoss()
    x, y = _make_batch()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()
    latest = optimizer.latest_diagnostics()
    expected = {
        "kinetic_energy",
        "potential_energy",
        "total_hamiltonian",
        "energy_drift",
        "relative_energy_drift",
        "inverse_mass_mean",
        "inverse_mass_std",
        "mass_trust",
        "mass_shock",
    }
    assert expected.issubset(latest.keys())


def test_real_hamiltonian_benchmark_schema(tmp_path: Path) -> None:
    row = _train_single_run(
        suite_name="real_hamiltonian_schema",
        task_name="harmonic_oscillator_objective",
        optimizer_name="real_hamiltonian_adam",
        hyperparameters={},
        seed=11,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        save_trace=True,
        epoch_scale=0.4,
    )
    expected = {
        "mean_energy_drift",
        "mean_relative_energy_drift",
        "mean_total_hamiltonian",
        "mean_inverse_mass_mean",
    }
    assert expected.issubset(row.keys())


def test_real_hamiltonian_ablation_schema(tmp_path: Path) -> None:
    frame = run_real_hamiltonian_ablation(
        {
            "output_dir": str(tmp_path),
            "device": "cpu",
            "seeds": [11],
            "ablation_tasks": ["harmonic_oscillator_objective"],
            "ablation_epoch_scale": 0.35,
        }
    )
    assert not frame.empty
    assert {"variant_name", "reference_optimizer", "variant_overrides"}.issubset(frame.columns)


def test_real_hamiltonian_report_generation(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "task": "demo_task",
                "optimizer": name,
                "task_family": "demo",
                "problem_type": "regression",
                "final_val_loss": 0.5 if name != "real_hamiltonian_adam" else 0.4,
                "best_val_loss": 0.5 if name != "real_hamiltonian_adam" else 0.4,
                "final_val_accuracy": float("nan"),
                "best_val_accuracy": float("nan"),
                "steps_to_target_loss": 10,
                "steps_to_target_accuracy": float("nan"),
                "training_stability": 1.0,
                "loss_variance": 0.1,
                "gradient_norm_stability": 0.1,
                "update_norm_stability": 0.1,
                "generalization_gap": 0.0,
                "runtime_seconds": 0.1,
                "diverged": 0,
                "gradient_norm_variance": 0.1,
                "update_norm_variance": 0.1,
                "mean_oscillation_score": 0.1,
                "mean_energy_drift": 0.02,
                "mean_relative_energy_drift": 0.01 if name == "real_hamiltonian_adam" else 0.03,
                "mean_normalized_total_energy": 0.2,
                "mean_kinetic_energy": 0.1,
                "mean_potential_energy": 0.2,
                "mean_total_hamiltonian": 0.3,
                "mean_momentum_norm": 0.4,
                "mean_parameter_step_norm": 0.05,
                "mean_inverse_mass_mean": 0.9,
                "mean_inverse_mass_std": 0.1,
                "mean_effective_damping": 0.02,
                "mean_alignment_scale": float("nan"),
                "mean_effective_lr_scale": float("nan"),
                "seed": 11,
                "selection_score": -0.4 if name == "real_hamiltonian_adam" else -0.5,
                "trace_path": None,
            }
            for name in ["hamiltonian_adam", "real_hamiltonian_adam", "adamw", "rmsprop", "topological_adam"]
        ]
    )
    rows.to_csv(tmp_path / "benchmark_results.csv", index=False)
    rows.to_csv(tmp_path / "energy_tests.csv", index=False)
    pd.DataFrame(
        [
            {"variant_name": "real_full", "selection_score": -0.4},
            {"variant_name": "no_leapfrog_closure", "selection_score": -0.5},
        ]
    ).to_csv(tmp_path / "ablation_results.csv", index=False)
    export_real_hamiltonian_report(tmp_path)
    assert (tmp_path / "final_report.md").exists()


def test_real_hamiltonian_no_hardcoded_wins() -> None:
    aggregated = aggregate_results(
        pd.DataFrame(
            [
                {
                    "task": "demo",
                    "optimizer": "real_hamiltonian_adam",
                    "task_family": "demo",
                    "problem_type": "regression",
                    "final_val_loss": 1.2,
                    "best_val_loss": 1.2,
                    "final_val_accuracy": float("nan"),
                    "best_val_accuracy": float("nan"),
                    "steps_to_target_loss": 20,
                    "steps_to_target_accuracy": float("nan"),
                    "training_stability": 1.0,
                    "loss_variance": 0.4,
                    "gradient_norm_stability": 0.2,
                    "update_norm_stability": 0.2,
                    "generalization_gap": 0.0,
                    "runtime_seconds": 0.1,
                    "diverged": 0,
                },
                {
                    "task": "demo",
                    "optimizer": "adamw",
                    "task_family": "demo",
                    "problem_type": "regression",
                    "final_val_loss": 0.8,
                    "best_val_loss": 0.8,
                    "final_val_accuracy": float("nan"),
                    "best_val_accuracy": float("nan"),
                    "steps_to_target_loss": 10,
                    "steps_to_target_accuracy": float("nan"),
                    "training_stability": 1.0,
                    "loss_variance": 0.1,
                    "gradient_norm_stability": 0.1,
                    "update_norm_stability": 0.1,
                    "generalization_gap": 0.0,
                    "runtime_seconds": 0.1,
                    "diverged": 0,
                },
            ]
        )
    )
    wins = compute_meaningful_wins(aggregated, "real_hamiltonian_adam", "adamw")
    assert bool(wins.iloc[0]["win"]) is False
