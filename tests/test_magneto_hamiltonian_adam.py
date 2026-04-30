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
from optimizer_research.magneto_hamiltonian_suite import (  # noqa: E402
    export_magneto_hamiltonian_report,
    run_magneto_hamiltonian_ablation,
)
from optimizer_research.reporting import aggregate_results, compute_meaningful_wins  # noqa: E402
from optimizers import (  # noqa: E402
    CoherentMomentumOptimizer,
    CoherentMomentumOptimizerImproved,
    HamiltonianAdamReal,
    MagnetoHamiltonianAdam,
    MagnetoHamiltonianAdamImproved,
)
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
    y = 0.5 * x[:, :1] - 0.3 * x[:, 1:2] + 0.2
    return x, y


def test_magneto_hamiltonian_initializes_and_steps() -> None:
    set_global_seed(5)
    model = TinyRegressor()
    optimizer = MagnetoHamiltonianAdam(model.parameters(), lr=0.03)
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


def test_coherent_momentum_aliases_import_and_step() -> None:
    set_global_seed(6)
    model = TinyRegressor()
    optimizer = CoherentMomentumOptimizer(model.parameters(), lr=0.03)
    improved = CoherentMomentumOptimizerImproved(model.parameters(), lr=0.02, diagnostics_every_n_steps=2)
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
    assert isinstance(improved, MagnetoHamiltonianAdamImproved)


def test_magneto_hamiltonian_improved_initializes_and_steps() -> None:
    set_global_seed(7)
    model = TinyRegressor()
    optimizer = MagnetoHamiltonianAdamImproved(model.parameters(), lr=0.03, diagnostics_every_n_steps=2)
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
    latest = optimizer.latest_diagnostics()
    assert latest["optimizer"] == "MagnetoHamiltonianAdamImproved"
    assert latest["step"] == 1


def test_magneto_hamiltonian_neutral_is_close_to_real_hamiltonian() -> None:
    set_global_seed(11)
    x, y = _make_batch()
    criterion = torch.nn.MSELoss()
    model_a = TinyRegressor()
    model_b = TinyRegressor()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = HamiltonianAdamReal(model_a.parameters(), lr=0.03)
    opt_b = MagnetoHamiltonianAdam(
        model_b.parameters(),
        lr=0.03,
        alignment_strength=0.0,
        coherence_strength=0.0,
        conflict_damping=0.0,
        rotation_penalty=0.0,
        projection_strength=0.0,
        max_projection=0.0,
        min_alignment_scale=1.0,
        max_alignment_scale=1.0,
    )

    opt_a.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    loss_a = criterion(model_a(x), y)
    loss_b = criterion(model_b(x), y)
    opt_a.set_current_loss(loss_a.item())
    opt_b.set_current_loss(loss_b.item())
    loss_a.backward()
    loss_b.backward()
    opt_a.step()
    opt_b.step()

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(param_a, param_b, atol=1e-6, rtol=1e-5)


def test_magneto_hamiltonian_leapfrog_closure_recomputes_gradient() -> None:
    set_global_seed(13)
    model = TinyRegressor()
    optimizer = MagnetoHamiltonianAdam(model.parameters(), lr=0.03, mode="leapfrog_with_closure")
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
    assert latest["parameter_step_norm"] > 0.0


def test_magneto_hamiltonian_no_nans_and_state_dict_round_trip() -> None:
    set_global_seed(17)
    model = TinyRegressor()
    optimizer = MagnetoHamiltonianAdam(model.parameters(), lr=0.03)
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
    reloaded_optimizer = MagnetoHamiltonianAdam(reloaded_model.parameters(), lr=0.03)
    reloaded_optimizer.load_state_dict(state)
    assert "physical_global_state" in reloaded_optimizer.state_dict()


def test_magneto_hamiltonian_diagnostics_exist() -> None:
    set_global_seed(23)
    model = TinyRegressor()
    optimizer = MagnetoHamiltonianAdam(model.parameters(), lr=0.03)
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
        "relative_energy_drift",
        "grad_momentum_cosine",
        "force_momentum_cosine",
        "rotation_score",
        "rotation_gate",
        "magneto_activation",
        "alignment_scale",
        "magneto_projection_strength",
    }
    assert expected.issubset(latest.keys())


def test_magneto_hamiltonian_benchmark_schema(tmp_path: Path) -> None:
    row = _train_single_run(
        suite_name="mh_schema",
        task_name="oscillatory_valley",
        optimizer_name="magneto_hamiltonian_adam",
        hyperparameters={},
        seed=11,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        save_trace=True,
        epoch_scale=0.4,
    )
    expected = {
        "mean_relative_energy_drift",
        "mean_total_hamiltonian",
        "mean_alignment_scale",
        "mean_grad_momentum_cosine",
        "mean_rotation_score",
        "mean_magneto_activation",
    }
    assert expected.issubset(row.keys())


def test_magneto_hamiltonian_ablation_schema(tmp_path: Path) -> None:
    frame = run_magneto_hamiltonian_ablation(
        {
            "output_dir": str(tmp_path),
            "device": "cpu",
            "seeds": [11],
            "ablation_tasks": ["oscillatory_valley"],
            "ablation_epoch_scale": 0.35,
        }
    )
    assert not frame.empty
    assert {"variant_name", "reference_optimizer", "variant_overrides"}.issubset(frame.columns)


def test_magneto_hamiltonian_report_generation(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "task": "demo_task",
                "optimizer": name,
                "task_family": "demo",
                "problem_type": "regression",
                "final_val_loss": 0.5 if name != "magneto_hamiltonian_adam" else 0.35,
                "best_val_loss": 0.5 if name != "magneto_hamiltonian_adam" else 0.35,
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
                "mean_energy_drift": 0.01,
                "mean_relative_energy_drift": 0.005 if name == "magneto_hamiltonian_adam" else 0.02,
                "mean_normalized_total_energy": 0.2,
                "mean_kinetic_energy": 0.1,
                "mean_potential_energy": 0.2,
                "mean_total_hamiltonian": 0.3,
                "mean_momentum_norm": 0.4,
                "mean_parameter_step_norm": 0.05,
                "mean_inverse_mass_mean": 0.95,
                "mean_inverse_mass_std": 0.08,
                "mean_effective_damping": 0.02,
                "mean_alignment_scale": 1.02 if name == "magneto_hamiltonian_adam" else 1.0,
                "mean_grad_momentum_cosine": 0.2 if name == "magneto_hamiltonian_adam" else float("nan"),
                "mean_rotation_score": 0.1 if name == "magneto_hamiltonian_adam" else float("nan"),
                "seed": 11,
                "selection_score": -0.35 if name == "magneto_hamiltonian_adam" else -0.5,
                "trace_path": None,
            }
            for name in [
                "magneto_hamiltonian_adam",
                "real_hamiltonian_adam",
                "magneto_adam",
                "hamiltonian_adam",
                "adamw",
                "rmsprop",
                "topological_adam",
            ]
        ]
    )
    rows.to_csv(tmp_path / "benchmark_results.csv", index=False)
    rows.to_csv(tmp_path / "energy_tests.csv", index=False)
    pd.DataFrame(
        [
            {"variant_name": "combined_full", "selection_score": -0.35},
            {"variant_name": "no_projection", "selection_score": -0.45},
        ]
    ).to_csv(tmp_path / "ablation_results.csv", index=False)
    export_magneto_hamiltonian_report(tmp_path)
    assert (tmp_path / "final_report.md").exists()


def test_magneto_hamiltonian_no_hardcoded_wins() -> None:
    aggregated = aggregate_results(
        pd.DataFrame(
            [
                {
                    "task": "demo",
                    "optimizer": "magneto_hamiltonian_adam",
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
                    "steps_to_target_loss": 12,
                    "steps_to_target_accuracy": float("nan"),
                    "training_stability": 1.0,
                    "loss_variance": 0.2,
                    "gradient_norm_stability": 0.1,
                    "update_norm_stability": 0.1,
                    "generalization_gap": 0.0,
                    "runtime_seconds": 0.1,
                    "diverged": 0,
                },
            ]
        )
    )
    wins = compute_meaningful_wins(aggregated, "magneto_hamiltonian_adam", "adamw")
    assert not bool(wins["win"].iloc[0])
