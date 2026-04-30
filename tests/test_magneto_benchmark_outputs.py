from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.magneto_gpu_suite import (  # noqa: E402
    export_magneto_gpu_report,
    run_magneto_gpu_ablation,
    run_magneto_gpu_benchmarks,
    run_magneto_gpu_cnn_benchmarks,
    run_magneto_gpu_multitask_benchmarks,
    run_magneto_gpu_smoke,
    run_magneto_gpu_stress_benchmarks,
)
from optimizer_research import (  # noqa: E402
    export_cnn_credibility_report,
    export_directional_instability_report,
    run_cnn_credibility_benchmark,
    run_directional_instability_benchmark,
)


def _load_export_script_module():
    script_path = ROOT / "scripts" / "export_magneto_hamiltonian_adam_report.py"
    spec = importlib.util.spec_from_file_location("export_magneto_hamiltonian_adam_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_magneto_gpu_suite_writes_expected_outputs(tmp_path: Path) -> None:
    base_config = {
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seeds": [11],
        "search_budget": 1,
        "optimizers": [
            "magneto_hamiltonian_adam",
            "magneto_hamiltonian_adam_improved",
            "adamw",
            "rmsprop",
            "sgd_momentum",
        ],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.25,
        "benchmark_epoch_scale": 0.3,
        "stress_epoch_scale": 0.3,
        "ablation_epoch_scale": 0.25,
    }

    run_magneto_gpu_smoke(
        {
            **base_config,
            "smoke_tasks": ["linear_regression", "breast_cancer_mlp"],
            "smoke_optimizers": ["magneto_hamiltonian_adam", "magneto_hamiltonian_adam_improved", "adamw"],
        }
    )
    run_magneto_gpu_benchmarks(
        {
            **base_config,
            "tuning_tasks": ["breast_cancer_mlp"],
            "benchmark_tasks": ["breast_cancer_mlp", "wine_mlp"],
        }
    )
    run_magneto_gpu_cnn_benchmarks(
        {
            **base_config,
            "tuning_tasks": ["digits_cnn"],
            "benchmark_tasks": ["digits_cnn"],
        }
    )
    run_magneto_gpu_stress_benchmarks(
        {
            **base_config,
            "tuning_tasks": ["oscillatory_valley"],
            "stress_tasks": ["oscillatory_valley", "saddle_objective"],
        }
    )
    run_magneto_gpu_multitask_benchmarks(
        {
            **base_config,
            "tuning_tasks": ["conflicting_batches_classification"],
            "benchmark_tasks": ["conflicting_batches_classification"],
        }
    )
    run_magneto_gpu_ablation(
        {
            **base_config,
            "ablation_tasks": ["oscillatory_valley", "conflicting_batches_classification"],
        }
    )
    export_magneto_gpu_report(tmp_path)

    expected_files = [
        "gpu_smoke_results.csv",
        "gpu_benchmark_results.csv",
        "gpu_cnn_results.csv",
        "gpu_stress_results.csv",
        "gpu_multitask_results.csv",
        "gpu_ablation_results.csv",
        "runtime_memory_results.csv",
        "best_by_task.csv",
        "win_flags.csv",
        "final_magneto_gpu_report.md",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists(), filename

    benchmark_frame = pd.read_csv(tmp_path / "gpu_benchmark_results.csv")
    assert {"optimizer", "device", "device_name", "optimizer_step_time_ms", "peak_device_memory_mb"}.issubset(benchmark_frame.columns)

    ablation_frame = pd.read_csv(tmp_path / "gpu_ablation_results.csv")
    assert {"variant_name", "reference_optimizer", "variant_overrides"}.issubset(ablation_frame.columns)


def test_directional_instability_suite_writes_expected_outputs(tmp_path: Path) -> None:
    config = {
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seeds": [11],
        "tuning_seeds": [11],
        "search_budget": 1,
        "optimizers": [
            "magneto_hamiltonian_adam",
            "magneto_hamiltonian_adam_improved",
            "adamw",
            "rmsprop",
            "sgd_momentum",
        ],
        "benchmark_tasks": ["oscillatory_valley", "direction_reversal_objective", "small_batch_instability"],
        "tuning_tasks": ["oscillatory_valley"],
        "include_optional_modern_baselines": False,
        "tuning_epoch_scale": 0.2,
        "benchmark_epoch_scale": 0.25,
    }
    run_directional_instability_benchmark(config)
    export_directional_instability_report(tmp_path)
    for filename in ["benchmark_results.csv", "best_by_task.csv", "win_flags.csv", "final_report.md"]:
        assert (tmp_path / filename).exists(), filename
    benchmark_frame = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert {"optimizer", "task", "best_val_loss", "runtime_per_step_ms"}.issubset(benchmark_frame.columns)


def test_cnn_credibility_suite_writes_expected_outputs(tmp_path: Path) -> None:
    config = {
        "output_dir": str(tmp_path),
        "device": "cpu",
        "seeds": [11],
        "tuning_seeds": [11],
        "search_budget": 1,
        "optimizers": [
            "magneto_hamiltonian_adam",
            "magneto_hamiltonian_adam_improved",
            "adamw",
            "rmsprop",
            "sgd_momentum",
        ],
        "benchmark_tasks": ["digits_cnn"],
        "tuning_tasks": ["digits_cnn"],
        "include_optional_modern_baselines": False,
        "tuning_epoch_scale": 0.2,
        "benchmark_epoch_scale": 0.25,
    }
    run_cnn_credibility_benchmark(config)
    export_cnn_credibility_report(tmp_path)
    for filename in ["benchmark_results.csv", "best_by_task.csv", "final_report.md"]:
        assert (tmp_path / filename).exists(), filename
    benchmark_frame = pd.read_csv(tmp_path / "benchmark_results.csv")
    assert {"optimizer", "task", "best_val_accuracy", "optimizer_step_time_ms"}.issubset(benchmark_frame.columns)


def test_mainline_export_script_falls_back_to_accepted_snapshot(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports" / "accepted_magneto_hamiltonian"
    reports_dir.mkdir(parents=True)
    for filename in ["benchmark_results.csv", "energy_tests.csv", "ablation_results.csv"]:
        (reports_dir / filename).write_text("placeholder\n")
    module = _load_export_script_module()
    resolved = module._resolve_export_dir(tmp_path)
    assert resolved == reports_dir
