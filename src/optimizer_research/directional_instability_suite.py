from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from optimizers.optimizer_utils import resolve_device

from .baselines import available_optimizer_names
from .benchmarking import run_benchmark_suite, run_tuning_suite
from .config import ensure_output_dir
from .reporting import (
    _load_trace_frames,
    _markdown_table,
    _plot_bar,
    _plot_heatmap,
    _plot_metric,
    aggregate_results,
    best_by_task,
    compute_meaningful_wins,
)
from .tasks import build_task_registry


DIRECTIONAL_TASKS = [
    "oscillatory_valley",
    "saddle_objective",
    "rosenbrock_valley",
    "narrow_valley_objective",
    "direction_reversal_objective",
    "noisy_quadratic_objective",
    "plateau_escape_objective",
    "conflicting_gradient_toy",
    "small_batch_instability",
    "label_noise_breast_cancer",
    "nonstationary_moons",
]

REQUIRED_BASELINES = [
    "sgd",
    "sgd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "lion",
    "real_hamiltonian_adam",
    "magneto_adam",
    "topological_adam",
]

OPTIONAL_MODERN_BASELINES = [
    "adabelief",
    "muon_hybrid",
    "sam_adamw",
    "asam_adamw",
    "schedulefree_adamw",
]

PRIMARY_OPTIMIZERS = [
    "magneto_hamiltonian_adam",
    "magneto_hamiltonian_adam_improved",
]


def _resolve_task_list(requested: list[str]) -> tuple[list[str], list[str]]:
    registry = build_task_registry()
    available = [task for task in requested if task in registry]
    skipped = [task for task in requested if task not in registry]
    return available, skipped


def _resolve_optimizer_list(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    optimizers = list(config.get("optimizers", PRIMARY_OPTIMIZERS + REQUIRED_BASELINES))
    if bool(config.get("include_optional_modern_baselines", True)):
        for name in config.get("optional_modern_optimizers", OPTIONAL_MODERN_BASELINES):
            if name not in optimizers:
                optimizers.append(name)
    return available_optimizer_names(optimizers)


def run_directional_instability_benchmark(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    task_names, skipped_tasks = _resolve_task_list(list(config.get("benchmark_tasks", DIRECTIONAL_TASKS)))
    optimizer_names, skipped_optimizers = _resolve_optimizer_list(config)
    run_tuning = bool(config.get("run_tuning", int(config.get("search_budget", 2)) > 1))

    if run_tuning:
        tuning_config = dict(config)
        tuning_config["device"] = str(device)
        tuning_config["optimizers"] = optimizer_names
        tuning_config["tuning_tasks"] = list(config.get("tuning_tasks", task_names))
        run_tuning_suite(tuning_config)

    benchmark_config = dict(config)
    benchmark_config["device"] = str(device)
    benchmark_config["optimizers"] = optimizer_names
    benchmark_config["benchmark_tasks"] = task_names
    benchmark_config["use_tuning_results"] = run_tuning
    frame = run_benchmark_suite(benchmark_config)
    frame.to_csv(output_dir / "benchmark_results.csv", index=False)

    metadata = {
        "device": str(device),
        "optimizers": optimizer_names,
        "skipped_optimizers": skipped_optimizers,
        "tasks": task_names,
        "skipped_tasks": skipped_tasks,
        "run_tuning": run_tuning,
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame


def export_directional_instability_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    metadata_path = output_path / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    comparison_targets = [
        "adamw",
        "rmsprop",
        "sgd_momentum",
        "lion",
        "real_hamiltonian_adam",
        "magneto_adam",
        "topological_adam",
        "adabelief",
        "muon_hybrid",
        "sam_adamw",
        "asam_adamw",
        "schedulefree_adamw",
    ]
    win_frames: list[pd.DataFrame] = []
    for optimizer_name in PRIMARY_OPTIMIZERS:
        if optimizer_name not in set(aggregated["optimizer"]):
            continue
        for baseline_name in comparison_targets:
            if baseline_name in set(aggregated["optimizer"]):
                win_frames.append(compute_meaningful_wins(aggregated, optimizer_name, baseline_name))
    win_frame = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame()
    win_frame.to_csv(output_path / "win_flags.csv", index=False)

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame)
    plot_optimizers = [name for name in ["magneto_hamiltonian_adam_improved", "magneto_hamiltonian_adam", "adamw", "rmsprop", "sgd_momentum"] if name in set(aggregated["optimizer"])]
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "stress_task_curves.png",
        title="Directional Instability Loss Curves",
        metric="train_loss",
        tasks=["oscillatory_valley", "direction_reversal_objective", "saddle_objective"],
        optimizers=plot_optimizers,
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "validation_accuracy_curves.png",
        title="Validation Accuracy Under Directional Instability",
        metric="val_accuracy",
        tasks=["label_noise_breast_cancer", "nonstationary_moons", "small_batch_instability"],
        optimizers=plot_optimizers,
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "direction_cosine_curves.png",
        title="Gradient-Momentum Cosine",
        metric="grad_momentum_cosine",
        tasks=["oscillatory_valley", "direction_reversal_objective"],
        optimizers=[name for name in plot_optimizers if "magneto" in name or name in {"adamw", "rmsprop"}],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "rotation_conflict_curves.png",
        title="Rotation and Conflict Signals",
        metric="rotation_score",
        tasks=["oscillatory_valley", "conflicting_gradient_toy"],
        optimizers=[name for name in plot_optimizers if "magneto" in name or name in {"adamw", "rmsprop"}],
    )
    runtime_rows = aggregated[aggregated["optimizer"].isin(plot_optimizers)][["task", "optimizer", "mean_runtime_per_step_ms"]]
    _plot_bar(runtime_rows, figure_dir / "runtime_comparison.png", "Runtime per Step", "task", "mean_runtime_per_step_ms", "optimizer")
    _plot_heatmap(aggregated[aggregated["optimizer"].isin(plot_optimizers + ["lion", "adabelief", "muon_hybrid"])], figure_dir / "win_loss_heatmap.png")

    def _best_row_for(optimizer_name: str) -> pd.Series | None:
        subset = aggregated[aggregated["optimizer"] == optimizer_name]
        if subset.empty:
            return None
        if subset["mean_best_val_accuracy"].notna().any():
            return subset.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
        return subset.sort_values(["mean_best_val_loss", "mean_runtime_per_step_ms"], ascending=[True, True]).iloc[0]

    cmo_row = _best_row_for("magneto_hamiltonian_adam")
    improved_row = _best_row_for("magneto_hamiltonian_adam_improved")
    adamw_row = _best_row_for("adamw")
    rmsprop_row = _best_row_for("rmsprop")
    sgdm_row = _best_row_for("sgd_momentum")

    cmo_vs_adamw = compute_meaningful_wins(aggregated, "magneto_hamiltonian_adam_improved", "adamw") if "magneto_hamiltonian_adam_improved" in set(aggregated["optimizer"]) and "adamw" in set(aggregated["optimizer"]) else pd.DataFrame()
    cmo_vs_rmsprop = compute_meaningful_wins(aggregated, "magneto_hamiltonian_adam_improved", "rmsprop") if "magneto_hamiltonian_adam_improved" in set(aggregated["optimizer"]) and "rmsprop" in set(aggregated["optimizer"]) else pd.DataFrame()
    cmo_vs_sgdm = compute_meaningful_wins(aggregated, "magneto_hamiltonian_adam_improved", "sgd_momentum") if "magneto_hamiltonian_adam_improved" in set(aggregated["optimizer"]) and "sgd_momentum" in set(aggregated["optimizer"]) else pd.DataFrame()

    report_lines = [
        "# Directional Instability Benchmark",
        "",
        "## Narrow Claim",
        "Coherent Momentum Optimizer is a specialist optimizer for training regimes where gradient direction is unreliable: oscillation, reversal, conflict, noisy small-batch drift, or saddle and narrow-valley dynamics.",
        "",
        "## Benchmark Scope",
        "- Tasks: " + ", ".join(metadata.get("tasks", DIRECTIONAL_TASKS)),
        "- Optimizers: " + ", ".join(metadata.get("optimizers", [])),
        "- Skipped optional optimizers: " + (json.dumps(metadata.get("skipped_optimizers", {}), sort_keys=True) if metadata.get("skipped_optimizers") else "none"),
        "",
        "## Best By Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
        "",
        "## Best Rows",
    ]
    for label, row in [
        ("CMO current", cmo_row),
        ("CMO improved", improved_row),
        ("AdamW", adamw_row),
        ("RMSProp", rmsprop_row),
        ("SGD momentum", sgdm_row),
    ]:
        if row is None:
            continue
        report_lines.append(
            f"- {label}: `{row['task']}` | best val loss `{row['mean_best_val_loss']:.6f}` | best val acc `{row['mean_best_val_accuracy'] if pd.notna(row['mean_best_val_accuracy']) else 'nan'}`"
        )
    report_lines += [
        "",
        "## Focused Comparison",
        f"- Improved CMO meaningful wins vs AdamW: `{int(cmo_vs_adamw['win'].sum()) if not cmo_vs_adamw.empty else 0}`",
        f"- Improved CMO meaningful wins vs RMSProp: `{int(cmo_vs_rmsprop['win'].sum()) if not cmo_vs_rmsprop.empty else 0}`",
        f"- Improved CMO meaningful wins vs SGD momentum: `{int(cmo_vs_sgdm['win'].sum()) if not cmo_vs_sgdm.empty else 0}`",
        "",
        "## Honest Read",
        "- This benchmark is the strongest narrow proof story for the optimizer family.",
        "- If AdamW, RMSProp, or SGD momentum win more tasks here, that is a real result and should not be hidden.",
        "- The optimizer should be described as a specialist unless it wins across these instability families broadly.",
    ]
    report_text = "\n".join(report_lines)
    (output_path / "final_report.md").write_text(report_text, encoding="utf-8")
    return {
        "aggregated": aggregated,
        "best_by_task": best_frame,
        "win_flags": win_frame,
    }
