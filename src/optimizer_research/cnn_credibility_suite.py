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


BASE_CNN_TASKS = [
    "digits_cnn",
    "digits_cnn_label_noise",
    "digits_cnn_input_noise",
    "mnist_small_cnn",
    "mnist_deeper_cnn",
    "fashion_mnist_small_cnn",
    "fashion_mnist_deeper_cnn",
]

GPU_CNN_TASKS = [
    "cifar10_subset_small_cnn",
    "cifar10_subset_resnetlike",
]

CNN_BASELINES = [
    "sgd_momentum",
    "rmsprop",
    "adamw",
]

OPTIONAL_CNN_BASELINES = [
    "schedulefree_adamw",
]

PRIMARY_CNN_OPTIMIZERS = [
    "coherent_momentum_optimizer",
    "coherent_momentum_optimizer_improved",
]


def _resolve_cnn_tasks(device: str) -> tuple[list[str], list[str]]:
    registry = build_task_registry()
    requested = list(BASE_CNN_TASKS)
    if device == "cuda":
        requested.extend(GPU_CNN_TASKS)
    available = [task for task in requested if task in registry]
    skipped = [task for task in requested if task not in registry]
    return available, skipped


def run_cnn_credibility_benchmark(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    task_names, skipped_tasks = _resolve_cnn_tasks(device.type)
    run_tuning = bool(config.get("run_tuning", int(config.get("search_budget", 2)) > 1))

    requested_optimizers = list(config.get("optimizers", PRIMARY_CNN_OPTIMIZERS + CNN_BASELINES))
    if bool(config.get("include_optional_modern_baselines", True)):
        for name in config.get("optional_modern_optimizers", OPTIONAL_CNN_BASELINES):
            if name not in requested_optimizers:
                requested_optimizers.append(name)
    optimizer_names, skipped_optimizers = available_optimizer_names(requested_optimizers)

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
        "tasks": task_names,
        "skipped_tasks": skipped_tasks,
        "optimizers": optimizer_names,
        "skipped_optimizers": skipped_optimizers,
        "run_tuning": run_tuning,
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame


def export_cnn_credibility_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    metadata_path = output_path / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    cmo_vs_adamw = compute_meaningful_wins(aggregated, "coherent_momentum_optimizer_improved", "adamw") if {"coherent_momentum_optimizer_improved", "adamw"}.issubset(set(aggregated["optimizer"])) else pd.DataFrame()
    cmo_vs_rmsprop = compute_meaningful_wins(aggregated, "coherent_momentum_optimizer_improved", "rmsprop") if {"coherent_momentum_optimizer_improved", "rmsprop"}.issubset(set(aggregated["optimizer"])) else pd.DataFrame()
    cmo_vs_sgdm = compute_meaningful_wins(aggregated, "coherent_momentum_optimizer_improved", "sgd_momentum") if {"coherent_momentum_optimizer_improved", "sgd_momentum"}.issubset(set(aggregated["optimizer"])) else pd.DataFrame()

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame)
    plot_optimizers = [name for name in ["coherent_momentum_optimizer_improved", "coherent_momentum_optimizer", "adamw", "rmsprop", "sgd_momentum"] if name in set(aggregated["optimizer"])]
    focus_tasks = [task for task in ["digits_cnn", "digits_cnn_label_noise", "digits_cnn_input_noise", "mnist_small_cnn", "fashion_mnist_small_cnn"] if task in set(benchmark_frame["task"])]
    if focus_tasks:
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "validation_accuracy_curves.png",
            title="CNN Validation Accuracy",
            metric="val_accuracy",
            tasks=focus_tasks[:3],
            optimizers=plot_optimizers,
            event="val",
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "validation_loss_curves.png",
            title="CNN Validation Loss",
            metric="val_loss",
            tasks=focus_tasks[:3],
            optimizers=plot_optimizers,
            event="val",
        )
    runtime_rows = aggregated[aggregated["optimizer"].isin(plot_optimizers)][["task", "optimizer", "mean_runtime_per_step_ms"]]
    _plot_bar(runtime_rows, figure_dir / "runtime_comparison.png", "CNN Runtime per Step", "task", "mean_runtime_per_step_ms", "optimizer")
    memory_rows = aggregated[aggregated["optimizer"].isin(plot_optimizers)][["task", "optimizer", "mean_peak_device_memory_mb"]]
    _plot_bar(memory_rows, figure_dir / "memory_comparison.png", "CNN Peak Device Memory", "task", "mean_peak_device_memory_mb", "optimizer")
    _plot_heatmap(aggregated[aggregated["optimizer"].isin(plot_optimizers)], figure_dir / "win_loss_heatmap.png")

    report_lines = [
        "# CNN Credibility Benchmark",
        "",
        "## Benchmark Scope",
        "- Tasks run: " + ", ".join(metadata.get("tasks", [])),
        "- Skipped tasks: " + (", ".join(metadata.get("skipped_tasks", [])) if metadata.get("skipped_tasks") else "none"),
        "- Optimizers: " + ", ".join(metadata.get("optimizers", [])),
        "- Skipped optional optimizers: " + (json.dumps(metadata.get("skipped_optimizers", {}), sort_keys=True) if metadata.get("skipped_optimizers") else "none"),
        "",
        "## Best By Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
        "",
        "## CMO vs Standard CNN Baselines",
        f"- Improved CMO wins vs AdamW: `{int(cmo_vs_adamw['win'].sum()) if not cmo_vs_adamw.empty else 0}`",
        f"- Improved CMO wins vs RMSProp: `{int(cmo_vs_rmsprop['win'].sum()) if not cmo_vs_rmsprop.empty else 0}`",
        f"- Improved CMO wins vs SGD momentum: `{int(cmo_vs_sgdm['win'].sum()) if not cmo_vs_sgdm.empty else 0}`",
        "",
        "## Honest Read",
        "- This benchmark exists to test the known weak point of the optimizer family.",
        "- If AdamW, RMSProp, or SGD momentum still dominate the CNN rows, the gap is still open.",
        "- Digits and optional torchvision tasks should be read as credibility checks, not as a claim of broad vision strength.",
    ]
    (output_path / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "aggregated": aggregated,
        "best_by_task": best_frame,
    }
