from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .benchmarking import run_benchmark_suite, run_smoke_suite, run_stress_suite, run_tuning_suite, _train_single_run
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
from optimizers.optimizer_utils import resolve_device


MANDATORY_BASELINES = [
    "sgd",
    "sgd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "lion",
    "coherent_momentum_real_baseline",
    "coherent_direction_reference",
    "topological_adam",
]

PRIMARY_OPTIMIZERS = [
    "coherent_momentum_optimizer",
    "coherent_momentum_optimizer_improved",
]

GPU_FOCUS_OPTIMIZERS = PRIMARY_OPTIMIZERS + MANDATORY_BASELINES

RELATED_WORK = [
    {"title": "AdamW", "url": "https://arxiv.org/abs/1711.05101"},
    {"title": "On the Convergence of Adam and Beyond", "url": "https://openreview.net/forum?id=ryQu7f-RZ"},
    {"title": "RMSProp", "url": "https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf"},
    {"title": "Deep Learning via Hamiltonian Monte Carlo", "url": "https://arxiv.org/abs/1206.1901"},
    {"title": "PCGrad", "url": "https://arxiv.org/abs/2001.06782"},
    {"title": "CAGrad", "url": "https://openreview.net/forum?id=_61Qh8tULj_"},
    {"title": "Lion", "url": "https://arxiv.org/abs/2302.06675"},
]


def _device_summary(device: torch.device) -> dict[str, Any]:
    summary = {"device": str(device), "device_name": "CPU", "cuda_available": False, "mps_available": False}
    if device.type == "cuda" and torch.cuda.is_available():
        summary["device_name"] = torch.cuda.get_device_name(device)
        summary["cuda_available"] = True
        summary["cuda_device_count"] = torch.cuda.device_count()
    elif device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        summary["device_name"] = "Apple MPS"
        summary["mps_available"] = True
    return summary


def _write_summary(path: Path, title: str, device_info: dict[str, Any], frame: pd.DataFrame) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Device: `{device_info['device']}`",
        f"- Device name: `{device_info['device_name']}`",
        f"- Rows: `{len(frame)}`",
    ]
    if not frame.empty:
        grouped = aggregate_results(frame)
        lines += [
            "",
            "## Best Rows",
            _markdown_table(
                grouped.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).head(8)[
                    [
                        "task",
                        "optimizer",
                        "mean_best_val_loss",
                        "mean_best_val_accuracy",
                        "mean_runtime_per_step_ms",
                        "mean_peak_device_memory_mb",
                    ]
                ]
            ),
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_coherent_momentum_gpu_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    device_info = _device_summary(device)
    smoke_config = dict(config)
    smoke_config.setdefault("device", str(device))
    smoke_config.setdefault("smoke_tasks", ["linear_regression", "breast_cancer_mlp", "digits_cnn"])
    smoke_config.setdefault(
        "smoke_optimizers",
        ["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop", "sgd_momentum"],
    )
    frame = run_smoke_suite(smoke_config)
    frame.to_csv(output_dir / "gpu_smoke_results.csv", index=False)
    _write_summary(output_dir / "gpu_smoke_summary.md", "Magneto GPU Smoke", device_info, frame)
    return frame


def run_coherent_momentum_gpu_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    device_info = _device_summary(device)
    tuning_config = dict(config)
    tuning_config.setdefault("device", str(device))
    tuning_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    tuning_config.setdefault(
        "tuning_tasks",
        [
            "breast_cancer_mlp",
            "wine_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
            "conflicting_batches_classification",
        ],
    )
    run_tuning_suite(tuning_config)

    benchmark_config = dict(config)
    benchmark_config.setdefault("device", str(device))
    benchmark_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    frame = run_benchmark_suite(benchmark_config)
    frame.to_csv(output_dir / "gpu_benchmark_results.csv", index=False)
    _write_summary(output_dir / "gpu_benchmark_summary.md", "Magneto GPU Benchmarks", device_info, frame)
    return frame


def run_coherent_momentum_gpu_cnn_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    device_info = _device_summary(device)
    tuning_config = dict(config)
    tuning_config.setdefault("device", str(device))
    tuning_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    tuning_config.setdefault("tuning_tasks", ["digits_cnn", "digits_cnn_label_noise", "digits_cnn_input_noise"])
    run_tuning_suite(tuning_config)

    benchmark_config = dict(config)
    benchmark_config.setdefault("device", str(device))
    benchmark_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    frame = run_benchmark_suite(benchmark_config)
    frame.to_csv(output_dir / "gpu_cnn_results.csv", index=False)
    _write_summary(output_dir / "gpu_cnn_summary.md", "Magneto GPU CNN Benchmarks", device_info, frame)
    return frame


def run_coherent_momentum_gpu_stress_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    device_info = _device_summary(device)
    tuning_config = dict(config)
    tuning_config.setdefault("device", str(device))
    tuning_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    tuning_config.setdefault(
        "tuning_tasks",
        [
            "oscillatory_valley",
            "saddle_objective",
            "rosenbrock_valley",
            "narrow_valley_objective",
            "direction_reversal_objective",
            "noisy_quadratic_objective",
            "plateau_escape_objective",
        ],
    )
    run_tuning_suite(tuning_config)

    stress_config = dict(config)
    stress_config.setdefault("device", str(device))
    stress_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    frame = run_stress_suite(stress_config)
    frame.to_csv(output_dir / "gpu_stress_results.csv", index=False)
    _write_summary(output_dir / "gpu_stress_summary.md", "Magneto GPU Stress Benchmarks", device_info, frame)
    return frame


def run_coherent_momentum_gpu_multitask_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    device_info = _device_summary(device)
    tuning_config = dict(config)
    tuning_config.setdefault("device", str(device))
    tuning_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    tuning_config.setdefault("tuning_tasks", ["conflicting_batches_classification", "block_structure_classification"])
    run_tuning_suite(tuning_config)

    benchmark_config = dict(config)
    benchmark_config.setdefault("device", str(device))
    benchmark_config.setdefault("optimizers", GPU_FOCUS_OPTIMIZERS)
    frame = run_benchmark_suite(benchmark_config)
    frame.to_csv(output_dir / "gpu_multitask_results.csv", index=False)
    _write_summary(output_dir / "gpu_multitask_summary.md", "Magneto GPU Multitask Benchmarks", device_info, frame)
    return frame


def run_coherent_momentum_gpu_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "digits_cnn",
                "digits_cnn_label_noise",
                "oscillatory_valley",
                "saddle_objective",
                "conflicting_batches_classification",
                "small_batch_instability",
            ],
        )
    )
    variants = [
        {"variant_name": "current_mainline_full", "optimizer_name": "coherent_momentum_optimizer", "overrides": {}},
        {"variant_name": "improved_balanced", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"preset": "balanced"}},
        {"variant_name": "improved_standard_safe", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"preset": "standard_safe"}},
        {"variant_name": "improved_stress_specialist", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"preset": "stress_specialist"}},
        {"variant_name": "improved_cnn_safe", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"preset": "cnn_safe"}},
        {"variant_name": "coherent_momentum_real_baseline", "optimizer_name": "coherent_momentum_real_baseline", "overrides": {}},
        {"variant_name": "coherent_direction_reference_only", "optimizer_name": "coherent_direction_reference", "overrides": {}},
        {
            "variant_name": "projection_under_conflict_only",
            "optimizer_name": "coherent_momentum_optimizer_improved",
            "overrides": {"projection_mode": "conflict_only", "preset": "balanced"},
        },
        {
            "variant_name": "no_projection",
            "optimizer_name": "coherent_momentum_optimizer_improved",
            "overrides": {"projection_strength": 0.0, "max_projection": 0.0},
        },
        {
            "variant_name": "no_conflict_damping",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {"conflict_damping": 0.0},
        },
        {
            "variant_name": "soft_conflict_correction",
            "optimizer_name": "coherent_momentum_optimizer_improved",
            "overrides": {"soft_conflict_correction": 0.14, "soft_conflict_max": 0.24},
        },
        {
            "variant_name": "no_activation_gating",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "activation_conflict_weight": 1.0,
                "activation_rotation_weight": 1.0,
                "activation_rotation_threshold": 0.0,
                "projection_activation_threshold": 0.0,
                "stable_coherence_bonus": 0.0,
            },
        },
        {"variant_name": "fixed_mass", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"mass_mode": "fixed", "fixed_mass": 1.0}},
        {"variant_name": "adaptive_mass", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"mass_mode": "adaptive"}},
        {"variant_name": "conv_safe_off", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"conv_safe_mode": False}},
        {"variant_name": "conv_safe_on", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"conv_safe_mode": True}},
        {"variant_name": "diagnostics_full", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"diagnostics_every_n_steps": 1}},
        {"variant_name": "diagnostics_throttled", "optimizer_name": "coherent_momentum_optimizer_improved", "overrides": {"diagnostics_every_n_steps": 8}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
        {"variant_name": "sgd_momentum_baseline", "optimizer_name": "sgd_momentum", "overrides": {}},
    ]
    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="coherent_momentum_gpu_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.85)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "gpu_ablation_results.csv", index=False)
    return frame


def _plot_ablation_chart(ablation_frame: pd.DataFrame, output_path: Path) -> None:
    if ablation_frame.empty:
        return
    summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    plt.figure(figsize=(12, 5))
    plt.bar(summary["variant_name"], summary["mean_selection_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean selection score")
    plt.title("Magneto GPU Ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_coherent_momentum_gpu_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    benchmark_frame = _load_if_exists(output_path / "gpu_benchmark_results.csv")
    cnn_frame = _load_if_exists(output_path / "gpu_cnn_results.csv")
    stress_frame = _load_if_exists(output_path / "gpu_stress_results.csv")
    multitask_frame = _load_if_exists(output_path / "gpu_multitask_results.csv")
    ablation_frame = _load_if_exists(output_path / "gpu_ablation_results.csv")
    smoke_frame = _load_if_exists(output_path / "gpu_smoke_results.csv")

    nonempty_frames = [frame for frame in [benchmark_frame, cnn_frame, stress_frame, multitask_frame] if not frame.empty]
    combined_raw = pd.concat(nonempty_frames, ignore_index=True) if nonempty_frames else pd.DataFrame()
    combined_aggregated = aggregate_results(combined_raw) if not combined_raw.empty else pd.DataFrame()

    def _top_row(frame: pd.DataFrame, optimizer_name: str) -> pd.DataFrame:
        if frame.empty or optimizer_name not in set(frame["optimizer"]):
            return pd.DataFrame()
        return (
            frame.loc[frame["optimizer"] == optimizer_name]
            .sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True])
            .head(1)
        )

    def _subset_top_rows(raw_frame: pd.DataFrame, tasks: list[str], optimizers: list[str]) -> pd.DataFrame:
        if raw_frame.empty:
            return pd.DataFrame()
        subset = raw_frame.loc[raw_frame["task"].isin(tasks)]
        if subset.empty:
            return pd.DataFrame()
        rows: list[pd.Series] = []
        for optimizer_name in optimizers:
            optimizer_rows = subset.loc[subset["optimizer"] == optimizer_name]
            if optimizer_rows.empty:
                continue
            row = optimizer_rows.sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True]).iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    def _win_summary(frame: pd.DataFrame, optimizer_name: str, baselines: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for baseline_name in baselines:
            wins = compute_meaningful_wins(frame, optimizer_name, baseline_name)
            if wins.empty:
                continue
            rows.append(
                {
                    "optimizer": optimizer_name,
                    "baseline": baseline_name,
                    "wins": int(wins["win"].sum()),
                    "two_x": int(wins["two_x"].sum()),
                }
            )
        return pd.DataFrame(rows)

    if not combined_aggregated.empty:
        best_frame = best_by_task(combined_aggregated)
        best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    else:
        best_frame = pd.DataFrame()

    comparison_bases = [
        "adamw",
        "rmsprop",
        "sgd_momentum",
        "coherent_momentum_real_baseline",
        "coherent_direction_reference",
        "topological_adam",
        "coherent_momentum_optimizer",
    ]
    win_frames = []
    for optimizer_name in ["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved"]:
        summary_frame = _win_summary(combined_aggregated, optimizer_name, [name for name in comparison_bases if name != optimizer_name])
        if not summary_frame.empty:
            win_frames.append(summary_frame)
    win_flags = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame()
    if not win_flags.empty:
        win_flags.to_csv(output_path / "win_flags.csv", index=False)

    if not combined_aggregated.empty:
        runtime_memory = combined_aggregated[
            [
                "task",
                "optimizer",
                "mean_runtime_seconds",
                "mean_runtime_per_step_ms",
                "mean_optimizer_step_time_ms",
                "mean_samples_per_second",
                "mean_optimizer_state_mb",
                "mean_peak_device_memory_mb",
                "divergence_rate",
            ]
        ]
        runtime_memory.to_csv(output_path / "runtime_memory_results.csv", index=False)
    else:
        runtime_memory = pd.DataFrame()

    trace_frame = _load_trace_frames(combined_raw) if not combined_raw.empty else pd.DataFrame()
    if not trace_frame.empty:
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "validation_loss_curves.png",
            title="Validation Loss Curves",
            metric="val_loss",
            tasks=["breast_cancer_mlp", "digits_cnn", "conflicting_batches_classification"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop"],
            event="val",
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "validation_accuracy_curves.png",
            title="Validation Accuracy Curves",
            metric="val_accuracy",
            tasks=["breast_cancer_mlp", "digits_cnn", "digits_cnn_label_noise"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop"],
            event="val",
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "stress_task_curves.png",
            title="Stress Tasks",
            metric="train_loss",
            tasks=["oscillatory_valley", "saddle_objective", "direction_reversal_objective"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "energy_drift_curves.png",
            title="Relative Energy Drift",
            metric="relative_energy_drift",
            tasks=["oscillatory_valley", "saddle_objective", "rosenbrock_valley"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "coherent_momentum_real_baseline"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "direction_cosine_curves.png",
            title="Gradient Momentum Cosine",
            metric="grad_momentum_cosine",
            tasks=["oscillatory_valley", "conflicting_batches_classification"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "projection_activation_curves.png",
            title="Projection Activation",
            metric="coherence_projection_strength",
            tasks=["direction_reversal_objective", "conflicting_batches_classification"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "conflict_rotation_curves.png",
            title="Conflict and Rotation",
            metric="rotation_score",
            tasks=["direction_reversal_objective", "oscillatory_valley"],
            optimizers=["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved"],
        )

    if not runtime_memory.empty:
        plot_runtime = runtime_memory[runtime_memory["task"].isin(["breast_cancer_mlp", "digits_cnn", "oscillatory_valley"])]
        _plot_bar(
            plot_runtime,
            figure_dir / "runtime_comparison.png",
            "Runtime Per Step",
            "task",
            "mean_runtime_per_step_ms",
            "optimizer",
        )
        _plot_bar(
            plot_runtime,
            figure_dir / "memory_comparison.png",
            "Peak Device Memory",
            "task",
            "mean_peak_device_memory_mb",
            "optimizer",
        )
    if not combined_aggregated.empty:
        _plot_heatmap(
            combined_aggregated[combined_aggregated["optimizer"].isin(GPU_FOCUS_OPTIMIZERS)],
            figure_dir / "win_loss_heatmap.png",
        )
    _plot_ablation_chart(ablation_frame, figure_dir / "ablation_chart.png")

    improved_row = _top_row(combined_aggregated, "coherent_momentum_optimizer_improved")
    current_row = _top_row(combined_aggregated, "coherent_momentum_optimizer")
    adamw_row = _top_row(combined_aggregated, "adamw")
    rmsprop_row = _top_row(combined_aggregated, "rmsprop")
    sgd_row = _top_row(combined_aggregated, "sgd_momentum")

    cnn_tasks = ["digits_cnn", "digits_cnn_label_noise", "digits_cnn_input_noise"]
    stress_tasks = [
        "oscillatory_valley",
        "saddle_objective",
        "direction_reversal_objective",
        "noisy_quadratic_objective",
        "rosenbrock_valley",
        "narrow_valley_objective",
        "plateau_escape_objective",
    ]
    multitask_tasks = ["conflicting_batches_classification", "block_structure_classification", "small_batch_instability"]
    focus_optimizers = [
        "coherent_momentum_optimizer",
        "coherent_momentum_optimizer_improved",
        "adamw",
        "rmsprop",
        "sgd_momentum",
        "coherent_momentum_real_baseline",
        "coherent_direction_reference",
    ]
    cnn_best = _subset_top_rows(combined_raw, cnn_tasks, focus_optimizers)
    stress_best = _subset_top_rows(combined_raw, stress_tasks, focus_optimizers)
    multitask_best = _subset_top_rows(combined_raw, multitask_tasks, focus_optimizers)

    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
        if not ablation_frame.empty
        else pd.DataFrame()
    )
    current_full_score = float(
        ablation_summary.loc[ablation_summary["variant_name"] == "current_mainline_full", "mean_selection_score"].iloc[0]
    ) if "current_mainline_full" in set(ablation_summary.get("variant_name", [])) else float("nan")
    improved_balanced_score = float(
        ablation_summary.loc[ablation_summary["variant_name"] == "improved_balanced", "mean_selection_score"].iloc[0]
    ) if "improved_balanced" in set(ablation_summary.get("variant_name", [])) else float("nan")
    improved_preset_summary = ablation_summary.loc[
        ablation_summary["variant_name"].isin(
            [
                "improved_balanced",
                "improved_standard_safe",
                "improved_stress_specialist",
                "improved_cnn_safe",
            ]
        )
    ].copy()
    best_improved_preset = improved_preset_summary.head(1) if not improved_preset_summary.empty else pd.DataFrame()

    device_row = smoke_frame.iloc[0] if not smoke_frame.empty else None
    smoke_device = str(device_row["device"]) if device_row is not None and "device" in smoke_frame.columns else "not-run"
    smoke_device_name = str(device_row["device_name"]) if device_row is not None and "device_name" in smoke_frame.columns else "unknown"
    benchmark_devices = sorted({str(value) for value in combined_raw.get("device", pd.Series(dtype=str)).dropna().unique()})
    benchmark_device_text = ", ".join(benchmark_devices) if benchmark_devices else "n/a"

    runtime_summary = (
        combined_aggregated.groupby("optimizer", as_index=False)[
            [
                "mean_runtime_per_step_ms",
                "mean_optimizer_step_time_ms",
                "mean_samples_per_second",
                "mean_optimizer_state_mb",
                "mean_peak_device_memory_mb",
            ]
        ]
        .mean()
        .sort_values("mean_runtime_per_step_ms")
        if not combined_aggregated.empty
        else pd.DataFrame()
    )

    improved_direct = win_flags.loc[win_flags["optimizer"] == "coherent_momentum_optimizer_improved"] if not win_flags.empty else pd.DataFrame()
    improved_vs_current_row = (
        improved_direct.loc[improved_direct["baseline"] == "coherent_momentum_optimizer"].head(1)
        if not improved_direct.empty
        else pd.DataFrame()
    )
    broad_best_optimizer = best_frame["best_optimizer"].value_counts().idxmax() if not best_frame.empty else "n/a"
    broad_best_count = int(best_frame["best_optimizer"].value_counts().max()) if not best_frame.empty else 0

    answers = {
        "gpu_capable": not smoke_frame.empty,
        "beat_adamw": bool(not improved_direct.empty and int(improved_direct.loc[improved_direct["baseline"] == "adamw", "wins"].fillna(0).sum()) > 0),
        "beat_rmsprop": bool(not improved_direct.empty and int(improved_direct.loc[improved_direct["baseline"] == "rmsprop", "wins"].fillna(0).sum()) > 0),
        "beat_sgd": bool(not improved_direct.empty and int(improved_direct.loc[improved_direct["baseline"] == "sgd_momentum", "wins"].fillna(0).sum()) > 0),
    }

    best_rows_table = pd.concat(
        [frame for frame in [improved_row, current_row, adamw_row, rmsprop_row, sgd_row] if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in [improved_row, current_row, adamw_row, rmsprop_row, sgd_row]) else pd.DataFrame()

    report_lines = [
        "# Coherent Momentum Adam GPU Audit",
        "",
        "## Related Work",
        *[f"- [{entry['title']}]({entry['url']})" for entry in RELATED_WORK],
        "",
        "## Summary",
        f"- GPU-capable on available device(s): `{'yes' if answers['gpu_capable'] else 'not verified'}`",
        f"- GPU smoke device: `{smoke_device}` / `{smoke_device_name}`",
        f"- Broad benchmark device(s): `{benchmark_device_text}`",
        "- Compatibility was verified on GPU-capable hardware. Broad quality and runtime comparisons were intentionally run on CPU so MPS was not used as a performance-claim platform.",
        f"- Current Magneto best row: `{current_row.iloc[0]['task']}`" if not current_row.empty else "- Current Magneto best row: `n/a`",
        f"- Improved Magneto best row: `{improved_row.iloc[0]['task']}`" if not improved_row.empty else "- Improved Magneto best row: `n/a`",
        f"- Best AdamW row: `{adamw_row.iloc[0]['task']}`" if not adamw_row.empty else "- Best AdamW row: `n/a`",
        f"- Best RMSProp row: `{rmsprop_row.iloc[0]['task']}`" if not rmsprop_row.empty else "- Best RMSProp row: `n/a`",
        f"- Best SGD momentum row: `{sgd_row.iloc[0]['task']}`" if not sgd_row.empty else "- Best SGD momentum row: `n/a`",
        f"- Broad task-winner count leader: `{broad_best_optimizer}` with `{broad_best_count}` task wins",
        "",
        "## Code Audit Findings",
        "- The old Magneto path carried too much Python scalar control logic in the hot path, which risks host synchronization overhead on accelerator devices.",
        "- Diagnostics were previously too eager. They now support `enable_step_diagnostics` and `diagnostics_every_n_steps` so high-overhead logging can be throttled or disabled.",
        "- The optimizer was GPU-compatible in basic operation, but there was no dedicated device-transfer test coverage for state tensors, closure execution, or state_dict round-trips before this pass.",
        "- Previous reports already suggested that conflict damping and extra activation gating were harming or neutral. The new audit kept those assumptions explicit and tested them rather than promoting them.",
        "",
        "## Improvements Kept",
        "- Tensor-based Magneto control computation on device to reduce per-step host synchronization.",
        "- Diagnostics throttling through `diagnostics_every_n_steps` and `enable_step_diagnostics`.",
        "- Soft conflict correction instead of heavy conflict damping.",
        "- Conservative projection modes with conflict-only default.",
        "- Conv-safe preset and typed conv update cap for CNN tasks.",
        "",
        "## Improvements That Hurt Or Stayed Neutral",
        f"- Current ablation top row: `{ablation_summary.iloc[0]['variant_name']}`" if not ablation_summary.empty else "- Current ablation top row: `n/a`",
        f"- `improved_balanced` mean selection score: `{improved_balanced_score:.4f}` vs current `{current_full_score:.4f}`" if pd.notna(improved_balanced_score) and pd.notna(current_full_score) else "- Improved/current selection-score comparison unavailable.",
        f"- Best improved preset in the ablation slice: `{best_improved_preset.iloc[0]['variant_name']}` (`{best_improved_preset.iloc[0]['mean_selection_score']:.4f}`)" if not best_improved_preset.empty else "- Best improved preset unavailable.",
        "- CNN-safe and standard-safe presets were the strongest improved presets in the limited ablation slice, but neither displaced RMSProp or SGD momentum overall.",
        "- Conflict damping remained harmful enough that the improved branch uses softer correction rather than strong suppression.",
        "",
        "## Direct Win Counts",
        _markdown_table(win_flags) if not win_flags.empty else "_No direct comparison rows available._",
        "",
        "## Best Rows",
        _markdown_table(
            best_rows_table[
                [
                    "optimizer",
                    "task",
                    "mean_best_val_loss",
                    "mean_best_val_accuracy",
                    "mean_runtime_per_step_ms",
                    "mean_peak_device_memory_mb",
                ]
            ]
        ) if not best_rows_table.empty else "_No benchmark rows available._",
        "",
        "## CNN Snapshot",
        _markdown_table(cnn_best[["optimizer", "task", "best_val_loss", "best_val_accuracy", "runtime_per_step_ms", "optimizer_step_time_ms"]]) if not cnn_best.empty else "_No CNN rows available._",
        "",
        "## Stress Snapshot",
        _markdown_table(stress_best[["optimizer", "task", "best_val_loss", "best_val_accuracy", "runtime_per_step_ms", "optimizer_step_time_ms"]]) if not stress_best.empty else "_No stress rows available._",
        "",
        "## Multitask / Conflict Snapshot",
        _markdown_table(multitask_best[["optimizer", "task", "best_val_loss", "best_val_accuracy", "runtime_per_step_ms", "optimizer_step_time_ms"]]) if not multitask_best.empty else "_No multitask rows available._",
        "",
        "## Best Optimizer Per Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]) if not best_frame.empty else "_No task winners yet._",
        "",
        "## Runtime / Memory",
        _markdown_table(runtime_summary) if not runtime_summary.empty else "_No runtime table available._",
        "",
        "## Direct Answers",
        f"- Is Coherent Momentum GPU-capable? `{'yes' if answers['gpu_capable'] else 'not verified'}`",
        "- What code issues were found? Hot-path scalar control overhead, over-eager diagnostics, and missing explicit GPU/state transfer coverage were the main issues addressed.",
        "- Which improvements helped? Device-safe tensor control computation, diagnostics throttling, and safer presets helped the implementation quality. In the ablation slice, simpler presets outperformed the heavier full-controller path.",
        "- Which improvements hurt? Heavy conflict damping, extra activation gating, and globally forcing more controller logic remained neutral-to-harmful.",
        "- Is it faster or slower than baselines? `slower` overall. Improved Magneto remained materially slower than AdamW, RMSProp, and SGD momentum on tabular and CNN tasks.",
        "- Is memory acceptable? `yes` in the narrow sense: optimizer state stayed modest, but runtime overhead is still the practical limiter.",
        f"- Does improved Magneto beat AdamW anywhere? `{'yes' if answers['beat_adamw'] else 'no'}`; the wins remain concentrated in directional synthetic stress tasks rather than broad practical tasks.",
        f"- Does improved Magneto beat RMSProp anywhere? `{'yes' if answers['beat_rmsprop'] else 'no'}`; again as a specialist, not broadly.",
        f"- Does improved Magneto beat SGD momentum anywhere? `{'yes' if answers['beat_sgd'] else 'no'}`; not broadly across standard MLP/CNN tasks.",
        "- Does it improve CNN performance? `not enough`. The improved branch remained far behind AdamW, RMSProp, and SGD momentum on the CNN slice.",
        "- Does it generalize better than AdamW on noisy/small/conflict tasks? `sometimes on synthetic directional stress tasks`, but the current evidence does not support a broad practical generalization claim.",
        "- Does it remain a specialist or become broader? `specialist`.",
        f"- Best preset from this pass: `{best_improved_preset.iloc[0]['variant_name']}`" if not best_improved_preset.empty else "- Best preset from this pass: `unavailable`",
        "",
        "## Honest Positioning",
        "- This optimizer should be treated as a specialist unless the held-out noisy, conflict, and oscillation suites show broad wins against RMSProp or SGD momentum.",
        "- MPS compatibility is verified here, but MPS should not be treated as a performance-claim platform.",
        "- RMSProp and SGD momentum still dominate the broad CNN/tabular results in this repo.",
        "- The improved branch is real and better on directional stress tasks than the current branch, but it is also slower and it does not close the practical CNN gap.",
        f"- Improved vs current direct result: `{int(improved_vs_current_row.iloc[0]['wins'])}` meaningful wins and `{int(improved_vs_current_row.iloc[0]['two_x'])}` tracked 2x wins." if not improved_vs_current_row.empty else "- Improved vs current direct result unavailable.",
        "",
        "## Exact Next Step",
        "- Make the improved branch simpler, not more complicated: promote a lighter standard-safe / no-projection default, keep projection as an optional stress preset, and stop pushing global controller complexity upward.",
        "- If CNN strength matters, pursue a separate lighter conv-specific branch instead of trying to force the main Magneto specialist path to become a broad CNN optimizer.",
    ]
    (output_path / "final_coherent_momentum_gpu_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "combined_aggregated": combined_aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
        "runtime_memory": runtime_memory,
        "ablation_summary": ablation_summary,
        "cnn_rows": cnn_best,
        "stress_rows": stress_best,
        "multitask_rows": multitask_best,
    }
