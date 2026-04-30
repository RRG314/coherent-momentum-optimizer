from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmarking import (
    _load_best_tuning_map,
    _train_single_run,
    run_benchmark_suite,
    run_smoke_suite,
    run_stress_suite,
    run_tuning_suite,
)
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


FOCUS_OPTIMIZERS = [
    "hamiltonian_adam",
    "hamiltonian_adam_v2",
    "adam",
    "adamw",
    "rmsprop",
    "sgd",
    "sgd_momentum",
    "lion",
    "topological_adam",
    "thermodynamic_adam",
    "diffusion_adam",
]


def run_hamiltonian_v2_smoke(config: dict[str, Any]) -> pd.DataFrame:
    return run_smoke_suite(config)


def run_hamiltonian_v2_tuning(config: dict[str, Any]) -> pd.DataFrame:
    return run_tuning_suite(config)


def run_hamiltonian_v2_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    return run_benchmark_suite(config)


def run_hamiltonian_v2_stress(config: dict[str, Any]) -> pd.DataFrame:
    frame = run_stress_suite(config)
    output_dir = ensure_output_dir(config)
    frame.to_csv(output_dir / "stress_results.csv", index=False)
    return frame


def run_hamiltonian_v2_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [71, 89, 107, 131, 149]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "noisy_regression",
                "overfit_small_wine",
                "saddle_objective",
                "oscillatory_valley",
                "conflicting_batches_classification",
                "rosenbrock_valley",
            ],
        )
    )
    tuning_map = _load_best_tuning_map(output_dir / "tuning_results.csv")
    variants: list[dict[str, Any]] = [
        {"variant_name": "v1_hamiltonian", "optimizer_name": "hamiltonian_adam", "overrides": {}},
        {"variant_name": "v2_full", "optimizer_name": "hamiltonian_adam_v2", "overrides": {}},
        {
            "variant_name": "v2_no_normalized_energy",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {
                "use_normalized_energy": False,
                "normalized_energy_weight": 0.0,
                "update_energy_weight": 0.0,
                "force_energy_weight": 0.0,
                "loss_change_weight": 0.0,
            },
        },
        {
            "variant_name": "v2_no_energy_trend",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_energy_trend": False, "energy_ema_decay": 1.0, "drift_ema_decay": 1.0},
        },
        {
            "variant_name": "v2_no_predictive_damping",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_predictive_damping": False, "predictive_damping_strength": 0.0},
        },
        {
            "variant_name": "v2_no_alignment_scaling",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {
                "use_alignment_scaling": False,
                "alignment_boost": 0.0,
                "misalignment_damping": 0.0,
                "alignment_min_scale": 1.0,
                "alignment_max_scale": 1.0,
            },
        },
        {
            "variant_name": "v2_no_adaptive_step_scale",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_adaptive_step_scale": False, "min_step_scale": 1.0, "max_step_scale": 1.0},
        },
        {
            "variant_name": "v2_no_symplectic_correction",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_symplectic_correction": False},
        },
        {
            "variant_name": "v2_no_oscillation_damping",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_oscillation_damping": False, "oscillation_damping": 0.0},
        },
        {
            "variant_name": "v2_no_friction",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"use_friction": False, "friction": 0.0},
        },
        {
            "variant_name": "v2_rmsprop_force",
            "optimizer_name": "hamiltonian_adam_v2",
            "overrides": {"force_mode": "rmsprop"},
        },
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
        {"variant_name": "topological_baseline", "optimizer_name": "topological_adam", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            optimizer_name = str(variant["optimizer_name"])
            tuned = tuning_map.get((task_name, optimizer_name), {})
            merged = dict(tuned)
            merged.update(dict(variant["overrides"]))
            for seed in seeds:
                row = _train_single_run(
                    suite_name="hamiltonian_ablation",
                    task_name=task_name,
                    optimizer_name=optimizer_name,
                    hyperparameters=merged,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.8)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = optimizer_name
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def _ablation_component_deltas(ablation_frame: pd.DataFrame) -> pd.DataFrame:
    frame = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
    )
    full_score = float(frame.loc[frame["variant_name"] == "v2_full", "mean_selection_score"].iloc[0])
    frame["delta_vs_v2_full"] = frame["mean_selection_score"] - full_score
    return frame


def _best_row_for_optimizer(aggregated: pd.DataFrame, optimizer_name: str) -> dict[str, Any] | None:
    frame = aggregated[aggregated["optimizer"] == optimizer_name]
    if frame.empty:
        return None
    if frame["mean_best_val_accuracy"].notna().any():
        row = frame.sort_values(
            ["mean_best_val_accuracy", "mean_best_val_loss", "mean_runtime_seconds"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        row = frame.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0]
    return row.to_dict()


def _competitive_vs_rmsprop(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_frame in aggregated.groupby("task"):
        if "hamiltonian_adam_v2" not in set(task_frame["optimizer"]) or "rmsprop" not in set(task_frame["optimizer"]):
            continue
        candidate = task_frame.loc[task_frame["optimizer"] == "hamiltonian_adam_v2"].iloc[0]
        baseline = task_frame.loc[task_frame["optimizer"] == "rmsprop"].iloc[0]
        use_accuracy = pd.notna(candidate["mean_best_val_accuracy"]) and pd.notna(baseline["mean_best_val_accuracy"])
        if use_accuracy:
            margin = float(candidate["mean_best_val_accuracy"] - baseline["mean_best_val_accuracy"])
            competitive = margin >= -0.005
            rationale = "within 0.5 accuracy points" if competitive else "accuracy gap too large"
        else:
            ratio = float(candidate["mean_best_val_loss"] / max(float(baseline["mean_best_val_loss"]), 1e-12))
            competitive = ratio <= 1.05
            rationale = "within 5% loss" if competitive else "loss gap too large"
        rows.append({"task": task, "competitive": competitive, "rationale": rationale})
    return pd.DataFrame(rows)


def _plot_ablation_chart(ablation_frame: pd.DataFrame, output_path: Path) -> None:
    if ablation_frame.empty:
        return
    summary = _ablation_component_deltas(ablation_frame)
    summary = summary[summary["variant_name"] != "v1_hamiltonian"]
    plt.figure(figsize=(10, 5))
    plt.bar(summary["variant_name"], summary["delta_vs_v2_full"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("selection score delta vs V2 full")
    plt.title("Hamiltonian V2 Ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_hamiltonian_v2_report(output_dir: str | Path, related_work: list[dict[str, str]] | None = None) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    stress_path = output_path / "stress_results.csv"
    if not stress_path.exists():
        stress_path = output_path / "stress_test_results.csv"
    stress_frame = pd.read_csv(stress_path)
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    combined_raw = pd.concat([benchmark_frame, stress_frame], ignore_index=True)
    combined_aggregated = aggregate_results(combined_raw)
    best_frame = best_by_task(combined_aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    v2_vs_v1 = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam_v2", "hamiltonian_adam")
    v2_vs_adamw = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam_v2", "adamw")
    v2_vs_rmsprop = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam_v2", "rmsprop")
    v2_vs_topological = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam_v2", "topological_adam")
    competitive_vs_rmsprop = _competitive_vs_rmsprop(combined_aggregated)

    two_x_events = pd.concat(
        [
            v2_vs_v1.assign(comparison="v2_vs_v1"),
            v2_vs_adamw.assign(comparison="v2_vs_adamw"),
            v2_vs_rmsprop.assign(comparison="v2_vs_rmsprop"),
            v2_vs_topological.assign(comparison="v2_vs_topological"),
        ],
        ignore_index=True,
    )
    two_x_events = two_x_events[two_x_events["two_x"] == True]  # noqa: E712
    two_x_events.to_csv(output_path / "two_x_events.csv", index=False)

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(combined_raw)
    if not trace_frame.empty and "task" in trace_frame.columns and "optimizer" in trace_frame.columns:
        focus_curve_optimizers = ["hamiltonian_adam", "hamiltonian_adam_v2", "adamw", "rmsprop", "sgd_momentum", "lion", "topological_adam"]
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "loss_curves.png",
            title="Hamiltonian Focus Loss Curves",
            metric="train_loss",
            tasks=["noisy_regression", "oscillatory_valley", "overfit_small_wine"],
            optimizers=focus_curve_optimizers,
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "validation_curves.png",
            title="Validation Curves",
            metric="val_accuracy",
            tasks=["breast_cancer_mlp", "wine_mlp", "circles_mlp"],
            optimizers=focus_curve_optimizers,
            event="val",
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "rmsprop_vs_v2_comparison_curves.png",
            title="RMSProp vs HamiltonianAdamV2",
            metric="train_loss",
            tasks=["noisy_regression", "small_batch_instability", "stagnating_regression"],
            optimizers=["rmsprop", "hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "energy_drift_curves.png",
            title="Energy Drift Curves",
            metric="energy_drift",
            tasks=["oscillatory_valley", "rosenbrock_valley"],
            optimizers=["hamiltonian_adam", "hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "normalized_energy_curves.png",
            title="Normalized Energy",
            metric="normalized_total_energy",
            tasks=["noisy_regression", "rosenbrock_valley"],
            optimizers=["hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "oscillation_curves.png",
            title="Oscillation Score",
            metric="oscillation_score",
            tasks=["oscillatory_valley", "conflicting_batches_classification"],
            optimizers=["hamiltonian_adam", "hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "damping_curves.png",
            title="Effective Damping",
            metric="effective_damping",
            tasks=["oscillatory_valley", "small_batch_instability"],
            optimizers=["hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "alignment_scale_curves.png",
            title="Alignment Scale",
            metric="alignment_scale",
            tasks=["conflicting_batches_classification", "nonstationary_moons"],
            optimizers=["hamiltonian_adam_v2"],
        )
        _plot_metric(
            trace_frame,
            output_path=figure_dir / "adaptive_step_scale_curves.png",
            title="Adaptive Step Scale",
            metric="adaptive_step_scale",
            tasks=["noisy_regression", "stagnating_regression"],
            optimizers=["hamiltonian_adam_v2"],
        )

    steps_frame = combined_aggregated[
        combined_aggregated["optimizer"].isin(["hamiltonian_adam", "hamiltonian_adam_v2", "adamw", "rmsprop", "sgd_momentum", "topological_adam"])
    ][["task", "optimizer", "mean_steps_to_target_loss"]]
    _plot_bar(steps_frame, figure_dir / "steps_to_target_chart.png", "Steps To Target Loss", "task", "mean_steps_to_target_loss", "optimizer")
    _plot_heatmap(
        combined_aggregated[
            combined_aggregated["optimizer"].isin(["hamiltonian_adam", "hamiltonian_adam_v2", "adamw", "rmsprop", "sgd_momentum", "lion", "topological_adam"])
        ],
        figure_dir / "win_loss_heatmap.png",
    )
    _plot_ablation_chart(ablation_frame, figure_dir / "ablation_chart.png")

    ablation_summary = _ablation_component_deltas(ablation_frame)
    v2_ablation_only = ablation_summary[ablation_summary["variant_name"].str.startswith("v2_") & (ablation_summary["variant_name"] != "v2_full")]
    most_helpful_row = v2_ablation_only.sort_values("delta_vs_v2_full").iloc[0]
    most_harmful_row = v2_ablation_only.sort_values("delta_vs_v2_full", ascending=False).iloc[0]

    related_lines = []
    for item in related_work or []:
        related_lines.append(f"- [{item['title']}]({item['url']}): {item['summary']}")

    best_v1 = _best_row_for_optimizer(combined_aggregated, "hamiltonian_adam")
    best_v2 = _best_row_for_optimizer(combined_aggregated, "hamiltonian_adam_v2")
    best_rmsprop = _best_row_for_optimizer(combined_aggregated, "rmsprop")
    best_adamw = _best_row_for_optimizer(combined_aggregated, "adamw")
    best_topological = _best_row_for_optimizer(combined_aggregated, "topological_adam")

    report_lines = [
        "# Hamiltonian Adam V2 Report",
        "",
        "## 1. What changed from HamiltonianAdam V1",
        "- Added normalized energy instead of relying on raw loss-scaled total energy.",
        "- Added EMA trend control for energy and drift, predictive damping before blow-up, and bounded directional alignment scaling.",
        "- Added a bounded adaptive step-scale controller and thresholded symplectic-style correction.",
        "",
        "## 2. Exact equations/signals added",
        "- `kinetic_norm = kinetic / (|loss_ema| + eps)` when loss is available, otherwise normalized by force energy.",
        "- `relative_loss_change = (loss_t - loss_ema) / (|loss_ema| + eps)`.",
        "- `normalized_total_energy = w_k * kinetic_norm + w_u * (update_energy / (force_energy + eps)) + w_f * force_energy + w_l * |relative_loss_change|`.",
        "- `sustained_drift_score = max(0, drift_ema - drift_threshold)` with EMA-smoothed energy and drift.",
        "- Predictive damping uses gradient acceleration, force acceleration, oscillation rise, update-norm rise, and momentum/gradient misalignment.",
        "- Alignment scaling uses cosine alignment of `(grad, momentum)`, `(force, momentum)`, `(update, prev_update)`, and `(grad, prev_grad)`.",
        "",
        "## Related Work",
        *(related_lines or ["- No external references were embedded."]),
        "",
        "## Baselines Tested",
        "- " + ", ".join(FOCUS_OPTIMIZERS),
        "",
        "## Tasks Tested",
        "- Benchmark tasks: " + ", ".join(sorted(benchmark_frame["task"].unique())),
        "- Stress tasks: " + ", ".join(sorted(stress_frame["task"].unique())),
        "",
        "## Best Optimizer Per Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
        "",
        "## 3. Whether V2 beats V1",
        f"- Meaningful wins: {int(v2_vs_v1['win'].sum())}",
        "",
        "## 4. Whether V2 beats AdamW",
        f"- Meaningful wins: {int(v2_vs_adamw['win'].sum())}",
        "",
        "## 5. Whether V2 beats TopologicalAdam",
        f"- Meaningful wins: {int(v2_vs_topological['win'].sum())}",
        "",
        "## 6. Whether V2 is competitive with RMSProp",
        f"- Meaningful wins vs RMSProp: {int(v2_vs_rmsprop['win'].sum())}",
        f"- Competitive-or-better tasks vs RMSProp: {int(competitive_vs_rmsprop['competitive'].sum())}/{len(competitive_vs_rmsprop) if not competitive_vs_rmsprop.empty else 0}",
        "",
        "## 7. Whether any 2x results survive held-out seeds",
        f"- Surviving 2x events: {len(two_x_events)}",
        _markdown_table(two_x_events[["task", "baseline", "rationale"]]) if not two_x_events.empty else "_No 2x events survived held-out seeds._",
        "",
        "## 8. Which component mattered most",
        f"- {most_helpful_row['variant_name']} changed selection score by {most_helpful_row['delta_vs_v2_full']:.4f} vs full V2.",
        "",
        "## 9. Which component hurt",
        f"- {most_harmful_row['variant_name']} changed selection score by {most_harmful_row['delta_vs_v2_full']:.4f} vs full V2.",
        "",
        "## 10. Why RMSProp still wins if it wins",
        "- RMSProp remains hard to beat when square-gradient normalization alone is enough to suppress noisy, high-curvature updates without the extra controller complexity.",
        "- The report compares V2 directly against RMSProp on loss variance, update variance, and steps to target through the raw CSV outputs and figures.",
        "",
        "## 11. Whether V2 should become the main optimizer",
        "- Promote V2 only if it beats both V1 and AdamW and is at least competitive with RMSProp on the focused suite.",
        "",
        "## 12. Next exact improvement target",
        "- If RMSProp still leads, the next target is stronger force smoothing or a better momentum-force blend rather than adding more physical branding.",
        "",
        "## Best Rows",
        f"- Best V1 row: {best_v1['task']} / loss {best_v1['mean_best_val_loss']:.4f} / acc {best_v1['mean_best_val_accuracy'] if best_v1 and best_v1['mean_best_val_accuracy'] == best_v1['mean_best_val_accuracy'] else float('nan'):.4f}" if best_v1 else "- Best V1 row unavailable.",
        f"- Best V2 row: {best_v2['task']} / loss {best_v2['mean_best_val_loss']:.4f} / acc {best_v2['mean_best_val_accuracy'] if best_v2 and best_v2['mean_best_val_accuracy'] == best_v2['mean_best_val_accuracy'] else float('nan'):.4f}" if best_v2 else "- Best V2 row unavailable.",
        f"- Best RMSProp row: {best_rmsprop['task']} / loss {best_rmsprop['mean_best_val_loss']:.4f} / acc {best_rmsprop['mean_best_val_accuracy'] if best_rmsprop and best_rmsprop['mean_best_val_accuracy'] == best_rmsprop['mean_best_val_accuracy'] else float('nan'):.4f}" if best_rmsprop else "- Best RMSProp row unavailable.",
        f"- Best AdamW row: {best_adamw['task']} / loss {best_adamw['mean_best_val_loss']:.4f} / acc {best_adamw['mean_best_val_accuracy'] if best_adamw and best_adamw['mean_best_val_accuracy'] == best_adamw['mean_best_val_accuracy'] else float('nan'):.4f}" if best_adamw else "- Best AdamW row unavailable.",
        f"- Best Topological row: {best_topological['task']} / loss {best_topological['mean_best_val_loss']:.4f} / acc {best_topological['mean_best_val_accuracy'] if best_topological and best_topological['mean_best_val_accuracy'] == best_topological['mean_best_val_accuracy'] else float('nan'):.4f}" if best_topological else "- Best Topological row unavailable.",
    ]
    (output_path / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "combined_aggregated": combined_aggregated,
        "best_by_task": best_frame,
        "v2_vs_v1": v2_vs_v1,
        "v2_vs_adamw": v2_vs_adamw,
        "v2_vs_rmsprop": v2_vs_rmsprop,
        "v2_vs_topological": v2_vs_topological,
        "competitive_vs_rmsprop": competitive_vs_rmsprop,
        "best_v1": best_v1,
        "best_v2": best_v2,
        "best_rmsprop": best_rmsprop,
        "best_adamw": best_adamw,
        "best_topological": best_topological,
        "tuning_frame": tuning_frame,
        "ablation_frame": ablation_frame,
    }
