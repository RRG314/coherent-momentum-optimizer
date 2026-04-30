from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in frame.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("nan")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join("| " + " | ".join(items) + " |" for items in rows)


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    aggregations: dict[str, tuple[str, str]] = {
        "mean_final_val_loss": ("final_val_loss", "mean"),
        "mean_best_val_loss": ("best_val_loss", "mean"),
        "mean_final_val_accuracy": ("final_val_accuracy", "mean"),
        "mean_best_val_accuracy": ("best_val_accuracy", "mean"),
        "mean_steps_to_target_loss": ("steps_to_target_loss", "mean"),
        "mean_steps_to_target_accuracy": ("steps_to_target_accuracy", "mean"),
        "mean_training_stability": ("training_stability", "mean"),
        "mean_loss_variance": ("loss_variance", "mean"),
        "mean_grad_norm_stability": ("gradient_norm_stability", "mean"),
        "mean_update_norm_stability": ("update_norm_stability", "mean"),
        "mean_generalization_gap": ("generalization_gap", "mean"),
        "mean_runtime_seconds": ("runtime_seconds", "mean"),
        "divergence_rate": ("diverged", "mean"),
        "seed_consistency_loss": ("final_val_loss", "std"),
        "seed_consistency_accuracy": ("final_val_accuracy", "std"),
    }
    optional_metrics = {
        "mean_gradient_norm_variance": "gradient_norm_variance",
        "mean_update_norm_variance": "update_norm_variance",
        "mean_oscillation_score": "mean_oscillation_score",
        "mean_energy_drift": "mean_energy_drift",
        "mean_relative_energy_drift": "mean_relative_energy_drift",
        "mean_normalized_total_energy": "mean_normalized_total_energy",
        "mean_kinetic_energy": "mean_kinetic_energy",
        "mean_potential_energy": "mean_potential_energy",
        "mean_total_hamiltonian": "mean_total_hamiltonian",
        "mean_momentum_norm": "mean_momentum_norm",
        "mean_parameter_step_norm": "mean_parameter_step_norm",
        "mean_inverse_mass_mean": "mean_inverse_mass_mean",
        "mean_inverse_mass_std": "mean_inverse_mass_std",
        "mean_effective_damping": "mean_effective_damping",
        "mean_alignment_scale": "mean_alignment_scale",
        "mean_effective_lr_scale": "mean_effective_lr_scale",
        "mean_force_momentum_cosine": "mean_force_momentum_cosine",
        "mean_grad_previous_grad_cosine": "mean_grad_previous_grad_cosine",
        "mean_update_previous_update_cosine": "mean_update_previous_update_cosine",
        "mean_rotation_gate": "mean_rotation_gate",
        "mean_coherence_score": "mean_coherence_score",
        "mean_conflict_score": "mean_conflict_score",
        "mean_conflict_gate": "mean_conflict_gate",
        "mean_magneto_activation": "mean_magneto_activation",
        "mean_stable_gate": "mean_stable_gate",
        "mean_field_strength": "mean_field_strength",
        "mean_magneto_projection_strength": "mean_magneto_projection_strength",
        "mean_magneto_friction_multiplier": "mean_magneto_friction_multiplier",
        "mean_soft_conflict_correction": "mean_soft_conflict_correction",
        "mean_recovery_score": "mean_recovery_score",
        "mean_direction_coherence": "mean_direction_coherence",
        "mean_rotation_score_structural": "mean_rotation_score_structural",
        "mean_trust_scale": "mean_trust_scale",
        "mean_relative_gradient_scale": "mean_relative_gradient_scale",
        "mean_constraint_agreement": "mean_constraint_agreement",
        "mean_recoverability_score": "mean_recoverability_score",
        "mean_support_balance": "mean_support_balance",
        "mean_consensus_strength": "mean_consensus_strength",
        "mean_component_conflict": "mean_component_conflict",
        "mean_memory_alignment": "mean_memory_alignment",
        "mean_residual_alignment": "mean_residual_alignment",
        "mean_block_coherence": "mean_block_coherence",
        "mean_block_trust_scale": "mean_block_trust_scale",
        "mean_block_norm_ratio": "mean_block_norm_ratio",
        "mean_memory_ratio": "mean_memory_ratio",
        "mean_block_conflict": "mean_block_conflict",
        "mean_filter_support": "mean_filter_support",
        "mean_conv_trust_bonus": "mean_conv_trust_bonus",
        "mean_conv_step_multiplier": "mean_conv_step_multiplier",
        "mean_trust_score": "trust_score",
        "mean_block_step_norm": "block_step_norm",
        "mean_fallback_rate": "fallback_rate",
        "mean_runtime_overhead_ms": "runtime_overhead_ms",
        "mean_observation_recoverability": "mean_observation_recoverability",
        "mean_observation_disagreement": "mean_observation_disagreement",
        "mean_view_support": "mean_view_support",
        "mean_runtime_per_step_ms": "runtime_per_step_ms",
        "mean_optimizer_step_time_ms": "optimizer_step_time_ms",
        "mean_samples_per_second": "samples_per_second",
        "mean_optimizer_state_mb": "optimizer_state_mb",
        "mean_peak_device_memory_mb": "peak_device_memory_mb",
        "mean_diagnostics_rows": "diagnostics_rows",
    }
    for output_col, input_col in optional_metrics.items():
        if input_col in frame.columns:
            aggregations[output_col] = (input_col, "mean")
    aggregated = frame.groupby(["task", "optimizer", "task_family", "problem_type"], as_index=False).agg(**aggregations).fillna(np.nan)
    return aggregated


def best_by_task(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_frame in aggregated.groupby("task"):
        use_accuracy = task_frame["mean_best_val_accuracy"].notna().any()
        if use_accuracy:
            best_row = task_frame.sort_values(
                ["mean_best_val_accuracy", "mean_runtime_seconds"],
                ascending=[False, True],
            ).iloc[0]
            score = best_row["mean_best_val_accuracy"]
        else:
            best_row = task_frame.sort_values(
                ["mean_best_val_loss", "mean_runtime_seconds"],
                ascending=[True, True],
            ).iloc[0]
            score = best_row["mean_best_val_loss"]
        rows.append(
            {
                "task": task,
                "best_optimizer": best_row["optimizer"],
                "task_family": best_row["task_family"],
                "score": score,
                "mean_best_val_loss": best_row["mean_best_val_loss"],
                "mean_best_val_accuracy": best_row["mean_best_val_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def compute_meaningful_wins(aggregated: pd.DataFrame, optimizer_name: str, baseline_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_frame in aggregated.groupby("task"):
        if optimizer_name not in set(task_frame["optimizer"]) or baseline_name not in set(task_frame["optimizer"]):
            continue
        candidate = task_frame.loc[task_frame["optimizer"] == optimizer_name].iloc[0]
        baseline = task_frame.loc[task_frame["optimizer"] == baseline_name].iloc[0]
        use_accuracy = not math.isnan(float(candidate["mean_best_val_accuracy"])) and not math.isnan(float(baseline["mean_best_val_accuracy"]))

        win = False
        two_x = False
        rationale = "no clear advantage"
        if use_accuracy:
            acc_delta = float(candidate["mean_best_val_accuracy"] - baseline["mean_best_val_accuracy"])
            steps_candidate = candidate["mean_steps_to_target_accuracy"]
            steps_baseline = baseline["mean_steps_to_target_accuracy"]
            if acc_delta >= 0.01:
                win = True
                rationale = "higher best validation accuracy"
            elif (
                not math.isnan(float(steps_candidate))
                and not math.isnan(float(steps_baseline))
                and abs(acc_delta) <= 0.005
                and float(steps_candidate) <= 0.8 * float(steps_baseline)
            ):
                win = True
                rationale = "matched accuracy with fewer steps"
            if (
                not math.isnan(float(steps_candidate))
                and not math.isnan(float(steps_baseline))
                and abs(acc_delta) <= 0.005
                and float(steps_candidate) <= 0.5 * float(steps_baseline)
            ):
                two_x = True
        else:
            loss_ratio = float(candidate["mean_best_val_loss"] / max(float(baseline["mean_best_val_loss"]), 1e-12))
            steps_candidate = candidate["mean_steps_to_target_loss"]
            steps_baseline = baseline["mean_steps_to_target_loss"]
            if loss_ratio <= 0.97:
                win = True
                rationale = "lower best validation loss"
            elif (
                not math.isnan(float(steps_candidate))
                and not math.isnan(float(steps_baseline))
                and loss_ratio <= 1.03
                and float(steps_candidate) <= 0.8 * float(steps_baseline)
            ):
                win = True
                rationale = "matched loss with fewer steps"
            if (
                not math.isnan(float(steps_candidate))
                and not math.isnan(float(steps_baseline))
                and loss_ratio <= 1.03
                and float(steps_candidate) <= 0.5 * float(steps_baseline)
            ):
                two_x = True

        if not win:
            divergence_ratio = float(candidate["divergence_rate"] + 1e-12) / float(baseline["divergence_rate"] + 1e-12)
            if divergence_ratio <= 0.5:
                win = True
                rationale = "lower divergence rate"
                two_x = True
        if not win:
            variance_ratio = float(candidate["mean_loss_variance"] + 1e-12) / float(baseline["mean_loss_variance"] + 1e-12)
            if variance_ratio <= 0.5 and candidate["mean_best_val_loss"] <= baseline["mean_best_val_loss"] * 1.03:
                win = True
                rationale = "lower loss variance with comparable score"
                two_x = True

        rows.append(
            {
                "task": task,
                "optimizer": optimizer_name,
                "baseline": baseline_name,
                "win": win,
                "two_x": two_x,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def summarize_ablations(ablation_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    if ablation_frame.empty:
        return [], []
    helpful: list[str] = []
    harmful: list[str] = []
    grouped = (
        ablation_frame.groupby(["base_optimizer", "variant_name"], as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
    )
    for optimizer_name, frame in grouped.groupby("base_optimizer"):
        if "base" not in set(frame["variant_name"]):
            continue
        base_score = float(frame.loc[frame["variant_name"] == "base", "mean_selection_score"].iloc[0])
        for _, row in frame.iterrows():
            if row["variant_name"] == "base":
                continue
            delta = float(row["mean_selection_score"]) - base_score
            label = f"{optimizer_name}:{row['variant_name']}"
            if delta < -0.01:
                helpful.append(label)
            elif delta > 0.01:
                harmful.append(label)
    return helpful, harmful


def _load_trace_frames(result_frame: pd.DataFrame) -> pd.DataFrame:
    trace_paths = [Path(path) for path in result_frame["trace_path"].dropna().unique()]
    frames = []
    for path in trace_paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_metric(
    trace_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    metric: str,
    tasks: list[str],
    optimizers: list[str],
    event: str = "train",
) -> None:
    required_columns = {"task", "optimizer"}
    if trace_frame.empty or not required_columns.issubset(trace_frame.columns):
        return
    subset = trace_frame[(trace_frame["task"].isin(tasks)) & (trace_frame["optimizer"].isin(optimizers))]
    if event:
        subset = subset[subset["event"] == event]
    if subset.empty or metric not in subset.columns:
        return
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 3.5 * len(tasks)), sharex=False)
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks):
        task_frame = subset[subset["task"] == task]
        for optimizer in optimizers:
            opt_frame = task_frame[task_frame["optimizer"] == optimizer]
            if opt_frame.empty or metric not in opt_frame.columns:
                continue
            mean_curve = opt_frame.groupby("step")[metric].mean().dropna()
            if mean_curve.empty:
                continue
            axis.plot(mean_curve.index, mean_curve.values, label=optimizer)
        axis.set_title(task)
        axis.set_ylabel(metric)
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bar(frame: pd.DataFrame, output_path: Path, title: str, x_col: str, y_col: str, hue_col: str) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index=x_col, columns=hue_col, values=y_col)
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_heatmap(aggregated: pd.DataFrame, output_path: Path) -> None:
    if aggregated.empty:
        return
    rows = []
    tasks = sorted(aggregated["task"].unique())
    optimizers = sorted(aggregated["optimizer"].unique())
    for task in tasks:
        task_frame = aggregated[aggregated["task"] == task].set_index("optimizer")
        if "adamw" not in task_frame.index:
            continue
        baseline = task_frame.loc["adamw"]
        values = []
        for optimizer in optimizers:
            if optimizer not in task_frame.index:
                values.append(np.nan)
                continue
            row = task_frame.loc[optimizer]
            if not math.isnan(float(row["mean_best_val_accuracy"])) and not math.isnan(float(baseline["mean_best_val_accuracy"])):
                values.append(float(row["mean_best_val_accuracy"] - baseline["mean_best_val_accuracy"]))
            else:
                values.append(float(baseline["mean_best_val_loss"] - row["mean_best_val_loss"]))
        rows.append(values)
    if not rows:
        return
    matrix = np.array(rows)
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="relative score vs AdamW")
    plt.xticks(range(len(optimizers)), optimizers, rotation=45, ha="right")
    plt.yticks(range(len(tasks)), tasks)
    plt.title("Win/Loss Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    stability_frame = pd.read_csv(output_path / "stability_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated_benchmark = aggregate_results(benchmark_frame)
    aggregated_stability = aggregate_results(stability_frame)
    combined_aggregated = pd.concat([aggregated_benchmark, aggregated_stability], ignore_index=True)
    best_frame = best_by_task(aggregated_benchmark)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    thermo_vs_adamw = compute_meaningful_wins(combined_aggregated, "thermodynamic_adam", "adamw")
    diffusion_vs_adamw = compute_meaningful_wins(combined_aggregated, "diffusion_adam", "adamw")
    hamiltonian_vs_adamw = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam", "adamw")
    thermo_vs_topo = compute_meaningful_wins(combined_aggregated, "thermodynamic_adam", "topological_adam")
    diffusion_vs_topo = compute_meaningful_wins(combined_aggregated, "diffusion_adam", "topological_adam")
    hamiltonian_vs_topo = compute_meaningful_wins(combined_aggregated, "hamiltonian_adam", "topological_adam")
    helpful_signals, harmful_signals = summarize_ablations(ablation_frame)

    figure_dir = output_path / "figures"
    trace_frame = _load_trace_frames(pd.concat([benchmark_frame, stability_frame], ignore_index=True))
    physical_optimizers = ["adamw", "topological_adam", "thermodynamic_adam", "diffusion_adam", "hamiltonian_adam"]
    _plot_metric(trace_frame, output_path=figure_dir / "loss_curves.png", title="Loss Curves", metric="train_loss", tasks=["moons_mlp", "digits_cnn", "pinn_harmonic_oscillator"], optimizers=physical_optimizers)
    _plot_metric(trace_frame, output_path=figure_dir / "validation_accuracy_curves.png", title="Validation Accuracy Curves", metric="val_accuracy", tasks=["moons_mlp", "digits_cnn", "breast_cancer_mlp"], optimizers=physical_optimizers, event="val")
    _plot_metric(trace_frame, output_path=figure_dir / "gradient_norm_curves.png", title="Gradient Norm Curves", metric="grad_norm", tasks=["noisy_gradients_classification", "nonstationary_moons"], optimizers=physical_optimizers)
    _plot_metric(trace_frame, output_path=figure_dir / "update_norm_curves.png", title="Update Norm Curves", metric="update_norm", tasks=["noisy_gradients_classification", "saddle_objective"], optimizers=physical_optimizers)
    _plot_metric(trace_frame, output_path=figure_dir / "temperature_curves.png", title="Thermodynamic Temperature", metric="temperature", tasks=["moons_mlp", "noisy_gradients_classification"], optimizers=["thermodynamic_adam"])
    _plot_metric(trace_frame, output_path=figure_dir / "diffusion_noise_curves.png", title="Diffusion Noise", metric="noise_norm", tasks=["moons_mlp", "noisy_gradients_classification"], optimizers=["diffusion_adam"])
    _plot_metric(trace_frame, output_path=figure_dir / "energy_drift_curves.png", title="Hamiltonian Energy Drift", metric="energy_drift", tasks=["saddle_objective", "pinn_harmonic_oscillator"], optimizers=["hamiltonian_adam"])
    stability_summary = aggregated_stability[aggregated_stability["optimizer"].isin(physical_optimizers)][["task", "optimizer", "mean_training_stability"]]
    _plot_bar(stability_summary, figure_dir / "stability_comparison.png", "Stability Comparison", "task", "mean_training_stability", "optimizer")
    steps_summary = aggregated_benchmark[aggregated_benchmark["optimizer"].isin(physical_optimizers)][["task", "optimizer", "mean_steps_to_target_loss"]]
    _plot_bar(steps_summary, figure_dir / "steps_to_target_comparison.png", "Steps to Target Loss", "task", "mean_steps_to_target_loss", "optimizer")
    _plot_heatmap(combined_aggregated[combined_aggregated["optimizer"].isin(physical_optimizers)], figure_dir / "win_loss_heatmap.png")

    best_baseline = aggregated_benchmark[
        aggregated_benchmark["optimizer"].isin(["adamw", "topological_adam", "adam", "lion", "radam", "nadam", "rmsprop", "sgd", "sgd_momentum"])
    ]
    best_baseline_row = best_baseline.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    report_lines = [
        "# Physical Adam Final Report",
        "",
        "## 1. What was implemented",
        "- Thermodynamic Adam, Diffusion Adam, and Hamiltonian Adam were implemented as real PyTorch optimizers with step-level diagnostics.",
        "- The project includes smoke, tuning, benchmark, stability, ablation, and report-export scripts.",
        "",
        "## 2. Whether existing Topological Adam was found and reused",
        "- Yes. The existing editable `topological-adam` package from `repos/topological-adam/` was detected and reused as the benchmark baseline.",
        "",
        "## 3. Baselines tested",
        "- SGD, SGD with momentum, RMSProp, Adam, AdamW, NAdam, RAdam, Lion, and Topological Adam.",
        "",
        "## 4. Tasks tested",
        "- Benchmark tasks: " + ", ".join(sorted(benchmark_frame["task"].unique())),
        "- Stability tasks: " + ", ".join(sorted(stability_frame["task"].unique())),
        "",
        "## 5. Best optimizer per task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
        "",
        "## 6. Whether Thermodynamic Adam beat AdamW",
        f"- {int(thermo_vs_adamw['win'].sum())} task-level meaningful wins across benchmark and stability suites.",
        "",
        "## 7. Whether Diffusion Adam beat AdamW",
        f"- {int(diffusion_vs_adamw['win'].sum())} task-level meaningful wins across benchmark and stability suites.",
        "",
        "## 8. Whether Hamiltonian Adam beat AdamW",
        f"- {int(hamiltonian_vs_adamw['win'].sum())} task-level meaningful wins across benchmark and stability suites.",
        "",
        "## 9. Whether any beat Topological Adam",
        f"- Thermodynamic Adam wins vs Topological Adam: {int(thermo_vs_topo['win'].sum())}",
        f"- Diffusion Adam wins vs Topological Adam: {int(diffusion_vs_topo['win'].sum())}",
        f"- Hamiltonian Adam wins vs Topological Adam: {int(hamiltonian_vs_topo['win'].sum())}",
        "",
        "## 10. Whether any showed a real 2x advantage",
        f"- 2x events vs AdamW: Thermodynamic {int(thermo_vs_adamw['two_x'].sum())}, Diffusion {int(diffusion_vs_adamw['two_x'].sum())}, Hamiltonian {int(hamiltonian_vs_adamw['two_x'].sum())}.",
        "",
        "## 11. Which physical signal actually helped",
        "- Signals whose removal hurt ablation score: " + (", ".join(helpful_signals) if helpful_signals else "none detected strongly enough"),
        "",
        "## 12. Which signal hurt",
        "- Signals whose removal improved ablation score: " + (", ".join(harmful_signals) if harmful_signals else "none detected strongly enough"),
        "",
        "## 13. Ablation results",
        _markdown_table(
            ablation_frame.groupby(["base_optimizer", "variant_name"], as_index=False)["selection_score"]
            .mean()
            .sort_values(["base_optimizer", "selection_score"], ascending=[True, False])
            .head(18)
        ),
        "",
        "## 14. Failure modes",
        "- Divergence and weak seed consistency were tracked explicitly in the benchmark outputs.",
        "- See `stability_results.csv` for divergence rates and stability metrics per optimizer-task pair.",
        "",
        "## 15. Recommendation",
        f"- Strongest baseline row observed: {best_baseline_row['optimizer']} on {best_baseline_row['task']}.",
        f"- Pursue Thermodynamic Adam: {'yes' if thermo_vs_adamw['win'].sum() > 0 else 'no'}",
        f"- Pursue Diffusion Adam: {'yes' if diffusion_vs_adamw['win'].sum() > 0 else 'no'}",
        f"- Pursue Hamiltonian Adam: {'yes' if hamiltonian_vs_adamw['win'].sum() > 0 else 'no'}",
        f"- Combine with Topological Adam: {'only if future controlled ablations justify it' if not helpful_signals else 'consider targeted follow-ups only'}",
        "- Abandon variants with zero meaningful wins or unstable ablations.",
    ]
    report_text = "\n".join(report_lines)
    (output_path / "final_report.md").write_text(report_text, encoding="utf-8")

    return {
        "aggregated_benchmark": aggregated_benchmark,
        "aggregated_stability": aggregated_stability,
        "best_by_task": best_frame,
        "thermo_vs_adamw": thermo_vs_adamw,
        "diffusion_vs_adamw": diffusion_vs_adamw,
        "hamiltonian_vs_adamw": hamiltonian_vs_adamw,
    }
