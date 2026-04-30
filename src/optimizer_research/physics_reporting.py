from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .reporting import (
    _load_trace_frames,
    _markdown_table,
    _plot_bar,
    _plot_heatmap,
    _plot_metric,
    aggregate_results,
    best_by_task,
    compute_meaningful_wins,
    summarize_ablations,
)


PHYSICS_OPTIMIZERS = [
    "sds_adam",
    "magneto_adam",
    "thermodynamic_adam",
    "diffusion_adam",
    "hamiltonian_adam",
    "uncertainty_adam",
]

BASELINE_OPTIMIZERS = [
    "sgd",
    "sgd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "nadam",
    "radam",
    "lion",
    "topological_adam",
]


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _combined_aggregate(benchmark_frame: pd.DataFrame, stress_frame: pd.DataFrame) -> pd.DataFrame:
    raw_frames = [frame for frame in (benchmark_frame, stress_frame) if not frame.empty]
    if not raw_frames:
        return pd.DataFrame()
    return aggregate_results(pd.concat(raw_frames, ignore_index=True))


def _strongest_task_row(frame: pd.DataFrame, optimizer_name: str) -> dict[str, Any] | None:
    subset = frame[frame["optimizer"] == optimizer_name]
    if subset.empty:
        return None
    if subset["mean_best_val_accuracy"].notna().any():
        row = subset.sort_values(
            ["mean_best_val_accuracy", "mean_best_val_loss", "mean_runtime_seconds"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        row = subset.sort_values(
            ["mean_best_val_loss", "mean_runtime_seconds"],
            ascending=[True, True],
        ).iloc[0]
    return row.to_dict()


def _win_summary(combined_aggregated: pd.DataFrame, baseline_name: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    mapping: dict[str, pd.DataFrame] = {}
    empty_template = pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale", "comparison"])
    for optimizer_name in PHYSICS_OPTIMIZERS:
        wins = compute_meaningful_wins(combined_aggregated, optimizer_name, baseline_name)
        if wins.empty:
            wins = empty_template.copy()
        wins["comparison"] = f"{optimizer_name}_vs_{baseline_name}"
        mapping[optimizer_name] = wins
        frames.append(wins)
    if not frames:
        return mapping, empty_template.copy()
    return mapping, pd.concat(frames, ignore_index=True)


def _generate_figures(output_path: Path, benchmark_frame: pd.DataFrame, stress_frame: pd.DataFrame) -> None:
    figure_dir = output_path / "figures"
    trace_frame = _load_trace_frames(pd.concat([frame for frame in [benchmark_frame, stress_frame] if not frame.empty], ignore_index=True))
    if trace_frame.empty or "task" not in trace_frame.columns or "optimizer" not in trace_frame.columns:
        return
    comparison_optimizers = ["adamw", "topological_adam"] + PHYSICS_OPTIMIZERS

    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Loss Curves",
        metric="train_loss",
        tasks=["moons_mlp", "wine_mlp", "pinn_harmonic_oscillator"],
        optimizers=comparison_optimizers,
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "validation_accuracy_curves.png",
        title="Validation Accuracy Curves",
        metric="val_accuracy",
        tasks=["moons_mlp", "breast_cancer_mlp", "digits_cnn"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "gradient_norm_curves.png",
        title="Gradient Norm Curves",
        metric="grad_norm",
        tasks=["noisy_gradients_classification", "unstable_deep_mlp"],
        optimizers=comparison_optimizers,
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "update_norm_curves.png",
        title="Update Norm Curves",
        metric="update_norm",
        tasks=["direction_reversal_objective", "rosenbrock_valley"],
        optimizers=comparison_optimizers,
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "sds_horizon_state_curves.png",
        title="SDS Horizon State",
        metric="horizon_code",
        tasks=["stagnating_regression", "unstable_deep_mlp"],
        optimizers=["sds_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "magneto_alignment_curves.png",
        title="Magneto Alignment",
        metric="grad_momentum_cosine",
        tasks=["nonstationary_moons", "direction_reversal_objective"],
        optimizers=["magneto_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "magneto_rotation_curves.png",
        title="Magneto Rotation",
        metric="rotation_score",
        tasks=["nonstationary_moons", "direction_reversal_objective"],
        optimizers=["magneto_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "thermodynamic_temperature_curves.png",
        title="Thermodynamic Temperature",
        metric="temperature",
        tasks=["loss_shock_classification", "small_batch_instability"],
        optimizers=["thermodynamic_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "diffusion_noise_scale_curves.png",
        title="Diffusion Noise Scale",
        metric="diffusion_scale",
        tasks=["plateau_escape_objective", "stagnating_regression"],
        optimizers=["diffusion_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "hamiltonian_energy_drift_curves.png",
        title="Hamiltonian Energy Drift",
        metric="energy_drift",
        tasks=["rosenbrock_valley", "oscillatory_valley"],
        optimizers=["hamiltonian_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "quantum_uncertainty_curves.png",
        title="Quantum Uncertainty",
        metric="uncertainty_score",
        tasks=["label_noise_breast_cancer", "sparse_gradients_linear"],
        optimizers=["uncertainty_adam"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "quantum_interference_curves.png",
        title="Quantum Interference",
        metric="interference_score",
        tasks=["label_noise_breast_cancer", "nonstationary_moons"],
        optimizers=["uncertainty_adam"],
    )


def export_physics_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)

    benchmark_frame = _read_optional_csv(output_path / "benchmark_results.csv")
    stress_frame = _read_optional_csv(output_path / "stress_test_results.csv")
    tuning_frame = _read_optional_csv(output_path / "tuning_results.csv")
    ablation_frame = _read_optional_csv(output_path / "ablation_results.csv")

    combined_aggregated = _combined_aggregate(benchmark_frame, stress_frame)
    best_frame = best_by_task(combined_aggregated) if not combined_aggregated.empty else pd.DataFrame()
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    wins_vs_adamw_map, wins_vs_adamw = _win_summary(combined_aggregated, "adamw")
    wins_vs_topo_map, wins_vs_topo = _win_summary(combined_aggregated, "topological_adam")
    win_flags = pd.concat([wins_vs_adamw, wins_vs_topo], ignore_index=True) if not wins_vs_adamw.empty or not wins_vs_topo.empty else pd.DataFrame()
    win_flags.to_csv(output_path / "win_flags.csv", index=False)

    helpful_signals, harmful_signals = summarize_ablations(ablation_frame)
    _generate_figures(output_path, benchmark_frame, stress_frame)
    _plot_heatmap(
        combined_aggregated[combined_aggregated["optimizer"].isin(["adamw", "topological_adam"] + PHYSICS_OPTIMIZERS)],
        output_path / "figures" / "win_loss_heatmap.png",
    )

    steps_frame = combined_aggregated[
        combined_aggregated["optimizer"].isin(["adamw", "topological_adam"] + PHYSICS_OPTIMIZERS)
    ][["task", "optimizer", "mean_steps_to_target_loss"]]
    _plot_bar(
        steps_frame,
        output_path / "figures" / "steps_to_target_chart.png",
        "Steps To Target Loss",
        "task",
        "mean_steps_to_target_loss",
        "optimizer",
    )

    stability_frame = combined_aggregated[
        combined_aggregated["optimizer"].isin(["adamw", "topological_adam"] + PHYSICS_OPTIMIZERS)
    ][["task", "optimizer", "mean_training_stability"]]
    _plot_bar(
        stability_frame,
        output_path / "figures" / "stability_comparison.png",
        "Training Stability",
        "task",
        "mean_training_stability",
        "optimizer",
    )

    strongest_adamw = _strongest_task_row(combined_aggregated, "adamw")
    strongest_topological = _strongest_task_row(combined_aggregated, "topological_adam")

    signal_descriptions = {
        "sds_adam": (
            "Update ratio, gradient ratio, gradient entropy, and validation gap.",
            "AdamW direction is scaled by a smooth inner/outer horizon controller that reheats low-update stagnation and cools oversized updates.",
        ),
        "magneto_adam": (
            "Cosine alignment between gradient, momentum, previous gradient, and previous update, plus rotation and coherence scores.",
            "AdamW direction is reweighted by directional coherence and damped when gradients rotate or oppose momentum.",
        ),
        "thermodynamic_adam": (
            "Gradient energy, update energy, gradient entropy, temperature EMA, and heat spikes.",
            "AdamW learning rate is cooled by energy/entropy spikes and reheated modestly under low-temperature stagnation.",
        ),
        "diffusion_adam": (
            "Diffusion scale, entropy-conditioned noise scale, stagnation counter, and bounded noise-to-update ratio.",
            "AdamW update is perturbed by annealed Gaussian noise with optional gradient alignment, capped so noise cannot dominate.",
        ),
        "hamiltonian_adam": (
            "Kinetic proxy from momentum, potential proxy from loss EMA, total energy drift, and oscillation score.",
            "AdamW direction is damped or amplified based on energy drift and oscillation, preserving momentum only when the trajectory is stable.",
        ),
        "uncertainty_adam": (
            "Gradient variance proxy, directional interference, and reliability from repeated alignment.",
            "AdamW step scale shrinks under uncertainty/conflict and sharpens under reliable repeated direction, with bounded exploration noise.",
        ),
    }

    report_lines = [
        "# Physics Adam Final Report",
        "",
        "## 1. Optimizers implemented",
        "- SDS Adam / Horizon Adam",
        "- Magneto Adam",
        "- Thermodynamic Adam",
        "- Diffusion Adam",
        "- Hamiltonian Adam",
        "- Quantum / Uncertainty Adam",
        "",
        "## 2. Exact mathematical signal each one uses",
    ]
    for optimizer_name in PHYSICS_OPTIMIZERS:
        title = optimizer_name.replace("_", " ")
        report_lines.append(f"- {title}: {signal_descriptions[optimizer_name][0]}")

    report_lines.extend(
        [
            "",
            "## 3. How each update differs from AdamW",
        ]
    )
    for optimizer_name in PHYSICS_OPTIMIZERS:
        title = optimizer_name.replace("_", " ")
        report_lines.append(f"- {title}: {signal_descriptions[optimizer_name][1]}")

    report_lines.extend(
        [
            "",
            "## 4. Whether Topological Adam was found and compared",
            "- Yes. The existing implementation from `repos/topological-adam/` was reused through the local adapter and was not overwritten.",
            "",
            "## 5. Baselines tested",
            "- " + ", ".join(BASELINE_OPTIMIZERS),
            "",
            "## 6. Tasks tested",
            "- Benchmark tasks: " + (", ".join(sorted(benchmark_frame["task"].unique())) if not benchmark_frame.empty else "none"),
            "- Stress tasks: " + (", ".join(sorted(stress_frame["task"].unique())) if not stress_frame.empty else "none"),
            "",
            "## 7. Best optimizer per task",
            _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]) if not best_frame.empty else "_No rows available._",
            "",
            "## 8. Whether SDS Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['sds_adam']['win'].sum()) if 'sds_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 9. Whether Magneto Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['magneto_adam']['win'].sum()) if 'magneto_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 10. Whether Thermodynamic Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['thermodynamic_adam']['win'].sum()) if 'thermodynamic_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 11. Whether Diffusion Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['diffusion_adam']['win'].sum()) if 'diffusion_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 12. Whether Hamiltonian Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['hamiltonian_adam']['win'].sum()) if 'hamiltonian_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 13. Whether Quantum/Uncertainty Adam beat AdamW",
            f"- {int(wins_vs_adamw_map['uncertainty_adam']['win'].sum()) if 'uncertainty_adam' in wins_vs_adamw_map else 0} meaningful task wins.",
            "",
            "## 14. Whether any beat Topological Adam",
            f"- Total meaningful wins vs Topological Adam: {int(wins_vs_topo['win'].sum()) if not wins_vs_topo.empty else 0}",
            "",
            "## 15. Whether any 2x result exists",
            f"- Total 2x flags vs AdamW: {int(wins_vs_adamw['two_x'].sum()) if not wins_vs_adamw.empty else 0}",
            f"- Total 2x flags vs Topological Adam: {int(wins_vs_topo['two_x'].sum()) if not wins_vs_topo.empty else 0}",
            "",
            "## 16. Which physical/math signal actually helped",
            "- " + (", ".join(helpful_signals) if helpful_signals else "No ablation signal cleared the reporting threshold strongly enough."),
            "",
            "## 17. Which signal hurt",
            "- " + (", ".join(harmful_signals) if harmful_signals else "No ablation signal clearly hurt enough to stand out above noise."),
            "",
            "## 18. Failure modes",
            "- Divergence, high variance, and loss-instability were logged directly in the raw CSV outputs and trace files.",
            "- Weak signals are treated as failures rather than wins when they do not survive the meaningful-win criteria.",
            "",
            "## 19. Recommendation",
            f"- Strongest AdamW row: {strongest_adamw['task']} (loss {strongest_adamw['mean_best_val_loss']:.4f}, acc {strongest_adamw['mean_best_val_accuracy'] if strongest_adamw['mean_best_val_accuracy'] == strongest_adamw['mean_best_val_accuracy'] else float('nan'):.4f})" if strongest_adamw else "- Strongest AdamW row unavailable.",
            f"- Strongest Topological Adam row: {strongest_topological['task']} (loss {strongest_topological['mean_best_val_loss']:.4f}, acc {strongest_topological['mean_best_val_accuracy'] if strongest_topological and strongest_topological['mean_best_val_accuracy'] == strongest_topological['mean_best_val_accuracy'] else float('nan'):.4f})" if strongest_topological else "- Strongest Topological Adam row unavailable.",
            "- Pursue the optimizer with repeated wins and stable ablation support, not the one with the flashiest single run.",
            "- Abandon variants whose wins disappear after seed averaging or whose neutral setting matches the full variant.",
        ]
    )
    report_text = "\n".join(report_lines)
    (output_path / "final_report.md").write_text(report_text, encoding="utf-8")

    return {
        "combined_aggregated": combined_aggregated,
        "best_by_task": best_frame,
        "wins_vs_adamw": wins_vs_adamw,
        "wins_vs_topological": wins_vs_topo,
        "strongest_adamw": strongest_adamw,
        "strongest_topological": strongest_topological,
        "tuning_rows": tuning_frame,
        "ablation_rows": ablation_frame,
    }
