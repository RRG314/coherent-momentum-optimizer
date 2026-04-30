from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .benchmarking import _train_single_run, run_benchmark_suite, run_smoke_suite, run_tuning_suite
from .config import ensure_output_dir
from .coherent_momentum_real_baseline_suite import RELATED_WORK, _best_row_for_optimizer, _competitive_vs_rmsprop
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
    "coherent_momentum_optimizer",
    "coherent_momentum_real_baseline",
    "coherent_direction_reference",
    "coherent_momentum_physical_baseline",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "lion",
    "topological_adam",
]


def run_coherent_momentum_smoke(config: dict[str, Any]) -> pd.DataFrame:
    return run_smoke_suite(config)


def run_coherent_momentum_tuning(config: dict[str, Any]) -> pd.DataFrame:
    return run_tuning_suite(config)


def run_coherent_momentum_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    return run_benchmark_suite(config)


def run_coherent_momentum_energy_tests(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "energy_tasks",
            [
                "harmonic_oscillator_objective",
                "quadratic_bowl_objective",
                "rosenbrock_valley",
                "saddle_objective",
                "narrow_valley_objective",
                "noisy_quadratic_objective",
                "oscillatory_valley",
                "direction_reversal_objective",
            ],
        )
    )
    optimizers = list(config.get("optimizers", FOCUS_OPTIMIZERS))
    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for optimizer_name in optimizers:
            for seed in seeds:
                rows.append(
                    _train_single_run(
                        suite_name="coherent_momentum_energy",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters={},
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=True,
                        epoch_scale=float(config.get("energy_epoch_scale", 0.9)),
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "energy_tests.csv", index=False)
    return frame


def run_coherent_momentum_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "rosenbrock_valley",
                "saddle_objective",
                "oscillatory_valley",
                "conflicting_batches_classification",
                "wine_mlp",
                "overfit_small_wine",
            ],
        )
    )
    variants: list[dict[str, Any]] = [
        {"variant_name": "combined_full", "optimizer_name": "coherent_momentum_optimizer", "overrides": {}},
        {
            "variant_name": "hamiltonian_real_baseline",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {},
        },
        {"variant_name": "direction_reference_baseline", "optimizer_name": "coherent_direction_reference", "overrides": {}},
        {
            "variant_name": "lr_scale_only_coherence",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "alignment_strength": 0.18,
                "coherence_strength": 0.12,
                "conflict_damping": 0.0,
                "rotation_penalty": 0.22,
                "projection_strength": 0.0,
                "max_projection": 0.0,
            },
        },
        {
            "variant_name": "damping_only_coherence",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "conflict_damping": 0.28,
                "rotation_penalty": 0.22,
                "projection_strength": 0.0,
                "max_projection": 0.0,
                "min_alignment_scale": 1.0,
                "max_alignment_scale": 1.0,
            },
        },
        {
            "variant_name": "projection_only_coherence",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "conflict_damping": 0.0,
                "rotation_penalty": 0.0,
                "projection_strength": 0.18,
                "max_projection": 0.35,
                "min_alignment_scale": 1.0,
                "max_alignment_scale": 1.0,
            },
        },
        {
            "variant_name": "no_alignment_scaling",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "rotation_penalty": 0.0,
                "min_alignment_scale": 1.0,
                "max_alignment_scale": 1.0,
            },
        },
        {
            "variant_name": "no_conflict_damping",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {"conflict_damping": 0.0},
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
        {
            "variant_name": "no_projection",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {"projection_strength": 0.0, "max_projection": 0.0},
        },
        {
            "variant_name": "fixed_mass_combined",
            "optimizer_name": "coherent_momentum_optimizer",
            "overrides": {
                "mass_mode": "fixed",
                "fixed_mass": 1.0,
                "use_adam_preconditioning": False,
            },
        },
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
        {"variant_name": "topological_baseline", "optimizer_name": "topological_adam", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="coherent_momentum_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.8)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
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
    plt.figure(figsize=(10, 5))
    plt.bar(summary["variant_name"], summary["mean_selection_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean selection score")
    plt.title("Coherent Momentum Adam Ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_coherent_momentum_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    energy_frame = pd.read_csv(output_path / "energy_tests.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    combined_raw = pd.concat([benchmark_frame, energy_frame], ignore_index=True)
    combined_aggregated = aggregate_results(combined_raw)
    best_frame = best_by_task(combined_aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    combo_vs_real = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "coherent_momentum_real_baseline")
    combo_vs_direction_reference = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "coherent_direction_reference")
    combo_vs_v1 = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "coherent_momentum_physical_baseline")
    combo_vs_adamw = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "adamw")
    combo_vs_rmsprop = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "rmsprop")
    combo_vs_topological = compute_meaningful_wins(combined_aggregated, "coherent_momentum_optimizer", "topological_adam")
    competitive_vs_rmsprop = _competitive_vs_rmsprop(combined_aggregated, optimizer_name="coherent_momentum_optimizer")

    combo_row = _best_row_for_optimizer(combined_aggregated, "coherent_momentum_optimizer")
    real_row = _best_row_for_optimizer(combined_aggregated, "coherent_momentum_real_baseline")
    direction_reference_row = _best_row_for_optimizer(combined_aggregated, "coherent_direction_reference")
    adamw_row = _best_row_for_optimizer(combined_aggregated, "adamw")
    rmsprop_row = _best_row_for_optimizer(combined_aggregated, "rmsprop")
    topo_row = _best_row_for_optimizer(combined_aggregated, "topological_adam")

    energy_agg = aggregate_results(energy_frame)
    combo_energy_drift = float(
        energy_agg.loc[energy_agg["optimizer"] == "coherent_momentum_optimizer", "mean_relative_energy_drift"].mean()
    )
    real_energy_drift = float(
        energy_agg.loc[energy_agg["optimizer"] == "coherent_momentum_real_baseline", "mean_relative_energy_drift"].mean()
    )

    competitive_tasks = set(competitive_vs_rmsprop.loc[competitive_vs_rmsprop["competitive"], "task"]) if not competitive_vs_rmsprop.empty else set()
    real_win_tasks = set(combo_vs_real.loc[combo_vs_real["win"], "task"])
    adamw_win_tasks = set(combo_vs_adamw.loc[combo_vs_adamw["win"], "task"])
    strong_win_tasks = sorted(real_win_tasks & adamw_win_tasks & competitive_tasks)
    two_x_survivors = sorted(
        (set(combo_vs_real.loc[combo_vs_real["two_x"], "task"]) | set(combo_vs_adamw.loc[combo_vs_adamw["two_x"], "task"]))
        & competitive_tasks
    )

    trace_frame = _load_trace_frames(combined_raw)
    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Loss Curves",
        metric="train_loss",
        tasks=["wine_mlp", "oscillatory_valley", "rosenbrock_valley"],
        optimizers=["coherent_momentum_optimizer", "coherent_momentum_real_baseline", "coherent_direction_reference", "rmsprop"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "validation_curves.png",
        title="Validation Accuracy Curves",
        metric="val_accuracy",
        tasks=["wine_mlp", "breast_cancer_mlp", "conflicting_batches_classification"],
        optimizers=["coherent_momentum_optimizer", "coherent_momentum_real_baseline", "coherent_direction_reference", "adamw"],
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "energy_drift_curves.png",
        title="Energy Drift",
        metric="relative_energy_drift",
        tasks=["harmonic_oscillator_objective", "rosenbrock_valley", "oscillatory_valley"],
        optimizers=["coherent_momentum_optimizer", "coherent_momentum_real_baseline", "coherent_momentum_physical_baseline"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "alignment_curves.png",
        title="Magneto Alignment Scale",
        metric="alignment_scale",
        tasks=["oscillatory_valley", "conflicting_batches_classification", "direction_reversal_objective"],
        optimizers=["coherent_momentum_optimizer", "coherent_direction_reference"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "rotation_curves.png",
        title="Rotation Score",
        metric="rotation_score",
        tasks=["oscillatory_valley", "direction_reversal_objective"],
        optimizers=["coherent_momentum_optimizer", "coherent_direction_reference"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "rmsprop_vs_combined_curves.png",
        title="RMSProp vs Coherent Momentum",
        metric="train_loss",
        tasks=["small_batch_instability", "overfit_small_wine", "oscillatory_valley"],
        optimizers=["coherent_momentum_optimizer", "rmsprop", "adamw", "coherent_momentum_real_baseline"],
    )
    energy_bar = energy_agg[
        energy_agg["optimizer"].isin(["coherent_momentum_optimizer", "coherent_momentum_real_baseline", "adamw", "rmsprop"])
    ][["task", "optimizer", "mean_relative_energy_drift"]]
    _plot_bar(
        energy_bar,
        figure_dir / "relative_energy_drift_bar.png",
        "Relative Energy Drift",
        "task",
        "mean_relative_energy_drift",
        "optimizer",
    )
    _plot_ablation_chart(ablation_frame, figure_dir / "ablation_chart.png")
    _plot_heatmap(
        combined_aggregated[combined_aggregated["optimizer"].isin(FOCUS_OPTIMIZERS)],
        figure_dir / "win_loss_heatmap.png",
    )

    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    best_ablation_variant = str(ablation_summary.iloc[0]["variant_name"]) if not ablation_summary.empty else "n/a"
    full_score = float(
        ablation_summary.loc[ablation_summary["variant_name"] == "combined_full", "mean_selection_score"].iloc[0]
    ) if "combined_full" in set(ablation_summary["variant_name"]) else float("nan")
    helpful_component = "n/a"
    harmful_component = "n/a"
    if "no_projection" in set(ablation_summary["variant_name"]):
        no_projection_score = float(
            ablation_summary.loc[ablation_summary["variant_name"] == "no_projection", "mean_selection_score"].iloc[0]
        )
        helpful_component = "projection helped" if no_projection_score < full_score else "projection hurt"
    if "no_conflict_damping" in set(ablation_summary["variant_name"]):
        no_damping_score = float(
            ablation_summary.loc[ablation_summary["variant_name"] == "no_conflict_damping", "mean_selection_score"].iloc[0]
        )
        harmful_component = "conflict damping helped" if no_damping_score < full_score else "conflict damping hurt"
    activation_component = "activation gating not evaluated"
    if "no_activation_gating" in set(ablation_summary["variant_name"]):
        no_activation_score = float(
            ablation_summary.loc[ablation_summary["variant_name"] == "no_activation_gating", "mean_selection_score"].iloc[0]
        )
        activation_component = "activation gating helped" if no_activation_score < full_score else "activation gating hurt or stayed neutral"

    report_lines = [
        "# Coherent Momentum Adam Report",
        "",
        "## Related Work",
        *[f"- [{entry['title']}]({entry['url']}): {entry['summary']}" for entry in RELATED_WORK],
        "- Gradient alignment and conflict methods in multitask optimization are the nearest conceptual relatives for the coherence controller; this branch is best viewed as a directional-coherence add-on to the real Hamiltonian optimizer rather than a novelty claim.",
        "",
        "## 1. What changed from the physical baseline",
        "- `CoherentMomentumOptimizer` keeps the stabilized physical core and adds directional coherence signals: gradient-momentum cosine, force-momentum cosine, gradient history cosine, update history cosine, rotation score, and a bounded projection back toward the force direction during conflict.",
        "- The current branch also adds activation gating so the coherence controller can stay closer to the physical baseline on ordinary tasks and only fully activate when conflict or rotation rises.",
        "",
        "## 2. Whether it beat the strengthened physical baseline",
        f"- Meaningful wins vs `coherent_momentum_real_baseline`: {int(combo_vs_real['win'].sum())}",
        "",
        "## 3. Whether it beat CoherentDirectionReferenceOptimizer alone",
        f"- Meaningful wins vs `coherent_direction_reference`: {int(combo_vs_direction_reference['win'].sum())}",
        "",
        "## 4. Whether it beat AdamW",
        f"- Meaningful wins vs `adamw`: {int(combo_vs_adamw['win'].sum())}",
        "",
        "## 5. Whether it beat RMSProp",
        f"- Meaningful wins vs `rmsprop`: {int(combo_vs_rmsprop['win'].sum())}",
        f"- Tasks where it beat the physical baseline and AdamW while staying competitive with RMSProp: {', '.join(strong_win_tasks) if strong_win_tasks else 'none'}",
        "",
        "## 6. Whether it beat Topological Adam",
        f"- Meaningful wins vs `topological_adam`: {int(combo_vs_topological['win'].sum())}",
        "",
        "## 7. Energy drift compared to the physical baseline branch",
        f"- Mean relative energy drift on direct energy tests: Real `{real_energy_drift:.6f}` vs Coherent Momentum `{combo_energy_drift:.6f}`",
        "",
        "## 8. Whether any 2x event survived",
        f"- Surviving 2x events under the stricter RMSProp-competitive filter: {', '.join(two_x_survivors) if two_x_survivors else 'none'}",
        "",
        "## 9. Which coherence component mattered most",
        f"- Best ablation variant by mean selection score: `{best_ablation_variant}`",
        f"- Activation gating interpretation: `{activation_component}`",
        f"- Projection interpretation: `{helpful_component}`",
        f"- Conflict damping interpretation: `{harmful_component}`",
        "",
        "## 10. Recommendation",
        "- Keep this branch separate from the real Hamiltonian baseline.",
        "- Use it where oscillation, direction reversal, or conflicting-batch behavior matters.",
        "- Do not replace RMSProp or the strengthened real-Hamiltonian baseline globally unless the controller wins survive broader held-out tasks.",
        "",
        "## Best Rows",
        _markdown_table(
            pd.DataFrame(
                [
                    {"optimizer": "coherent_momentum_optimizer", **(combo_row or {})},
                    {"optimizer": "coherent_momentum_real_baseline", **(real_row or {})},
                    {"optimizer": "coherent_direction_reference", **(direction_reference_row or {})},
                    {"optimizer": "adamw", **(adamw_row or {})},
                    {"optimizer": "rmsprop", **(rmsprop_row or {})},
                    {"optimizer": "topological_adam", **(topo_row or {})},
                ]
            )[
                ["optimizer", "task", "mean_best_val_loss", "mean_best_val_accuracy", "mean_relative_energy_drift"]
            ]
        ),
        "",
        "## Best Optimizer Per Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
    ]
    (output_path / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "combined_aggregated": combined_aggregated,
        "best_by_task": best_frame,
        "combo_vs_real": combo_vs_real,
        "combo_vs_direction_reference": combo_vs_direction_reference,
        "combo_vs_v1": combo_vs_v1,
        "combo_vs_adamw": combo_vs_adamw,
        "combo_vs_rmsprop": combo_vs_rmsprop,
        "combo_vs_topological": combo_vs_topological,
        "strong_win_tasks": strong_win_tasks,
        "two_x_survivors": two_x_survivors,
    }
