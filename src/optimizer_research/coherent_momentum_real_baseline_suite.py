from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .benchmarking import _train_single_run, run_benchmark_suite, run_smoke_suite, run_tuning_suite
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
    "coherent_momentum_physical_baseline",
    "coherent_momentum_real_baseline",
    "adam",
    "adamw",
    "rmsprop",
    "sgd",
    "sgd_momentum",
    "lion",
    "topological_adam",
]

RELATED_WORK = [
    {
        "title": "Decoupled Weight Decay Regularization",
        "url": "https://arxiv.org/abs/1711.05101",
        "summary": "AdamW is the practical Adam-family baseline because it decouples weight decay from the adaptive update rather than mixing it into the gradient moments.",
    },
    {
        "title": "Stochastic Gradient Hamiltonian Monte Carlo",
        "url": "https://arxiv.org/abs/1402.4102",
        "summary": "SGHMC is the canonical stochastic optimizer/sampler that introduces Hamiltonian momentum with friction to handle minibatch-noisy gradients.",
    },
    {
        "title": "On Symplectic Optimization",
        "url": "https://arxiv.org/abs/1802.03653",
        "summary": "This work is the clearest nearby reference for translating continuous-time Hamiltonian dynamics into discrete optimization algorithms with symplectic integrators.",
    },
    {
        "title": "Hamiltonian Descent Methods",
        "url": "https://arxiv.org/abs/1809.05042",
        "summary": "A family of dissipative Hamiltonian optimization methods that formalizes momentum dynamics as conformal Hamiltonian systems rather than heuristic damping.",
    },
    {
        "title": "A Physics-Inspired Optimizer: Velocity Regularized Adam",
        "url": "https://arxiv.org/abs/2505.13196",
        "summary": "Recent nearby work that uses physically motivated velocity regularization to damp large Adam updates; relevant because it already occupies some of the stability-control design space.",
    },
]


def run_coherent_momentum_real_baseline_smoke(config: dict[str, Any]) -> pd.DataFrame:
    return run_smoke_suite(config)


def run_real_baseline_tuning(config: dict[str, Any]) -> pd.DataFrame:
    return run_tuning_suite(config)


def run_coherent_momentum_real_baseline_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    return run_benchmark_suite(config)


def run_coherent_momentum_real_baseline_energy_tests(config: dict[str, Any]) -> pd.DataFrame:
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
                        suite_name="energy",
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


def run_coherent_momentum_real_baseline_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "harmonic_oscillator_objective",
                "rosenbrock_valley",
                "saddle_objective",
                "wine_mlp",
                "circles_mlp",
                "overfit_small_wine",
            ],
        )
    )
    variants: list[dict[str, Any]] = [
        {"variant_name": "real_full", "optimizer_name": "coherent_momentum_real_baseline", "overrides": {}},
        {
            "variant_name": "no_adam_preconditioning",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"use_adam_preconditioning": False, "mass_mode": "fixed", "fixed_mass": 1.0},
        },
        {
            "variant_name": "no_friction",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"friction": 0.0, "use_friction": False},
        },
        {
            "variant_name": "no_energy_correction",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"energy_correction_strength": 0.0, "use_energy_correction": False},
        },
        {
            "variant_name": "no_leapfrog_closure",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"mode": "symplectic_euler"},
        },
        {
            "variant_name": "symplectic_euler_only",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"mode": "symplectic_euler", "friction": 0.0, "use_friction": False},
        },
        {
            "variant_name": "fixed_mass",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"mass_mode": "fixed", "fixed_mass": 1.0},
        },
        {
            "variant_name": "adaptive_mass",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"mass_mode": "adaptive", "use_adam_preconditioning": True},
        },
        {
            "variant_name": "dissipative_hamiltonian",
            "optimizer_name": "coherent_momentum_real_baseline",
            "overrides": {"mode": "dissipative_hamiltonian", "use_energy_correction": True, "friction": 0.03},
        },
        {"variant_name": "v1_hamiltonian", "optimizer_name": "coherent_momentum_physical_baseline", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
        {"variant_name": "topological_baseline", "optimizer_name": "topological_adam", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="real_baseline_ablation",
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


def _competitive_vs_rmsprop(aggregated: pd.DataFrame, optimizer_name: str = "coherent_momentum_real_baseline") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_frame in aggregated.groupby("task"):
        if optimizer_name not in set(task_frame["optimizer"]) or "rmsprop" not in set(task_frame["optimizer"]):
            continue
        candidate = task_frame.loc[task_frame["optimizer"] == optimizer_name].iloc[0]
        baseline = task_frame.loc[task_frame["optimizer"] == "rmsprop"].iloc[0]
        use_accuracy = pd.notna(candidate["mean_best_val_accuracy"]) and pd.notna(baseline["mean_best_val_accuracy"])
        if use_accuracy:
            delta = float(candidate["mean_best_val_accuracy"] - baseline["mean_best_val_accuracy"])
            competitive = delta >= -0.005
        else:
            loss_ratio = float(candidate["mean_best_val_loss"] / max(float(baseline["mean_best_val_loss"]), 1e-12))
            competitive = loss_ratio <= 1.05
        rows.append({"task": task, "competitive": competitive})
    return pd.DataFrame(rows)


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
    plt.title("Real Hamiltonian Adam Ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_coherent_momentum_real_baseline_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    energy_frame = pd.read_csv(output_path / "energy_tests.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    combined_raw = pd.concat([benchmark_frame, energy_frame], ignore_index=True)
    combined_aggregated = aggregate_results(combined_raw)
    best_frame = best_by_task(combined_aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    real_vs_v1 = compute_meaningful_wins(combined_aggregated, "coherent_momentum_real_baseline", "coherent_momentum_physical_baseline")
    real_vs_adamw = compute_meaningful_wins(combined_aggregated, "coherent_momentum_real_baseline", "adamw")
    real_vs_rmsprop = compute_meaningful_wins(combined_aggregated, "coherent_momentum_real_baseline", "rmsprop")
    real_vs_topological = compute_meaningful_wins(combined_aggregated, "coherent_momentum_real_baseline", "topological_adam")
    competitive_vs_rmsprop = _competitive_vs_rmsprop(combined_aggregated)

    v1_row = _best_row_for_optimizer(combined_aggregated, "coherent_momentum_physical_baseline")
    real_row = _best_row_for_optimizer(combined_aggregated, "coherent_momentum_real_baseline")
    adamw_row = _best_row_for_optimizer(combined_aggregated, "adamw")
    rmsprop_row = _best_row_for_optimizer(combined_aggregated, "rmsprop")
    topo_row = _best_row_for_optimizer(combined_aggregated, "topological_adam")

    energy_agg = aggregate_results(energy_frame)
    v1_energy_drift = float(energy_agg.loc[energy_agg["optimizer"] == "coherent_momentum_physical_baseline", "mean_relative_energy_drift"].mean())
    real_energy_drift = float(energy_agg.loc[energy_agg["optimizer"] == "coherent_momentum_real_baseline", "mean_relative_energy_drift"].mean())

    competitive_tasks = set(competitive_vs_rmsprop.loc[competitive_vs_rmsprop["competitive"], "task"]) if not competitive_vs_rmsprop.empty else set()
    v1_win_tasks = set(real_vs_v1.loc[real_vs_v1["win"], "task"])
    adamw_win_tasks = set(real_vs_adamw.loc[real_vs_adamw["win"], "task"])
    strong_win_tasks = sorted(v1_win_tasks & adamw_win_tasks & competitive_tasks)
    two_x_survivors = sorted(
        set(real_vs_v1.loc[real_vs_v1["two_x"], "task"]) & competitive_tasks
        | (set(real_vs_adamw.loc[real_vs_adamw["two_x"], "task"]) & competitive_tasks)
    )

    trace_frame = _load_trace_frames(combined_raw)
    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Loss Curves",
        metric="train_loss",
        tasks=["wine_mlp", "harmonic_oscillator_objective", "rosenbrock_valley"],
        optimizers=["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline", "adamw", "rmsprop"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "validation_curves.png",
        title="Validation Accuracy Curves",
        metric="val_accuracy",
        tasks=["wine_mlp", "breast_cancer_mlp", "circles_mlp"],
        optimizers=["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline", "adamw", "rmsprop"],
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "energy_drift_curves.png",
        title="Energy Drift",
        metric="energy_drift",
        tasks=["harmonic_oscillator_objective", "rosenbrock_valley", "narrow_valley_objective"],
        optimizers=["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "relative_energy_drift_curves.png",
        title="Relative Energy Drift",
        metric="relative_energy_drift",
        tasks=["harmonic_oscillator_objective", "quadratic_bowl_objective", "noisy_quadratic_objective"],
        optimizers=["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "total_hamiltonian_curves.png",
        title="Total Hamiltonian",
        metric="total_hamiltonian",
        tasks=["harmonic_oscillator_objective", "rosenbrock_valley"],
        optimizers=["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "rmsprop_vs_real_curves.png",
        title="RMSProp vs Real Hamiltonian",
        metric="train_loss",
        tasks=["rosenbrock_valley", "small_batch_instability", "overfit_small_wine"],
        optimizers=["coherent_momentum_real_baseline", "rmsprop", "adamw", "sgd_momentum"],
    )
    energy_bar = energy_agg[energy_agg["optimizer"].isin(["coherent_momentum_real_baseline", "coherent_momentum_physical_baseline", "adamw", "rmsprop"])][
        ["task", "optimizer", "mean_relative_energy_drift"]
    ]
    _plot_bar(energy_bar, figure_dir / "relative_energy_drift_bar.png", "Relative Energy Drift", "task", "mean_relative_energy_drift", "optimizer")
    _plot_ablation_chart(ablation_frame, figure_dir / "ablation_chart.png")
    _plot_heatmap(combined_aggregated[combined_aggregated["optimizer"].isin(FOCUS_OPTIMIZERS)], figure_dir / "win_loss_heatmap.png")

    closure_rate = 0.0
    leapfrog_rate = 0.0
    if not trace_frame.empty:
        real_trace = trace_frame[(trace_frame["optimizer"] == "coherent_momentum_real_baseline") & (trace_frame["event"] == "train")]
        if not real_trace.empty:
            if "closure_recomputed_gradient" in real_trace.columns:
                closure_rate = float(real_trace["closure_recomputed_gradient"].dropna().mean())
            if "leapfrog_enabled" in real_trace.columns:
                leapfrog_rate = float(real_trace["leapfrog_enabled"].dropna().mean())

    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    best_ablation_variant = str(ablation_summary.iloc[0]["variant_name"]) if not ablation_summary.empty else "n/a"
    harmful_variant = "n/a"
    real_full_score = float(ablation_summary.loc[ablation_summary["variant_name"] == "real_full", "mean_selection_score"].iloc[0]) if "real_full" in set(ablation_summary["variant_name"]) else float("nan")
    if "no_leapfrog_closure" in set(ablation_summary["variant_name"]):
        no_leapfrog_score = float(ablation_summary.loc[ablation_summary["variant_name"] == "no_leapfrog_closure", "mean_selection_score"].iloc[0])
        harmful_variant = "leapfrog closure helped" if no_leapfrog_score < real_full_score else "leapfrog closure hurt"

    mode_text = "unknown"
    real_benchmark_rows = benchmark_frame[benchmark_frame["optimizer"] == "coherent_momentum_real_baseline"]
    if not real_benchmark_rows.empty and "hyperparameters" in real_benchmark_rows.columns:
        modes: list[str] = []
        for value in real_benchmark_rows["hyperparameters"]:
            try:
                params = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                continue
            mode_value = params.get("mode")
            if isinstance(mode_value, str):
                modes.append(mode_value)
        if modes:
            mode_text = max(set(modes), key=modes.count)

    report_lines = [
        "# Coherent Momentum Real Baseline Report",
        "",
        "## Related Work",
        *[
            f"- [{entry['title']}]({entry['url']}): {entry['summary']}"
            for entry in RELATED_WORK
        ],
        "",
        "## 1. What changed from the reactive baseline",
        "- The reactive baseline used simple AdamW-style damping. `CoherentMomentumRealBaseline` adds explicit position-momentum dynamics, kinetic and potential tracking, adaptive diagonal mass from Adam's second moment, and a leapfrog/symplectic-Euler integrator.",
        "",
        "## 2. Whether the new optimizer uses real Hamiltonian dynamics",
        "- Yes, in the limited optimizer sense used here: parameters are positions, optimizer state is physical momentum, kinetic energy is computed from momentum and inverse mass, and updates use symplectic-Euler or leapfrog-style kick-drift-kick integration.",
        "- The report does not call the closure-free path full leapfrog. It is labeled a symplectic-Euler approximation.",
        "",
        "## 3. Which update mode was used",
        f"- Default benchmark mode: `{mode_text}` with adaptive mass and closure-driven leapfrog whenever the harness provided a closure.",
        "",
        "## 4. Whether leapfrog closure worked",
        f"- Mean `closure_recomputed_gradient` across real-Hamiltonian traces: {closure_rate:.3f}",
        f"- Mean `leapfrog_enabled` across real-Hamiltonian traces: {leapfrog_rate:.3f}",
        "",
        "## 5. Energy drift compared to the reactive baseline",
        f"- Mean relative energy drift on direct energy tests: reactive baseline `{v1_energy_drift:.6f}` vs real baseline `{real_energy_drift:.6f}`",
        "",
        "## 6. Whether it beats AdamW",
        f"- Meaningful wins vs AdamW: {int(real_vs_adamw['win'].sum())}",
        "",
        "## 7. Whether it beats RMSProp",
        f"- Meaningful wins vs RMSProp: {int(real_vs_rmsprop['win'].sum())}",
        f"- Tasks where it beat the reactive baseline and AdamW while staying competitive with RMSProp: {', '.join(strong_win_tasks) if strong_win_tasks else 'none'}",
        "",
        "## 8. Whether it beats Topological Adam",
        f"- Meaningful wins vs Topological Adam: {int(real_vs_topological['win'].sum())}",
        "",
        "## 9. Whether any 2x event appeared",
        f"- Surviving 2x events under the stricter RMSProp-competitive filter: {', '.join(two_x_survivors) if two_x_survivors else 'none'}",
        "",
        "## 10. Whether the more faithful Hamiltonian math helped or hurt",
        f"- Best ablation variant by mean selection score: `{best_ablation_variant}`",
        f"- Interpretation: `{harmful_variant}`",
        "",
        "## 11. Whether to keep the reactive baseline, keep the physical baseline, or combine them",
        "- Keep both for now.",
        "- The reactive baseline remains the simpler stability baseline.",
        "- `CoherentMomentumPhysicalBaseline` is the more mathematically faithful branch, but only keep pushing it if the energy-drift gains survive without giving back too much to RMSProp on practical ML tasks.",
        "",
        "## Best Rows",
        _markdown_table(
            pd.DataFrame(
                [
                    {"optimizer": "coherent_momentum_reactive_baseline", **(v1_row or {})},
                    {"optimizer": "coherent_momentum_real_baseline", **(real_row or {})},
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
        "real_vs_v1": real_vs_v1,
        "real_vs_adamw": real_vs_adamw,
        "real_vs_rmsprop": real_vs_rmsprop,
        "real_vs_topological": real_vs_topological,
        "strong_win_tasks": strong_win_tasks,
        "two_x_survivors": two_x_survivors,
    }
