from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .benchmarking import run_benchmark_suite, run_smoke_suite, run_tuning_suite, _train_single_run
from .config import ensure_output_dir
from .reporting import (
    _load_trace_frames,
    _markdown_table,
    _plot_heatmap,
    _plot_metric,
    aggregate_results,
    best_by_task,
    compute_meaningful_wins,
)
from optimizers.optimizer_utils import resolve_device


PINN_OPTIMIZERS = [
    "sgd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "lbfgs",
    "adamw_lbfgs_hybrid",
    "topological_adam",
    "real_hamiltonian_adam",
    "magneto_hamiltonian_adam",
    "constraint_consensus_optimizer",
]

PINN_RELATED_WORK = [
    {
        "title": "Challenges in Training PINNs: A Loss Landscape Perspective",
        "url": "https://arxiv.org/abs/2402.01868",
        "summary": "Adam+L-BFGS and second-order methods often outperform plain Adam because PINNs are ill-conditioned and composite-loss optimization is stiff.",
    },
    {
        "title": "Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective",
        "url": "https://arxiv.org/abs/2502.00604",
        "summary": "Directional conflicts across PINN loss terms are a major bottleneck; SOAP and other quasi-second-order methods can resolve them much better than first-order baselines.",
    },
    {
        "title": "PDE-aware Optimizer for Physics-informed Neural Networks",
        "url": "https://arxiv.org/abs/2507.08118",
        "summary": "Residual-gradient variance can be used to adapt updates in PINNs, showing that PDE-specific optimizer design is a live and relevant research area.",
    },
    {
        "title": "Gradient Statistics-Based Multi-Objective Optimization in Physics-Informed Neural Networks",
        "url": "https://www.mdpi.com/1424-8220/23/21/8665",
        "summary": "Adaptive weighting based on gradient statistics already covers part of the multi-objective PINN optimization space.",
    },
    {
        "title": "Improving physics-informed neural networks with meta-learned optimization",
        "url": "https://www.jmlr.org/papers/volume25/23-0356/23-0356.pdf",
        "summary": "Meta-learned optimizers can outperform hand-crafted ones on PINNs, so novelty claims for a hand-designed method must be more specific than 'better optimizer for PINNs'.",
    },
]


def pinn_optimizer_default_config() -> dict[str, Any]:
    return {
        "output_dir": "reports/pinn_optimizer_research",
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 3,
        "search_seed": 2026,
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.45,
        "benchmark_epoch_scale": 0.85,
        "ablation_epoch_scale": 0.75,
        "optimizers": list(PINN_OPTIMIZERS),
        "smoke_tasks": ["pinn_harmonic_oscillator", "pinn_poisson_1d"],
        "smoke_optimizers": ["adamw", "lbfgs", "constraint_consensus_optimizer"],
        "tuning_tasks": ["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"],
        "benchmark_tasks": [
            "pinn_harmonic_oscillator",
            "pinn_poisson_1d",
            "pinn_heat_equation",
            "pinn_poisson_1d_small_batch",
            "pinn_heat_equation_noisy_initial",
        ],
        "use_tuning_results": True,
    }


def run_pinn_optimizer_smoke(config: dict[str, Any]) -> pd.DataFrame:
    merged = pinn_optimizer_default_config()
    merged.update(config)
    return run_smoke_suite(merged)


def run_pinn_optimizer_tuning(config: dict[str, Any]) -> pd.DataFrame:
    merged = pinn_optimizer_default_config()
    merged.update(config)
    return run_tuning_suite(merged)


def run_pinn_optimizer_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    merged = pinn_optimizer_default_config()
    merged.update(config)
    return run_benchmark_suite(merged)


def run_pinn_optimizer_ablation(config: dict[str, Any]) -> pd.DataFrame:
    merged = pinn_optimizer_default_config()
    merged.update(config)
    output_dir = ensure_output_dir(merged)
    device = resolve_device(str(merged.get("device", "cpu")))
    seeds = list(merged.get("seeds", [11, 29, 47]))
    task_names = list(merged.get("ablation_tasks", ["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"]))
    variants = [
        {"variant_name": "full", "optimizer_name": "constraint_consensus_optimizer", "overrides": {}},
        {
            "variant_name": "no_recoverability",
            "optimizer_name": "constraint_consensus_optimizer",
            "overrides": {"recoverability_strength": 0.0},
        },
        {
            "variant_name": "no_agreement",
            "optimizer_name": "constraint_consensus_optimizer",
            "overrides": {"agreement_strength": 0.0, "balance_strength": 0.0},
        },
        {
            "variant_name": "no_memory",
            "optimizer_name": "constraint_consensus_optimizer",
            "overrides": {"use_memory": False, "memory_strength": 0.0},
        },
        {
            "variant_name": "no_projection",
            "optimizer_name": "constraint_consensus_optimizer",
            "overrides": {"use_projection": False, "projection_strength": 0.0},
        },
        {
            "variant_name": "neutral",
            "optimizer_name": "constraint_consensus_optimizer",
            "overrides": {
                "agreement_strength": 0.0,
                "recoverability_strength": 0.0,
                "balance_strength": 0.0,
                "conflict_penalty": 0.0,
                "memory_decay": 1.0,
                "memory_strength": 0.0,
                "projection_strength": 0.0,
                "max_update_ratio": 1.0,
                "min_scale": 1.0,
                "max_scale": 1.0,
                "use_memory": False,
                "use_projection": False,
            },
        },
        {"variant_name": "adamw", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop", "optimizer_name": "rmsprop", "overrides": {}},
        {"variant_name": "sgd_momentum", "optimizer_name": "sgd_momentum", "overrides": {}},
        {"variant_name": "lbfgs", "optimizer_name": "lbfgs", "overrides": {}},
        {"variant_name": "adamw_lbfgs_hybrid", "optimizer_name": "adamw_lbfgs_hybrid", "overrides": {}},
        {"variant_name": "topological_adam", "optimizer_name": "topological_adam", "overrides": {}},
        {"variant_name": "real_hamiltonian_adam", "optimizer_name": "real_hamiltonian_adam", "overrides": {}},
        {"variant_name": "magneto_hamiltonian_adam", "optimizer_name": "magneto_hamiltonian_adam", "overrides": {}},
    ]
    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="pinn_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(merged.get("ablation_epoch_scale", 0.75)),
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
    row = frame.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0]
    return row.to_dict()


def export_pinn_optimizer_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    prototype_name = "constraint_consensus_optimizer"
    baselines = ["adamw", "rmsprop", "sgd_momentum", "lbfgs", "adamw_lbfgs_hybrid", "topological_adam", "real_hamiltonian_adam", "magneto_hamiltonian_adam"]
    win_rows = []
    for baseline_name in baselines:
        if baseline_name in set(aggregated["optimizer"]):
            win_rows.append(compute_meaningful_wins(aggregated, prototype_name, baseline_name))
    win_flags = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    win_flags.to_csv(output_path / "win_flags.csv", index=False)

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame)
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="PINN Loss Curves",
        metric="train_loss",
        tasks=["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"],
        optimizers=["adamw", "lbfgs", "adamw_lbfgs_hybrid", "constraint_consensus_optimizer"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "recoverability_curves.png",
        title="Recoverability Score",
        metric="recoverability_score",
        tasks=["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"],
        optimizers=["constraint_consensus_optimizer"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "agreement_curves.png",
        title="Constraint Agreement",
        metric="constraint_agreement",
        tasks=["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"],
        optimizers=["constraint_consensus_optimizer"],
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "consensus_strength_curves.png",
        title="Consensus Strength",
        metric="consensus_strength",
        tasks=["pinn_harmonic_oscillator", "pinn_poisson_1d", "pinn_heat_equation"],
        optimizers=["constraint_consensus_optimizer"],
    )
    _plot_heatmap(aggregated[aggregated["optimizer"].isin(["adamw", "rmsprop", "lbfgs", "adamw_lbfgs_hybrid", "constraint_consensus_optimizer"])], figure_dir / "win_loss_heatmap.png")

    baseline_pool = aggregated[aggregated["optimizer"].isin(baselines)]
    strongest_baseline = baseline_pool.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0] if not baseline_pool.empty else None
    prototype_best = _best_row_for_optimizer(aggregated, prototype_name)
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )

    wins_summary = {}
    for baseline_name in baselines:
        frame = win_flags[win_flags["baseline"] == baseline_name] if not win_flags.empty else pd.DataFrame()
        wins_summary[baseline_name] = int(frame["win"].sum()) if not frame.empty else 0
    two_x_total = int(win_flags["two_x"].sum()) if not win_flags.empty else 0
    best_task = prototype_best["task"] if prototype_best is not None else "n/a"
    best_loss = prototype_best["mean_best_val_loss"] if prototype_best is not None else float("nan")

    report_lines = [
        "# PINN Optimizer Report",
        "",
        "## Strategy Alignment",
        "- The broad novel-optimizer search did not honestly beat conventional optimizers overall, so this suite pivots to a narrower and more plausible direction: a PINN-specific optimizer for conflicting residual/boundary/initial objectives.",
        "",
        "## Related Work",
        *[f"- [{entry['title']}]({entry['url']}): {entry['summary']}" for entry in PINN_RELATED_WORK],
        "",
        "## Optimizer Implemented",
        "- `ConstraintConsensusOptimizer`: update along the recoverable consensus of constraint gradients, rather than Adam moments or scalar loss reweighting.",
        "- Mathematical signal: pairwise directional agreement across PINN constraint gradients, residual recoverability under deterministic collocation perturbation, and support balance across constraint projections onto the chosen update direction.",
        "",
        "## What This Is Close To",
        "- It overlaps with gradient-conflict and loss-balancing PINN literature, so this is not claimed as a clean novelty result yet.",
        "- The potentially distinct part is using collocation-perturbed residual recoverability to select a consensus direction, instead of only reweighting losses or applying second-order preconditioning.",
        "",
        "## Tasks Tested",
        "- " + ", ".join(sorted(benchmark_frame["task"].unique())),
        "",
        "## Best Optimizer Per Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss"]]),
        "",
        "## Prototype Summary",
        f"- Best prototype task: `{best_task}` with mean best validation loss `{best_loss:.6f}`." if prototype_best is not None else "- Prototype did not run.",
        f"- Wins vs AdamW: {wins_summary['adamw']}",
        f"- Wins vs RMSProp: {wins_summary['rmsprop']}",
        f"- Wins vs SGD momentum: {wins_summary['sgd_momentum']}",
        f"- Wins vs L-BFGS: {wins_summary['lbfgs']}",
        f"- Wins vs AdamW→L-BFGS hybrid: {wins_summary['adamw_lbfgs_hybrid']}",
        f"- Wins vs Topological Adam: {wins_summary['topological_adam']}",
        f"- Wins vs Real Hamiltonian Adam: {wins_summary['real_hamiltonian_adam']}",
        "",
        f"## Two-X Events",
        f"- Total tracked 2x events: {two_x_total}",
        "",
        "## Strongest Baseline",
        (
            f"- `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best validation loss `{strongest_baseline['mean_best_val_loss']:.6f}`."
            if strongest_baseline is not None
            else "- No baseline rows available."
        ),
        "",
        "## Ablation Snapshot",
        _markdown_table(ablation_summary.head(12)),
        "",
        "## Recommendation",
        "- Keep this direction only if it wins on PINNs without hiding behind weaker baselines.",
        "- If it only beats AdamW but not L-BFGS or the AdamW→L-BFGS hybrid, treat it as a useful diagnostic optimizer, not a publishable replacement yet.",
        "- If the recoverability and agreement terms both help in ablation, the next step is a cleaner block-structured V2 rather than a broader all-domain optimizer.",
    ]
    (output_path / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "aggregated": aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
        "prototype_best": prototype_best,
    }

