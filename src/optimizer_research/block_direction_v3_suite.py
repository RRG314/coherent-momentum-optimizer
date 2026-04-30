from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .benchmarking import _train_single_run, run_benchmark_suite, run_smoke_suite, run_tuning_suite
from .block_direction_v2_suite import LITERATURE_ROWS
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v3"
V2_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v2"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v3",
    "block_direction_optimizer_v2",
    "block_direction_optimizer",
    "magneto_hamiltonian_adam",
    "real_hamiltonian_adam",
    "adamw",
    "adam",
    "rmsprop",
    "sgd",
    "sgd_momentum",
    "lion",
    "muon_hybrid",
    "topological_adam",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer_v2",
    "block_direction_optimizer",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "muon_hybrid",
    "topological_adam",
    "real_hamiltonian_adam",
    "magneto_hamiltonian_adam",
]


def block_direction_v3_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 4,
        "search_seed": 2037,
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "breast_cancer_mlp",
            "moons_mlp",
            "wine_mlp",
            "oscillatory_valley",
            "saddle_objective",
            "block_structure_classification",
            "low_rank_matrix_objective",
            "sparse_gradients_linear",
        ],
        "benchmark_tasks": [
            "breast_cancer_mlp",
            "wine_mlp",
            "moons_mlp",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
            "direction_reversal_objective",
            "block_structure_classification",
            "low_rank_matrix_objective",
        ],
        "smoke_tasks": ["oscillatory_valley", "breast_cancer_mlp", "block_structure_classification"],
        "smoke_optimizers": ["block_direction_optimizer_v3", "block_direction_optimizer_v2", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.35,
        "tuning_epoch_scale": 0.55,
        "benchmark_epoch_scale": 0.82,
        "ablation_epoch_scale": 0.42,
        "rule_search_epoch_scale": 0.38,
        "use_tuning_results": True,
    }


def _best_row_for_optimizer(aggregated: pd.DataFrame, optimizer_name: str) -> pd.Series | None:
    frame = aggregated[aggregated["optimizer"] == optimizer_name]
    if frame.empty:
        return None
    if frame["mean_best_val_accuracy"].notna().any():
        return frame.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
    return frame.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0]


def _prepare_docs(output_dir: Path) -> None:
    write_block_direction_v3_current_state(output_dir)
    write_block_direction_v3_literature_scan(output_dir)
    write_block_direction_v3_math_definition(output_dir)


def write_block_direction_v3_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV3 Current State",
        "",
        "BlockDirectionOptimizerV3 is the report-backed successor to V2, not a replacement for V1 or V2.",
        "",
        "## What the existing reports said to change",
        "",
        "- Keep blockwise direction memory and trust-scored candidate selection.",
        "- Keep recoverability only as a gate, because it helped weakly at best in V2.",
        "- Remove projection and orthogonal escape from the default path because they did not help broadly.",
        "- Prefer simple winner-take-all selection over richer blending by default.",
        "- Strengthen the matrix story with a better structured candidate instead of generic low-rank rhetoric.",
        "",
        "## V2 benchmark state that motivated V3",
        "",
        "- V2 already beat V1 clearly and beat AdamW often on oscillatory, saddle, plateau, and direction-reversal tasks.",
        "- V2 did not broadly replace RMSProp or SGD momentum.",
        "- The strongest V2 rule-search variant was `scalar_wta`, which showed the branch wanted a sharper default path.",
        "- The weakest remaining weakness was structure quality on low-rank and matrix-heavy tasks.",
        "",
        "## What V3 changes",
        "",
        "- Default selection mode is `winner_take_all`.",
        "- Default block strategy is `smart_v3`, which uses row blocks for matrices and scalar blocks for vectors.",
        "- The `low_rank_matrix` slot is repurposed into a row/column consensus matrix candidate by default.",
        "- Projection and orthogonal escape remain available only as stress-specialist variants, not the default rule.",
        "- A stable-consensus candidate lets ordinary blocks follow the intersection of gradient, memory, and matrix agreement instead of falling back to brittle stress candidates.",
        "- A regime gate suppresses stress-specialist candidates when the block is stable, then re-enables them when oscillation, conflict, or gradient shock rises.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")
    return {}


def write_block_direction_v3_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BlockDirectionOptimizerV3 Literature Scan",
        "",
        "V3 inherits the same core literature pressure as V2: strong optimizers usually transform one gradient direction, while this branch tries to **select** a direction at block level.",
        "",
        _markdown_table(
            frame[
                [
                    "family",
                    "representative_method",
                    "direction_or_transform",
                    "scope",
                    "difference_from_block_direction_v2",
                    "required_baseline",
                ]
            ]
        ),
        "",
        "## What matters most for V3",
        "",
        "- Muon/Shampoo/K-FAC remain the main structure-aware comparison pressure.",
        "- PCGrad/CAGrad remain the most relevant direction-selection relatives when explicit conflict gradients are available.",
        "- SAM remains the closest perturbation-based trust baseline, but it perturbs the objective rather than gating candidate directions.",
        "- The novelty opening is still only plausible if V3 wins through **blockwise candidate selection with structured matrix candidates**, not by resembling Adam or Muon with extra words.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v3_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV3 Mathematical Definition",
        "",
        "Split parameters into blocks `B_i` with block parameters `q_i` and gradients `g_i`.",
        "",
        "## Candidate directions",
        "",
        "For each block `i`, build candidates",
        "",
        "`D_i = {d_i^(grad), d_i^(norm), d_i^(cons), d_i^(trust), d_i^(smooth), d_i^(topk), d_i^(matrix), d_i^(sign)}`",
        "",
        "with optional stress-specialist candidates `d_i^(proj)` and `d_i^(orth)` kept outside the default path.",
        "",
        "The stable-consensus candidate is",
        "",
        "`d_i^(cons) = normalize(d_i^(grad) + a_i d_i^(trust) + b_i d_i^(smooth) + c_i d_i^(matrix))`",
        "",
        "where the coefficients are gated by positive alignment with the current negative gradient direction.",
        "",
        "The default structured matrix candidate for 2D parameters is a row/column consensus direction:",
        "",
        "`R = row_normalize(-G)`",
        "",
        "`C = col_normalize(-G)`",
        "",
        "`d^(matrix) = alpha * R * ||G||_row^gamma + (1-alpha) * C^T * ||G||_col^(1-gamma)`",
        "",
        "where `alpha` is the row/column mix and `gamma` controls how much row-vs-column energy enters the candidate.",
        "",
        "## Trust score",
        "",
        "For candidate `c` in block `i`, the trust score is",
        "",
        "`T_i(c) = w_d A_i(c) + w_m M_i(c) + w_q Q_i(c) + w_r G_i(c)R_i(c) + w_s S_i(c) - w_o O_i(c) - w_c C_i(c) - w_k cost(c)`",
        "",
        "where:",
        "",
        "- `A_i(c)` is predicted descent alignment with the current negative gradient direction",
        "- `M_i(c)` is coherence with trusted direction memory",
        "- `Q_i(c)` is improvement-history memory for that candidate",
        "- `R_i(c)` is the recoverability score under cheap masking/noise/drop perturbations",
        "- `G_i(c)` is the recoverability gate thresholding `R_i(c)`",
        "- `S_i(c)` is a norm-stability score",
        "- `O_i(c)` is oscillation against previous gradient/update directions",
        "- `C_i(c)` is conflict against trusted/update memory",
        "- `Z_i` is a stress gate from oscillation, trusted-direction conflict, and gradient-norm shock",
        "",
        "## Selection",
        "",
        "V3 defaults to",
        "",
        "`c_i^* = argmax_c T_i(c)`",
        "",
        "with `winner_take_all` selection. `top2_blend`, `softmax_weighted_average`, and `fallback_to_gradient` remain ablation/search modes.",
        "",
        "## Step magnitude",
        "",
        "The chosen block direction `u_i = normalize(d_i^(c_i^*))` is scaled by",
        "",
        "`alpha_i = min( eta * lambda_i * ||g_i|| * (0.4 + 0.6 A_i(c_i^*)) / |B_i|^p, rho (||q_i|| + eps) )`",
        "",
        "where `lambda_i` is a bounded trust-scale derived from the selected trust score, recoverability EMA, and stable-consensus support. Stress-specialist candidates also pay an extra penalty when `Z_i` is low.",
        "",
        "## Why V3 is still non-Adam",
        "",
        "- No Adam first-moment EMA defines the main direction.",
        "- No Adam second-moment preconditioner defines the main step.",
        "- Block memories exist only to score and stabilize candidate directions.",
        "- The core decision is still blockwise direction choice.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v3_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v3_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v3", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v3_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v3_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "oscillatory_valley",
                "saddle_objective",
                "plateau_escape_objective",
                "direction_reversal_objective",
                "conflicting_batches_classification",
                "block_structure_classification",
                "low_rank_matrix_objective",
            ],
        )
    )
    variants = [
        {"variant_name": "v3_full", "optimizer_name": "block_direction_optimizer_v3", "overrides": {}},
        {
            "variant_name": "no_recoverability_gate",
            "optimizer_name": "block_direction_optimizer_v3",
            "overrides": {
                "recoverability_weight": 0.0,
                "recoverability_samples": 0,
                "recoverability_noise_scale": 0.0,
                "recoverability_drop_fraction": 0.0,
            },
        },
        {
            "variant_name": "no_direction_memory",
            "optimizer_name": "block_direction_optimizer_v3",
            "overrides": {
                "use_trusted_direction_candidate": False,
                "use_smoothed_direction_candidate": False,
                "coherence_weight": 0.0,
                "memory_decay": 1.0,
                "smooth_decay": 1.0,
            },
        },
        {
            "variant_name": "no_stable_consensus",
            "optimizer_name": "block_direction_optimizer_v3",
            "overrides": {
                "use_stable_consensus_candidate": False,
                "stable_consensus_bonus": 0.0,
                "matrix_consensus_bonus": 0.0,
            },
        },
        {
            "variant_name": "no_regime_gating",
            "optimizer_name": "block_direction_optimizer_v3",
            "overrides": {
                "stress_gate_threshold": 1.0,
                "stress_candidate_penalty": 0.0,
            },
        },
        {"variant_name": "no_matrix_consensus", "optimizer_name": "block_direction_optimizer_v3", "overrides": {"use_low_rank_candidate": False}},
        {"variant_name": "no_block_structure", "optimizer_name": "block_direction_optimizer_v3", "overrides": {"block_strategy": "tensor"}},
        {"variant_name": "no_conflict_penalty", "optimizer_name": "block_direction_optimizer_v3", "overrides": {"conflict_penalty": 0.0}},
        {"variant_name": "v2_baseline", "optimizer_name": "block_direction_optimizer_v2", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v3_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.42)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def search_block_direction_v3_rules(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "rule_search_tasks",
            [
                "oscillatory_valley",
                "saddle_objective",
                "plateau_escape_objective",
                "direction_reversal_objective",
                "block_structure_classification",
                "low_rank_matrix_objective",
            ],
        )
    )
    variants = [
        {"variant_name": "default_wta", "overrides": {}},
        {"variant_name": "scalar_wta", "overrides": {"block_strategy": "scalar", "selection_mode": "winner_take_all"}},
        {
            "variant_name": "consensus_default",
            "overrides": {
                "selection_mode": "winner_take_all",
                "use_stable_consensus_candidate": True,
                "stable_consensus_bonus": 0.16,
                "matrix_consensus_bonus": 0.08,
            },
        },
        {"variant_name": "rank1_svd", "overrides": {"matrix_candidate_mode": "rank1_svd"}},
        {
            "variant_name": "recoverability_gated",
            "overrides": {
                "recoverability_weight": 0.12,
                "recoverability_samples": 2,
                "selection_mode": "winner_take_all",
            },
        },
        {
            "variant_name": "stress_specialist",
            "overrides": {
                "selection_mode": "top2_blend",
                "use_projection_candidate": True,
                "use_orthogonal_escape_candidate": True,
                "projection_strength": 0.35,
                "orthogonal_strength": 0.35,
            },
        },
        {
            "variant_name": "magneto_informed",
            "overrides": {
                "coherence_weight": 0.30,
                "conflict_penalty": 0.20,
                "oscillation_penalty": 0.20,
                "use_projection_candidate": True,
                "projection_strength": 0.25,
            },
        },
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v3_rule_search",
                    task_name=task_name,
                    optimizer_name="block_direction_optimizer_v3",
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("rule_search_epoch_scale", 0.38)),
                )
                row["variant_name"] = variant["variant_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "rule_search_results.csv", index=False)
    return frame


def _plot_bar(frame: pd.DataFrame, output_path: Path, x_col: str, y_col: str, title: str, rotation: int = 35) -> None:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return
    ordered = frame.sort_values(y_col, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(ordered[x_col], ordered[y_col])
    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _variant_delta(frame: pd.DataFrame, full_name: str, compare_name: str) -> float | None:
    summary = frame.groupby("variant_name", as_index=False)["selection_score"].mean()
    if full_name not in set(summary["variant_name"]) or compare_name not in set(summary["variant_name"]):
        return None
    full_score = float(summary.loc[summary["variant_name"] == full_name, "selection_score"].iloc[0])
    compare_score = float(summary.loc[summary["variant_name"] == compare_name, "selection_score"].iloc[0])
    return full_score - compare_score


def _format_optional_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def export_block_direction_v3_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")
    rule_frame = pd.read_csv(output_path / "rule_search_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v3_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v3")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum", "lion", "muon_hybrid", "topological_adam"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v3", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }

    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    rule_summary = (
        rule_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )

    recoverability_delta = _variant_delta(ablation_frame, "v3_full", "no_recoverability_gate")
    stable_consensus_delta = _variant_delta(ablation_frame, "v3_full", "no_stable_consensus")
    regime_delta = _variant_delta(ablation_frame, "v3_full", "no_regime_gating")
    block_delta = _variant_delta(ablation_frame, "v3_full", "no_block_structure")
    matrix_delta = _variant_delta(ablation_frame, "v3_full", "no_matrix_consensus")
    projection_delta = _variant_delta(ablation_frame, "v3_full", "projection_stress_preset")
    orthogonal_delta = _variant_delta(ablation_frame, "v3_full", "stress_escape_preset")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "block_direction_optimizer_v3",
        "block_direction_optimizer_v2",
        "magneto_hamiltonian_adam",
        "real_hamiltonian_adam",
        "adamw",
        "rmsprop",
        "sgd_momentum",
    ]
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Validation loss curves",
        metric="val_loss",
        tasks=["oscillatory_valley", "block_structure_classification", "low_rank_matrix_objective"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "accuracy_curves.png",
        title="Validation accuracy curves",
        metric="val_accuracy",
        tasks=["breast_cancer_mlp", "moons_mlp", "conflicting_batches_classification"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png")
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV3 ablation", 45)
    _plot_bar(rule_summary, figure_dir / "rule_search_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV3 rule search", 45)
    runtime_summary = aggregated[aggregated["optimizer"].isin(comparison_optimizers)][["optimizer", "mean_runtime_per_step_ms", "mean_optimizer_state_mb"]]
    _plot_bar(runtime_summary, figure_dir / "runtime_comparison.png", "optimizer", "mean_runtime_per_step_ms", "Runtime per step")
    _plot_bar(runtime_summary, figure_dir / "memory_comparison.png", "optimizer", "mean_optimizer_state_mb", "Optimizer state size")

    family_summary = (
        benchmark_frame.groupby(["task_family", "optimizer"], as_index=False)["selection_score"]
        .mean()
        .sort_values(["task_family", "selection_score"], ascending=[True, False])
    )
    _plot_bar(
        family_summary[family_summary["optimizer"].isin(comparison_optimizers)],
        figure_dir / "task_family_ranking.png",
        "optimizer",
        "selection_score",
        "Task-family selection score ranking",
    )

    v3_vs = {name: int(frame["win"].sum()) for name, frame in win_map.items()}
    v3_two_x = {name: int(frame["two_x"].sum()) for name, frame in win_map.items()}
    best_rule = None if rule_summary.empty else str(rule_summary.iloc[0]["variant_name"])
    best_ablation = None if ablation_summary.empty else str(ablation_summary.iloc[0]["variant_name"])

    task_family_scores = (
        benchmark_frame[benchmark_frame["optimizer"] == "block_direction_optimizer_v3"]
        .groupby("task_family", as_index=False)["selection_score"]
        .mean()
        .sort_values("selection_score", ascending=False)
    )
    best_family = None if task_family_scores.empty else str(task_family_scores.iloc[0]["task_family"])

    lines = [
        "# BlockDirectionOptimizerV3 Final Report",
        "",
        "## 1. What BlockDirectionOptimizerV3 is",
        "",
        "- V3 is a blockwise candidate-direction optimizer that keeps the V2 trust-scoring framework but simplifies the default path.",
        "- It is not an Adam-family optimizer: no first-moment EMA or second-moment preconditioner defines the main direction.",
        "",
        "## 2. How V3 differs from V2 and the baselines",
        "",
        "- V3 defaults to `winner_take_all` rather than blended selection.",
        "- V3 keeps recoverability as a gate, not a central generator.",
        "- V3 turns off projection and orthogonal escape by default and reserves them for stress-specialist variants.",
        "- V3 adds a row/column matrix-consensus candidate so the structured branch has a concrete matrix story beyond generic low-rank language.",
        "- V3 adds a stable-consensus candidate and a regime gate so ordinary tasks can stay close to gradient-plus-memory consensus while stress tasks still unlock escape behavior.",
        "",
        "## 3. Literature findings",
        "",
        "- The literature still does not support a novelty claim for 'block structure' alone.",
        "- The plausible distinct part is blockwise candidate-direction choice with explicit trust, structured matrix candidates, and non-Adam state.",
        "- The closest pressure remains Muon/Shampoo/K-FAC on structure and PCGrad/CAGrad on conflict-aware direction choice.",
        "",
        "## 4. Strongest baseline comparison",
        "",
        f"- Strongest baseline row in this suite: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best validation loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best validation accuracy `{float(strongest_baseline['mean_best_val_accuracy']) if pd.notna(strongest_baseline['mean_best_val_accuracy']) else float('nan'):.4f}`.",
        f"- Best V3 row: `{None if v3_row is None else v3_row['task']}` with mean best validation loss `{float(v3_row['mean_best_val_loss']) if v3_row is not None else float('nan'):.6f}` and mean best validation accuracy `{float(v3_row['mean_best_val_accuracy']) if v3_row is not None and pd.notna(v3_row['mean_best_val_accuracy']) else float('nan'):.4f}`.",
        "",
        "## 5. Comparison counts",
        "",
        f"- V3 vs V2: `{v3_vs.get('block_direction_optimizer_v2', 0)}` meaningful wins and `{v3_two_x.get('block_direction_optimizer_v2', 0)}` tracked 2x events",
        f"- V3 vs V1: `{v3_vs.get('block_direction_optimizer', 0)}` meaningful wins and `{v3_two_x.get('block_direction_optimizer', 0)}` tracked 2x events",
        f"- V3 vs AdamW: `{v3_vs.get('adamw', 0)}` meaningful wins and `{v3_two_x.get('adamw', 0)}` tracked 2x events",
        f"- V3 vs RMSProp: `{v3_vs.get('rmsprop', 0)}` meaningful wins and `{v3_two_x.get('rmsprop', 0)}` tracked 2x events",
        f"- V3 vs SGD momentum: `{v3_vs.get('sgd_momentum', 0)}` meaningful wins and `{v3_two_x.get('sgd_momentum', 0)}` tracked 2x events",
        f"- V3 vs Muon hybrid: `{v3_vs.get('muon_hybrid', 0)}` meaningful wins and `{v3_two_x.get('muon_hybrid', 0)}` tracked 2x events",
        f"- V3 vs Real Hamiltonian Adam: `{v3_vs.get('real_hamiltonian_adam', 0)}` meaningful wins and `{v3_two_x.get('real_hamiltonian_adam', 0)}` tracked 2x events",
        f"- V3 vs MagnetoHamiltonianAdam: `{v3_vs.get('magneto_hamiltonian_adam', 0)}` meaningful wins and `{v3_two_x.get('magneto_hamiltonian_adam', 0)}` tracked 2x events",
        "",
        "## 6. Ablation findings",
        "",
        _markdown_table(ablation_summary.head(10)),
        "",
        f"- Recoverability helped broadly: `{bool(recoverability_delta is not None and recoverability_delta > 0.01)}` (delta full - no_recoverability = `{_format_optional_delta(recoverability_delta)}`).",
        f"- Stable consensus helped broadly: `{bool(stable_consensus_delta is not None and stable_consensus_delta > 0.01)}` (delta full - no_stable_consensus = `{_format_optional_delta(stable_consensus_delta)}`).",
        f"- Regime gating helped broadly: `{bool(regime_delta is not None and regime_delta > 0.01)}` (delta full - no_regime_gating = `{_format_optional_delta(regime_delta)}`).",
        f"- Block structure helped broadly: `{bool(block_delta is not None and block_delta > 0.01)}` (delta full - no_block_structure = `{_format_optional_delta(block_delta)}`).",
        f"- Matrix consensus helped broadly: `{bool(matrix_delta is not None and matrix_delta > 0.01)}` (delta full - no_matrix_consensus = `{_format_optional_delta(matrix_delta)}`).",
        f"- Projection helped broadly: `{bool(projection_delta is not None and projection_delta > 0.01)}` (delta full - projection_stress_preset = `{_format_optional_delta(projection_delta)}`).",
        f"- Orthogonal escape helped broadly: `{bool(orthogonal_delta is not None and orthogonal_delta > 0.01)}` (delta full - stress_escape_preset = `{_format_optional_delta(orthogonal_delta)}`).",
        "",
        "## 7. Rule-search findings",
        "",
        _markdown_table(rule_summary.head(10)),
        "",
        f"- Best discovered rule combination: `{best_rule}`.",
        f"- Best ablation row overall: `{best_ablation}`.",
        "",
        "## 8. Specialist domain",
        "",
        f"- Best task family for V3 in this suite: `{best_family}`.",
        "- The branch is expected to help most when useful directions persist at block level but raw gradients conflict, rotate, or become noisy.",
        "",
        "## 9. Verdict",
        "",
        "- If V3 beats V2 and at least one strong baseline on held-out stress or structure tasks, it is a serious specialist branch.",
        "- If it still loses broadly to RMSProp/SGD momentum on standard tasks, it remains a specialist rather than a default optimizer.",
        "- Novelty is still prototype-level unless the structured-matrix and direction-stress gains remain stable under repeated runs.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "best_v3": None if v3_row is None else v3_row.to_dict(),
        "strongest_baseline": strongest_baseline.to_dict(),
        "wins": v3_vs,
        "two_x": v3_two_x,
        "best_rule": best_rule,
        "best_family": best_family,
        "recoverability_delta": recoverability_delta,
        "stable_consensus_delta": stable_consensus_delta,
        "regime_delta": regime_delta,
        "block_delta": block_delta,
        "matrix_delta": matrix_delta,
        "projection_delta": projection_delta,
        "orthogonal_delta": orthogonal_delta,
        "tuning_rows": int(len(tuning_frame)),
    }
