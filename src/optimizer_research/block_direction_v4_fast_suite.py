from __future__ import annotations

import json
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v4_fast"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v4_fast",
    "block_direction_optimizer_v4_fast_legacy",
    "block_direction_optimizer_v42",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer_v4_fast_legacy",
    "block_direction_optimizer_v42",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]


def block_direction_v4_fast_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 3,
        "search_seed": 2051,
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "breast_cancer_mlp",
            "moons_mlp",
            "wine_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
            "low_rank_matrix_objective",
        ],
        "benchmark_tasks": [
            "breast_cancer_mlp",
            "moons_mlp",
            "wine_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
            "direction_reversal_objective",
            "block_structure_classification",
            "low_rank_matrix_objective",
        ],
        "smoke_tasks": ["oscillatory_valley", "breast_cancer_mlp"],
        "smoke_optimizers": ["block_direction_optimizer_v4_fast", "block_direction_optimizer_v4_fast_legacy", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.35,
        "tuning_epoch_scale": 0.50,
        "benchmark_epoch_scale": 0.75,
        "ablation_epoch_scale": 0.40,
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
    write_block_direction_v4_fast_current_state(output_dir)
    write_block_direction_v4_fast_literature_scan(output_dir)
    write_block_direction_v4_fast_math_definition(output_dir)


def write_block_direction_v4_fast_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4Fast Current State",
        "",
        "V4Fast now serves as the consolidated main block branch and keeps the legacy fast path and V4.2 as explicit comparison baselines.",
        "",
        "## Why V4Fast exists",
        "",
        "- Earlier block branches showed two durable strengths: V4Fast had the best dense/stress speed-quality balance, while V4.2 had the best CNN/block behavior.",
        "- The heavy V4.2/V4.3/V4.4/V4.5 additions did not all pay for themselves.",
        "- The pieces that kept validating were simpler: block structure, trusted direction memory, stable consensus, typed conv/dense profiles, and conv-safe step control.",
        "",
        "## What V4Fast changes",
        "",
        "- Keeps only four default candidates: `gradient`, `stable_consensus`, `trusted_direction`, and `low_rank_matrix`.",
        "- Uses `winner_take_all` by default.",
        "- Removes per-candidate recoverability from the default hot path.",
        "- Uses typed conv/dense profiles inside the same optimizer class.",
        "- Uses cheap conv structure support and conv-safe scaling instead of the heavier V4.2 support path.",
        "- Keeps `smart_v4` grouping and block-energy normalization.",
        "",
        "## What V4Fast is trying to prove",
        "",
        "- Preserve the old V4Fast stress-task edge.",
        "- Absorb the useful CNN/general-task behavior from V4.2 without taking on the full V4.2 runtime and logic cost.",
        "- Keep one main block optimizer line instead of multiple near-duplicate descendants.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")


def write_block_direction_v4_fast_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BlockDirectionOptimizerV4Fast Literature Scan",
        "",
        "V4Fast keeps the same literature pressure as the V2 and V3 block-direction branch: it is only interesting if blockwise direction **selection** does useful work that standard gradient **transforms** are not already doing.",
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
        "## V4Fast-specific interpretation",
        "",
        "- Muon, Shampoo, and K-FAC are still the main structure-aware baseline pressure.",
        "- SAM remains the most relevant perturbation-trust baseline, but V4Fast does not perturb the objective itself.",
        "- The novelty opening is still the same narrow one: blockwise candidate-direction choice with lightweight structure and memory, not another adaptive-gradient transform.",
        "- The merged V4Fast path is specifically testing whether typed tensor profiles and cheap structure-aware step control improve practicality without collapsing the branch into Adam-like logic.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v4_fast_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4Fast Mathematical Definition",
        "",
        "Split parameters into blocks `B_i` and compute block gradients `g_i`.",
        "",
        "## Candidate set",
        "",
        "V4Fast keeps a small candidate set per block:",
        "",
        "`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`",
        "",
        "where:",
        "",
        "- `d_i^(grad)` is the normalized negative gradient direction",
        "- `d_i^(trust)` is the normalized trusted-direction memory",
        "- `d_i^(matrix)` is a structured row/column matrix consensus direction for 2D tensors",
        "- `d_i^(cons)` is a stable consensus direction built from the candidates that currently agree",
        "",
        "For the consensus candidate:",
        "",
        "`d_i^(cons) = normalize(d_i^(grad) + a_i d_i^(trust) + b_i d_i^(smooth) + c_i d_i^(matrix))`",
        "",
        "with coefficients only contributing when they align positively with the current descent direction.",
        "",
        "## Trust score",
        "",
        "Each candidate receives",
        "",
        "`T_i(d) = w_d A_i(d) + w_m M_i(d) + w_q Q_i(d) + w_s S_i(d) + w_c C_i(d) - w_o O_i(d) - w_f F_i(d) - w_k cost(d)`",
        "",
        "where:",
        "",
        "- `A_i(d)` is descent alignment with the current negative gradient direction",
        "- `M_i(d)` is coherence with trusted and smoothed direction memory",
        "- `Q_i(d)` is candidate improvement-history memory",
        "- `S_i(d)` is norm stability relative to block gradient EMA",
        "- `C_i(d)` is consensus support",
        "- `O_i(d)` is oscillation against previous gradient and update directions",
        "- `F_i(d)` is conflict against trusted direction and previous update memory",
        "",
        "## Selection",
        "",
        "V4Fast defaults to",
        "",
        "`d_i^* = argmax_{d in D_i} T_i(d)`",
        "",
        "with winner-take-all selection.",
        "",
        "## Typed conv/dense profiles",

        "The merged V4Fast branch keeps one optimizer class but applies a typed profile split.",
        "",
        "- Dense/vector tensors stay on the original V4Fast-style scoring and scaling defaults.",
        "- Convolutional tensors use a safer profile: lower effective step cap, slightly stronger stability/coherence emphasis, and conv-safe scaling derived from cheap structure support.",
        "",
        "For convolutional tensors, cheap structure support is estimated from filter-slice coherence:",
        "",
        "- channel coherence from normalized input-channel slices",
        "- spatial coherence from normalized spatial-position slices",
        "- bank coherence from alignment with the mean filter direction",
        "",
        "These are combined into `support_i in [0, 1]` for each conv block.",
        "",
        "## Step magnitude",
        "",
        "The chosen direction is scaled by blockwise energy normalization:",
        "",
        "`alpha_i = eta * lambda_i * ||g_i|| / (sqrt(E_i) + eps)^p`",
        "",
        "where `E_i` is an EMA of mean squared gradient energy in block `i`, `p` is `energy_power`, and `lambda_i` is the bounded trust scale. A parameter-relative cap prevents destructive overshoot.",
        "",
        "For conv blocks, V4Fast now uses support-aware scaling:",
        "",
        "`alpha_i^(conv) = alpha_i * (f + (1-f) support_i)`",
        "",
        "and raises the effective energy exponent when support is weak, so noisy conv filters get cooled without changing the candidate-direction rule itself.",
        "",
        "For conv blocks, V4Fast now uses support-aware scaling:",
        "",
        "`alpha_i^(conv) = alpha_i * (f + (1-f) support_i)`",
        "",
        "and raises the effective energy exponent when support is weak, so noisy conv filters get cooled without changing the candidate-direction rule itself.",
        "",
        "## Why V4Fast is still non-Adam",
        "",
        "- No Adam first-moment EMA chooses the direction.",
        "- No Adam per-coordinate second-moment preconditioner chooses the direction.",
        "- The gradient-energy EMA only scales block step size; the novelty remains blockwise direction selection.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v4_fast_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v4_fast_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v4_fast", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v4_fast_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v4_fast_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(config.get("ablation_tasks", ["breast_cancer_mlp", "digits_cnn", "oscillatory_valley", "low_rank_matrix_objective"]))
    variants = [
        {"variant_name": "v4_fast_full", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {}},
        {"variant_name": "v4_fast_legacy", "optimizer_name": "block_direction_optimizer_v4_fast_legacy", "overrides": {}},
        {"variant_name": "v42_baseline", "optimizer_name": "block_direction_optimizer_v42", "overrides": {}},
        {"variant_name": "no_conv_profile_split", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_typed_profiles": False}},
        {"variant_name": "no_conv_structure_support", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_conv_structure_support": False, "conv_energy_power_bonus": 0.0, "conv_step_floor": 1.0}},
        {"variant_name": "no_conv_support_bonus", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"conv_consensus_bonus": 0.0, "conv_memory_bonus": 0.0}},
        {"variant_name": "no_conv_fallback_relaxation", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"conv_fallback_relaxation": 0.0}},
        {"variant_name": "no_stable_consensus", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_stable_consensus_candidate": False, "stable_consensus_bonus": 0.0}},
        {"variant_name": "no_trusted_direction", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_trusted_direction_candidate": False, "coherence_weight": 0.0}},
        {"variant_name": "no_matrix_consensus", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_low_rank_candidate": False, "matrix_consensus_bonus": 0.0}},
        {"variant_name": "no_block_structure", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"block_strategy": "tensor"}},
        {"variant_name": "recoverability_periodic", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {"use_recoverability_gate": True, "recoverability_weight": 0.10, "recoverability_interval": 8}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v4_fast_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.40)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
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


def export_block_direction_v4_fast_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v4_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v4_fast")
    legacy_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v4_fast_legacy")
    v42_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v42")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v4_fast", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )

    profile_split_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_conv_profile_split")
    conv_support_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_conv_structure_support")
    conv_bonus_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_conv_support_bonus")
    conv_fallback_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_conv_fallback_relaxation")
    stable_consensus_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_stable_consensus")
    memory_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_trusted_direction")
    matrix_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_matrix_consensus")
    block_delta = _variant_delta(ablation_frame, "v4_fast_full", "no_block_structure")
    recoverability_delta = _variant_delta(ablation_frame, "v4_fast_full", "recoverability_periodic")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "block_direction_optimizer_v4_fast",
        "block_direction_optimizer_v4_fast_legacy",
        "block_direction_optimizer_v42",
        "magneto_hamiltonian_adam",
        "adamw",
        "rmsprop",
        "sgd_momentum",
    ]
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Validation loss curves",
        metric="val_loss",
        tasks=["breast_cancer_mlp", "digits_cnn", "oscillatory_valley"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png")
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV4Fast ablation", 45)
    runtime_summary = aggregated[aggregated["optimizer"].isin(comparison_optimizers)][["optimizer", "mean_runtime_per_step_ms", "mean_optimizer_state_mb"]]
    _plot_bar(runtime_summary, figure_dir / "runtime_comparison.png", "optimizer", "mean_runtime_per_step_ms", "Runtime per step")
    _plot_bar(runtime_summary, figure_dir / "memory_comparison.png", "optimizer", "mean_optimizer_state_mb", "Optimizer state size")

    v4_vs = {name: int(frame["win"].sum()) for name, frame in win_map.items()}
    v4_two_x = {name: int(frame["two_x"].sum()) for name, frame in win_map.items()}
    best_ablation = None if ablation_summary.empty else str(ablation_summary.iloc[0]["variant_name"])

    family_summary = (
        benchmark_frame[benchmark_frame["optimizer"] == "block_direction_optimizer_v4_fast"]
        .groupby("task_family", as_index=False)["selection_score"]
        .mean()
        .sort_values("selection_score", ascending=False)
    )
    best_family = None if family_summary.empty else str(family_summary.iloc[0]["task_family"])

    lines = [
        "# BlockDirectionOptimizerV4Fast Final Report",
        "",
        "## 1. What V4Fast is",
        "",
        "- V4Fast is now the consolidated main block branch.",
        "- It keeps the novel blockwise direction-selection principle, preserves the fast dense/stress core, and folds in the lowest-cost validated CNN/general-task elements from V4.2.",
        "",
        "## 2. How merged V4Fast differs from the older fast path",
        "",
        "- Candidate set stays reduced to gradient, stable consensus, trusted direction, and matrix consensus.",
        "- Winner-take-all selection stays the default.",
        "- Recoverability stays out of the default hot path.",
        "- Dense/vector tensors use the fast original profile.",
        "- Convolutional tensors use a typed conv profile, cheap structure-aware conv-safe scaling, and a low-cost conv-aware step rule.",
        "",
        "## 3. Best rows",
        "",
        f"- Best V4Fast row: `{v4_row['task']}` with mean best val loss `{float(v4_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v4_row['mean_best_val_accuracy']):.6f}`." if v4_row is not None else "- Best V4Fast row: unavailable.",
        f"- Best legacy V4Fast row: `{legacy_row['task']}` with mean best val loss `{float(legacy_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(legacy_row['mean_best_val_accuracy']):.6f}`." if legacy_row is not None else "- Best legacy V4Fast row: unavailable.",
        f"- Best V4.2 row: `{v42_row['task']}` with mean best val loss `{float(v42_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v42_row['mean_best_val_accuracy']):.6f}`." if v42_row is not None else "- Best V4.2 row: unavailable.",
        f"- Strongest baseline row: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(strongest_baseline['mean_best_val_accuracy']):.6f}`.",
        "",
        "## 4. Competitive summary",
        "",
        f"- V4Fast wins vs legacy V4Fast: `{v4_vs.get('block_direction_optimizer_v4_fast_legacy', 0)}` meaningful wins.",
        f"- V4Fast wins vs V4.2: `{v4_vs.get('block_direction_optimizer_v42', 0)}` meaningful wins.",
        f"- V4Fast wins vs AdamW: `{v4_vs.get('adamw', 0)}` meaningful wins; tracked 2x wins `{v4_two_x.get('adamw', 0)}`.",
        f"- V4Fast wins vs RMSProp: `{v4_vs.get('rmsprop', 0)}` meaningful wins.",
        f"- V4Fast wins vs SGD momentum: `{v4_vs.get('sgd_momentum', 0)}` meaningful wins.",
        f"- V4Fast wins vs MagnetoHamiltonianAdam: `{v4_vs.get('magneto_hamiltonian_adam', 0)}` meaningful wins.",
        "",
        "## 5. Runtime",
        "",
        f"- Mean V4Fast runtime per step: `{float(v4_row['mean_runtime_per_step_ms']):.4f} ms`." if v4_row is not None else "- Mean V4Fast runtime per step: unavailable.",
        f"- Mean legacy V4Fast runtime per step: `{float(legacy_row['mean_runtime_per_step_ms']):.4f} ms`." if legacy_row is not None else "- Mean legacy V4Fast runtime per step: unavailable.",
        f"- Mean V4.2 runtime per step: `{float(v42_row['mean_runtime_per_step_ms']):.4f} ms`." if v42_row is not None else "- Mean V4.2 runtime per step: unavailable.",
        "",
        "## 6. Ablation findings",
        "",
        f"- Typed conv/dense split delta vs removal: `{_format_optional_delta(profile_split_delta)}`.",
        f"- Cheap conv structure support delta vs removal: `{_format_optional_delta(conv_support_delta)}`.",
        f"- Conv support bonus delta vs removal: `{_format_optional_delta(conv_bonus_delta)}`.",
        f"- Conv fallback-relaxation delta vs removal: `{_format_optional_delta(conv_fallback_delta)}`.",
        f"- Stable consensus delta vs removal: `{_format_optional_delta(stable_consensus_delta)}`.",
        f"- Trusted-direction memory delta vs removal: `{_format_optional_delta(memory_delta)}`.",
        f"- Matrix consensus delta vs removal: `{_format_optional_delta(matrix_delta)}`.",
        f"- Block structure delta vs tensor collapse: `{_format_optional_delta(block_delta)}`.",
        f"- Recoverability periodic gate delta: `{_format_optional_delta(recoverability_delta)}`.",
        f"- Best ablation row overall: `{best_ablation}`." if best_ablation is not None else "- Best ablation row overall: unavailable.",
        "",
        "## 7. Honest conclusion",
        "",
        f"- Best task family for V4Fast: `{best_family}`." if best_family is not None else "- Best task family for V4Fast: unavailable.",
        "- The merged V4Fast only counts as a success if it keeps the old stress-task strengths and improves the broader task mix relative to legacy V4Fast and V4.2.",
        "- If RMSProp or SGD momentum still dominate broadly, V4Fast remains a specialist optimizer rather than a new default.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "best_v4_task": None if v4_row is None else str(v4_row["task"]),
        "best_v4_loss": None if v4_row is None else float(v4_row["mean_best_val_loss"]),
        "best_v4_accuracy": None if v4_row is None else float(v4_row["mean_best_val_accuracy"]),
        "best_family": best_family,
        "v4_vs": v4_vs,
        "v4_two_x": v4_two_x,
        "best_ablation": best_ablation,
        "benchmark_rows": int(len(benchmark_frame)),
        "tuning_rows": int(len(tuning_frame)),
        "ablation_rows": int(len(ablation_frame)),
        "win_flag_rows": int(len(win_flags)),
    }
    return summary
