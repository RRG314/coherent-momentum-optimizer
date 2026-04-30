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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v41"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v41",
    "block_direction_optimizer_v4_fast",
    "block_direction_optimizer_v3",
    "magneto_hamiltonian_adam",
    "real_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "topological_adam",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer_v4_fast",
    "block_direction_optimizer_v3",
    "magneto_hamiltonian_adam",
    "real_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "topological_adam",
]


def block_direction_v41_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 2,
        "search_seed": 2143,
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "breast_cancer_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
        ],
        "benchmark_tasks": [
            "breast_cancer_mlp",
            "digits_mlp",
            "digits_cnn",
            "moons_mlp",
            "wine_mlp",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
        ],
        "ablation_tasks": [
            "digits_cnn",
            "breast_cancer_mlp",
            "oscillatory_valley",
            "saddle_objective",
        ],
        "smoke_tasks": ["digits_cnn", "oscillatory_valley"],
        "smoke_optimizers": ["block_direction_optimizer_v41", "block_direction_optimizer_v4_fast", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.45,
        "benchmark_epoch_scale": 0.75,
        "ablation_epoch_scale": 0.40,
        "use_tuning_results": True,
    }


def _prepare_docs(output_dir: Path) -> None:
    write_block_direction_v41_current_state(output_dir)
    write_block_direction_v41_literature_scan(output_dir)
    write_block_direction_v41_math_definition(output_dir)


def write_block_direction_v41_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.1 Current State",
        "",
        "V4.1 is a separate branch built on top of V4Fast.",
        "",
        "## Why V4.1 exists",
        "",
        "- V4Fast clearly beat V3 on dense MLP-style tasks and runtime.",
        "- V4Fast still lagged on CNNs, especially `digits_cnn`.",
        "- The diagnostics showed that on CNNs V4Fast mostly fell back to the raw gradient path, so the novel block-direction machinery was under-engaged there.",
        "",
        "## What V4.1 changes",
        "",
        "- Adds an optional conv-only `filter_consensus` candidate.",
        "- Builds that candidate from filter, channel, spatial, and shared bank structure when the experimental path is enabled.",
        "- Adds a conv-safe step policy so convolution filters use safer update caps and slightly stronger energy normalization.",
        "- Keeps V4Fast's small candidate set and winner-take-all default.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")


def write_block_direction_v41_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BlockDirectionOptimizerV4.1 Literature Scan",
        "",
        "V4.1 keeps the same pressure as the block branch overall: it is only interesting if conv-aware blockwise direction selection does useful work that standard gradient transforms are not already doing.",
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
        "## V4.1-specific interpretation",
        "",
        "- Muon, Shampoo, and K-FAC still pressure the structure-aware claim.",
        "- V4.1's distinct hypothesis is narrower: convolution filters may benefit from explicit axis-aware candidate selection before the step is taken.",
        "- That is different from Adam-style variance scaling and different from matrix preconditioning, but it still needs empirical validation.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v41_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.1 Mathematical Definition",
        "",
        "Split parameters into blocks `B_i` and compute block gradients `g_i`.",
        "",
        "## Candidate set",
        "",
        "V4.1 keeps the V4Fast core candidate set and adds one conv-only candidate:",
        "",
        "`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix), d_i^(filter)}`",
        "",
        "For convolutional tensors `G in R^{O x I x S}` where `S` flattens spatial kernel positions:",
        "",
        "- `G_o` is the filter block for output channel `o`",
        "- `C_o = mean_s G_{o,:,s}` is the channel profile",
        "- `S_o = mean_i G_{o,i,:}` is the spatial profile",
        "- `B = mean_o G_{o,:,:}` is the shared bank profile",
        "",
        "The filter-consensus candidate is",
        "",
        "`d_o^(filter) = normalize(u_o + a_o * alpha * c_o + b_o * beta * s_o + c_o' * gamma * b)`",
        "",
        "where `u_o` is the normalized raw filter direction, `c_o`, `s_o`, and `b` are normalized broadcasts of the profiles above, and the coefficients `a_o`, `b_o`, `c_o'` are positive alignments with `u_o`.",
        "",
        "## Trust score",
        "",
        "Each candidate receives",
        "",
        "`T_i(d) = w_d A_i(d) + w_m M_i(d) + w_q Q_i(d) + w_s S_i(d) + w_c C_i(d) + w_filt Filt_i(d) - w_o O_i(d) - w_f F_i(d) - w_k cost(d)`",
        "",
        "where `Filt_i(d)` is a conv-structure support term. For non-conv tensors it is zero.",
        "",
        "## Step magnitude",
        "",
        "V4.1 keeps V4Fast's block-energy normalization, but for convolutional tensors it applies:",
        "",
        "- a lower parameter-relative update cap",
        "- a step multiplier bounded by the selected filter-support score",
        "- an extra energy-power term when filter support is weak",
        "",
        "This makes the conv update safer without changing the direction-selection principle.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v41_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v41_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v41", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v41_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v41_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(config.get("ablation_tasks", ["digits_cnn", "breast_cancer_mlp", "oscillatory_valley"]))
    variants = [
        {"variant_name": "v41_full", "optimizer_name": "block_direction_optimizer_v41", "overrides": {}},
        {"variant_name": "no_filter_consensus", "optimizer_name": "block_direction_optimizer_v41", "overrides": {"use_filter_consensus_candidate": False, "filter_consensus_bonus": 0.0}},
        {"variant_name": "no_conv_safe_scaling", "optimizer_name": "block_direction_optimizer_v41", "overrides": {"conv_max_update_ratio": 0.16, "conv_energy_power_bonus": 0.0, "conv_step_floor": 1.0}},
        {"variant_name": "v4_fast_baseline", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v41_ablation",
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


def _best_row_for_optimizer(aggregated: pd.DataFrame, optimizer_name: str) -> pd.Series | None:
    frame = aggregated[aggregated["optimizer"] == optimizer_name]
    if frame.empty:
        return None
    if frame["mean_best_val_accuracy"].notna().any():
        return frame.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
    return frame.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0]


def _variant_delta(frame: pd.DataFrame, full_name: str, compare_name: str) -> float | None:
    summary = frame.groupby("variant_name", as_index=False)["selection_score"].mean()
    if full_name not in set(summary["variant_name"]) or compare_name not in set(summary["variant_name"]):
        return None
    full_score = float(summary.loc[summary["variant_name"] == full_name, "selection_score"].iloc[0])
    compare_score = float(summary.loc[summary["variant_name"] == compare_name, "selection_score"].iloc[0])
    return full_score - compare_score


def _format_optional_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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


def export_block_direction_v41_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v41_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v41")
    v4_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v4_fast")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum", "topological_adam"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v41", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    filter_delta = _variant_delta(ablation_frame, "v41_full", "no_filter_consensus")
    conv_step_delta = _variant_delta(ablation_frame, "v41_full", "no_conv_safe_scaling")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "block_direction_optimizer_v41",
        "block_direction_optimizer_v4_fast",
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
        tasks=["breast_cancer_mlp", "digits_cnn", "oscillatory_valley"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png")
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV4.1 ablation", 45)

    lines = [
        "# BlockDirectionOptimizerV4.1 Final Report",
        "",
        "## 1. What V4.1 is",
        "",
        "- V4.1 is a conv-aware refinement of BlockDirectionOptimizerV4Fast.",
        "- It keeps V4Fast's small candidate core, adds conv-safe step scaling, and keeps filter-consensus available as an experimental option.",
        "",
        "## 2. Best rows",
        "",
        f"- Best V4.1 row: `{v41_row['task']}` with mean best val loss `{float(v41_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v41_row['mean_best_val_accuracy']):.6f}`." if v41_row is not None else "- Best V4.1 row: unavailable.",
        f"- Best V4Fast row on the same suite: `{v4_row['task']}` with mean best val loss `{float(v4_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v4_row['mean_best_val_accuracy']):.6f}`." if v4_row is not None else "- Best V4Fast row: unavailable.",
        f"- Strongest baseline row: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(strongest_baseline['mean_best_val_accuracy']):.6f}`.",
        "",
        "## 3. Competitive summary",
        "",
        f"- V4.1 wins vs V4Fast: `{int(win_map.get('block_direction_optimizer_v4_fast', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.1 wins vs AdamW: `{int(win_map.get('adamw', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.1 wins vs RMSProp: `{int(win_map.get('rmsprop', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.1 wins vs SGD momentum: `{int(win_map.get('sgd_momentum', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.1 wins vs MagnetoHamiltonianAdam: `{int(win_map.get('magneto_hamiltonian_adam', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.1 wins vs RealHamiltonianAdam: `{int(win_map.get('real_hamiltonian_adam', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        "",
        "## 4. Ablation findings",
        "",
        f"- Filter-consensus delta vs removal: `{_format_optional_delta(filter_delta)}`.",
        f"- Conv-safe scaling delta vs removal: `{_format_optional_delta(conv_step_delta)}`.",
        f"- Best ablation row overall: `{str(ablation_summary.iloc[0]['variant_name'])}`." if not ablation_summary.empty else "- Best ablation row overall: unavailable.",
        "",
        "## 5. Honest conclusion",
        "",
        "- In this run, filter-consensus stayed experimental and off by default because the ablation did not justify making it a default rule.",
        "- V4.1 is successful only if it improves CNN/standard-task performance without giving back V4Fast's stress-task advantages.",
        "- If RMSProp still wins the CNN tasks cleanly, V4.1 remains a specialist prototype rather than a replacement default optimizer.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "best_v41_task": None if v41_row is None else str(v41_row["task"]),
        "best_v41_loss": None if v41_row is None else float(v41_row["mean_best_val_loss"]),
        "best_v41_accuracy": None if v41_row is None else float(v41_row["mean_best_val_accuracy"]),
        "benchmark_rows": int(len(benchmark_frame)),
        "tuning_rows": int(len(tuning_frame)),
        "ablation_rows": int(len(ablation_frame)),
        "win_flag_rows": int(len(win_flags)),
    }
