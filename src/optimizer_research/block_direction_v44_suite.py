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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v44"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v44",
    "block_direction_optimizer_v42",
    "block_direction_optimizer_v4_fast",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer_v42",
    "block_direction_optimizer_v4_fast",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]


def block_direction_v44_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 1,
        "search_seed": 2404,
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "breast_cancer_mlp",
            "wine_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
        ],
        "benchmark_tasks": [
            "breast_cancer_mlp",
            "wine_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
        ],
        "ablation_tasks": [
            "breast_cancer_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
        ],
        "smoke_tasks": ["digits_cnn", "oscillatory_valley"],
        "smoke_optimizers": ["block_direction_optimizer_v44", "block_direction_optimizer_v42", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.45,
        "benchmark_epoch_scale": 0.75,
        "ablation_epoch_scale": 0.40,
        "use_tuning_results": True,
    }


def _prepare_docs(output_dir: Path) -> None:
    write_block_direction_v44_current_state(output_dir)
    write_block_direction_v44_literature_scan(output_dir)
    write_block_direction_v44_math_definition(output_dir)


def write_block_direction_v44_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.4 Current State",
        "",
        "V4.4 is the synthesis branch built from the validated pieces of the earlier block line.",
        "",
        "## Why V4.4 exists",
        "",
        "- V4Fast gave the strongest dense/stress core and the cleanest hot path.",
        "- V4.2 gave the strongest CNN-side improvement through conv-safe scaling.",
        "- V4.3 showed that conv/dense profile separation is a sound architectural move, but default conv trust bonuses did not validate.",
        "- Magneto-Hamiltonian reinforced a separate lesson: extra directional machinery should only survive if it helps cleanly under ablation.",
        "",
        "## What V4.4 keeps",
        "",
        "- V4Fast candidate core and winner-take-all selector.",
        "- Stable consensus, trusted memory, and matrix consensus.",
        "- Conv-safe scaling from V4.2.",
        "- A true internal split between dense and convolutional profiles.",
        "",
        "## What V4.4 leaves experimental",
        "",
        "- Conv trust bonuses.",
        "- Conv fallback relaxation.",
        "- Recoverability in the default path.",
        "- Projection or stress-specialist extras from older branches.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")


def write_block_direction_v44_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BlockDirectionOptimizerV4.4 Literature Scan",
        "",
        "V4.4 stays on the same plausible novelty path as the block family overall: typed blockwise direction selection with lightweight structure-aware control, not another Adam-style controller stack.",
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
        "## V4.4-specific interpretation",
        "",
        "- Compared with V4Fast, V4.4 adds typed internal profiles rather than new candidate directions.",
        "- Compared with V4.2, V4.4 keeps conv-safe scaling but refuses to make weak conv trust bonuses part of the default path.",
        "- Compared with Muon/Shampoo/K-FAC, V4.4 still uses blockwise direction selection rather than matrix preconditioning.",
        "- Compared with Magneto-Hamiltonian, the branch remains structure-typed rather than dynamics-typed.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v44_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.4 Mathematical Definition",
        "",
        "V4.4 keeps the reduced block-direction candidate set",
        "",
        "`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`",
        "",
        "with winner-take-all blockwise selection.",
        "",
        "The block trust score remains the V4Fast score, while parameter types receive distinct internal profiles:",
        "",
        "- dense/vector tensors use the V4Fast-style profile",
        "- convolutional tensors use a conv profile with its own coherence, stability, consensus, and energy settings",
        "",
        "The step rule is typed rather than coordinate-wise adaptive:",
        "",
        "- dense blocks use the V4Fast energy-normalized step rule",
        "- conv blocks use the same chosen direction but a smaller conv update cap, a stronger energy-power term, and a bounded conv step floor",
        "",
        "Conv structure support is still measured from filter/channel/spatial/bank agreement, but in the default V4.4 path it is used only as a diagnostic and inside conv-safe scaling, not as a direct trust bonus.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v44_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v44_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v44", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v44_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v44_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(config.get("ablation_tasks", ["breast_cancer_mlp", "digits_mlp", "digits_cnn", "oscillatory_valley"]))
    variants = [
        {"variant_name": "v44_full", "optimizer_name": "block_direction_optimizer_v44", "overrides": {}},
        {"variant_name": "no_conv_profile_split", "optimizer_name": "block_direction_optimizer_v44", "overrides": {"conv_lr_scale": 1.0, "conv_coherence_weight": 0.18, "conv_stability_weight": 0.16, "conv_consensus_memory_mix": 0.38, "conv_consensus_matrix_mix": 0.14, "conv_stable_consensus_bonus": 0.10, "conv_matrix_consensus_bonus": 0.05, "conv_small_matrix_cutoff": 1024, "conv_energy_power": 0.50}},
        {"variant_name": "no_conv_safe_scaling", "optimizer_name": "block_direction_optimizer_v44", "overrides": {"conv_max_update_ratio": 0.16, "conv_energy_power_bonus": 0.0, "conv_step_floor": 1.0}},
        {"variant_name": "v42_baseline", "optimizer_name": "block_direction_optimizer_v42", "overrides": {}},
        {"variant_name": "v4_fast_baseline", "optimizer_name": "block_direction_optimizer_v4_fast", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v44_ablation",
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


def export_block_direction_v44_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v44_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v44")
    v42_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v42")
    v4_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v4_fast")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v44", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    split_delta = _variant_delta(ablation_frame, "v44_full", "no_conv_profile_split")
    scaling_delta = _variant_delta(ablation_frame, "v44_full", "no_conv_safe_scaling")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "block_direction_optimizer_v44",
        "block_direction_optimizer_v42",
        "block_direction_optimizer_v4_fast",
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
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV4.4 ablation", 45)

    lines = [
        "# BlockDirectionOptimizerV4.4 Final Report",
        "",
        "## 1. What V4.4 is",
        "",
        "- V4.4 is the typed-profile synthesis branch: V4Fast dense core plus V4.2 conv-safe control.",
        "- It keeps the V4Fast candidate set and uses conv-safe scaling with an explicit conv/dense profile split.",
        "",
        "## 2. Best rows",
        "",
        f"- Best V4.4 row: `{v44_row['task']}` with mean best val loss `{float(v44_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v44_row['mean_best_val_accuracy']):.6f}`." if v44_row is not None else "- Best V4.4 row: unavailable.",
        f"- Best V4.2 row on the same suite: `{v42_row['task']}` with mean best val loss `{float(v42_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v42_row['mean_best_val_accuracy']):.6f}`." if v42_row is not None else "- Best V4.2 row: unavailable.",
        f"- Best V4Fast row on the same suite: `{v4_row['task']}` with mean best val loss `{float(v4_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v4_row['mean_best_val_accuracy']):.6f}`." if v4_row is not None else "- Best V4Fast row: unavailable.",
        f"- Strongest baseline row: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(strongest_baseline['mean_best_val_accuracy']):.6f}`.",
        "",
        "## 3. Competitive summary",
        "",
        f"- V4.4 wins vs V4.2: `{int(win_map.get('block_direction_optimizer_v42', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.4 wins vs V4Fast: `{int(win_map.get('block_direction_optimizer_v4_fast', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.4 wins vs AdamW: `{int(win_map.get('adamw', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.4 wins vs RMSProp: `{int(win_map.get('rmsprop', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.4 wins vs SGD momentum: `{int(win_map.get('sgd_momentum', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.4 wins vs MagnetoHamiltonianAdam: `{int(win_map.get('magneto_hamiltonian_adam', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        "",
        "## 4. Ablation findings",
        "",
        f"- Conv profile split delta vs removal: `{_format_optional_delta(split_delta)}`.",
        f"- Conv safe scaling delta vs removal: `{_format_optional_delta(scaling_delta)}`.",
        f"- Best ablation row overall: `{str(ablation_summary.iloc[0]['variant_name'])}`." if not ablation_summary.empty else "- Best ablation row overall: unavailable.",
        "",
        "## 5. Honest conclusion",
        "",
        "- V4.4 is only successful if it preserves the V4Fast dense/stress story while keeping or improving the V4.2 CNN gain.",
        "- If RMSProp still wins the mixed suite broadly, V4.4 remains a specialist branch rather than a new default optimizer.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "best_v44_task": None if v44_row is None else str(v44_row["task"]),
        "best_v44_loss": None if v44_row is None else float(v44_row["mean_best_val_loss"]),
        "best_v44_accuracy": None if v44_row is None else float(v44_row["mean_best_val_accuracy"]),
        "benchmark_rows": int(len(benchmark_frame)),
        "tuning_rows": int(len(tuning_frame)),
        "ablation_rows": int(len(ablation_frame)),
        "win_flag_rows": int(len(win_flags)),
    }
