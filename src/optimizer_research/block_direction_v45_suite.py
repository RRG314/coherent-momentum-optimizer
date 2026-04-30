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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v45"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v45",
    "block_direction_optimizer_v44",
    "block_direction_optimizer_v42",
    "block_direction_optimizer_v4_fast",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer_v44",
    "block_direction_optimizer_v42",
    "block_direction_optimizer_v4_fast",
    "magneto_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]


def block_direction_v45_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 1,
        "search_seed": 2405,
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
        "smoke_optimizers": ["block_direction_optimizer_v45", "block_direction_optimizer_v44", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.45,
        "benchmark_epoch_scale": 0.75,
        "ablation_epoch_scale": 0.40,
        "use_tuning_results": True,
    }


def _prepare_docs(output_dir: Path) -> None:
    write_block_direction_v45_current_state(output_dir)
    write_block_direction_v45_literature_scan(output_dir)
    write_block_direction_v45_math_definition(output_dir)


def write_block_direction_v45_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.5 Current State",
        "",
        "V4.5 is the regime-routed typed block branch.",
        "",
        "## Why V4.5 exists",
        "",
        "- V4Fast provided the strongest dense/stress block core.",
        "- V4.2 showed that conv-safe scaling is the strongest CNN-side keeper.",
        "- V4.4 proved that typed conv/dense internal profiles help.",
        "- The remaining gap suggested a more novel move: let blocks choose among a few low-cost scoring policies instead of hand-sharing one score profile everywhere.",
        "",
        "## What V4.5 keeps",
        "",
        "- V4Fast candidate core and winner-take-all selector.",
        "- Stable consensus, trusted memory, and matrix consensus.",
        "- Conv-safe scaling and conv/dense profile split.",
        "",
        "## What V4.5 adds",
        "",
        "- A low-cost per-block router with three policies: dense_stable, conv_structured, and stress_response.",
        "- The router changes score weights and fallback pressure without adding more default candidate directions.",
        "",
        "## What stays experimental",
        "",
        "- Recoverability in the default hot path.",
        "- Projection or orthogonal escape extras.",
        "- Direct magneto-style controller stacking inside the block optimizer.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")


def write_block_direction_v45_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BlockDirectionOptimizerV4.5 Literature Scan",
        "",
        "V4.5 stays on the same plausible novelty path as the block family overall: typed blockwise direction selection, now with regime-routed scoring rather than one fixed score profile for every tensor and every local regime.",
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
        "## V4.5-specific interpretation",
        "",
        "- Compared with V4.4, V4.5 does not add more candidates. It adds a low-cost router over the existing candidates.",
        "- Compared with Muon/Shampoo/K-FAC, V4.5 still uses direction selection rather than matrix preconditioning.",
        "- Compared with Magneto-Hamiltonian, V4.5 stays structure-typed and route-typed rather than continuous-dynamics-typed.",
        "- Compared with trust-region or mixture-of-experts ideas, the router is local, cheap, and policy-weighted rather than solving a second optimization problem.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v45_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV4.5 Mathematical Definition",
        "",
        "V4.5 keeps the reduced candidate set",
        "",
        "`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`",
        "",
        "and still uses winner-take-all blockwise selection.",
        "",
        "The new addition is a block policy router with three route weights per block:",
        "",
        "`r_i = softmax([l_i^(stable), l_i^(structured), l_i^(stress)])`",
        "",
        "The route logits depend on measured local signals:",
        "",
        "- memory coherence",
        "- consensus support",
        "- conflict",
        "- oscillation",
        "- structure support",
        "- gradient-stress / update-pressure",
        "",
        "Those route weights modulate the block trust score rather than replacing it:",
        "",
        "- dense_stable increases coherence and stability emphasis",
        "- conv_structured increases matrix/structure emphasis",
        "- stress_response increases descent emphasis and oscillation/conflict penalties",
        "",
        "Dense and convolutional tensors still use different typed profiles, and convolutional tensors still use conv-safe scaling for the final bounded step magnitude.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v45_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v45_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v45", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v45_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v45_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(config.get("ablation_tasks", ["breast_cancer_mlp", "digits_mlp", "digits_cnn", "oscillatory_valley"]))
    variants = [
        {"variant_name": "v45_full", "optimizer_name": "block_direction_optimizer_v45", "overrides": {}},
        {"variant_name": "no_policy_router", "optimizer_name": "block_direction_optimizer_v45", "overrides": {"stable_route_weight": 0.0, "structured_route_weight": 0.0, "stress_route_weight": 0.0, "stable_route_bonus": 0.0, "structured_route_bonus": 0.0, "stress_route_bonus": 0.0, "stress_memory_penalty": 0.0, "route_threshold_relaxation": 0.0, "route_step_gain": 0.0, "route_stress_damping": 0.0}},
        {"variant_name": "no_conv_profile_split", "optimizer_name": "block_direction_optimizer_v45", "overrides": {"conv_lr_scale": 1.0, "conv_coherence_weight": 0.18, "conv_stability_weight": 0.16, "conv_consensus_memory_mix": 0.38, "conv_consensus_matrix_mix": 0.14, "conv_stable_consensus_bonus": 0.10, "conv_matrix_consensus_bonus": 0.05, "conv_small_matrix_cutoff": 1024, "conv_energy_power": 0.50}},
        {"variant_name": "no_conv_safe_scaling", "optimizer_name": "block_direction_optimizer_v45", "overrides": {"conv_max_update_ratio": 0.16, "conv_energy_power_bonus": 0.0, "conv_step_floor": 1.0}},
        {"variant_name": "v44_baseline", "optimizer_name": "block_direction_optimizer_v44", "overrides": {}},
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
                    suite_name="block_direction_v45_ablation",
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


def export_block_direction_v45_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v45_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v45")
    v44_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v44")
    v42_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v42")
    v4_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v4_fast")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v45", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    router_delta = _variant_delta(ablation_frame, "v45_full", "no_policy_router")
    split_delta = _variant_delta(ablation_frame, "v45_full", "no_conv_profile_split")
    scaling_delta = _variant_delta(ablation_frame, "v45_full", "no_conv_safe_scaling")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "block_direction_optimizer_v45",
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
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BlockDirectionOptimizerV4.5 ablation", 45)

    lines = [
        "# BlockDirectionOptimizerV4.5 Final Report",
        "",
        "## 1. What V4.5 is",
        "",
        "- V4.5 is the regime-routed typed block branch: V4Fast dense core plus V4.2 conv-safe control plus a low-cost three-policy block router.",
        "- It keeps the small candidate set and uses the router to modulate block trust rather than adding more default candidate directions.",
        "",
        "## 2. Best rows",
        "",
        f"- Best V4.5 row: `{v45_row['task']}` with mean best val loss `{float(v45_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v45_row['mean_best_val_accuracy']):.6f}`." if v45_row is not None else "- Best V4.5 row: unavailable.",
        f"- Best V4.4 row on the same suite: `{v44_row['task']}` with mean best val loss `{float(v44_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v44_row['mean_best_val_accuracy']):.6f}`." if v44_row is not None else "- Best V4.4 row: unavailable.",
        f"- Best V4.2 row on the same suite: `{v42_row['task']}` with mean best val loss `{float(v42_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v42_row['mean_best_val_accuracy']):.6f}`." if v42_row is not None else "- Best V4.2 row: unavailable.",
        f"- Best V4Fast row on the same suite: `{v4_row['task']}` with mean best val loss `{float(v4_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v4_row['mean_best_val_accuracy']):.6f}`." if v4_row is not None else "- Best V4Fast row: unavailable.",
        f"- Strongest baseline row: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(strongest_baseline['mean_best_val_accuracy']):.6f}`.",
        "",
        "## 3. Competitive summary",
        "",
        f"- V4.5 wins vs V4.4: `{int(win_map.get('block_direction_optimizer_v44', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs V4.2: `{int(win_map.get('block_direction_optimizer_v42', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs V4Fast: `{int(win_map.get('block_direction_optimizer_v4_fast', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs AdamW: `{int(win_map.get('adamw', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs RMSProp: `{int(win_map.get('rmsprop', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs SGD momentum: `{int(win_map.get('sgd_momentum', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- V4.5 wins vs MagnetoHamiltonianAdam: `{int(win_map.get('magneto_hamiltonian_adam', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        "",
        "## 4. Ablation findings",
        "",
        f"- Policy router delta vs removal: `{_format_optional_delta(router_delta)}`.",
        f"- Conv profile split delta vs removal: `{_format_optional_delta(split_delta)}`.",
        f"- Conv safe scaling delta vs removal: `{_format_optional_delta(scaling_delta)}`.",
        f"- Best ablation row overall: `{str(ablation_summary.iloc[0]['variant_name'])}`." if not ablation_summary.empty else "- Best ablation row overall: unavailable.",
        "",
        "## 5. Honest conclusion",
        "",
        "- V4.5 is only successful if the routed policy keeps the V4Fast dense/stress story while improving the typed mixed-suite behavior over V4.4.",
        "- If RMSProp still wins the mixed suite broadly, V4.5 remains a specialist branch rather than a new default optimizer.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "best_v45_task": None if v45_row is None else str(v45_row["task"]),
        "best_v45_loss": None if v45_row is None else float(v45_row["mean_best_val_loss"]),
        "best_v45_accuracy": None if v45_row is None else float(v45_row["mean_best_val_accuracy"]),
        "benchmark_rows": int(len(benchmark_frame)),
        "tuning_rows": int(len(tuning_frame)),
        "ablation_rows": int(len(ablation_frame)),
        "win_flag_rows": int(len(win_flags)),
    }
