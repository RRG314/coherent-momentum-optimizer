from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .reporting import _markdown_table, aggregate_results, compute_meaningful_wins


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
PAPER_DIR = ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

ACCEPTED_DIR = REPORTS_DIR / "accepted_coherent_momentum"
DIRECTIONAL_DIR = REPORTS_DIR / "directional_instability"
GPU_DIR = REPORTS_DIR / "coherent_momentum_gpu"
CNN_DIR = REPORTS_DIR / "cnn_credibility"
DEMO_DIR = REPORTS_DIR / "demo_directional_instability"

SELECTED_ACCEPTED_OPTIMIZERS = [
    "coherent_momentum_optimizer",
    "coherent_momentum_real_baseline",
    "coherent_direction_reference",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]
SELECTED_FOCUSED_OPTIMIZERS = [
    "coherent_momentum_optimizer",
    "coherent_momentum_optimizer_improved",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]
ACCEPTED_BASELINES = [
    "coherent_momentum_real_baseline",
    "coherent_direction_reference",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "topological_adam",
]
FOCUSED_BASELINES = ["adamw", "rmsprop", "sgd_momentum"]


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_table(name: str, frame: pd.DataFrame, sources: list[Path], intro: str) -> None:
    table_frame = frame.copy()
    table_frame["source_csv"] = " | ".join(_relative(path) for path in sources)
    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"
    table_frame.to_csv(csv_path, index=False)
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        intro,
        "",
        "Source CSVs:",
    ]
    lines.extend(f"- `{_relative(path)}`" for path in sources)
    lines.extend(["", _markdown_table(table_frame)])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_grouped_bar(
    frame: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: str,
    output_path: Path,
    ylabel: str,
) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index=category_col, columns=group_col, values=value_col)
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title=group_col, fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_simple_bar(frame: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path, ylabel: str) -> None:
    if frame.empty:
        return
    ordered = frame.sort_values(value_col, ascending=False)
    plt.figure(figsize=(9, 4.5))
    plt.bar(ordered[label_col], ordered[value_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _summarize_wins(aggregated: pd.DataFrame, optimizer_name: str, baselines: list[str], benchmark_group: str, source_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    present = set(aggregated["optimizer"])
    if optimizer_name not in present:
        return pd.DataFrame(columns=["benchmark_group", "optimizer", "baseline", "meaningful_wins", "two_x_wins", "source_csv"])
    for baseline in baselines:
        if baseline not in present:
            continue
        wins = compute_meaningful_wins(aggregated, optimizer_name, baseline)
        rows.append(
            {
                "benchmark_group": benchmark_group,
                "optimizer": optimizer_name,
                "baseline": baseline,
                "meaningful_wins": int(wins["win"].sum()),
                "two_x_wins": int(wins["two_x"].sum()) if "two_x" in wins.columns else 0,
                "source_csv": _relative(source_csv),
            }
        )
    return pd.DataFrame(rows)


def _accepted_task_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    tasks = [
        "breast_cancer_mlp",
        "conflicting_batches_classification",
        "direction_reversal_objective",
        "oscillatory_valley",
        "rosenbrock_valley",
        "saddle_objective",
        "small_batch_instability",
    ]
    frame = aggregated[
        aggregated["task"].isin(tasks) & aggregated["optimizer"].isin(SELECTED_ACCEPTED_OPTIMIZERS)
    ][
        [
            "task",
            "optimizer",
            "task_family",
            "mean_best_val_loss",
            "mean_best_val_accuracy",
            "mean_runtime_per_step_ms",
            "mean_optimizer_state_mb",
        ]
    ].sort_values(["task", "optimizer"])
    return frame


def _focused_task_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[
        aggregated["optimizer"].isin(SELECTED_FOCUSED_OPTIMIZERS)
    ][
        [
            "task",
            "optimizer",
            "task_family",
            "mean_best_val_loss",
            "mean_best_val_accuracy",
            "mean_runtime_per_step_ms",
        ]
    ].sort_values(["task", "optimizer"])


def _runtime_table(runtime_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        runtime_frame.groupby("optimizer", as_index=False)[
            [
                "mean_runtime_per_step_ms",
                "mean_optimizer_step_time_ms",
                "mean_samples_per_second",
                "mean_optimizer_state_mb",
                "mean_peak_device_memory_mb",
                "divergence_rate",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("mean_runtime_per_step_ms")
    )


def _cnn_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[
        aggregated["optimizer"].isin(SELECTED_FOCUSED_OPTIMIZERS)
    ][
        [
            "task",
            "optimizer",
            "mean_best_val_loss",
            "mean_best_val_accuracy",
            "mean_runtime_per_step_ms",
        ]
    ].sort_values(["task", "optimizer"])


def _ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in ["best_val_loss", "best_val_accuracy", "runtime_per_step_ms", "selection_score"] if col in frame.columns]
    grouped = frame.groupby("variant_name", as_index=False)[value_cols].mean(numeric_only=True)
    rename_map = {
        "best_val_loss": "mean_best_val_loss",
        "best_val_accuracy": "mean_best_val_accuracy",
        "runtime_per_step_ms": "mean_runtime_per_step_ms",
        "selection_score": "mean_selection_score",
    }
    grouped = grouped.rename(columns=rename_map)
    sort_col = "mean_selection_score" if "mean_selection_score" in grouped.columns else "mean_best_val_loss"
    ascending = False if sort_col == "mean_selection_score" else True
    return grouped.sort_values(sort_col, ascending=ascending)


def _write_paper_results_summary(rows: list[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(rows)
    output_path = PAPER_DIR / "paper_results_summary.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def _write_claims_audit(
    accepted_win_summary: pd.DataFrame,
    focused_win_summary: pd.DataFrame,
    cnn_win_summary: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> Path:
    accepted_adamw = accepted_win_summary.loc[accepted_win_summary["baseline"] == "adamw", "meaningful_wins"]
    focused_improved_adamw = focused_win_summary[
        (focused_win_summary["optimizer"] == "coherent_momentum_optimizer_improved")
        & (focused_win_summary["baseline"] == "adamw")
    ]["meaningful_wins"]
    cnn_improved_adamw = cnn_win_summary[
        (cnn_win_summary["optimizer"] == "coherent_momentum_optimizer_improved")
        & (cnn_win_summary["baseline"] == "adamw")
    ]["meaningful_wins"]
    runtime_lookup = runtime_table.set_index("optimizer")
    improved_runtime = runtime_lookup.loc["coherent_momentum_optimizer_improved", "mean_runtime_per_step_ms"] if "coherent_momentum_optimizer_improved" in runtime_lookup.index else float("nan")
    adamw_runtime = runtime_lookup.loc["adamw", "mean_runtime_per_step_ms"] if "adamw" in runtime_lookup.index else float("nan")
    lines = [
        "# Coherent Momentum Claims Audit",
        "",
        "This file is the paper-facing claim check for the current repository state. It is intentionally narrower than a README overview.",
        "",
        "## Supported claims",
        "",
        "The checked-in reports support describing Coherent Momentum as a directional-coherence optimizer for unstable gradient-direction regimes. The strongest evidence comes from the accepted historical line in `reports/accepted_coherent_momentum/` and the focused newcomer-facing slice in `reports/directional_instability/`.",
        "",
        f"The accepted historical line still shows `{int(accepted_adamw.iloc[0]) if not accepted_adamw.empty else 0}` meaningful wins over AdamW across its broader stress-inclusive suite, and the focused directional-instability slice shows `{int(focused_improved_adamw.iloc[0]) if not focused_improved_adamw.empty else 0}` meaningful wins for the improved branch over AdamW on the narrower instability benchmark.",
        "",
        "The checked-in GPU audit also supports a more limited engineering claim: the branch can be made device-safe, diagnostics can be throttled, and the improved branch can win more synthetic directional stress comparisons than the stable mainline without changing the public identity of the method.",
        "",
        "## Claims the repository does not support",
        "",
        "The repository does not support claiming that Coherent Momentum is a general replacement for RMSProp, SGD with momentum, or AdamW. The accepted historical line still shows RMSProp and SGD with momentum as stronger practical baselines on many ordinary tasks, and the focused directional-instability slice does not reverse that conclusion.",
        "",
        "The repository also does not support presenting Coherent Momentum as a competitive CNN optimizer yet. The CNN credibility report remains visible precisely because the gap is still open.",
        "",
        "## Failure cases that should remain visible",
        "",
        "Clean classification, mainstream CNN training, and closure-friendly PINN-style problems are still failure checks for this repository, not public win conditions. The current docs should continue to say that directly.",
        "",
        "## Unfair claims",
        "",
        "It would be unfair to claim that the improved branch is the new default public optimizer, that the CNN problem is solved, or that the repo shows broad superiority over RMSProp or SGD with momentum. Those claims are not supported by the checked-in CSVs.",
        "",
        "## Safe public wording",
        "",
        "Coherent Momentum Optimizer is a specialist optimizer for unstable gradient-direction regimes. In the checked-in reports it improves on AdamW in selected instability slices, but it remains slower than simpler baselines and does not replace RMSProp, SGD with momentum, or mainstream CNN optimizers as a default choice.",
        "",
        "## Runtime context",
        "",
        f"The improved branch currently averages `{_fmt(improved_runtime, 4)} ms` per step against `{_fmt(adamw_runtime, 4)} ms` for AdamW in the checked-in runtime audit, which is why runtime cost remains part of the honest public story.",
        "",
        "## Sources",
        "",
        "- `reports/accepted_coherent_momentum/benchmark_results.csv`",
        "- `reports/directional_instability/benchmark_results.csv`",
        "- `reports/coherent_momentum_gpu/runtime_memory_results.csv`",
        "- `reports/cnn_credibility/benchmark_results.csv`",
    ]
    output_path = PAPER_DIR / "paper_claims_audit.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_paper_draft(
    accepted_best_task: pd.Series,
    focused_win_summary: pd.DataFrame,
    accepted_win_summary: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> Path:
    focused_improved_adamw = focused_win_summary[
        (focused_win_summary["optimizer"] == "coherent_momentum_optimizer_improved")
        & (focused_win_summary["baseline"] == "adamw")
    ]["meaningful_wins"]
    focused_improved_rmsprop = focused_win_summary[
        (focused_win_summary["optimizer"] == "coherent_momentum_optimizer_improved")
        & (focused_win_summary["baseline"] == "rmsprop")
    ]["meaningful_wins"]
    focused_improved_sgdm = focused_win_summary[
        (focused_win_summary["optimizer"] == "coherent_momentum_optimizer_improved")
        & (focused_win_summary["baseline"] == "sgd_momentum")
    ]["meaningful_wins"]
    accepted_adamw = accepted_win_summary.loc[accepted_win_summary["baseline"] == "adamw", "meaningful_wins"]
    runtime_lookup = runtime_table.set_index("optimizer")
    improved_runtime = runtime_lookup.loc["coherent_momentum_optimizer_improved", "mean_runtime_per_step_ms"] if "coherent_momentum_optimizer_improved" in runtime_lookup.index else float("nan")
    rms_runtime = runtime_lookup.loc["rmsprop", "mean_runtime_per_step_ms"] if "rmsprop" in runtime_lookup.index else float("nan")
    sgdm_runtime = runtime_lookup.loc["sgd_momentum", "mean_runtime_per_step_ms"] if "sgd_momentum" in runtime_lookup.index else float("nan")
    text = f"""# Coherent Momentum Optimizer: A Directional-Coherence Optimizer for Unstable Gradient Regimes

## Abstract

This draft documents the current narrow claim supported by the repository: Coherent Momentum Optimizer is a specialist optimizer for training regimes where the update direction becomes unreliable. The method builds on a real Hamiltonian momentum baseline and adds bounded directional controls driven by gradient-momentum coherence, force alignment, rotation, and conflict signals. The checked-in evidence supports a limited claim. The accepted historical benchmark line shows improvement over the real Hamiltonian baseline and over AdamW on selected stress-oriented tasks, while the focused directional-instability benchmark shows that the improved branch can beat AdamW on parts of the instability slice. The same report set does not support broad superiority over RMSProp or SGD with momentum, and the CNN credibility report remains negative. This draft therefore presents Coherent Momentum as a targeted optimizer family rather than as a default replacement for standard first-order methods.

## Introduction

Many practical optimizers assume that the gradient already points in a usable direction and focus on smoothing, scaling, or preconditioning that direction. This assumption works well in many settings, but it becomes fragile when optimization trajectories oscillate, reverse, or become inconsistent under noisy or conflicting minibatches. Coherent Momentum targets that failure mode directly. The public method asks whether the direction itself is coherent enough to trust before it applies bounded control to the underlying Hamiltonian step.

The repository is intentionally not making a universal optimizer claim. The accepted historical report still shows strong ordinary-task performance for RMSProp and SGD with momentum. The value of this branch therefore depends on a narrower question: can directional-coherence control help when the direction becomes unreliable?

## Related Work

The comparison burden in this repository is anchored to standard first-order baselines and nearby stability methods. SGD and momentum remain the cleanest raw-direction baselines (Bottou, 2010; Sutskever et al., 2013). RMSProp, Adam, and AdamW are the main adaptive baselines because they change magnitude and smoothing while still committing to one update direction (Hinton, 2012; Kingma and Ba, 2015; Loshchilov and Hutter, 2019). SGHMC, symplectic optimization, and Hamiltonian descent are the closest conceptual references for the real Hamiltonian substrate beneath Coherent Momentum (Chen et al., 2014; Maddox et al., 2018; Wilson et al., 2018). SAM, ASAM, PCGrad, and CAGrad matter as alternative stability or conflict-aware approaches, but they target a different control surface than the one used here.

## Method

Let `g_t` be the current gradient, `p_t` the Hamiltonian momentum, and `f_t = M_t^{{-1}} p_t` the current force direction under inverse mass `M_t^{{-1}}`. The real baseline produces a base step from position-momentum dynamics, friction, and optional energy correction. Coherent Momentum augments that base with directional observables:

- `c_t^{{gm}} = cos(g_t, p_t)`
- `c_t^{{fm}} = cos(f_t, p_t)`
- `c_t^{{gg-1}} = cos(g_t, g_{{t-1}})`
- `c_t^{{uu-1}} = cos(u_t, u_{{t-1}})`
- rotation and conflict scores derived from these alignments

These observables drive bounded control values for friction, alignment, and optional projection toward the force direction. The practical update is therefore still a Hamiltonian-momentum update, but with a controller that only intervenes when the direction appears unstable. The exact thresholds and preset balances remain empirical. The repository does not claim a formal proof that the current controller schedule is optimal.

## Experimental Setup

This draft uses only local checked-in artifacts:

- accepted historical mainline: `reports/accepted_coherent_momentum/benchmark_results.csv`
- focused directional-instability slice: `reports/directional_instability/benchmark_results.csv`
- improved/GPU audit: `reports/coherent_momentum_gpu/*.csv`
- CNN credibility check: `reports/cnn_credibility/benchmark_results.csv`

The accepted historical line covers a broader stress-inclusive suite with three seeds. The newcomer-facing directional-instability benchmark is intentionally smaller and easier to rerun. The improved/GPU audit is treated as an engineering and branch-follow-up report rather than as a replacement for the accepted historical line.

## Benchmarks

The accepted historical line includes ordinary tabular neural tasks, label-noise and conflicting-batch cases, and several explicit stress objectives. The focused directional benchmark isolates a smaller instability slice intended to match the narrow public claim. The CNN credibility benchmark remains separate because it is a failure check for the method, not a surface to hide.

## Results

The best accepted Coherent Momentum row in the historical benchmark is `{accepted_best_task['task']}` with mean best validation loss `{_fmt(accepted_best_task['mean_best_val_loss'], 6)}` and mean best validation accuracy `{_fmt(accepted_best_task['mean_best_val_accuracy'], 6)}`. The accepted historical line still shows `{int(accepted_adamw.iloc[0]) if not accepted_adamw.empty else 0}` meaningful wins against AdamW, which is enough to keep the branch interesting. The focused newcomer-facing instability slice is stricter: the improved branch shows `{int(focused_improved_adamw.iloc[0]) if not focused_improved_adamw.empty else 0}` meaningful wins against AdamW there, but `{int(focused_improved_rmsprop.iloc[0]) if not focused_improved_rmsprop.empty else 0}` against RMSProp and `{int(focused_improved_sgdm.iloc[0]) if not focused_improved_sgdm.empty else 0}` against SGD with momentum.

That result pattern matters. The repository supports the claim that directional-coherence control can help on selected instability slices. It does not support a claim that the branch broadly displaces the strongest simple baselines.

## Ablations

The accepted ablation report remains important because it distinguishes the core method from the controller details. Projection survives as a useful optional control. Heavy conflict damping does not. Extra activation gating does not earn default status as a paper-level claim. The improved branch should therefore be described as an engineering refinement, not as proof that every additional controller term matters.

## Runtime and Memory Costs

The improved branch remains materially slower than simpler baselines. In the checked-in runtime audit it averages `{_fmt(improved_runtime, 4)} ms` per step, against `{_fmt(rms_runtime, 4)} ms` for RMSProp and `{_fmt(sgdm_runtime, 4)} ms` for SGD with momentum. Optimizer-state memory remains modest in the local audit, but runtime overhead is still the main practical cost.

## Failure Cases

The repository should keep its failure cases visible. CNN performance remains weak in the current checked-in credibility benchmark. Ordinary clean supervised tasks often favor RMSProp or SGD with momentum. PINN-style closure-heavy workloads are still better described as failure checks than as win conditions for this method.

## Limitations

This draft is based on checked-in local artifacts rather than on a fresh paper-scale rerun of every suite. The focused directional benchmark is a representative slice, not the largest possible instability sweep. The improved branch is still slower than the practical baselines it is being compared against. The CNN gap remains open.

## Reproducibility Statement

The repository includes focused tests, example scripts, a `build_paper_artifacts.py` script that regenerates these paper tables and figures from local CSVs, and a `run_paper_smoke.py` script that checks imports, examples, focused tests, and artifact generation without rerunning the full benchmark suite.

## Conclusion

The strongest defensible reading of the current repository is narrow. Coherent Momentum is a directional-coherence optimizer for unstable gradient-direction regimes. It remains interesting because it improves on AdamW in selected instability slices and because its real Hamiltonian baseline keeps the design conceptually cleaner than a generic “Adam with more gates” story. It should still be treated as a specialist optimizer rather than as a broad default replacement.

## Figures and Tables

- Accepted historical summary: `paper/tables/accepted_mainline_summary.md`
- Focused directional benchmark: `paper/tables/directional_instability_summary.md`
- GPU runtime and memory summary: `paper/tables/gpu_runtime_memory_summary.md`
- CNN credibility summary: `paper/tables/cnn_credibility_summary.md`
- Ablation summary: `paper/tables/accepted_ablation_summary.md`
- Figures: `paper/figures/`

## References

- Bottou, Léon. “Large-Scale Machine Learning with Stochastic Gradient Descent.” 2010. <https://leon.bottou.org/papers/bottou-2010>
- Sutskever, Ilya, James Martens, George Dahl, and Geoffrey Hinton. “On the Importance of Initialization and Momentum in Deep Learning.” ICML 2013. <https://proceedings.mlr.press/v28/sutskever13.html>
- Hinton, Geoffrey. “Neural Networks for Machine Learning, Lecture 6e.” 2012. <https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf>
- Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” 2015. <https://arxiv.org/abs/1412.6980>
- Loshchilov, Ilya, and Frank Hutter. “Decoupled Weight Decay Regularization.” 2019. <https://arxiv.org/abs/1711.05101>
- Chen, Tianqi, Emily Fox, and Carlos Guestrin. “Stochastic Gradient Hamiltonian Monte Carlo.” 2014. <https://arxiv.org/abs/1402.4102>
- Maddox, William, et al. “On Symplectic Optimization.” 2018. <https://arxiv.org/abs/1802.03653>
- Wilson, Ashia C., et al. “Hamiltonian Descent Methods.” 2018. <https://arxiv.org/abs/1809.05042>
- Foret, Pierre, et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization.” 2021. <https://openreview.net/forum?id=6Tm1mposlrM>
- Yu, Tianhe, et al. “Gradient Surgery for Multi-Task Learning.” 2020. <https://arxiv.org/abs/2001.06782>
"""
    output_path = PAPER_DIR / "cmo_draft.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_paper_artifacts() -> dict[str, Path]:
    _ensure_dirs()

    accepted_benchmark_path = ACCEPTED_DIR / "benchmark_results.csv"
    accepted_ablation_path = ACCEPTED_DIR / "ablation_results.csv"
    directional_benchmark_path = DIRECTIONAL_DIR / "benchmark_results.csv"
    gpu_benchmark_paths = [
        GPU_DIR / "gpu_benchmark_results.csv",
        GPU_DIR / "gpu_cnn_results.csv",
        GPU_DIR / "gpu_stress_results.csv",
        GPU_DIR / "gpu_multitask_results.csv",
    ]
    gpu_runtime_path = GPU_DIR / "runtime_memory_results.csv"
    gpu_ablation_path = GPU_DIR / "gpu_ablation_results.csv"
    cnn_benchmark_path = CNN_DIR / "benchmark_results.csv"

    accepted_raw = _read_csv(accepted_benchmark_path)
    accepted_agg = aggregate_results(accepted_raw)
    directional_raw = _read_csv(directional_benchmark_path)
    directional_agg = aggregate_results(directional_raw)
    gpu_combined_raw = pd.concat([_read_csv(path) for path in gpu_benchmark_paths if path.exists()], ignore_index=True)
    gpu_agg = aggregate_results(gpu_combined_raw)
    gpu_runtime = _read_csv(gpu_runtime_path)
    gpu_ablation = _read_csv(gpu_ablation_path)
    cnn_raw = _read_csv(cnn_benchmark_path)
    cnn_agg = aggregate_results(cnn_raw)
    accepted_ablation = _read_csv(accepted_ablation_path)

    accepted_table = _accepted_task_table(accepted_agg)
    directional_table = _focused_task_table(directional_agg)
    runtime_table = _runtime_table(gpu_runtime)
    cnn_table = _cnn_table(cnn_agg)
    accepted_ablation_table = _ablation_table(accepted_ablation)
    gpu_ablation_table = _ablation_table(gpu_ablation)

    accepted_win_summary = _summarize_wins(
        accepted_agg,
        "coherent_momentum_optimizer",
        ACCEPTED_BASELINES,
        "accepted_historical",
        accepted_benchmark_path,
    )
    focused_win_summary = pd.concat(
        [
            _summarize_wins(directional_agg, "coherent_momentum_optimizer", FOCUSED_BASELINES, "directional_instability", directional_benchmark_path),
            _summarize_wins(directional_agg, "coherent_momentum_optimizer_improved", FOCUSED_BASELINES, "directional_instability", directional_benchmark_path),
        ],
        ignore_index=True,
    )
    gpu_win_summary = pd.concat(
        [
            _summarize_wins(gpu_agg, "coherent_momentum_optimizer_improved", ["coherent_momentum_optimizer", "adamw", "rmsprop", "sgd_momentum"], "gpu_improved_branch", gpu_benchmark_paths[0]),
        ],
        ignore_index=True,
    )
    cnn_win_summary = _summarize_wins(
        cnn_agg,
        "coherent_momentum_optimizer_improved",
        ["adamw", "rmsprop", "sgd_momentum"],
        "cnn_credibility",
        cnn_benchmark_path,
    )

    _write_table(
        "accepted_mainline_summary",
        accepted_table,
        [accepted_benchmark_path],
        "Accepted historical Coherent Momentum rows aggregated from the checked-in mainline benchmark CSV.",
    )
    _write_table(
        "directional_instability_summary",
        directional_table,
        [directional_benchmark_path],
        "Focused newcomer-facing directional-instability benchmark slice. This is the narrowest public proof benchmark in the repository.",
    )
    _write_table(
        "gpu_runtime_memory_summary",
        runtime_table,
        [gpu_runtime_path],
        "Runtime and memory summary from the checked-in improved/GPU audit.",
    )
    _write_table(
        "cnn_credibility_summary",
        cnn_table,
        [cnn_benchmark_path],
        "CNN credibility results kept visible as a failure check rather than folded into a broad average.",
    )
    _write_table(
        "accepted_ablation_summary",
        accepted_ablation_table,
        [accepted_ablation_path],
        "Accepted historical ablation summary for the stable public line.",
    )
    _write_table(
        "gpu_ablation_summary",
        gpu_ablation_table,
        [gpu_ablation_path],
        "Improved/GPU branch ablation summary. This is useful engineering context but does not replace the accepted historical line.",
    )
    _write_table(
        "accepted_win_summary",
        accepted_win_summary,
        [accepted_benchmark_path],
        "Meaningful win counts for the accepted historical line, computed from the checked-in benchmark CSV.",
    )
    _write_table(
        "focused_and_branch_win_summary",
        pd.concat([focused_win_summary, gpu_win_summary, cnn_win_summary], ignore_index=True),
        [directional_benchmark_path, gpu_benchmark_paths[0], cnn_benchmark_path],
        "Focused directional-instability, improved/GPU branch, and CNN credibility win summaries.",
    )

    _plot_grouped_bar(
        accepted_table[accepted_table["optimizer"].isin(["coherent_momentum_optimizer", "adamw", "rmsprop", "sgd_momentum"])],
        "task",
        "mean_best_val_loss",
        "optimizer",
        "Accepted Historical Best Validation Loss",
        FIGURES_DIR / "accepted_historical_best_val_loss.png",
        "mean best val loss",
    )
    _plot_grouped_bar(
        directional_table,
        "task",
        "mean_best_val_loss",
        "optimizer",
        "Directional Instability Benchmark",
        FIGURES_DIR / "directional_instability_comparison.png",
        "mean best val loss",
    )
    _plot_simple_bar(
        runtime_table[runtime_table["optimizer"].isin(["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop", "sgd_momentum"])],
        "optimizer",
        "mean_runtime_per_step_ms",
        "Runtime Per Step",
        FIGURES_DIR / "runtime_per_step_comparison.png",
        "ms per step",
    )
    _plot_simple_bar(
        runtime_table[runtime_table["optimizer"].isin(["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop", "sgd_momentum"])],
        "optimizer",
        "mean_optimizer_state_mb",
        "Optimizer State Size",
        FIGURES_DIR / "memory_state_size_comparison.png",
        "MB",
    )
    _plot_grouped_bar(
        cnn_table,
        "task",
        "mean_best_val_accuracy",
        "optimizer",
        "CNN Credibility Benchmark",
        FIGURES_DIR / "cnn_credibility_failure_plot.png",
        "mean best val accuracy",
    )
    _plot_simple_bar(
        accepted_ablation_table.head(8),
        "variant_name",
        "mean_selection_score" if "mean_selection_score" in accepted_ablation_table.columns else "mean_best_val_loss",
        "Accepted Historical Ablation Impact",
        FIGURES_DIR / "ablation_impact_plot.png",
        "mean selection score" if "mean_selection_score" in accepted_ablation_table.columns else "mean best val loss",
    )
    _plot_simple_bar(
        accepted_win_summary,
        "baseline",
        "meaningful_wins",
        "Accepted Historical Win Counts",
        FIGURES_DIR / "accepted_historical_win_summary.png",
        "meaningful wins",
    )

    summary_rows: list[dict[str, Any]] = []
    for baseline in ACCEPTED_BASELINES:
        subset = accepted_win_summary[accepted_win_summary["baseline"] == baseline]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "section": "accepted_historical",
                "metric": f"meaningful_wins_vs_{baseline}",
                "value": int(subset["meaningful_wins"].iloc[0]),
                "source_csv": _relative(accepted_benchmark_path),
                "note": "computed from aggregate_results and compute_meaningful_wins",
            }
        )
    for baseline in FOCUSED_BASELINES:
        for optimizer_name in ["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved"]:
            subset = focused_win_summary[
                (focused_win_summary["optimizer"] == optimizer_name) & (focused_win_summary["baseline"] == baseline)
            ]
            if subset.empty:
                continue
            summary_rows.append(
                {
                    "section": "directional_instability",
                    "metric": f"{optimizer_name}_wins_vs_{baseline}",
                    "value": int(subset["meaningful_wins"].iloc[0]),
                    "source_csv": _relative(directional_benchmark_path),
                    "note": "focused newcomer-facing instability slice",
                }
            )
    for baseline in ["coherent_momentum_optimizer", "adamw", "rmsprop", "sgd_momentum"]:
        subset = gpu_win_summary[gpu_win_summary["baseline"] == baseline]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "section": "gpu_improved_branch",
                "metric": f"coherent_momentum_optimizer_improved_wins_vs_{baseline}",
                "value": int(subset["meaningful_wins"].iloc[0]),
                "source_csv": " | ".join(_relative(path) for path in gpu_benchmark_paths if path.exists()),
                "note": "combined benchmark, stress, cnn, and multitask slices from the improved/GPU audit",
            }
        )
    for optimizer_name in ["coherent_momentum_optimizer", "coherent_momentum_optimizer_improved", "adamw", "rmsprop", "sgd_momentum"]:
        subset = runtime_table[runtime_table["optimizer"] == optimizer_name]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "section": "runtime_memory",
                "metric": f"{optimizer_name}_mean_runtime_per_step_ms",
                "value": float(subset["mean_runtime_per_step_ms"].iloc[0]),
                "source_csv": _relative(gpu_runtime_path),
                "note": "average over runtime_memory_results rows",
            }
        )
        summary_rows.append(
            {
                "section": "runtime_memory",
                "metric": f"{optimizer_name}_mean_optimizer_state_mb",
                "value": float(subset["mean_optimizer_state_mb"].iloc[0]),
                "source_csv": _relative(gpu_runtime_path),
                "note": "average over runtime_memory_results rows",
            }
        )
    accepted_best_row = accepted_agg[accepted_agg["optimizer"] == "coherent_momentum_optimizer"].sort_values(
        ["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]
    ).iloc[0]
    summary_rows.append(
        {
            "section": "accepted_historical",
            "metric": "best_current_cmo_task",
            "value": accepted_best_row["task"],
            "source_csv": _relative(accepted_benchmark_path),
            "note": f"best val loss {_fmt(accepted_best_row['mean_best_val_loss'], 6)}, best val accuracy {_fmt(accepted_best_row['mean_best_val_accuracy'], 6)}",
        }
    )

    summary_path = _write_paper_results_summary(summary_rows)
    claims_path = _write_claims_audit(accepted_win_summary, focused_win_summary, cnn_win_summary, runtime_table)
    draft_path = _write_paper_draft(accepted_best_row, focused_win_summary, accepted_win_summary, runtime_table)

    manifest = {
        "summary_csv": _relative(summary_path),
        "claims_audit": _relative(claims_path),
        "draft": _relative(draft_path),
        "tables_dir": _relative(TABLES_DIR),
        "figures_dir": _relative(FIGURES_DIR),
    }
    (PAPER_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary_csv": summary_path,
        "claims_audit": claims_path,
        "draft": draft_path,
        "tables_dir": TABLES_DIR,
        "figures_dir": FIGURES_DIR,
    }
