from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmarking import _train_single_run, run_benchmark_suite, run_smoke_suite, run_tuning_suite
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "block_direction_v2"
EXISTING_RESEARCH_DIR = PROJECT_ROOT / "reports" / "optimizer_research"

FOCUS_OPTIMIZERS = [
    "block_direction_optimizer_v2",
    "block_direction_optimizer",
    "direction_recovery_optimizer",
    "observation_recovery_optimizer",
    "adamw",
    "adam",
    "rmsprop",
    "sgd",
    "sgd_momentum",
    "lion",
    "muon_hybrid",
    "topological_adam",
    "real_hamiltonian_adam",
    "magneto_hamiltonian_adam",
]

BASELINE_COMPARISONS = [
    "block_direction_optimizer",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "muon_hybrid",
    "topological_adam",
    "real_hamiltonian_adam",
    "magneto_hamiltonian_adam",
]

LITERATURE_ROWS = [
    {
        "family": "Momentum / SGD",
        "representative_method": "SGD with momentum",
        "source_title": "On the importance of initialization and momentum in deep learning",
        "source_url": "https://proceedings.mlr.press/v28/sutskever13.html",
        "what_it_does": "Accumulates a velocity vector and follows a smoothed gradient direction.",
        "direction_or_transform": "Transforms gradient into velocity; does not choose among multiple candidate directions.",
        "scope": "Global / per-parameter",
        "already_covers": "Low-state directional persistence and strong practical baselines.",
        "difference_from_block_direction_v2": "V2 explicitly selects blockwise directions from a candidate pool instead of using one momentum recursion.",
        "required_baseline": "sgd_momentum",
    },
    {
        "family": "RMSProp",
        "representative_method": "RMSProp",
        "source_title": "Neural Networks for Machine Learning, Lecture 6e",
        "source_url": "https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf",
        "what_it_does": "Normalizes updates by an EMA of squared gradients.",
        "direction_or_transform": "Per-parameter scaling of the gradient; no candidate selection.",
        "scope": "Per-parameter",
        "already_covers": "Cheap adaptivity and strong small/noisy benchmark performance.",
        "difference_from_block_direction_v2": "V2 keeps step magnitude separate and focuses on blockwise direction trust rather than coordinate-wise variance scaling.",
        "required_baseline": "rmsprop",
    },
    {
        "family": "Adam / AdamW",
        "representative_method": "AdamW",
        "source_title": "Decoupled Weight Decay Regularization",
        "source_url": "https://arxiv.org/abs/1711.05101",
        "what_it_does": "Uses first and second moments with decoupled weight decay.",
        "direction_or_transform": "Transforms the gradient through EMA momentum and variance normalization.",
        "scope": "Per-parameter",
        "already_covers": "The standard strong adaptive baseline.",
        "difference_from_block_direction_v2": "V2 is not an Adam-family moment method; it stores block trust and direction memories instead of Adam moments.",
        "required_baseline": "adamw",
    },
    {
        "family": "Lion",
        "representative_method": "Lion",
        "source_title": "Symbolic Discovery of Optimization Algorithms",
        "source_url": "https://arxiv.org/abs/2302.06675",
        "what_it_does": "Uses sign momentum updates with low state overhead.",
        "direction_or_transform": "Transforms the gradient into a signed momentum direction.",
        "scope": "Per-parameter",
        "already_covers": "Low-memory momentum-like direction control.",
        "difference_from_block_direction_v2": "V2 chooses blockwise directions from several structured candidates instead of committing to a single signed-momentum rule.",
        "required_baseline": "lion",
    },
    {
        "family": "Muon / matrix orthogonalization",
        "representative_method": "Muon",
        "source_title": "Muon is Scalable for LLM Training",
        "source_url": "https://arxiv.org/abs/2502.16982",
        "what_it_does": "Orthogonalizes matrix updates, typically via Newton-Schulz or related matrix steps.",
        "direction_or_transform": "Transforms matrix gradients into orthogonalized matrix directions.",
        "scope": "Matrix-wise / blockwise",
        "already_covers": "Matrix geometry and orthogonalized updates.",
        "difference_from_block_direction_v2": "V2 may include a Muon-like candidate, but only as one option inside a broader blockwise trust-and-selection rule.",
        "required_baseline": "muon_hybrid",
    },
    {
        "family": "Shampoo",
        "representative_method": "Shampoo",
        "source_title": "Shampoo: Preconditioned Stochastic Tensor Optimization",
        "source_url": "https://arxiv.org/abs/1802.09568",
        "what_it_does": "Uses tensor-mode preconditioners for structured second-order scaling.",
        "direction_or_transform": "Transforms gradients with block/tensor preconditioners.",
        "scope": "Tensor / block-wise",
        "already_covers": "Structured preconditioning with heavy compute/state cost.",
        "difference_from_block_direction_v2": "V2 is not a tensor preconditioner; it is a candidate-direction selector with lightweight block memory.",
        "required_baseline": "muon_hybrid",
    },
    {
        "family": "K-FAC / natural gradient",
        "representative_method": "K-FAC",
        "source_title": "Optimizing Neural Networks with Kronecker-factored Approximate Curvature",
        "source_url": "https://arxiv.org/abs/1503.05671",
        "what_it_does": "Approximates natural-gradient curvature with Kronecker factors.",
        "direction_or_transform": "Transforms gradients with an approximate inverse curvature matrix.",
        "scope": "Layer / block-wise",
        "already_covers": "Curvature-aware structured updates.",
        "difference_from_block_direction_v2": "V2 does not estimate curvature; it evaluates candidate directions by trust signals such as coherence and recoverability.",
        "required_baseline": "adamw",
    },
    {
        "family": "SAM / ASAM",
        "representative_method": "SAM",
        "source_title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "source_url": "https://arxiv.org/abs/2010.01412",
        "what_it_does": "Optimizes for flat neighborhoods by solving a local min-max problem.",
        "direction_or_transform": "Transforms the objective / gradient using a perturb-and-recompute step.",
        "scope": "Global / layer-wise perturbation",
        "already_covers": "Neighborhood robustness and flatness-aware updates.",
        "difference_from_block_direction_v2": "V2 uses cheap blockwise perturbation only to gate candidate trust, not to redefine the full objective.",
        "required_baseline": "adamw",
    },
    {
        "family": "Gradient surgery",
        "representative_method": "PCGrad",
        "source_title": "Gradient Surgery for Multi-Task Learning",
        "source_url": "https://arxiv.org/abs/2001.06782",
        "what_it_does": "Projects conflicting task gradients away from one another.",
        "direction_or_transform": "Selects/projectively edits a gradient combination across task losses.",
        "scope": "Task-global / shared-parameter",
        "already_covers": "Explicit conflict projection when multiple loss gradients are available.",
        "difference_from_block_direction_v2": "V2 works with single-task blocks too and compares a richer candidate pool than pairwise task projections.",
        "required_baseline": "magneto_hamiltonian_adam",
    },
    {
        "family": "Conflict-aware MTL",
        "representative_method": "CAGrad",
        "source_title": "Conflict-Averse Gradient Descent for Multi-task learning",
        "source_url": "https://openreview.net/forum?id=_61Qh8tULj_",
        "what_it_does": "Balances average descent with worst-task local improvement.",
        "direction_or_transform": "Chooses a joint gradient direction under conflict constraints.",
        "scope": "Task-global",
        "already_covers": "Conflict-aware direction selection for multi-loss settings.",
        "difference_from_block_direction_v2": "V2 operates blockwise and does not require multiple explicit task gradients to define its trust rule.",
        "required_baseline": "magneto_hamiltonian_adam",
    },
    {
        "family": "Lookahead",
        "representative_method": "Lookahead",
        "source_title": "Lookahead Optimizer: k steps forward, 1 step back",
        "source_url": "https://papers.nips.cc/paper_files/paper/2019/file/90fd4f88f588ae64038134f1eeaa023f-Paper.pdf",
        "what_it_does": "Averages fast inner optimizer trajectories with slower outer steps.",
        "direction_or_transform": "Wraps another optimizer rather than defining a new local direction rule.",
        "scope": "Global",
        "already_covers": "Trajectory smoothing and schedule robustness.",
        "difference_from_block_direction_v2": "V2 makes block-local direction choices directly; it is not a two-timescale wrapper.",
        "required_baseline": "adamw",
    },
    {
        "family": "Learned optimizers",
        "representative_method": "VeLO",
        "source_title": "VeLO: Training Versatile Learned Optimizers by Scaling Up",
        "source_url": "https://arxiv.org/abs/2211.09760",
        "what_it_does": "Meta-learns an optimizer across many tasks.",
        "direction_or_transform": "Learns update transformations from data instead of hand-specifying the rule.",
        "scope": "Global / meta-learned",
        "already_covers": "Large-scale learned optimization policies.",
        "difference_from_block_direction_v2": "V2 is hand-specified and interpretable, with explicit block trust components instead of a learned policy.",
        "required_baseline": "adamw",
    },
    {
        "family": "Block coordinate / direction methods",
        "representative_method": "Block coordinate descent",
        "source_title": "Efficiency of Coordinate Descent Methods on Huge-Scale Optimization Problems",
        "source_url": "https://link.springer.com/article/10.1007/s10107-012-0597-5",
        "what_it_does": "Optimizes by choosing coordinates or blocks to update.",
        "direction_or_transform": "Selects coordinates/blocks, often with simple local descent rules.",
        "scope": "Coordinate / block-wise",
        "already_covers": "Explicit block structure and block scheduling.",
        "difference_from_block_direction_v2": "V2 updates all blocks but selects different candidate directions inside each block based on trust.",
        "required_baseline": "sgd_momentum",
    },
    {
        "family": "Trust-region methods",
        "representative_method": "Trust-region gradient methods",
        "source_title": "Trust Region Methods",
        "source_url": "https://epubs.siam.org/doi/book/10.1137/1.9780898719857",
        "what_it_does": "Limits step size according to a local trust model.",
        "direction_or_transform": "Uses trust to scale or reject steps.",
        "scope": "Global / block-wise depending on method",
        "already_covers": "Trust gating and bounded steps.",
        "difference_from_block_direction_v2": "V2 uses trust to rank discrete candidate directions inside each block, not just to scale one search direction.",
        "required_baseline": "adamw",
    },
    {
        "family": "Evolutionary / black-box search",
        "representative_method": "CMA-ES",
        "source_title": "Completely Derandomized Self-Adaptation in Evolution Strategies",
        "source_url": "https://link.springer.com/article/10.1023/A:1009669705971",
        "what_it_does": "Searches update directions without gradients using covariance-adapted sampling.",
        "direction_or_transform": "Samples candidate directions rather than transforming the gradient.",
        "scope": "Global population-based",
        "already_covers": "Candidate-direction search without gradients.",
        "difference_from_block_direction_v2": "V2 is still first-order and cheap enough for neural training; it scores a small deterministic candidate set per block.",
        "required_baseline": "sgd_momentum",
    },
]


def block_direction_v2_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "linear_regression",
            "breast_cancer_mlp",
            "moons_mlp",
            "oscillatory_valley",
            "plateau_escape_objective",
            "saddle_objective",
            "conflicting_batches_classification",
            "block_structure_classification",
            "low_rank_matrix_objective",
            "sparse_gradients_linear",
        ],
        "benchmark_tasks": [
            "linear_regression",
            "logistic_regression",
            "breast_cancer_mlp",
            "wine_mlp",
            "moons_mlp",
            "circles_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
            "rosenbrock_valley",
            "narrow_valley_objective",
            "direction_reversal_objective",
            "noisy_gradients_classification",
            "label_noise_breast_cancer",
            "overfit_small_wine",
            "small_batch_instability",
            "noisy_regression",
            "conflicting_batches_classification",
            "block_structure_classification",
            "low_rank_matrix_objective",
            "sparse_gradients_linear",
            "nonstationary_moons",
        ],
        "smoke_tasks": ["oscillatory_valley", "breast_cancer_mlp", "low_rank_matrix_objective"],
        "smoke_optimizers": ["block_direction_optimizer_v2", "block_direction_optimizer", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.35,
        "search_budget": 3,
        "tuning_epoch_scale": 0.50,
        "benchmark_epoch_scale": 0.78,
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
    write_block_direction_v2_current_state(output_dir)
    write_block_direction_v2_literature_scan(output_dir)
    write_block_direction_v2_math_definition(output_dir)


def write_block_direction_v2_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    benchmark_path = EXISTING_RESEARCH_DIR / "benchmark_results.csv"
    stress_path = EXISTING_RESEARCH_DIR / "stress_test_results.csv"
    win_path = EXISTING_RESEARCH_DIR / "win_flags.csv"
    if not benchmark_path.exists() or not stress_path.exists() or not win_path.exists():
        current_state_text = "# Current BlockDirection State\n\nExisting optimizer research outputs were not found in `reports/optimizer_research`, so the current-state summary could not be regenerated automatically.\n"
        (output_path / "current_state.md").write_text(current_state_text, encoding="utf-8")
        return {}

    benchmark_frame = pd.read_csv(benchmark_path)
    stress_frame = pd.read_csv(stress_path)
    combined = pd.concat([benchmark_frame, stress_frame], ignore_index=True)
    aggregated = aggregate_results(combined)
    win_flags = pd.read_csv(win_path)

    v1_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer")
    recovery_row = _best_row_for_optimizer(aggregated, "direction_recovery_optimizer")
    observation_row = _best_row_for_optimizer(aggregated, "observation_recovery_optimizer")
    magneto_h_row = _best_row_for_optimizer(aggregated, "magneto_hamiltonian_adam")
    real_h_row = _best_row_for_optimizer(aggregated, "real_hamiltonian_adam")

    v1_wins = win_flags[win_flags["optimizer"] == "block_direction_optimizer"]
    summary = {
        "v1_best_task": None if v1_row is None else str(v1_row["task"]),
        "v1_best_val_loss": None if v1_row is None else float(v1_row["mean_best_val_loss"]),
        "v1_best_val_accuracy": None if v1_row is None or math.isnan(float(v1_row["mean_best_val_accuracy"])) else float(v1_row["mean_best_val_accuracy"]),
        "v1_wins_vs_adamw": int(v1_wins.loc[v1_wins["baseline"] == "adamw", "win"].sum()),
        "v1_wins_vs_rmsprop": int(v1_wins.loc[v1_wins["baseline"] == "rmsprop", "win"].sum()),
        "v1_wins_vs_sgd_momentum": int(v1_wins.loc[v1_wins["baseline"] == "sgd_momentum", "win"].sum()),
        "v1_wins_vs_muon": int(v1_wins.loc[v1_wins["baseline"] == "muon_hybrid", "win"].sum()),
    }

    rows = []
    for name, row in [
        ("block_direction_optimizer", v1_row),
        ("direction_recovery_optimizer", recovery_row),
        ("observation_recovery_optimizer", observation_row),
        ("real_hamiltonian_adam", real_h_row),
        ("magneto_hamiltonian_adam", magneto_h_row),
        ("adamw", _best_row_for_optimizer(aggregated, "adamw")),
        ("rmsprop", _best_row_for_optimizer(aggregated, "rmsprop")),
        ("sgd_momentum", _best_row_for_optimizer(aggregated, "sgd_momentum")),
        ("muon_hybrid", _best_row_for_optimizer(aggregated, "muon_hybrid")),
        ("topological_adam", _best_row_for_optimizer(aggregated, "topological_adam")),
    ]:
        if row is None:
            continue
        rows.append(
            {
                "optimizer": name,
                "best_task": row["task"],
                "mean_best_val_loss": row["mean_best_val_loss"],
                "mean_best_val_accuracy": row["mean_best_val_accuracy"],
                "mean_runtime_seconds": row["mean_runtime_seconds"],
                "mean_optimizer_state_mb": row.get("mean_optimizer_state_mb", np.nan),
            }
        )
    comparison_table = pd.DataFrame(rows)

    lines = [
        "# Current BlockDirectionOptimizer State",
        "",
        "## Current update rule",
        "",
        "- `BlockDirectionOptimizer` V1 is already a non-Adam blockwise optimizer. It stores a blockwise direction memory, forms a signed consensus between the current block gradient and memory, then scales the block step with coherence, norm-ratio, recoverability, and conflict signals.",
        "- It does **not** maintain Adam first/second moments. Its main step is `update_i = trust_i * consensus_i`, clipped by a block update-ratio cap.",
        "",
        "## Current candidate directions",
        "",
        "- V1 does **not** have an explicit candidate pool. It effectively uses one consensus direction built from the current gradient direction, the stored memory direction, and an opposite-signed consensus when memory and gradient disagree strongly.",
        "- That means V1 already shows the right *theme* for a novel branch, but it still lacks explicit direction search, blockwise winner/blend rules, and matrix-aware candidates.",
        "",
        "## Current scoring method",
        "",
        "- V1 uses one trust scale per block:",
        "  - positive terms: coherence with memory, gradient-norm surprise, recoverability EMA",
        "  - negative term: conflict with memory",
        "- It does **not** compare multiple candidate directions with separate trust scores.",
        "",
        "## Current diagnostics",
        "",
        "- `block_coherence`",
        "- `block_trust_scale`",
        "- `block_norm_ratio`",
        "- `memory_ratio` / `memory_support`",
        "- `recoverability_score`",
        "- `block_conflict`",
        "- `update_ratio`",
        "- `block_count`",
        "",
        "## Current benchmark state",
        "",
        _markdown_table(comparison_table),
        "",
        "## Current weaknesses",
        "",
        "- No explicit candidate pool, so it cannot test projection, orthogonal escape, sparse, low-rank, or Muon-like directions cleanly.",
        "- Recoverability is attached to the current gradient direction only; it is not a gate that can reject one candidate in favor of another.",
        "- V1 improved a lot on oscillatory, saddle, and plateau-style tasks, but it still underperforms on true matrix/low-rank structure tasks.",
        "- The strongest current local custom comparator is `MagnetoHamiltonianAdam`, which already showed that directional coherence and rotation cues matter; V1 should borrow those cues for scoring but remain non-Adam.",
        "",
        "## Why a V2 branch is justified",
        "",
        f"- Existing V1 best task: `{summary['v1_best_task']}` with mean best validation loss `{summary['v1_best_val_loss']:.6f}` and mean best validation accuracy `{summary['v1_best_val_accuracy']:.4f}`.",
        f"- Existing V1 meaningful wins: `{summary['v1_wins_vs_adamw']}` vs AdamW, `{summary['v1_wins_vs_rmsprop']}` vs RMSProp, `{summary['v1_wins_vs_sgd_momentum']}` vs SGD momentum, `{summary['v1_wins_vs_muon']}` vs the local Muon hybrid baseline.",
        "- That is enough signal to justify a serious V2, but not enough to claim a publishable optimizer yet.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def write_block_direction_v2_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)

    lines = [
        "# BlockDirectionOptimizerV2 Literature Scan",
        "",
        "This scan focuses on optimizer families that are most relevant to a **blockwise direction-selection** optimizer. The central distinction is whether a method *transforms one gradient direction* or *chooses among multiple candidate directions*.",
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
        "## Main findings",
        "",
        "- Most strong practical optimizers transform one direction: SGD momentum, RMSProp, AdamW, Lion, Shampoo, and K-FAC all start from a single gradient direction and change its scale or curvature.",
        "- The closest *direction-selection* relatives are multi-task conflict methods such as PCGrad and CAGrad, plus line-search/trust-region families. Those methods still differ from this branch because they usually assume multiple task gradients or a local model rather than a cheap blockwise candidate pool.",
        "- Muon is especially important because it already shows that matrix-shaped parameters can benefit from structured directions rather than plain coordinate-wise scaling.",
        "- SAM is relevant because it uses perturbation to test robustness, but it perturbs the objective neighborhood rather than using perturbation as a cheap gate on candidate block directions.",
        "",
        "## Novelty pressure from the literature",
        "",
        "- A novelty claim is **not** available for 'Adam plus more signals'. That area is heavily populated already.",
        "- A more plausible opening is a **fast blockwise candidate-direction optimizer** that stays first-order, keeps state light, and uses perturbation/recoverability only as a gate.",
        "- The most obvious redundancy risk is overlapping with block coordinate descent, Muon-style matrix updates, or gradient surgery, so the report must compare to those families explicitly and avoid overclaiming.",
        "",
        "## Sources",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']}): {row['what_it_does']}")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_block_direction_v2_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BlockDirectionOptimizerV2 Mathematical Definition",
        "",
        "Let the trainable parameters be partitioned into blocks `B_i`, with parameter block `q_i` and gradient block `g_i`.",
        "",
        "## Candidate directions",
        "",
        "For each block `i`, construct a candidate set",
        "",
        "`D_i = {d_i^(grad), d_i^(norm), d_i^(trust), d_i^(smooth), d_i^(proj), d_i^(orth), d_i^(topk), d_i^(lowrank), d_i^(sign), d_i^(muon)}`",
        "",
        "where candidates are enabled or disabled independently. In the default configuration:",
        "",
        "- `d_i^(grad) = -g_i`",
        "- `d_i^(norm) = -g_i / (||g_i|| + eps)`",
        "- `d_i^(trust)` is the previous trusted block direction",
        "- `d_i^(smooth)` is an EMA-smoothed block direction",
        "- `d_i^(proj)` is a projection-corrected direction that removes the conflicting component of the current direction relative to trusted/previous directions",
        "- `d_i^(orth)` is an orthogonal escape direction used when oscillation or conflict rises",
        "- `d_i^(topk)` keeps only the largest-magnitude block entries",
        "- `d_i^(lowrank)` is a rank-1 matrix approximation for 2D parameters",
        "- `d_i^(sign) = sign(-g_i)`",
        "- `d_i^(muon)` is an experimental orthogonalized matrix direction for 2D parameters",
        "",
        "Each candidate is normalized to a unit block direction `u_i^(c)` when scored or applied.",
        "",
        "## Trust signals",
        "",
        "For candidate `c` in block `i`, define:",
        "",
        "- Descent alignment",
        "",
        "  `A_i(c) = 0.5 * (1 + cos(u_i^(c), -g_i / (||g_i|| + eps)))`",
        "",
        "- Memory coherence",
        "",
        "  `M_i(c) = 0.5 * (1 + cos(u_i^(c), m_i))`",
        "",
        "  where `m_i` is the trusted block-memory direction if it exists, else `0`.",
        "",
        "- Improvement history",
        "",
        "  `Q_i(c)` is an EMA of observed post-step loss improvements attributed to the candidate previously selected on this block.",
        "",
        "- Recoverability gate",
        "",
        "  `R_i(c) = (1 / S) * sum_s 0.5 * (1 + cos(P_s(d_i^(c)), u_i^(c)))`",
        "",
        "  where `P_s` applies cheap perturbations such as masking, additive noise, or top-k dropping. This is used as a gate, not as a direction generator.",
        "",
        "- Gate function",
        "",
        "  `G_i(c) = clip((R_i(c) - r0) / (1 - r0), 0, 1)`",
        "",
        "- Stability score",
        "",
        "  `S_i(c) = 1 / (1 + |log((||g_i|| * A_i(c) + eps) / (gbar_i + eps))|)`",
        "",
        "  where `gbar_i` is an EMA of the block gradient norm.",
        "",
        "- Oscillation score",
        "",
        "  `O_i(c) = 0.5 * (1 - cos(u_i^(c), p_i)) + 0.5 * flip(u_i^(c), v_i)`",
        "",
        "  with `p_i` the previous gradient direction and `v_i` the previous update direction.",
        "",
        "- Conflict score",
        "",
        "  `C_i(c) = 0.5 * max(0, -cos(u_i^(c), m_i)) + 0.5 * max(0, -cos(u_i^(c), v_i))`",
        "",
        "## Trust function",
        "",
        "The raw trust score is",
        "",
        "`T_i(c) = w_d A_i(c) + w_m M_i(c) + w_q Q_i(c) + w_r G_i(c) R_i(c) + w_s S_i(c) - w_o O_i(c) - w_c C_i(c) - w_k cost(c)`",
        "",
        "where `cost(c)` is a small fixed complexity penalty for expensive candidates such as low-rank or Muon-like directions.",
        "",
        "## Selection rule",
        "",
        "Given scores `T_i(c)` over candidates, the optimizer uses one of four modes:",
        "",
        "- `winner_take_all`: choose `argmax_c T_i(c)`",
        "- `softmax_weighted_average`: blend candidates with weights proportional to `softmax(T_i / tau)`",
        "- `top2_blend`: blend only the top two candidates",
        "- `fallback_to_gradient`: choose the best candidate only if it clears a threshold, else fall back to the gradient direction",
        "",
        "## Step magnitude",
        "",
        "The default block step magnitude is",
        "",
        "`alpha_i = min( eta * lambda_i * ||g_i|| * (0.4 + 0.6 * A_i(c*)) / |B_i|^p , rho * (||q_i|| + eps) )`",
        "",
        "where",
        "",
        "`lambda_i = clip(1 + 0.55 * (T_i(c*) - 0.45) + 0.20 * (Rbar_i - 0.5), lambda_min, lambda_max)`",
        "",
        "with `Rbar_i` the recoverability EMA, `p` the dimension penalty, and `rho` the maximum block update ratio.",
        "",
        "An experimental ablation can instead use an RMSProp-like magnitude rule, but that is not the default.",
        "",
        "## Update",
        "",
        "`q_i <- q_i + alpha_i * u_i^(c*)`",
        "",
        "Weight decay remains decoupled and optional.",
        "",
        "## Why this is not Adam-like",
        "",
        "- No Adam first-moment EMA defines the main direction.",
        "- No Adam second-moment variance normalization defines the main step.",
        "- Block memories are used for trust and candidate generation, not as momentum/variance accumulators.",
        "- The primary decision is *which direction to trust for each block*, not how to rescale one universal gradient direction.",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_block_direction_v2_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_block_direction_v2_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_tuning_suite(config)
    return frame


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    win_frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "block_direction_optimizer_v2", baseline_name)
        if not wins.empty:
            win_frames.append(wins)
    frame = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_block_direction_v2_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_block_direction_v2_ablation(config: dict[str, Any]) -> pd.DataFrame:
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
                "conflicting_batches_classification",
                "block_structure_classification",
                "low_rank_matrix_objective",
                "breast_cancer_mlp",
            ],
        )
    )
    variants = [
        {"variant_name": "v2_full", "optimizer_name": "block_direction_optimizer_v2", "overrides": {}},
        {
            "variant_name": "no_recoverability_gate",
            "optimizer_name": "block_direction_optimizer_v2",
            "overrides": {
                "recoverability_weight": 0.0,
                "recoverability_samples": 0,
                "recoverability_noise_scale": 0.0,
                "recoverability_drop_fraction": 0.0,
            },
        },
        {
            "variant_name": "no_direction_memory",
            "optimizer_name": "block_direction_optimizer_v2",
            "overrides": {
                "use_trusted_direction_candidate": False,
                "use_smoothed_direction_candidate": False,
                "coherence_weight": 0.0,
                "memory_decay": 1.0,
                "smooth_decay": 1.0,
            },
        },
        {"variant_name": "no_projection", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"use_projection_candidate": False}},
        {"variant_name": "no_orthogonal_escape", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"use_orthogonal_escape_candidate": False}},
        {"variant_name": "no_block_structure", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"block_strategy": "tensor"}},
        {"variant_name": "per_parameter_version", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"block_strategy": "scalar"}},
        {"variant_name": "layerwise_only_version", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"block_strategy": "layer"}},
        {"variant_name": "winner_take_all_only", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"selection_mode": "winner_take_all"}},
        {"variant_name": "weighted_average_only", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"selection_mode": "softmax_weighted_average"}},
        {"variant_name": "fallback_to_gradient_only", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"selection_mode": "fallback_to_gradient", "fallback_threshold": 0.2}},
        {"variant_name": "no_trust_scaling", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"min_scale": 1.0, "max_scale": 1.0, "trust_decay": 1.0}},
        {"variant_name": "no_conflict_penalty", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"conflict_penalty": 0.0}},
        {"variant_name": "no_oscillation_penalty", "optimizer_name": "block_direction_optimizer_v2", "overrides": {"oscillation_penalty": 0.0}},
        {
            "variant_name": "muon_like_direction_only",
            "optimizer_name": "block_direction_optimizer_v2",
            "overrides": {
                "use_gradient_candidate": False,
                "use_normalized_gradient_candidate": False,
                "use_trusted_direction_candidate": False,
                "use_smoothed_direction_candidate": False,
                "use_projection_candidate": False,
                "use_orthogonal_escape_candidate": False,
                "use_sparse_topk_candidate": False,
                "use_low_rank_candidate": False,
                "use_sign_candidate": False,
                "use_muon_like_candidate": True,
                "selection_mode": "winner_take_all",
            },
        },
        {"variant_name": "v1_baseline", "optimizer_name": "block_direction_optimizer", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v2_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.72)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def search_block_direction_v2_rules(config: dict[str, Any]) -> pd.DataFrame:
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
                "conflicting_batches_classification",
                "block_structure_classification",
                "low_rank_matrix_objective",
            ],
        )
    )
    variants = [
        {"variant_name": "full_top2", "overrides": {}},
        {
            "variant_name": "gradient_only",
            "overrides": {
                "selection_mode": "fallback_to_gradient",
                "use_normalized_gradient_candidate": False,
                "use_trusted_direction_candidate": False,
                "use_smoothed_direction_candidate": False,
                "use_projection_candidate": False,
                "use_orthogonal_escape_candidate": False,
                "use_sparse_topk_candidate": False,
                "use_low_rank_candidate": False,
                "use_sign_candidate": False,
                "use_muon_like_candidate": False,
            },
        },
        {
            "variant_name": "gradient_memory",
            "overrides": {
                "use_projection_candidate": False,
                "use_orthogonal_escape_candidate": False,
                "use_sparse_topk_candidate": False,
                "use_low_rank_candidate": False,
                "use_sign_candidate": False,
                "use_muon_like_candidate": False,
            },
        },
        {
            "variant_name": "magneto_informed",
            "overrides": {
                "recoverability_weight": 0.0,
                "selection_mode": "top2_blend",
                "coherence_weight": 0.28,
                "projection_strength": 0.65,
                "orthogonal_strength": 0.45,
                "use_sparse_topk_candidate": False,
                "use_low_rank_candidate": False,
                "use_muon_like_candidate": False,
            },
        },
        {
            "variant_name": "recoverability_gated",
            "overrides": {
                "selection_mode": "winner_take_all",
                "recoverability_weight": 0.25,
                "coherence_weight": 0.12,
                "projection_strength": 0.0,
                "orthogonal_strength": 0.0,
                "use_projection_candidate": False,
                "use_orthogonal_escape_candidate": False,
            },
        },
        {
            "variant_name": "sparse_escape",
            "overrides": {
                "use_sparse_topk_candidate": True,
                "use_orthogonal_escape_candidate": True,
                "use_low_rank_candidate": False,
                "selection_mode": "softmax_weighted_average",
            },
        },
        {
            "variant_name": "matrix_low_rank",
            "overrides": {
                "block_strategy": "column",
                "use_low_rank_candidate": True,
                "use_muon_like_candidate": False,
                "selection_mode": "top2_blend",
            },
        },
        {
            "variant_name": "muon_matrix_experimental",
            "overrides": {
                "block_strategy": "column",
                "use_low_rank_candidate": False,
                "use_muon_like_candidate": True,
                "selection_mode": "winner_take_all",
            },
        },
        {
            "variant_name": "tensor_softmax",
            "overrides": {"block_strategy": "tensor", "selection_mode": "softmax_weighted_average"},
        },
        {
            "variant_name": "scalar_wta",
            "overrides": {"block_strategy": "scalar", "selection_mode": "winner_take_all"},
        },
        {
            "variant_name": "no_recovery_top2",
            "overrides": {"recoverability_weight": 0.0, "recoverability_samples": 0, "selection_mode": "top2_blend"},
        },
        {
            "variant_name": "winner_take_all_row",
            "overrides": {"block_strategy": "row", "selection_mode": "winner_take_all"},
        },
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="block_direction_v2_rule_search",
                    task_name=task_name,
                    optimizer_name="block_direction_optimizer_v2",
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("rule_search_epoch_scale", 0.68)),
                )
                row["variant_name"] = variant["variant_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "rule_search_results.csv", index=False)
    return frame


def _plot_selection_frequency(trace_frame: pd.DataFrame, output_path: Path) -> None:
    if trace_frame.empty or "selected_candidate_type" not in trace_frame.columns:
        return
    subset = trace_frame[(trace_frame["optimizer"] == "block_direction_optimizer_v2") & (trace_frame["event"] == "train")]
    if subset.empty:
        return
    counts = subset["selected_candidate_type"].value_counts()
    plt.figure(figsize=(9, 4))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("train-step frequency")
    plt.title("BlockDirectionOptimizerV2 dominant candidate frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_recovery_vs_performance(aggregated: pd.DataFrame, output_path: Path) -> None:
    subset = aggregated[aggregated["optimizer"] == "block_direction_optimizer_v2"]
    if subset.empty or "mean_recovery_score" not in subset.columns:
        return
    plt.figure(figsize=(6.5, 4.5))
    x = subset["mean_recovery_score"].astype(float)
    y = subset["mean_best_val_loss"].astype(float)
    plt.scatter(x, y)
    for _, row in subset.iterrows():
        plt.annotate(str(row["task"]), (float(row["mean_recovery_score"]), float(row["mean_best_val_loss"])), fontsize=7)
    plt.xlabel("mean recovery score")
    plt.ylabel("mean best validation loss")
    plt.title("Recovery score vs performance (V2)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_ablation_chart(ablation_frame: pd.DataFrame, output_path: Path) -> None:
    if ablation_frame.empty:
        return
    summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(summary["variant_name"], summary["mean_selection_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean selection score")
    plt.title("BlockDirectionOptimizerV2 ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_rule_search_chart(rule_frame: pd.DataFrame, output_path: Path) -> None:
    if rule_frame.empty:
        return
    summary = (
        rule_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(summary["variant_name"], summary["mean_selection_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean selection score")
    plt.title("BlockDirectionOptimizerV2 rule search")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_task_family_ranking(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    summary = (
        frame.groupby(["task_family", "optimizer"], as_index=False)["selection_score"]
        .mean()
        .sort_values(["task_family", "selection_score"], ascending=[True, False])
    )
    top_optimizers = ["block_direction_optimizer_v2", "block_direction_optimizer", "adamw", "rmsprop", "sgd_momentum", "muon_hybrid", "magneto_hamiltonian_adam"]
    summary = summary[summary["optimizer"].isin(top_optimizers)]
    if summary.empty:
        return
    pivot = summary.pivot(index="task_family", columns="optimizer", values="selection_score")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("mean selection score")
    plt.title("Task-family ranking")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_optimizer_metric_bar(frame: pd.DataFrame, output_path: Path, metric: str, title: str) -> None:
    if frame.empty or metric not in frame.columns:
        return
    ordered = frame.sort_values(metric, ascending=True)
    plt.figure(figsize=(9, 4.5))
    plt.bar(ordered["optimizer"], ordered[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _variant_helpfulness(frame: pd.DataFrame, full_name: str, compare_name: str) -> float | None:
    summary = frame.groupby("variant_name", as_index=False)["selection_score"].mean()
    if full_name not in set(summary["variant_name"]) or compare_name not in set(summary["variant_name"]):
        return None
    full_score = float(summary.loc[summary["variant_name"] == full_name, "selection_score"].iloc[0])
    compare_score = float(summary.loc[summary["variant_name"] == compare_name, "selection_score"].iloc[0])
    return full_score - compare_score


def export_block_direction_v2_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
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

    v2_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer_v2")
    v1_row = _best_row_for_optimizer(aggregated, "block_direction_optimizer")
    adamw_row = _best_row_for_optimizer(aggregated, "adamw")
    rmsprop_row = _best_row_for_optimizer(aggregated, "rmsprop")
    sgd_momentum_row = _best_row_for_optimizer(aggregated, "sgd_momentum")
    muon_row = _best_row_for_optimizer(aggregated, "muon_hybrid")
    magneto_h_row = _best_row_for_optimizer(aggregated, "magneto_hamiltonian_adam")
    real_h_row = _best_row_for_optimizer(aggregated, "real_hamiltonian_adam")

    v2_vs = {
        baseline: compute_meaningful_wins(aggregated, "block_direction_optimizer_v2", baseline)
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

    recoverability_delta = _variant_helpfulness(ablation_frame, "v2_full", "no_recoverability_gate")
    block_structure_delta = _variant_helpfulness(ablation_frame, "v2_full", "no_block_structure")
    projection_delta = _variant_helpfulness(ablation_frame, "v2_full", "no_projection")
    orthogonal_delta = _variant_helpfulness(ablation_frame, "v2_full", "no_orthogonal_escape")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame)
    comparison_optimizers = [
        "block_direction_optimizer_v2",
        "block_direction_optimizer",
        "adamw",
        "rmsprop",
        "sgd_momentum",
        "muon_hybrid",
        "magneto_hamiltonian_adam",
    ]
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "best_validation_loss_curves.png",
        title="Best validation-loss curves",
        metric="val_loss",
        tasks=["oscillatory_valley", "conflicting_batches_classification", "low_rank_matrix_objective"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "accuracy_curves.png",
        title="Validation accuracy curves",
        metric="val_accuracy",
        tasks=["breast_cancer_mlp", "wine_mlp", "block_structure_classification"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png")
    _plot_selection_frequency(trace_frame, figure_dir / "direction_selection_frequency.png")
    _plot_recovery_vs_performance(aggregated, figure_dir / "recovery_vs_performance.png")
    _plot_ablation_chart(ablation_frame, figure_dir / "ablation_chart.png")
    _plot_rule_search_chart(rule_frame, figure_dir / "rule_search_chart.png")
    _plot_task_family_ranking(benchmark_frame, figure_dir / "task_family_ranking.png")

    runtime_summary = aggregated[aggregated["optimizer"].isin(comparison_optimizers)][["optimizer", "mean_runtime_per_step_ms", "mean_optimizer_state_mb"]]
    _plot_optimizer_metric_bar(runtime_summary, figure_dir / "runtime_comparison.png", "mean_runtime_per_step_ms", "Runtime per step")
    _plot_optimizer_metric_bar(runtime_summary, figure_dir / "memory_state_size_comparison.png", "mean_optimizer_state_mb", "Optimizer state size")

    best_rule = None if rule_summary.empty else str(rule_summary.iloc[0]["variant_name"])
    best_ablation = None if ablation_summary.empty else str(ablation_summary.iloc[0]["variant_name"])
    v2_outperforms_v1 = int(v2_vs.get("block_direction_optimizer", pd.DataFrame())["win"].sum()) if "block_direction_optimizer" in v2_vs else 0
    v2_vs_adamw = int(v2_vs.get("adamw", pd.DataFrame())["win"].sum()) if "adamw" in v2_vs else 0
    v2_vs_rmsprop = int(v2_vs.get("rmsprop", pd.DataFrame())["win"].sum()) if "rmsprop" in v2_vs else 0
    v2_vs_sgd_momentum = int(v2_vs.get("sgd_momentum", pd.DataFrame())["win"].sum()) if "sgd_momentum" in v2_vs else 0
    v2_vs_muon = int(v2_vs.get("muon_hybrid", pd.DataFrame())["win"].sum()) if "muon_hybrid" in v2_vs else 0
    v2_vs_topological = int(v2_vs.get("topological_adam", pd.DataFrame())["win"].sum()) if "topological_adam" in v2_vs else 0
    v2_vs_magneto_h = int(v2_vs.get("magneto_hamiltonian_adam", pd.DataFrame())["win"].sum()) if "magneto_hamiltonian_adam" in v2_vs else 0
    v2_vs_real_h = int(v2_vs.get("real_hamiltonian_adam", pd.DataFrame())["win"].sum()) if "real_hamiltonian_adam" in v2_vs else 0
    total_two_x = int(win_flags["two_x"].sum()) if not win_flags.empty else 0

    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["sgd_momentum", "rmsprop", "adamw", "muon_hybrid", "lion", "topological_adam"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    lines = [
        "# BlockDirectionOptimizerV2 Final Report",
        "",
        "## 1. What BlockDirectionOptimizerV2 is",
        "",
        "- `BlockDirectionOptimizerV2` is a blockwise candidate-direction optimizer.",
        "- It does not use Adam first moments or Adam variance scaling as its core rule.",
        "- For each block it builds candidate directions, scores them with a trust function, then chooses or blends directions before applying a bounded step.",
        "",
        "## 2. How it differs from AdamW, RMSProp, SGD momentum, Muon, SAM, PCGrad/CAGrad, and V1",
        "",
        "- AdamW/RMSProp/Lion transform one gradient direction. V2 chooses among multiple blockwise candidate directions.",
        "- Muon orthogonalizes matrix updates. V2 may use a Muon-like candidate, but only as one option inside a blockwise selector.",
        "- SAM perturbs the objective neighborhood; V2 uses cheap perturbations only to gate candidate trust.",
        "- PCGrad/CAGrad operate on conflicting task gradients; V2 operates on blockwise candidate directions even in single-loss training.",
        "- V1 stored block direction memory and built one consensus direction. V2 adds an explicit candidate pool, trust function, selection modes, and matrix-aware candidates.",
        "",
        "## 3. Whether the idea is plausibly novel",
        "",
        "- The literature scan does not support a claim that 'block structure' or 'direction trust' alone is novel.",
        "- The plausibly distinct piece is the combination of: blockwise candidate-direction selection, cheap recoverability gating, magneto-style projection/rotation signals, and optional matrix candidates, without falling back to Adam moments as the main rule.",
        "- That is **prototype-level novelty**, not yet claim-ready novelty by itself.",
        "",
        "## 4. Whether it is fast enough",
        "",
        f"- Mean runtime per step for V2 best row: `{float(v2_row['mean_runtime_per_step_ms']) if v2_row is not None else float('nan'):.4f}` ms.",
        f"- Mean optimizer state size for V2 best row: `{float(v2_row['mean_optimizer_state_mb']) if v2_row is not None else float('nan'):.4f}` MB.",
        "- The extra cost is practical on the current CPU-scale suite, but matrix candidates and recoverability do add overhead relative to SGD/RMSProp.",
        "",
        "## 5. Strongest baseline comparison",
        "",
        f"- Strongest baseline in this suite: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best validation loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best validation accuracy `{float(strongest_baseline['mean_best_val_accuracy']) if pd.notna(strongest_baseline['mean_best_val_accuracy']) else float('nan'):.4f}`.",
        f"- Best V2 row: `{None if v2_row is None else v2_row['task']}` with mean best validation loss `{float(v2_row['mean_best_val_loss']) if v2_row is not None else float('nan'):.6f}` and mean best validation accuracy `{float(v2_row['mean_best_val_accuracy']) if v2_row is not None and pd.notna(v2_row['mean_best_val_accuracy']) else float('nan'):.4f}`.",
        "",
        "## 6. Comparison counts",
        "",
        f"- V2 vs V1: `{v2_outperforms_v1}` meaningful wins",
        f"- V2 vs AdamW: `{v2_vs_adamw}` meaningful wins",
        f"- V2 vs RMSProp: `{v2_vs_rmsprop}` meaningful wins",
        f"- V2 vs SGD momentum: `{v2_vs_sgd_momentum}` meaningful wins",
        f"- V2 vs Muon hybrid: `{v2_vs_muon}` meaningful wins",
        f"- V2 vs TopologicalAdam: `{v2_vs_topological}` meaningful wins",
        f"- V2 vs MagnetoHamiltonianAdam: `{v2_vs_magneto_h}` meaningful wins",
        f"- V2 vs HamiltonianAdamReal: `{v2_vs_real_h}` meaningful wins",
        f"- Total 2x events across tracked baseline comparisons: `{total_two_x}`",
        "",
        "## 7. Which task family it is best for",
        "",
        "- The current evidence should be read family-by-family, not as a claim of general superiority.",
        "- V2 is designed for noisy, conflicting, oscillatory, sparse, and structured blockwise tasks.",
        "- The most important question is whether it outperforms V1 and at least one strong baseline there. The answer is in the comparison counts and `best_by_task.csv`, not in isolated train-loss wins.",
        "",
        "## 8. Ablation findings",
        "",
        _markdown_table(ablation_summary.head(10)),
        "",
        f"- Recoverability helped: `{recoverability_delta is not None and recoverability_delta > 0.01}` (delta full - no_recoverability = `{recoverability_delta if recoverability_delta is not None else float('nan'):.4f}`)",
        f"- Block structure helped: `{block_structure_delta is not None and block_structure_delta > 0.01}` (delta full - no_block_structure = `{block_structure_delta if block_structure_delta is not None else float('nan'):.4f}`)",
        f"- Projection helped: `{projection_delta is not None and projection_delta > 0.01}` (delta full - no_projection = `{projection_delta if projection_delta is not None else float('nan'):.4f}`)",
        f"- Orthogonal escape helped: `{orthogonal_delta is not None and orthogonal_delta > 0.01}` (delta full - no_orthogonal_escape = `{orthogonal_delta if orthogonal_delta is not None else float('nan'):.4f}`)",
        "",
        "## 9. Rule-search findings",
        "",
        _markdown_table(rule_summary.head(10)),
        "",
        f"- Best discovered rule combination: `{best_rule}`",
        f"- Best ablation variant: `{best_ablation}`",
        "- This was a controlled rule-set search, not an exhaustive combinatorial search.",
        "",
        "## 10. Recommendation",
        "",
        "- If V2 beats V1 and at least one strong baseline on held-out stress or structure tasks, it is a **useful specialist branch**.",
        "- If it does not beat RMSProp or SGD momentum across a full task family, it is not a broad replacement yet.",
        "- The novelty claim should remain cautious until the branch shows consistent wins on its target family and the rule search confirms that the candidate-selection mechanism, not just one ad hoc candidate, is carrying the result.",
        "",
        "## 11. Exact next improvement target",
        "",
        "- If low-rank and block-structure tasks remain weak, the next branch should add explicit row-plus-column matrix consensus or a better low-rank candidate.",
        "- If recoverability is neutral or harmful, keep it as an experimental gate rather than a default rule.",
        "- If projection and orthogonal escape help mostly on oscillatory/saddle tasks, expose a narrower stress-specialist preset instead of forcing one default across all tasks.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "aggregated": aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
        "ablation_summary": ablation_summary,
        "rule_summary": rule_summary,
    }
