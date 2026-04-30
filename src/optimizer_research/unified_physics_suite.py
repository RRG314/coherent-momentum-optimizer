from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import build_optimizer_registry, sample_search_configs
from .benchmarking import (
    _aggregate_suite_rows,
    _load_best_tuning_map,
    _train_single_run,
    run_benchmark_suite,
    run_smoke_suite,
    run_stress_suite,
    run_tuning_suite,
)
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


UNIFIED_BASELINE_OPTIMIZERS = [
    "sgd",
    "sgd_momentum",
    "rmsprop",
    "adam",
    "adamw",
    "nadam",
    "radam",
    "lion",
    "topological_adam",
]

UNIFIED_PHYSICAL_OPTIMIZERS = [
    "sds_adam",
    "magneto_adam",
    "thermodynamic_adam",
    "diffusion_adam",
    "hamiltonian_adam",
    "hamiltonian_adam_v2",
    "uncertainty_adam",
    "unified_physics_adam",
]

CONTROLLERS = ["sds", "magneto", "thermodynamic", "diffusion", "hamiltonian", "uncertainty"]

ORIGINAL_COMBO_NAMES = {
    "neutral_adamw_equivalent",
    "unified_full",
    "sds_only",
    "magneto_only",
    "thermodynamic_only",
    "diffusion_only",
    "hamiltonian_only",
    "uncertainty_only",
    "sds_magneto",
    "sds_thermodynamic",
    "magneto_thermodynamic",
    "sds_magneto_thermodynamic",
    "full_without_diffusion",
    "full_without_hamiltonian",
    "full_without_uncertainty",
}

LITERATURE_REFERENCES = [
    {
        "category": "Adam / AdamW",
        "title": "Decoupled Weight Decay Regularization",
        "url": "https://openreview.net/forum?id=Bkg6RiCqY7",
        "summary": "AdamW is the practical Adam-family baseline because it decouples weight decay from the adaptive update rather than folding it into the gradient moments.",
    },
    {
        "category": "Adaptive baselines",
        "title": "Adaptive Methods for Nonconvex Optimization (Yogi)",
        "url": "https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf",
        "summary": "Yogi already targets Adam-style instability by changing how the second moment grows, so any stability claim needs to beat strong adaptive baselines rather than plain Adam only.",
    },
    {
        "category": "Adaptive baselines",
        "title": "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients",
        "url": "https://proceedings.neurips.cc/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf",
        "summary": "AdaBelief already uses gradient-surprise style information, which overlaps with uncertainty and reliability language.",
    },
    {
        "category": "Exploration / diffusion",
        "title": "Bayesian Learning via Stochastic Gradient Langevin Dynamics",
        "url": "https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf",
        "summary": "SGLD makes controlled Gaussian perturbations to stochastic-gradient steps explicit prior art for diffusion-style exploration.",
    },
    {
        "category": "Exploration / diffusion",
        "title": "Stochastic Gradient Hamiltonian Monte Carlo",
        "url": "https://proceedings.mlr.press/v32/cheni14.html",
        "summary": "SGHMC already combines momentum, damping, and injected noise, so unified diffusion-plus-Hamiltonian control is a recombination unless it earns a clear empirical niche.",
    },
    {
        "category": "Hamiltonian / symplectic",
        "title": "Hamiltonian Descent Methods",
        "url": "https://arxiv.org/abs/1809.05042",
        "summary": "Hamiltonian optimization with dissipation is established prior art for energy-based momentum control.",
    },
    {
        "category": "Hamiltonian / symplectic",
        "title": "On Symplectic Optimization",
        "url": "https://arxiv.org/abs/1802.03653",
        "summary": "Symplectic discretization is already a known way to control energy drift in optimization dynamics.",
    },
    {
        "category": "Entropy / temperature",
        "title": "Entropy-SGD: Biasing Gradient Descent Into Wide Valleys",
        "url": "https://arxiv.org/abs/1611.01838",
        "summary": "Entropy-shaped optimization and temperature-like control are already represented in the literature.",
    },
    {
        "category": "Gradient alignment / conflict",
        "title": "Gradient Surgery for Multi-Task Learning",
        "url": "https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf",
        "summary": "Gradient alignment and conflict mitigation are established, especially in multi-task settings.",
    },
    {
        "category": "Gradient alignment / conflict",
        "title": "Conflict-Averse Gradient Descent for Multi-task Learning",
        "url": "https://papers.nips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf",
        "summary": "Directional conflict can already be treated as a controllable optimization signal rather than a metaphor.",
    },
    {
        "category": "Physics-inspired Adam variants",
        "title": "A Physics-Inspired Optimizer: Velocity Regularized Adam",
        "url": "https://openreview.net/forum?id=6BhduwrCp3",
        "summary": "VRAdam is a direct near-neighbor because it applies physical velocity regularization to Adam-style dynamics.",
    },
    {
        "category": "Information geometry / curvature",
        "title": "New Insights and Perspectives on the Natural Gradient Method",
        "url": "https://www.jmlr.org/papers/volume21/17-678/17-678.pdf",
        "summary": "Natural-gradient and trust-region reasoning already cover geometry-aware scaling and stability control from an information-geometric perspective.",
    },
    {
        "category": "Sharpness / stability-aware training",
        "title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "url": "https://openreview.net/forum?id=6Tm1mposlrM",
        "summary": "Sharpness-aware updates are already a strong comparator for any stability-aware optimizer claim.",
    },
    {
        "category": "Control-theoretic adjacent work",
        "title": "Controlled gradient descent: A control theoretical perspective for optimization",
        "url": "https://www.sciencedirect.com/science/article/pii/S266672072400047X",
        "summary": "Control-theoretic framing of gradient dynamics already exists, even if it is not an AdamW variant.",
    },
]

REPO_SCAN_OBSERVATIONS = [
    {
        "path": "reports/novelty_assessment.md",
        "summary": "The repo already treats novelty conservatively and explicitly warns that recombining known geometry, thermodynamics, or recoverability ideas does not by itself justify novelty claims.",
    },
    {
        "path": "reports/physics_adam/final_report.md",
        "summary": "The prior physics-Adam suite found that magneto-style directional control had the strongest broad signal, while diffusion and uncertainty were weak and often fragile.",
    },
    {
        "path": "reports/hamiltonian_adam_v2/final_report.md",
        "summary": "The Hamiltonian v2 study showed that normalized energy trend control mattered, but RMSProp still remained the strongest overall baseline on many tasks.",
    },
]


def write_unified_physics_literature_scan(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "literature_scan.md"

    lines = [
        "# Unified Physics Adam Literature Scan",
        "",
        "This scan is claim discipline for the unified optimizer, not a novelty pitch.",
        "",
        "## Repo-local scan",
    ]
    for observation in REPO_SCAN_OBSERVATIONS:
        lines.append(f"- `{observation['path']}`: {observation['summary']}")

    lines.extend(
        [
            "",
            "## External literature anchors",
        ]
    )
    for reference in LITERATURE_REFERENCES:
        lines.append(f"- **{reference['category']}**: [{reference['title']}]({reference['url']})")
        lines.append(f"  {reference['summary']}")

    lines.extend(
        [
            "",
            "## What already exists",
            "- AdamW is the practical decoupled-weight-decay baseline.",
            "- SGLD and SGHMC already cover noise-driven exploration with stochastic dynamics.",
            "- Hamiltonian descent and symplectic optimization already cover energy, momentum, and drift language.",
            "- Entropy-SGD and related work already connect entropy/temperature ideas to optimization.",
            "- PCGrad and CAGrad already treat directional conflict and alignment as actionable optimization signals.",
            "- Natural gradient, curvature-aware methods, and SAM already occupy nearby geometry/stability territory.",
            "- Recent physics-inspired Adam variants like VRAdam directly overlap with velocity/stability regularization language.",
            "",
            "## What looks too close",
            "- Any claim that diffusion noise, Hamiltonian damping, entropy cooling, or uncertainty-weighted scaling is individually novel would be unsupported.",
            "- A unified optimizer that merely stacks these controls without ablation-backed benefit is best described as a recombination of known ideas.",
            "",
            "## What may still be distinct",
            "- A single AdamW-based optimizer with independently toggleable SDS, coherence, thermodynamic, diffusion, Hamiltonian, and uncertainty controllers is at least an unusual engineering combination.",
            "- The genuinely interesting question is not novelty-by-name but whether a conservative controller stack or a simpler discovered subset consistently beats AdamW and Topological Adam on held-out tasks.",
            "",
            "## Bottom line",
            "- The literature does not support claiming a new physical principle for optimization here.",
            "- At most, the unified suite can claim a careful empirical study of controller composition around AdamW.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_unified_physics_smoke(config: dict[str, Any]) -> pd.DataFrame:
    return run_smoke_suite(config)


def run_unified_physics_tuning(config: dict[str, Any]) -> pd.DataFrame:
    return run_tuning_suite(config)


def run_unified_physics_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    return run_benchmark_suite(config)


def run_unified_physics_stress(config: dict[str, Any]) -> pd.DataFrame:
    return run_stress_suite(config)


def _controller_overrides(active_controllers: list[str] | set[str]) -> dict[str, Any]:
    active = set(active_controllers)
    overrides: dict[str, Any] = {}
    for controller in CONTROLLERS:
        overrides[f"enable_{controller}"] = controller in active

    if "sds" not in active:
        overrides.update(
            {
                "sds_cooling_strength": 0.0,
                "sds_reheating_strength": 0.0,
                "sds_entropy_weight": 0.0,
            }
        )
    if "magneto" not in active:
        overrides.update(
            {
                "alignment_strength": 0.0,
                "coherence_strength": 0.0,
                "rotation_penalty": 0.0,
                "misalignment_damping": 0.0,
            }
        )
    if "thermodynamic" not in active:
        overrides.update(
            {
                "thermodynamic_entropy_weight": 0.0,
                "thermodynamic_energy_weight": 0.0,
                "thermodynamic_cooling_strength": 0.0,
                "thermodynamic_reheating_strength": 0.0,
            }
        )
    if "diffusion" not in active:
        overrides.update(
            {
                "diffusion_strength": 0.0,
                "max_noise": 0.0,
                "noise_to_update_cap": 0.0,
                "aligned_noise_weight": 0.0,
            }
        )
    if "hamiltonian" not in active:
        overrides.update(
            {
                "friction": 0.0,
                "energy_correction_strength": 0.0,
                "oscillation_damping": 0.0,
                "momentum_coupling": 0.0,
            }
        )
    if "uncertainty" not in active:
        overrides.update(
            {
                "uncertainty_weight": 0.0,
                "interference_weight": 0.0,
                "reliability_strength": 0.0,
                "exploration_strength": 0.0,
            }
        )
    return overrides


def _ablation_variants() -> list[dict[str, Any]]:
    return [
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "topological_baseline", "optimizer_name": "topological_adam", "overrides": {}},
        {"variant_name": "neutral_adamw_equivalent", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides([]) | {"min_step_scale": 1.0, "max_step_scale": 1.0}},
        {"variant_name": "unified_full", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(CONTROLLERS)},
        {"variant_name": "sds_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds"])},
        {"variant_name": "magneto_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["magneto"])},
        {"variant_name": "thermodynamic_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["thermodynamic"])},
        {"variant_name": "diffusion_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["diffusion"])},
        {"variant_name": "hamiltonian_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["hamiltonian"])},
        {"variant_name": "uncertainty_only", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["uncertainty"])},
        {"variant_name": "sds_magneto", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "magneto"])},
        {"variant_name": "sds_thermodynamic", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "thermodynamic"])},
        {"variant_name": "magneto_thermodynamic", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["magneto", "thermodynamic"])},
        {"variant_name": "sds_magneto_thermodynamic", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "magneto", "thermodynamic"])},
        {"variant_name": "full_without_diffusion", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "magneto", "thermodynamic", "hamiltonian", "uncertainty"])},
        {"variant_name": "full_without_hamiltonian", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "magneto", "thermodynamic", "diffusion", "uncertainty"])},
        {"variant_name": "full_without_uncertainty", "optimizer_name": "unified_physics_adam", "overrides": _controller_overrides(["sds", "magneto", "thermodynamic", "diffusion", "hamiltonian"])},
    ]


def run_unified_physics_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [71, 89, 107]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "moons_mlp",
                "small_batch_instability",
                "plateau_escape_objective",
                "oscillatory_valley",
                "conflicting_batches_classification",
                "pinn_harmonic_oscillator",
            ],
        )
    )
    tuning_map = _load_best_tuning_map(output_dir / "tuning_results.csv")

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in _ablation_variants():
            optimizer_name = str(variant["optimizer_name"])
            tuned = tuning_map.get((task_name, optimizer_name), {})
            hyperparameters = dict(tuned)
            hyperparameters.update(dict(variant["overrides"]))
            for seed in seeds:
                row = _train_single_run(
                    suite_name="unified_ablation",
                    task_name=task_name,
                    optimizer_name=optimizer_name,
                    hyperparameters=hyperparameters,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.75)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = optimizer_name
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def _combo_candidates() -> list[dict[str, Any]]:
    return [
        {"combo_name": "adamw_baseline", "optimizer_name": "adamw", "active": [], "codex_inferred": False},
        {"combo_name": "topological_baseline", "optimizer_name": "topological_adam", "active": [], "codex_inferred": False},
        {"combo_name": "neutral_adamw_equivalent", "optimizer_name": "unified_physics_adam", "active": [], "codex_inferred": False},
        {"combo_name": "unified_full", "optimizer_name": "unified_physics_adam", "active": CONTROLLERS, "codex_inferred": False},
        {"combo_name": "sds_only", "optimizer_name": "unified_physics_adam", "active": ["sds"], "codex_inferred": False},
        {"combo_name": "magneto_only", "optimizer_name": "unified_physics_adam", "active": ["magneto"], "codex_inferred": False},
        {"combo_name": "thermodynamic_only", "optimizer_name": "unified_physics_adam", "active": ["thermodynamic"], "codex_inferred": False},
        {"combo_name": "diffusion_only", "optimizer_name": "unified_physics_adam", "active": ["diffusion"], "codex_inferred": False},
        {"combo_name": "hamiltonian_only", "optimizer_name": "unified_physics_adam", "active": ["hamiltonian"], "codex_inferred": False},
        {"combo_name": "uncertainty_only", "optimizer_name": "unified_physics_adam", "active": ["uncertainty"], "codex_inferred": False},
        {"combo_name": "sds_magneto", "optimizer_name": "unified_physics_adam", "active": ["sds", "magneto"], "codex_inferred": False},
        {"combo_name": "sds_thermodynamic", "optimizer_name": "unified_physics_adam", "active": ["sds", "thermodynamic"], "codex_inferred": False},
        {"combo_name": "magneto_thermodynamic", "optimizer_name": "unified_physics_adam", "active": ["magneto", "thermodynamic"], "codex_inferred": False},
        {"combo_name": "sds_magneto_thermodynamic", "optimizer_name": "unified_physics_adam", "active": ["sds", "magneto", "thermodynamic"], "codex_inferred": False},
        {"combo_name": "full_without_diffusion", "optimizer_name": "unified_physics_adam", "active": ["sds", "magneto", "thermodynamic", "hamiltonian", "uncertainty"], "codex_inferred": False},
        {"combo_name": "full_without_hamiltonian", "optimizer_name": "unified_physics_adam", "active": ["sds", "magneto", "thermodynamic", "diffusion", "uncertainty"], "codex_inferred": False},
        {"combo_name": "full_without_uncertainty", "optimizer_name": "unified_physics_adam", "active": ["sds", "magneto", "thermodynamic", "diffusion", "hamiltonian"], "codex_inferred": False},
        {"combo_name": "magneto_hamiltonian", "optimizer_name": "unified_physics_adam", "active": ["magneto", "hamiltonian"], "codex_inferred": True},
        {"combo_name": "thermodynamic_hamiltonian", "optimizer_name": "unified_physics_adam", "active": ["thermodynamic", "hamiltonian"], "codex_inferred": True},
        {"combo_name": "magneto_hamiltonian_thermodynamic", "optimizer_name": "unified_physics_adam", "active": ["magneto", "hamiltonian", "thermodynamic"], "codex_inferred": True},
    ]


def run_controller_combination_search(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [71, 89, 107]))
    task_names = list(
        config.get(
            "combo_search_tasks",
            [
                "noisy_regression",
                "small_batch_instability",
                "conflicting_batches_classification",
                "oscillatory_valley",
            ],
        )
    )
    budget = int(config.get("search_budget", 2))
    search_seed = int(config.get("search_seed", 2031))
    registry = build_optimizer_registry()
    rows: list[dict[str, Any]] = []

    for candidate in _combo_candidates():
        optimizer_name = str(candidate["optimizer_name"])
        spec = registry[optimizer_name]
        sampled = sample_search_configs(spec, budget if optimizer_name == "unified_physics_adam" else 1, search_seed)
        for task_name in task_names:
            for trial_index, params in enumerate(sampled, start=1):
                trial_params = dict(params)
                if optimizer_name == "unified_physics_adam":
                    if candidate["combo_name"] == "neutral_adamw_equivalent":
                        trial_params.update(_controller_overrides([]))
                        trial_params["min_step_scale"] = 1.0
                        trial_params["max_step_scale"] = 1.0
                    else:
                        trial_params.update(_controller_overrides(candidate["active"]))
                seed_rows = [
                    _train_single_run(
                        suite_name="controller_combo_search",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters=trial_params,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=False,
                        epoch_scale=float(config.get("combo_search_epoch_scale", 0.65)),
                    )
                    for seed in seeds
                ]
                trial_frame = pd.DataFrame(seed_rows)
                rows.append(
                    {
                        "suite": "controller_combo_search",
                        "combo_name": candidate["combo_name"],
                        "optimizer": optimizer_name,
                        "task": task_name,
                        "trial_index": trial_index,
                        "hyperparameters": json.dumps(trial_params, sort_keys=True, default=str),
                        "active_controllers": json.dumps(candidate["active"]),
                        "codex_inferred": bool(candidate["codex_inferred"]),
                        "mean_final_val_loss": float(trial_frame["final_val_loss"].mean()),
                        "mean_best_val_loss": float(trial_frame["best_val_loss"].mean()),
                        "mean_final_val_accuracy": float(trial_frame["final_val_accuracy"].dropna().mean()) if trial_frame["final_val_accuracy"].notna().any() else np.nan,
                        "mean_best_val_accuracy": float(trial_frame["best_val_accuracy"].dropna().mean()) if trial_frame["best_val_accuracy"].notna().any() else np.nan,
                        "mean_steps_to_target_loss": float(trial_frame["steps_to_target_loss"].dropna().mean()) if trial_frame["steps_to_target_loss"].notna().any() else np.nan,
                        "mean_steps_to_target_accuracy": float(trial_frame["steps_to_target_accuracy"].dropna().mean()) if trial_frame["steps_to_target_accuracy"].notna().any() else np.nan,
                        "mean_runtime_seconds": float(trial_frame["runtime_seconds"].mean()),
                        "mean_loss_variance": float(trial_frame["loss_variance"].mean()),
                        "divergence_rate": float(trial_frame["diverged"].mean()),
                        "selection_score": float(trial_frame["selection_score"].mean()),
                    }
                )
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "controller_combo_results.csv", index=False)
    return frame


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_trial_per_combo_task(combo_frame: pd.DataFrame) -> pd.DataFrame:
    if combo_frame.empty:
        return combo_frame
    return (
        combo_frame.sort_values(["combo_name", "task", "selection_score"], ascending=[True, True, False])
        .groupby(["combo_name", "task"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _combo_summary(best_combo_trials: pd.DataFrame) -> pd.DataFrame:
    if best_combo_trials.empty:
        return pd.DataFrame()
    return (
        best_combo_trials.groupby(["combo_name", "optimizer", "codex_inferred"], as_index=False)[
            ["selection_score", "mean_best_val_loss", "mean_best_val_accuracy", "divergence_rate"]
        ]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )


def _combo_wins(best_combo_trials: pd.DataFrame, baseline_combo: str) -> pd.DataFrame:
    if best_combo_trials.empty:
        return pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    renamed = best_combo_trials.rename(
        columns={
            "combo_name": "optimizer",
            "mean_best_val_loss": "mean_best_val_loss",
            "mean_best_val_accuracy": "mean_best_val_accuracy",
            "mean_steps_to_target_loss": "mean_steps_to_target_loss",
            "mean_steps_to_target_accuracy": "mean_steps_to_target_accuracy",
            "mean_loss_variance": "mean_loss_variance",
        }
    ).copy()
    renamed["task_family"] = "combo_search"
    renamed["problem_type"] = "mixed"
    return compute_meaningful_wins(renamed, optimizer_name="unused", baseline_name=baseline_combo)


def _plot_ablation_heatmap(ablation_frame: pd.DataFrame, output_path: Path) -> None:
    if ablation_frame.empty or not {"task", "variant_name", "selection_score"}.issubset(ablation_frame.columns):
        return
    grouped = (
        ablation_frame.groupby(["task", "variant_name"], as_index=False)["selection_score"]
        .mean()
        .pivot(index="task", columns="variant_name", values="selection_score")
    )
    if grouped.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.imshow(grouped.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="mean selection score")
    plt.xticks(range(len(grouped.columns)), grouped.columns, rotation=45, ha="right")
    plt.yticks(range(len(grouped.index)), grouped.index)
    plt.title("Controller Ablation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_unified_physics_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    literature_path = output_path / "literature_scan.md"
    if not literature_path.exists():
        write_unified_physics_literature_scan(output_path)

    benchmark_frame = _read_optional_csv(output_path / "benchmark_results.csv")
    stress_frame = _read_optional_csv(output_path / "stress_test_results.csv")
    tuning_frame = _read_optional_csv(output_path / "tuning_results.csv")
    ablation_frame = _read_optional_csv(output_path / "ablation_results.csv")
    combo_frame = _read_optional_csv(output_path / "controller_combo_results.csv")

    combined_raw = pd.concat([frame for frame in [benchmark_frame, stress_frame] if not frame.empty], ignore_index=True)
    combined_aggregated = aggregate_results(combined_raw) if not combined_raw.empty else pd.DataFrame()
    best_frame = best_by_task(combined_aggregated) if not combined_aggregated.empty else pd.DataFrame()
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    unified_vs_adamw = compute_meaningful_wins(combined_aggregated, "unified_physics_adam", "adamw") if not combined_aggregated.empty else pd.DataFrame()
    unified_vs_topo = compute_meaningful_wins(combined_aggregated, "unified_physics_adam", "topological_adam") if not combined_aggregated.empty else pd.DataFrame()

    best_combo_trials = _best_trial_per_combo_task(combo_frame)
    combo_summary = _combo_summary(best_combo_trials)
    combo_aggregated = best_combo_trials.rename(columns={"combo_name": "optimizer"}).copy()
    if not combo_aggregated.empty:
        combo_aggregated["task_family"] = "combo_search"
        combo_aggregated["problem_type"] = "mixed"
        combo_aggregated = combo_aggregated.rename(
            columns={
                "mean_best_val_loss": "mean_best_val_loss",
                "mean_best_val_accuracy": "mean_best_val_accuracy",
                "mean_steps_to_target_loss": "mean_steps_to_target_loss",
                "mean_steps_to_target_accuracy": "mean_steps_to_target_accuracy",
                "mean_loss_variance": "mean_loss_variance",
            }
        )
    combo_wins_vs_adamw: list[pd.DataFrame] = []
    combo_wins_vs_topo: list[pd.DataFrame] = []
    if not combo_aggregated.empty and "adamw_baseline" in set(combo_aggregated["optimizer"]):
        for combo_name in sorted(set(combo_aggregated["optimizer"])):
            if combo_name == "adamw_baseline":
                continue
            wins = compute_meaningful_wins(combo_aggregated, combo_name, "adamw_baseline")
            if not wins.empty:
                wins["comparison"] = f"{combo_name}_vs_adamw_combo"
                combo_wins_vs_adamw.append(wins)
    if not combo_aggregated.empty and "topological_baseline" in set(combo_aggregated["optimizer"]):
        for combo_name in sorted(set(combo_aggregated["optimizer"])):
            if combo_name == "topological_baseline":
                continue
            wins = compute_meaningful_wins(combo_aggregated, combo_name, "topological_baseline")
            if not wins.empty:
                wins["comparison"] = f"{combo_name}_vs_topological_combo"
                combo_wins_vs_topo.append(wins)
    win_flags = pd.concat([frame for frame in [unified_vs_adamw, unified_vs_topo, *combo_wins_vs_adamw, *combo_wins_vs_topo] if not frame.empty], ignore_index=True) if any(not frame.empty for frame in [unified_vs_adamw, unified_vs_topo, *combo_wins_vs_adamw, *combo_wins_vs_topo]) else pd.DataFrame()
    win_flags.to_csv(output_path / "win_flags.csv", index=False)

    trace_frame = _load_trace_frames(combined_raw)
    comparison_optimizers = ["adamw", "rmsprop", "topological_adam", "hamiltonian_adam_v2", "magneto_adam", "unified_physics_adam"]
    if not trace_frame.empty and {"task", "optimizer"}.issubset(trace_frame.columns):
        _plot_metric(trace_frame, output_path=output_path / "figures" / "loss_curves.png", title="Loss Curves", metric="train_loss", tasks=["moons_mlp", "wine_mlp", "pinn_harmonic_oscillator"], optimizers=comparison_optimizers)
        _plot_metric(trace_frame, output_path=output_path / "figures" / "validation_accuracy_curves.png", title="Validation Accuracy Curves", metric="val_accuracy", tasks=["moons_mlp", "breast_cancer_mlp", "digits_mlp"], optimizers=comparison_optimizers, event="val")
        _plot_metric(trace_frame, output_path=output_path / "figures" / "gradient_norm_curves.png", title="Gradient Norm Curves", metric="grad_norm", tasks=["small_batch_instability", "unstable_deep_mlp"], optimizers=comparison_optimizers)
        _plot_metric(trace_frame, output_path=output_path / "figures" / "update_norm_curves.png", title="Update Norm Curves", metric="update_norm", tasks=["oscillatory_valley", "conflicting_batches_classification"], optimizers=comparison_optimizers)
        _plot_metric(trace_frame, output_path=output_path / "figures" / "sds_horizon_state_curves.png", title="SDS Horizon State", metric="horizon_code", tasks=["stagnating_regression", "small_batch_instability"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "magneto_alignment_curves.png", title="Magneto Alignment", metric="grad_momentum_cosine", tasks=["conflicting_batches_classification", "oscillatory_valley"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "magneto_rotation_curves.png", title="Magneto Rotation", metric="rotation_score", tasks=["conflicting_batches_classification", "oscillatory_valley"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "thermodynamic_entropy_temperature_curves.png", title="Thermodynamic Temperature", metric="temperature", tasks=["loss_shock_classification", "small_batch_instability"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "diffusion_noise_scale_curves.png", title="Diffusion Noise Scale", metric="diffusion_scale", tasks=["plateau_escape_objective", "stagnating_regression"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "hamiltonian_energy_drift_curves.png", title="Hamiltonian Energy Drift", metric="energy_drift", tasks=["rosenbrock_valley", "oscillatory_valley"], optimizers=["unified_physics_adam"])
        _plot_metric(trace_frame, output_path=output_path / "figures" / "uncertainty_interference_curves.png", title="Uncertainty Score", metric="uncertainty_score", tasks=["label_noise_breast_cancer", "sparse_gradients_linear"], optimizers=["unified_physics_adam"])
    _plot_heatmap(combined_aggregated[combined_aggregated["optimizer"].isin(comparison_optimizers)], output_path / "figures" / "win_loss_heatmap.png")
    steps_frame = combined_aggregated[combined_aggregated["optimizer"].isin(comparison_optimizers)][["task", "optimizer", "mean_steps_to_target_loss"]]
    _plot_bar(steps_frame, output_path / "figures" / "steps_to_target_chart.png", "Steps To Target Loss", "task", "mean_steps_to_target_loss", "optimizer")
    _plot_ablation_heatmap(ablation_frame, output_path / "figures" / "controller_ablation_heatmap.png")

    strongest_adamw = combined_aggregated[combined_aggregated["optimizer"] == "adamw"].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).head(1)
    strongest_topological = combined_aggregated[combined_aggregated["optimizer"] == "topological_adam"].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).head(1)
    strongest_unified = combined_aggregated[combined_aggregated["optimizer"] == "unified_physics_adam"].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).head(1)

    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    ) if not ablation_frame.empty else pd.DataFrame()

    best_single = combo_summary[combo_summary["combo_name"].str.endswith("_only")] if not combo_summary.empty else pd.DataFrame()
    best_single_row = best_single.head(1) if not best_single.empty else pd.DataFrame()
    candidate_combos = combo_summary[
        ~combo_summary["combo_name"].isin(["adamw_baseline", "topological_baseline", "neutral_adamw_equivalent"])
    ] if not combo_summary.empty else pd.DataFrame()
    best_combo_row = candidate_combos.head(1) if not candidate_combos.empty else pd.DataFrame()
    proposed_combo_summary = combo_summary[combo_summary["combo_name"].isin(ORIGINAL_COMBO_NAMES)] if not combo_summary.empty else pd.DataFrame()
    best_proposed = proposed_combo_summary.head(1) if not proposed_combo_summary.empty else pd.DataFrame()

    harmful_controller = "none clearly harmful"
    if not ablation_summary.empty and "unified_full" in set(ablation_summary["variant_name"]):
        full_score = float(ablation_summary.loc[ablation_summary["variant_name"] == "unified_full", "mean_selection_score"].iloc[0])
        removals = ablation_summary[ablation_summary["variant_name"].str.startswith("full_without_")].copy()
        if not removals.empty:
            removals["delta_vs_full"] = removals["mean_selection_score"] - full_score
            helpful_removal = removals.sort_values("delta_vs_full", ascending=False).iloc[0]
            if float(helpful_removal["delta_vs_full"]) > 0.01:
                harmful_controller = str(helpful_removal["variant_name"]).replace("full_without_", "")

    codex_found_better = False
    if not best_combo_row.empty and not best_proposed.empty:
        codex_found_better = bool(best_combo_row.iloc[0]["codex_inferred"]) and float(best_combo_row.iloc[0]["mean_selection_score"]) > float(best_proposed.iloc[0]["mean_selection_score"]) + 0.01

    any_combo_beats_adamw = False
    any_combo_beats_topo = False
    any_two_x = False
    if not win_flags.empty:
        any_combo_beats_adamw = bool(win_flags[win_flags["baseline"].isin(["adamw", "adamw_baseline"])]["win"].astype(bool).any())
        any_combo_beats_topo = bool(win_flags[win_flags["baseline"].isin(["topological_adam", "topological_baseline"])]["win"].astype(bool).any())
        any_two_x = bool(win_flags["two_x"].astype(bool).any())

    literature_text = literature_path.read_text(encoding="utf-8")
    literature_summary = literature_text.split("## What already exists", 1)[-1].split("## Bottom line", 1)[0].strip() if "## What already exists" in literature_text else "See literature_scan.md."

    report_lines = [
        "# Unified Physics Adam Final Report",
        "",
        "## 1. Literature scan summary",
        literature_summary,
        "",
        "## 2. What already exists",
        "- AdamW, SGLD/SGHMC, Hamiltonian descent, Entropy-SGD, PCGrad/CAGrad, natural-gradient methods, SAM, and VRAdam already cover most of the individual ingredients.",
        "",
        "## 3. What appears distinct or potentially novel",
        "- The individual signals are not novel.",
        "- At best, the distinct contribution here is a conservative AdamW controller stack with explicit toggles, ablations, and controller-combination search.",
        "",
        "## 4. Exact optimizer implemented",
        "- `UnifiedPhysicsAdam` / `HorizonFieldAdam`: AdamW base update plus independently switchable SDS, Magneto, Thermodynamic, Diffusion, Hamiltonian, and Uncertainty controllers.",
        "",
        "## 5. Exact mathematical signals used",
        "- SDS: update ratio, gradient ratio, entropy, validation gap.",
        "- Magneto: cosine alignment of gradient/momentum, gradient/previous-gradient, update/previous-update, plus rotation and coherence scores.",
        "- Thermodynamic: gradient energy, update energy, entropy, temperature EMA, heat spike.",
        "- Diffusion: stagnation counter, entropy, uncertainty, temperature, bounded noise-to-update ratio.",
        "- Hamiltonian: kinetic proxy, loss EMA, normalized energy drift, oscillation, predictive damping.",
        "- Uncertainty: second-moment variance proxy, interference, reliability, disagreement-driven exploration.",
        "",
        "## 6. How update differs from AdamW",
        "- Compute the ordinary AdamW direction.",
        "- Compute controller-specific bounded scale factors.",
        "- Combine active scales conservatively with a geometric mean.",
        "- Apply optional capped Langevin-style noise only when diffusion is enabled.",
        "- With all controller strengths zero and all toggles off, the optimizer behaves as AdamW.",
        "",
        "## 7. Whether Topological Adam was found and compared",
        "- Yes. The existing `repos/topological-adam/` implementation was reused and not overwritten.",
        "",
        "## 8. Baselines tested",
        "- " + ", ".join(UNIFIED_BASELINE_OPTIMIZERS + [name for name in UNIFIED_PHYSICAL_OPTIMIZERS if name != "unified_physics_adam"]),
        "",
        "## 9. Tasks tested",
        "- Benchmark tasks: " + (", ".join(sorted(benchmark_frame["task"].unique())) if not benchmark_frame.empty else "none"),
        "- Stress tasks: " + (", ".join(sorted(stress_frame["task"].unique())) if not stress_frame.empty else "none"),
        "- Combo-search tasks: " + (", ".join(sorted(combo_frame["task"].unique())) if not combo_frame.empty else "none"),
        "",
        "## 10. Best optimizer per task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]) if not best_frame.empty else "_No rows available._",
        "",
        "## 11. Whether UnifiedPhysicsAdam beat AdamW",
        f"- Meaningful wins: {int(unified_vs_adamw['win'].sum()) if not unified_vs_adamw.empty else 0}",
        "",
        "## 12. Whether it beat Topological Adam",
        f"- Meaningful wins: {int(unified_vs_topo['win'].sum()) if not unified_vs_topo.empty else 0}",
        "",
        "## 13. Whether any 2x result exists",
        f"- Any 2x event across unified and controller-combo comparisons: {'yes' if any_two_x else 'no'}",
        "",
        "## 14. Best controller alone",
        f"- {best_single_row.iloc[0]['combo_name']} with mean selection score {float(best_single_row.iloc[0]['mean_selection_score']):.4f}" if not best_single_row.empty else "- none",
        "",
        "## 15. Best controller combination",
        f"- {best_combo_row.iloc[0]['combo_name']} with mean selection score {float(best_combo_row.iloc[0]['mean_selection_score']):.4f}" if not best_combo_row.empty else "- none",
        "",
        "## 16. Worst/harmful controller",
        f"- {harmful_controller}",
        "",
        "## 17. Whether Codex found a better combination than the original proposal",
        f"- {'yes' if codex_found_better else 'no'}",
        "",
        "## 18. Failure modes",
        "- Full controller stacking can over-regularize or over-damp when simpler adaptive baselines already solve the task.",
        "- Diffusion and uncertainty controls remain the most likely to degrade convergence if enabled too aggressively.",
        "- If RMSProp or SGD momentum still dominate, the unified controller stack is probably too complex for the gain it provides.",
        "",
        "## 19. Recommendation",
        "- Pursue the full unified optimizer only if it clearly beats AdamW and Topological Adam on held-out seeds.",
        "- Otherwise prefer the best simple controller or the best small controller subset discovered by the combo search.",
        "- Treat novelty claims as unsupported unless the combined optimizer wins consistently and by a meaningful margin.",
    ]
    (output_path / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "combined_aggregated": combined_aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
        "combo_summary": combo_summary,
        "strongest_adamw": strongest_adamw,
        "strongest_topological": strongest_topological,
        "strongest_unified": strongest_unified,
    }
