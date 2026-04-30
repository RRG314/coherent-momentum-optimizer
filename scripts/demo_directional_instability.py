from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.benchmarking import _train_single_run  # noqa: E402
from optimizer_research.config import ensure_output_dir  # noqa: E402
from optimizer_research.reporting import _load_trace_frames, aggregate_results  # noqa: E402
from optimizers.optimizer_utils import resolve_device  # noqa: E402


def _load_best_tuning(output_dir: Path, task: str, optimizer: str) -> dict:
    tuning_path = output_dir / "tuning_results.csv"
    if not tuning_path.exists():
        return {}
    frame = pd.read_csv(tuning_path)
    subset = frame[(frame["task"] == task) & (frame["optimizer"] == optimizer)]
    if subset.empty:
        return {}
    best = subset.sort_values("selection_score", ascending=False).iloc[0]
    return json.loads(best["hyperparameters"])


if __name__ == "__main__":
    output_dir = ensure_output_dir({"output_dir": str(ROOT / "reports" / "demo_directional_instability")})
    device = resolve_device("auto")
    task_name = "direction_reversal_objective"
    optimizers = [
        "coherent_momentum_optimizer_improved",
        "adamw",
        "rmsprop",
        "sgd_momentum",
    ]
    directional_root = ROOT / "reports" / "directional_instability"
    rows = []
    for optimizer_name in optimizers:
        rows.append(
            _train_single_run(
                suite_name="directional_demo",
                task_name=task_name,
                optimizer_name=optimizer_name,
                hyperparameters=_load_best_tuning(directional_root, task_name, optimizer_name),
                seed=11,
                device=device,
                output_dir=output_dir,
                save_trace=True,
                epoch_scale=0.8,
            )
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "benchmark_results.csv", index=False)
    aggregated = aggregate_results(frame)
    aggregated.to_csv(output_dir / "summary.csv", index=False)
    trace_frame = _load_trace_frames(frame)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for optimizer_name in optimizers:
        subset = trace_frame[(trace_frame["optimizer"] == optimizer_name) & (trace_frame["event"] == "train")]
        if subset.empty:
            continue
        mean_curve = subset.groupby("step")["train_loss"].mean()
        plt.plot(mean_curve.index, mean_curve.values, label=optimizer_name)
    plt.title("Directional Instability Demo: Loss Curves")
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for optimizer_name in [name for name in optimizers if "coherence" in name or name in {"adamw", "rmsprop"}]:
        subset = trace_frame[(trace_frame["optimizer"] == optimizer_name) & (trace_frame["event"] == "train")]
        if subset.empty or "grad_momentum_cosine" not in subset.columns:
            continue
        mean_curve = subset.groupby("step")["grad_momentum_cosine"].mean()
        plt.plot(mean_curve.index, mean_curve.values, label=optimizer_name)
    plt.title("Directional Instability Demo: Gradient-Momentum Cosine")
    plt.xlabel("step")
    plt.ylabel("cosine")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "direction_cosine.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for optimizer_name in [name for name in optimizers if "coherence" in name or name in {"adamw", "rmsprop"}]:
        subset = trace_frame[(trace_frame["optimizer"] == optimizer_name) & (trace_frame["event"] == "train")]
        if subset.empty or "rotation_score" not in subset.columns:
            continue
        mean_curve = subset.groupby("step")["rotation_score"].mean()
        plt.plot(mean_curve.index, mean_curve.values, label=optimizer_name)
    plt.title("Directional Instability Demo: Rotation Score")
    plt.xlabel("step")
    plt.ylabel("rotation score")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "rotation_score.png", dpi=180)
    plt.close()

    best_row = aggregated.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True], na_position="last").iloc[0]
    report_lines = [
        "# Directional Instability Demo",
        "",
        f"- Task: `{task_name}`",
        f"- Device: `{device}`",
        f"- Best row in this demo: `{best_row['optimizer']}` on `{best_row['task']}`",
        "",
        "This demo is intended to show the optimizer niche on one unstable task. It is not a claim of general superiority.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
