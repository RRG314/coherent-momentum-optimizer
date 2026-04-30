from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer  # noqa: E402


REPORT_DIR = ROOT / "reports" / "demo_directional_instability"


def _loss(theta: torch.Tensor) -> torch.Tensor:
    x, y = theta[0], theta[1]
    valley = 4.0 * (y - 0.45 * x.square()).square() + 0.08 * x.square()
    oscillation = 0.18 * torch.sin(6.0 * x) * y + 0.06 * torch.cos(3.0 * y) * x
    return valley + oscillation


def _make_optimizer(name: str, parameter: torch.nn.Parameter):
    if name == "cmo":
        return CoherentMomentumOptimizer(
            [parameter],
            lr=0.03,
            mode="adam_preconditioned_hamiltonian",
        )
    if name == "adamw":
        return torch.optim.AdamW([parameter], lr=0.03)
    if name == "rmsprop":
        return torch.optim.RMSprop([parameter], lr=0.02, alpha=0.95)
    if name == "sgd_momentum":
        return torch.optim.SGD([parameter], lr=0.02, momentum=0.9)
    raise ValueError(name)


def _run(name: str) -> dict[str, list[float] | float]:
    theta = torch.nn.Parameter(torch.tensor([2.2, -1.4], dtype=torch.float32))
    optimizer = _make_optimizer(name, theta)
    losses: list[float] = []
    rotation: list[float] = []
    coherence: list[float] = []
    best_loss = float("inf")

    for _step in range(140):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(theta)
        if isinstance(optimizer, CoherentMomentumOptimizer):
            optimizer.set_current_loss(loss.item())
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        best_loss = min(best_loss, loss_value)
        diagnostics = optimizer.latest_diagnostics() if isinstance(optimizer, CoherentMomentumOptimizer) else {}
        rotation.append(float(diagnostics.get("rotation_score", float("nan"))))
        coherence.append(float(diagnostics.get("coherence_score", float("nan"))))

    return {
        "losses": losses,
        "rotation": rotation,
        "coherence": coherence,
        "best_loss": best_loss,
        "final_loss": losses[-1],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {
        "CoherentMomentumOptimizer": _run("cmo"),
        "AdamW": _run("adamw"),
        "RMSProp": _run("rmsprop"),
        "SGD+momentum": _run("sgd_momentum"),
    }

    plt.figure(figsize=(10, 4.5))
    for label, run in runs.items():
        plt.plot(run["losses"], label=label)
    plt.title("Directional instability demo: loss curves")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    loss_path = REPORT_DIR / "example_loss_curves.png"
    plt.savefig(loss_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(runs["CoherentMomentumOptimizer"]["rotation"], label="rotation_score")
    plt.plot(runs["CoherentMomentumOptimizer"]["coherence"], label="coherence_score")
    plt.title("Coherent Momentum diagnostics on the toy instability objective")
    plt.xlabel("step")
    plt.ylabel("diagnostic value")
    plt.legend()
    plt.tight_layout()
    diag_path = REPORT_DIR / "example_direction_diagnostics.png"
    plt.savefig(diag_path, dpi=180)
    plt.close()

    print("Directional instability demo")
    for label, run in runs.items():
        print(f"{label}: best_loss={run['best_loss']:.6f}, final_loss={run['final_loss']:.6f}")
    print(f"saved_loss_plot={loss_path}")
    print(f"saved_diagnostic_plot={diag_path}")


if __name__ == "__main__":
    main()
