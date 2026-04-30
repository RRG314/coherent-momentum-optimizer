from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import run_real_hamiltonian_smoke  # noqa: E402


if __name__ == "__main__":
    run_real_hamiltonian_smoke(
        {
            "output_dir": str(ROOT / "reports" / "real_hamiltonian_adam"),
            "device": "cpu",
            "smoke_seeds": [11],
            "smoke_epoch_scale": 0.45,
            "smoke_tasks": ["harmonic_oscillator_objective", "wine_mlp", "rosenbrock_valley"],
            "smoke_optimizers": ["hamiltonian_adam", "real_hamiltonian_adam", "adamw", "rmsprop"],
        }
    )
