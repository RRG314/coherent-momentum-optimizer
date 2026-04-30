from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import export_coherent_momentum_report  # noqa: E402


def _resolve_export_dir(root: Path) -> Path:
    primary = root / "reports" / "coherent_momentum_optimizer"
    accepted = root / "reports" / "accepted_coherent_momentum"
    required = ("benchmark_results.csv", "energy_tests.csv", "ablation_results.csv")

    if all((primary / name).exists() for name in required):
        return primary
    if all((accepted / name).exists() for name in required):
        return accepted
    missing_primary = [name for name in required if not (primary / name).exists()]
    missing_accepted = [name for name in required if not (accepted / name).exists()]
    raise FileNotFoundError(
        "Could not find a complete Coherent Momentum report directory. "
        f"Missing in reports/coherent_momentum_optimizer: {missing_primary}. "
        f"Missing in reports/accepted_coherent_momentum: {missing_accepted}."
    )


if __name__ == "__main__":
    export_coherent_momentum_report(_resolve_export_dir(ROOT))
