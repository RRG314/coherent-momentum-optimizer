from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import export_real_hamiltonian_report  # noqa: E402


if __name__ == "__main__":
    export_real_hamiltonian_report(ROOT / "reports" / "real_hamiltonian_adam")
