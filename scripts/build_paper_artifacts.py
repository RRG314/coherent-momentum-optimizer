from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research.paper_artifacts import build_paper_artifacts  # noqa: E402


if __name__ == "__main__":
    outputs = build_paper_artifacts()
    for label, path in outputs.items():
        print(f"{label}: {path}")
