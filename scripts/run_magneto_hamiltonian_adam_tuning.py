from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import load_yaml_config
from optimizer_research.magneto_hamiltonian_suite import run_magneto_hamiltonian_tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_magneto_hamiltonian_tuning(config)


if __name__ == "__main__":
    main()
