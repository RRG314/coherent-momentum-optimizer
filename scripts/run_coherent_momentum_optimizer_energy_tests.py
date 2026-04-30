from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("run_magneto_hamiltonian_adam_energy_tests.py")), run_name="__main__")
