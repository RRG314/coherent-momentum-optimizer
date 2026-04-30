from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import load_yaml_config, run_directional_instability_benchmark  # noqa: E402
from optimizers.optimizer_utils import resolve_device  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    device = resolve_device(str(config.get("device", "auto")))
    print(json.dumps({"device": str(device)}, indent=2))
    frame = run_directional_instability_benchmark(config)
    print(f"rows={len(frame)}")
