from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["config_path"] = str(path.resolve())
    data["project_root"] = str(path.resolve().parents[1])
    return data


def ensure_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", "reports/physical_adam")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    return output_dir
