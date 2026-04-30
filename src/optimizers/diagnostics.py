from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_value(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


@dataclass(slots=True)
class DiagnosticsHistory:
    optimizer_name: str
    rows: list[dict[str, float | int | str | bool | None]] = field(default_factory=list)

    def append(self, metrics: dict[str, Any]) -> dict[str, float | int | str | bool | None]:
        row = {"optimizer": self.optimizer_name}
        for key, value in metrics.items():
            row[key] = _safe_value(value)
        self.rows.append(row)
        return row

    def latest(self) -> dict[str, float | int | str | bool | None]:
        if not self.rows:
            return {"optimizer": self.optimizer_name}
        return dict(self.rows[-1])

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["optimizer"])
        return pd.DataFrame(self.rows)

    def save_csv(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output, index=False)
        return output

    def clear(self) -> None:
        self.rows.clear()
