from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _import_topological_adam():
    try:
        from topological_adam import TopologicalAdam, TopologicalAdamV2  # type: ignore

        return TopologicalAdam, TopologicalAdamV2
    except ImportError:
        search_roots = [
            Path(__file__).resolve().parents[2] / "repos" / "topological-adam",
            Path(__file__).resolve().parents[4] / "repos" / "topological-adam",
        ]
        for repo_root in search_roots:
            if repo_root.exists():
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from topological_adam import TopologicalAdam, TopologicalAdamV2  # type: ignore

                return TopologicalAdam, TopologicalAdamV2
        raise


TopologicalAdam, TopologicalAdamV2 = _import_topological_adam()
BaselineTopologicalAdam = TopologicalAdamV2


def topological_metrics(optimizer: Any) -> dict[str, float]:
    if hasattr(optimizer, "stats") and isinstance(optimizer.stats, dict):
        return {
            "topological_energy": float(optimizer.stats.get("energy", 0.0)),
            "topological_coupling": float(optimizer.stats.get("coupling", 0.0)),
            "topological_ratio": float(optimizer.stats.get("topo_ratio", 0.0)),
        }
    if hasattr(optimizer, "field_metrics"):
        metrics = optimizer.field_metrics()
        return {
            "topological_energy": float(metrics.get("energy", 0.0)),
            "topological_coupling": float(metrics.get("j_t", 0.0)),
            "topological_ratio": float(metrics.get("alpha_beta_corr", 0.0)),
        }
    return {
        "topological_energy": 0.0,
        "topological_coupling": 0.0,
        "topological_ratio": 0.0,
    }
