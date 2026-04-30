from __future__ import annotations

from .magneto_hamiltonian_adam import MagnetoHamiltonianAdam
from .magneto_hamiltonian_adam_improved import MagnetoHamiltonianAdamImproved


class CoherentMomentumOptimizer(MagnetoHamiltonianAdam):
    """Public alias for the current stable Magneto-Hamiltonian implementation."""


class CoherentMomentumOptimizerImproved(MagnetoHamiltonianAdamImproved):
    """Experimental improved branch with GPU-safe control cleanup and presets."""


__all__ = [
    "CoherentMomentumOptimizer",
    "CoherentMomentumOptimizerImproved",
]
