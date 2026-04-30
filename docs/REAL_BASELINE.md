# Real Hamiltonian Baseline

References for the Hamiltonian and adaptive baselines mentioned here are collected in [../REFERENCES.md](../REFERENCES.md).

This repository keeps `CoherentMomentumRealBaseline` in full because it is the baseline that makes the coherence branch meaningful.

## Why it matters

Without a real Hamiltonian base, the coherence branch would collapse into "Adam plus controls." The whole point of keeping `CoherentMomentumRealBaseline` visible is to show that the public coherent-momentum branch is built on top of a real position-momentum optimizer rather than on top of a renamed AdamW wrapper.

The base gives:

- explicit position-momentum dynamics
- mass handling
- friction handling
- energy correction
- leapfrog or symplectic-Euler distinction

That is the difference between having a physical baseline and merely borrowing physical language.

## How to interpret comparisons

- If `CoherentMomentumOptimizer` beats `CoherentMomentumRealBaseline`, that is evidence the directional controller is doing useful work.
- If it loses on broad tasks, that is evidence the extra controller complexity is not justified there.

This is why the repo keeps both the baseline and the public branch documented side by side.
