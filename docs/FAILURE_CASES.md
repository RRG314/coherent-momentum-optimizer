# Failure Cases

References for the baseline optimizers mentioned here are collected in [../REFERENCES.md](../REFERENCES.md).

This repository is more credible if it documents where the optimizer loses and what a user should do instead.

## Clean Classification

On ordinary clean tabular or small neural classification tasks, simpler baselines often win. The directional controller is not free, and on tasks where the raw direction is already stable it may simply be unnecessary.

- Likely stronger baselines:
  - `RMSProp`
  - `SGD+momentum`
  - sometimes `AdamW`
- Why:
  - the directional coherence machinery adds overhead
  - if the gradient direction is already stable, the extra control logic may not help

## CNNs

The CNN gap is still open. The current repo results are useful precisely because they do not hide this. The optimizer family can be interesting on instability-driven tasks and still be weak on mainstream convolutional training.

- Likely stronger baselines:
  - `RMSProp`
  - `SGD+momentum`
  - `AdamW`
- Why:
  - the current optimizer family is stronger on directional instability than on broad vision optimization
  - convolution-heavy workloads still favor simpler, well-tuned baselines in the current repo results

## PINNs and Scientific PDE Training

This optimizer should not currently be presented as a winning PINN optimizer. The repo includes those comparisons as a boundary condition, not as a marketing surface.

- Likely stronger baselines:
  - `LBFGS`
  - `AdamW -> LBFGS` hybrid schedules
- Why:
  - the Hamiltonian/coherence controls are not currently enough to beat the strong closure-based second-order baselines on these tasks

## Stable Smooth Problems

On stable smooth problems where the gradient direction is already reliable, the repo’s main optimizer family is usually not the right default. The public claim is about unreliable direction, so stable smooth settings are where the extra machinery has the least room to justify itself.

- Likely stronger choice:
  - `SGD+momentum` if you want simplicity and speed
  - `RMSProp` or `AdamW` if you want a dependable adaptive baseline
- Why:
  - the narrow claim here is about unreliable direction, not about every smooth optimization regime

## What To Use Instead

- Use `RMSProp` or `SGD+momentum` on ordinary standard tasks unless the instability benchmark suggests otherwise for your workload.
- Use `AdamW` when you want a strong familiar baseline with broad community support.
- Use `LBFGS` or `AdamW -> LBFGS` for PINN-style closure-friendly problems when those baselines dominate.

The purpose of this file is not to weaken the repository. It is to make the narrow claim precise and to prevent misuse.
