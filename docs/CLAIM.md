# Focused Claim

References for the baseline families named here are collected in [../REFERENCES.md](../REFERENCES.md).

The narrow claim for this repository is:

**Coherent Momentum Optimizer improves training mainly when gradient direction is unreliable.**

This repository uses "unreliable direction" in a specific sense:

- oscillation around narrow or curved valleys
- repeated direction reversal over time
- conflicting mini-batch gradients
- noisy or small-batch gradient variance
- saddle dynamics and related unstable curvature

The optimizer is designed to monitor directional signals such as gradient-momentum cosine, force-momentum cosine, coherence, conflict, and rotation. It intervenes when those signals indicate that the raw update direction is becoming unreliable.

The repository currently exposes that claim in two layers:

- a newcomer-facing default directional benchmark config with a representative subset of instability tasks
- broader instability task support in the task registry and benchmark harness for deeper follow-up runs

That distinction matters. The default benchmark path is meant to be reproducible by a newcomer cloning the repo, while the wider task registry remains available for heavier follow-up work.

## What This Optimizer Is Not

This repository does **not** claim that Coherent Momentum Optimizer is:

- a universal replacement for AdamW
- a universal replacement for RMSProp
- a general-purpose replacement for SGD with momentum
- a strong CNN optimizer across ordinary vision workloads yet
- a state-of-the-art claim

The intended interpretation is narrower:

- use it when the update direction itself appears unstable
- compare it against strong baselines on those instability regimes
- treat ordinary clean tasks and CNN-heavy tasks as failure checks, not as assumed win conditions

If the optimizer fails those failure checks, the repo should say so directly rather than treating the checks as optional.
