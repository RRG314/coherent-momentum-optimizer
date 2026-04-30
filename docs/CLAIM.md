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

## What the current evidence supports

The checked-in reports support a specialist claim. The accepted historical line shows that the stable Coherent Momentum branch can beat its real Hamiltonian baseline, can beat the lighter directional reference branch, and can beat AdamW on selected stress-oriented slices. The focused newcomer-facing directional benchmark supports a narrower follow-up claim: the improved branch can still beat AdamW on parts of the oscillation and reversal slice.

The GPU and improved-branch audit supports a separate engineering claim. It shows that the optimizer can be made device-safe, that diagnostics can be throttled without breaking the core method, and that the improved branch can win more directional synthetic stress comparisons than the stable branch. It does not turn the improved branch into the new public default automatically.

## What the current evidence does not support

The checked-in reports do **not** support claiming that Coherent Momentum is:

- a universal replacement for AdamW
- a universal replacement for RMSProp
- a universal replacement for SGD with momentum
- a strong general CNN optimizer yet
- a broad state-of-the-art optimizer claim
- a proof that every controller term in the improved branch deserves default status

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

## What would be unfair to claim

It would be unfair to claim that the improved branch is already the public default, that the CNN problem is solved, or that the repository shows broad superiority over RMSProp or SGD with momentum. Those claims would go past the checked-in evidence.

## Safe Public Wording

The safest short public sentence the current repository supports is:

> Coherent Momentum is a specialist optimizer for unstable gradient-direction regimes, with evidence of improvement over AdamW on selected instability slices, but not as a general replacement for RMSProp, SGD with momentum, or AdamW.

Anything broader than that requires stronger evidence than the current checked-in reports provide.
