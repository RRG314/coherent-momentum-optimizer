# Comparison Notes

This document explains what Coherent Momentum Optimizer is relative to the baseline families used throughout the repo. It is meant to answer the question “is this actually a different optimizer idea, or just another rewrite of AdamW?”

References for every optimizer family mentioned below are collected in [../REFERENCES.md](../REFERENCES.md).

## The Short Version

Coherent Momentum Optimizer is not defined by a new second-moment estimator and it is not defined by a matrix preconditioner. Its central move is to monitor whether the current update direction is trustworthy, then intervene only when the direction appears unstable.

That is different from the main baseline families in this repository:

- `SGD` uses the current gradient directly.
- `SGD+momentum` smooths direction across time.
- `RMSProp` rescales magnitude using squared-gradient history.
- `AdamW` combines momentum with adaptive magnitude scaling and decoupled weight decay.
- `CoherentMomentumRealBaseline` turns the adaptive force into a position-momentum system with mass, friction, and energy bookkeeping.
- `CoherentMomentumOptimizer` inherits the real Hamiltonian base and then adds bounded directional-coherence control on top of it.

## Versus AdamW

AdamW starts from one adaptive direction and changes its magnitude through the usual first- and second-moment machinery. Its geometry is still fundamentally “take this adaptive direction, then scale it.”

Coherent Momentum Optimizer starts from the same practical reality that AdamW-like forces are useful, but it asks a different question: is the direction itself still reliable? The code monitors gradient-momentum cosine, force-momentum cosine, gradient-history cosine, update-history cosine, rotation, and conflict. Those signals are then used to change damping, alignment scale, and projection strength only when the direction looks unstable.

So the difference is not a new adaptive denominator. The difference is that CMO treats directional reliability as a first-class control signal.

## Versus RMSProp

RMSProp is the strongest “don’t overcomplicate this” baseline in many of the included reports. It rescales updates well, is fast, and often wins broad practical tasks in this repository.

CMO is different in two ways. First, it keeps a real momentum-plus-mass state rather than only a gradient-magnitude history. Second, it adds explicit directional diagnostics. That makes it more expensive. It also means it only deserves to exist when those directional diagnostics help enough to justify the overhead.

This is why the README and reports are careful not to present CMO as a general RMSProp replacement.

## Versus Lion

Lion is a useful modern comparison because it changes update geometry with low state and low overhead. It is still fundamentally a compact first-order directional rule, built around signed momentum behavior rather than explicit reliability diagnostics.

CMO is more expensive and only deserves to exist where that extra cost buys something. The contrast with Lion is therefore practical as well as algorithmic: Lion asks how much useful behavior can be packed into a simple update rule, while CMO asks whether extra directional observables can help when the direction itself becomes unreliable.

## Versus SGD With Momentum

Momentum already does one important thing that CMO also cares about: it stabilizes direction through time. That makes `SGD+momentum` a serious comparison baseline, not a toy baseline.

The difference is that momentum uses the same smoothing mechanism all the time. CMO instead tries to detect when the smoothed direction itself has become unreliable because of reversal, oscillation, or batch conflict. When that happens, it changes the effective dynamics through bounded friction scaling, directional alignment scaling, and optional force projection.

In other words, `SGD+momentum` assumes temporal smoothing is enough. CMO tests whether extra directional control helps when smoothing alone is not enough.

## Versus CoherentMomentumRealBaseline

This comparison matters the most conceptually.

`CoherentMomentumRealBaseline` is the baseline that makes the coherent-momentum story meaningful. It already exposes position-momentum style dynamics, inverse mass, friction, energy correction, and optional leapfrog-style updates. Without that base, the public optimizer would be harder to distinguish from “Adam plus a few gates.”

CMO keeps that Hamiltonian base intact and adds a bounded control layer. That layer is the actual experiment. If CMO cannot beat `CoherentMomentumRealBaseline` in unstable-direction regimes, then the coherence controller is not earning its keep.

## Versus CoherentDirectionReferenceOptimizer

`CoherentDirectionReferenceOptimizer` is an internal repository baseline that isolates directional-coherence control without the full Hamiltonian machinery. It is useful because it tells you whether the repo’s directional signals help on their own or only help once they are attached to the stronger physical base.

If `CoherentMomentumOptimizer` beats `CoherentDirectionReferenceOptimizer`, that suggests the coherent controller is benefiting from the real Hamiltonian substrate rather than merely duplicating a softer Adam-like rule.

## Versus Muon, Shampoo, and K-FAC

`Muon`, `Shampoo`, and `K-FAC` matter because they are serious structure-aware baselines. They use matrix structure or approximate curvature to change update geometry.

CMO is not doing that. It does not estimate a structured inverse, it does not orthogonalize updates the way Muon-style methods do, and it does not try to approximate curvature the way K-FAC does. Its central mechanism is still directional reliability control on top of a Hamiltonian first-order base.

## Versus SAM and ASAM

Sharpness-aware methods modify the training objective around a perturbation neighborhood. That is a different route to stability from what CMO does.

CMO does not perturb the objective and recompute the gradient around that perturbation. It stays in the original objective and tries to judge whether the current direction is coherent enough to trust. That is why the repo treats SAM and ASAM as serious stability baselines, not as close algorithmic relatives.

## Versus PCGrad and CAGrad

Conflict-aware methods like `PCGrad` and `CAGrad` modify gradients in explicitly multitask settings. They matter here because CMO also talks about conflict, but the setting is different.

CMO does not need separate task gradients. It measures instability inside ordinary single-loss training as well as conflict-style settings. That gives it a broader ordinary-training surface than PCGrad or CAGrad, but it also means the comparison has to stay honest: CMO is not a drop-in multitask-gradient surgery method.

## What Makes It More Than A Rewrite

The repo should not claim novelty just because it has new parameter names. The serious distinction is narrower and more concrete:

1. The public optimizer retains a real Hamiltonian state with mass, momentum, energy drift, and optional leapfrog-style closure behavior.
2. It computes explicit directional diagnostics rather than assuming one adaptive direction is always good enough.
3. It uses bounded control values, not unbounded heuristics, to change friction, alignment, and projection.
4. It keeps a clearer failure boundary than a generic “better Adam” claim: if direction is already stable, the controller often does not help.

That is a legitimate algorithmic difference. It is also why the repository keeps both the coherent branch and the real Hamiltonian baseline visible side by side.
