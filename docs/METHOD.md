# Method Notes

References for the optimizer families and theory mentioned here are collected in [../REFERENCES.md](../REFERENCES.md).

## Core Principle

`MagnetoHamiltonianAdam`, and therefore the public alias `CoherentMomentumOptimizer`, is built around a narrow question: when the current update direction is unstable, can a bounded directional controller help more than simply rescaling magnitude?

That is the point of departure from ordinary Adam-family framing. The implementation does not start by claiming a new universal adaptive preconditioner. Instead, it starts from a real Hamiltonian base optimizer and then asks whether explicit directional reliability signals can improve that base in oscillatory or reversal-heavy regimes.

## Real Hamiltonian Base

The underlying base is `HamiltonianAdamReal`. It keeps state that is meant to be interpreted as actual optimizer dynamics rather than only as bookkeeping:

- momentum
- inverse mass
- kinetic energy
- potential energy
- total Hamiltonian
- energy drift

The practical consequence is that the optimizer already has a physical notion of update state before the coherent-momentum controller is added. This is important because it makes the public optimizer more than “Adam with a few extra gates.” The base optimizer already has its own step logic, mass handling, friction, energy correction, and optional closure-aware integration behavior.

## Directional Signals

On top of that base, the magneto layer computes directional observables from the current gradient, the current Hamiltonian momentum, the previous gradient, the previous update, and the current force direction implied by the inverse mass.

The implementation specifically tracks quantities such as:

- gradient-momentum cosine
- force-momentum cosine
- gradient-history cosine
- update-history cosine
- rotation score
- coherence score
- conflict score

These quantities are not treated as decoration. They are the input variables for the controller.

## Bounded Control Layer

The magneto layer is a bounded controller, not an unrestricted rewrite of the step.

In the current stable implementation, those directional signals are used to produce:

- a friction multiplier
- an alignment scale
- a projection strength toward the force direction

Those controls are clamped to limited ranges. That design choice matters. It means the branch is trying to correct an unreliable direction without letting the controller fully replace the base optimizer geometry.

## Improved Branch

`MagnetoHamiltonianAdamImproved` keeps the same conceptual structure but changes how the controller is executed.

The main engineering differences are:

- more control computation stays on tensors instead of moving through Python scalar logic
- diagnostics can be throttled
- presets such as `standard_safe`, `stress_specialist`, and `cnn_safe` are applied explicitly
- conflict handling is softened rather than relying on heavy suppression

This improved branch is therefore best interpreted as a device-safe and preset-aware refinement of the same idea, not as a different optimizer family.

## Why The Branch Exists

`HamiltonianAdamReal` is the cleaner baseline. The coherent-momentum branch exists only because the repo is testing whether a real gain appears when the direction itself becomes unreliable. If those gains do not appear, then the extra controller logic is not justified.

That is also why the public docs in this repo keep emphasizing a narrow claim. The coherent-momentum layer earns its place only when directional instability is the real problem.
