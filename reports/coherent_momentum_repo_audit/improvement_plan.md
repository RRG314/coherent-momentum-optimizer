# Coherent Momentum Improvement Plan

Optimizer families and baseline names mentioned below are referenced in [../../REFERENCES.md](../../REFERENCES.md).

## Main objective
Keep Coherent Momentum Adam as a specialist optimizer for unstable gradient-direction regimes while making it:
- GPU-safe
- easier to benchmark honestly
- less likely to over-intervene on standard tasks
- more usable on CNN-like workloads

## Planned changes

### 1. Keep the current branch intact
- Preserve `CoherentMomentumOptimizer` as the accepted reference implementation.
- Do not silently overwrite the existing method.

### 2. Add a separate improved branch
- Introduce `CoherentMomentumOptimizerImproved`.
- Compare old and new branches directly in the same harness.

### 3. Reduce host synchronization in the hot path
- Replace many control computations with tensor-valued scalar tensors on device.
- Delay conversion to Python floats until reporting/diagnostic aggregation.

### 4. Add diagnostics throttling
- Support:
  - `enable_step_diagnostics`
  - `diagnostics_every_n_steps`
- Keep full tracing available, but not mandatory on every step.

### 5. Replace harmful control defaults
- Avoid hard conflict damping as the default behavior.
- Use softer conflict correction through bounded momentum blending.
- Keep projection optional and conflict-gated.

### 6. Add standard-safe and CNN-safe behavior
- Standard-safe preset:
  - lower intervention on clean, coherent gradients
- CNN-safe preset:
  - lighter projection
  - conv-specific update cap
  - cheap per-filter coherence support

### 7. Expand benchmark instrumentation
- Add:
  - optimizer step time
  - throughput
  - peak device memory
  - diagnostics row count
  - device name

### 8. Add GPU compatibility tests
- GPU/MPS step changes parameters
- all state tensors remain on correct device
- CPU->GPU load works
- GPU->CPU load works
- no NaNs on GPU/MPS smoke path
- diagnostics disable/throttle works

### 9. Keep improvements modular
- If an improvement hurts:
  - remove it from default
  - keep it only as an experimental preset if still informative

## Success standard
- The repo should clearly show whether Magneto remains a specialist or broadens meaningfully.
- If it only wins in stress/conflict/oscillation regimes, the final report must say so explicitly.
