# Coherent Momentum Code Audit

Optimizer families and baseline names mentioned below are referenced in [../../REFERENCES.md](../../REFERENCES.md).

## Scope
- Optimizer core: `src/optimizers/coherent_momentum_optimizer.py`
- Real Hamiltonian dependency: `src/optimizers/coherent_momentum_physical_baseline.py`
- Shared diagnostics/utilities: `src/optimizers/base.py`, `src/optimizers/optimizer_utils.py`
- Benchmark harness: `src/optimizer_research/benchmarking.py`, `src/optimizer_research/coherent_momentum_suite.py`

## Findings

### 1. Mathematical correctness
- The existing Magneto implementation is mathematically consistent with its intended design: a stabilized real-Hamiltonian core plus directional control terms.
- The most credible mechanism remains bounded projection toward force direction under oscillation/conflict.
- Previous reports already showed that hard conflict damping and always-on controller stacking were harmful or neutral.

### 2. GPU compatibility
- State tensors in the Real Hamiltonian base are device-safe because they are created with `torch.zeros_like` and `torch.full_like`.
- Closure-based step modes are already compatible with autograd-enabled GPU execution.
- CPU/MPS/CUDA portability was incomplete at the package level because no dedicated GPU smoke or transfer tests existed.

### 3. Device safety issues
- The original Magneto hot path used many scalar conversions through `.item()` and `float(...)` inside `step`.
- Those conversions can trigger CPU synchronization on CUDA/MPS and distort optimizer step timing.

### 4. Runtime overhead
- The original Magneto implementation computed many diagnostic scalars every step and immediately converted them to Python floats.
- Benchmarking also lacked optimizer-step timing and device-memory accounting, making practical overhead harder to evaluate honestly.

### 5. Memory overhead
- Optimizer state size is moderate and primarily comes from:
  - `exp_avg`
  - `exp_avg_sq`
  - `hamiltonian_momentum`
  - `prev_update`
  - `prev_grad`
  - `inverse_mass_ema`
- This is acceptable for a specialist first-order optimizer but still heavier than SGD/RMSProp.

### 6. Diagnostics overhead
- Diagnostics were previously recorded every step with no throttling.
- This was especially problematic for GPU benchmarking because the logging pathway itself added avoidable host interaction.

### 7. Closure handling
- Closure logic already worked for leapfrog and dissipative modes, but it needed dedicated GPU tests.
- The closure path is still more expensive than single-pass steps, so it should be used only where the Hamiltonian mode needs it.

### 8. Mixed precision compatibility
- No dedicated AMP smoke tests existed.
- The implementation is mostly safe because it avoids custom fused kernels, but mixed precision should be treated as opportunistic rather than guaranteed on this machine because CUDA is unavailable here.

### 9. state_dict correctness
- `CoherentMomentumRealBaseline` already saves and restores `physical_global_state`.
- No dedicated device-transfer tests existed before this pass.

### 10. Baseline fairness
- The harness already supports tuned baselines and separate output directories.
- The main fairness issue was not baseline weakening, but missing GPU-oriented timing/memory metrics and inconsistent diagnostics overhead control.

### 11. Previously harmful controls
- Conflict damping: previously harmful or neutral.
- Activation gating: previously neutral or harmful.
- Over-stacked controller logic: previously harmful.
- Projection: sometimes helpful, especially in oscillatory/reversal regimes.

## Audit conclusion
- The core method was worth keeping.
- The main problems were practical:
  - too many host scalar syncs
  - diagnostics overhead every step
  - no GPU-oriented benchmark path
  - no explicit improved branch for honest A/B testing
