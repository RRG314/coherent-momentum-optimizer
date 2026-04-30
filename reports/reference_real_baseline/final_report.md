# Coherent Momentum Real Baseline Report

## Related Work
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW is the practical Adam-family baseline because it decouples weight decay from the adaptive update rather than mixing it into the gradient moments.
- [Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1402.4102): SGHMC is the canonical stochastic optimizer/sampler that introduces Hamiltonian momentum with friction to handle minibatch-noisy gradients.
- [On Symplectic Optimization](https://arxiv.org/abs/1802.03653): This work is the clearest nearby reference for translating continuous-time Hamiltonian dynamics into discrete optimization algorithms with symplectic integrators.
- [Hamiltonian Descent Methods](https://arxiv.org/abs/1809.05042): A family of dissipative Hamiltonian optimization methods that formalizes momentum dynamics as conformal Hamiltonian systems rather than heuristic damping.
- [A Physics-Inspired Optimizer: Velocity Regularized Adam](https://arxiv.org/abs/2505.13196): Recent nearby work that uses physically motivated velocity regularization to damp large Adam updates; relevant because it already occupies some of the stability-control design space.

## 1. What changed from the reactive baseline
- The reactive baseline used simple AdamW-style damping. `CoherentMomentumRealBaseline` adds explicit position-momentum dynamics, kinetic and potential tracking, adaptive diagonal mass from Adam's second moment, and a leapfrog/symplectic-Euler integrator.

## 2. Whether the new optimizer uses real Hamiltonian dynamics
- Yes, in the limited optimizer sense used here: parameters are positions, optimizer state is physical momentum, kinetic energy is computed from momentum and inverse mass, and updates use symplectic-Euler or leapfrog-style kick-drift-kick integration.
- The report does not call the closure-free path full leapfrog. It is labeled a symplectic-Euler approximation.

## 3. Which update mode was used
- Default benchmark mode: `dissipative_hamiltonian` with adaptive mass and closure-driven leapfrog whenever the harness provided a closure.

## 4. Whether leapfrog closure worked
- Mean `closure_recomputed_gradient` across real-Hamiltonian traces: 0.790
- Mean `leapfrog_enabled` across real-Hamiltonian traces: 0.790

## 5. Energy drift compared to the reactive baseline
- Mean relative energy drift on direct energy tests: reactive baseline `nan` vs real baseline `-0.001617`

## 6. Whether it beats AdamW
- Meaningful wins vs AdamW: 8

## 7. Whether it beats RMSProp
- Meaningful wins vs RMSProp: 2
- Tasks where it beat the reactive baseline and AdamW while staying competitive with RMSProp: oscillatory_valley, saddle_objective

## 8. Whether it beats Topological Adam
- Meaningful wins vs Topological Adam: 8

## 9. Whether any 2x event appeared
- Surviving 2x events under the stricter RMSProp-competitive filter: saddle_objective

## 10. Whether the more faithful Hamiltonian math helped or hurt
- Best ablation variant by mean selection score: `rmsprop_baseline`
- Interpretation: `leapfrog closure hurt`

## 11. Whether to keep the reactive baseline, keep the physical baseline, or combine them
- Keep both for now.
- The reactive baseline remains the simpler stability baseline.
- `CoherentMomentumPhysicalBaseline` is the more mathematically faithful branch, but only keep pushing it if the energy-drift gains survive without giving back too much to RMSProp on practical ML tasks.

## Best Rows
| optimizer | task | mean_best_val_loss | mean_best_val_accuracy | mean_relative_energy_drift |
| --- | --- | --- | --- | --- |
| coherent_momentum_physical_baseline | circles_mlp | 0.1029 | 0.9862 | nan |
| coherent_momentum_real_baseline | breast_cancer_mlp | 0.1141 | 0.9635 | -0.0065 |
| adamw | circles_mlp | 0.0926 | 0.9896 | nan |
| rmsprop | moons_mlp | 0.0488 | 0.9948 | nan |
| topological_adam | circles_mlp | 0.0926 | 0.9896 | nan |

## Best Optimizer Per Task
| task | best_optimizer | mean_best_val_loss | mean_best_val_accuracy |
| --- | --- | --- | --- |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 |
| circles_mlp | rmsprop | 0.0400 | 0.9944 |
| harmonic_oscillator_objective | rmsprop | 0.0000 | nan |
| high_curvature_regression | sgd_momentum | 0.0002 | nan |
| label_noise_breast_cancer | lion | 0.1786 | 0.9648 |
| moons_mlp | rmsprop | 0.0488 | 0.9948 |
| narrow_valley_objective | sgd_momentum | 0.0000 | nan |
| noisy_quadratic_objective | sgd_momentum | -0.0487 | nan |
| noisy_regression | sgd_momentum | 0.0632 | nan |
| oscillatory_valley | coherent_momentum_real_baseline | -0.2340 | nan |
| overfit_small_wine | sgd_momentum | 0.0783 | 0.9852 |
| quadratic_bowl_objective | sgd_momentum | 0.0299 | nan |
| rosenbrock_valley | rmsprop | 0.0004 | nan |
| saddle_objective | sgd_momentum | -4.1322 | nan |
| small_batch_instability | coherent_momentum_physical_baseline | 0.3799 | 0.8760 |
| wine_mlp | sgd_momentum | 0.0556 | 0.9852 |