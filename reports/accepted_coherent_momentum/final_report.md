# Coherent Momentum Adam Report

## Related Work
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW is the practical Adam-family baseline because it decouples weight decay from the adaptive update rather than mixing it into the gradient moments.
- [Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1402.4102): SGHMC is the canonical stochastic optimizer/sampler that introduces Hamiltonian momentum with friction to handle minibatch-noisy gradients.
- [On Symplectic Optimization](https://arxiv.org/abs/1802.03653): This work is the clearest nearby reference for translating continuous-time Hamiltonian dynamics into discrete optimization algorithms with symplectic integrators.
- [Hamiltonian Descent Methods](https://arxiv.org/abs/1809.05042): A family of dissipative Hamiltonian optimization methods that formalizes momentum dynamics as conformal Hamiltonian systems rather than heuristic damping.
- [A Physics-Inspired Optimizer: Velocity Regularized Adam](https://arxiv.org/abs/2505.13196): Recent nearby work that uses physically motivated velocity regularization to damp large Adam updates; relevant because it already occupies some of the stability-control design space.
- Gradient alignment and conflict methods in multitask optimization are the nearest conceptual relatives for the coherence controller; this branch is best viewed as a directional-coherence add-on to the real Hamiltonian optimizer rather than a novelty claim.

## 1. What changed from the physical baseline
- `CoherentMomentumOptimizer` keeps the stabilized physical core and adds directional coherence signals: gradient-momentum cosine, force-momentum cosine, gradient history cosine, update history cosine, rotation score, and a bounded projection back toward the force direction during conflict.
- The current branch also adds activation gating so the coherence controller can stay closer to the physical baseline on ordinary tasks and only fully activate when conflict or rotation rises.

## 2. Whether it beat the strengthened physical baseline
- Meaningful wins vs `coherent_momentum_real_baseline`: 13

## 3. Whether it beat CoherentDirectionReferenceOptimizer alone
- Meaningful wins vs `coherent_direction_reference`: 7

## 4. Whether it beat AdamW
- Meaningful wins vs `adamw`: 8

## 5. Whether it beat RMSProp
- Meaningful wins vs `rmsprop`: 4
- Tasks where it beat the physical baseline and AdamW while staying competitive with RMSProp: direction_reversal_objective, noisy_quadratic_objective, oscillatory_valley, quadratic_bowl_objective, saddle_objective

## 6. Whether it beat Topological Adam
- Meaningful wins vs `topological_adam`: 8

## 7. Energy drift compared to the physical baseline branch
- Mean relative energy drift on direct energy tests: Real `-0.001439` vs Coherent Momentum `-0.003154`

## 8. Whether any 2x event survived
- Surviving 2x events under the stricter RMSProp-competitive filter: oscillatory_valley, saddle_objective

## 9. Which coherence component mattered most
- Best ablation variant by mean selection score: `rmsprop_baseline`
- Activation gating interpretation: `activation gating hurt or stayed neutral`
- Projection interpretation: `projection helped`
- Conflict damping interpretation: `conflict damping hurt`

## 10. Recommendation
- Keep this branch separate from the real Hamiltonian baseline.
- Use it where oscillation, direction reversal, or conflicting-batch behavior matters.
- Do not replace RMSProp or the strengthened real-Hamiltonian baseline globally unless the controller wins survive broader held-out tasks.

## Best Rows
| optimizer | task | mean_best_val_loss | mean_best_val_accuracy | mean_relative_energy_drift |
| --- | --- | --- | --- | --- |
| coherent_momentum_optimizer | breast_cancer_mlp | 0.0799 | 0.9805 | -0.0011 |
| coherent_momentum_real_baseline | conflicting_batches_classification | 0.0857 | 0.9779 | -0.0035 |
| coherent_direction_reference | circles_mlp | 0.0961 | 0.9888 | nan |
| adamw | circles_mlp | 0.0987 | 0.9896 | nan |
| rmsprop | moons_mlp | 0.0489 | 0.9948 | nan |
| topological_adam | moons_mlp | 0.0684 | 0.9896 | nan |

## Best Optimizer Per Task
| task | best_optimizer | mean_best_val_loss | mean_best_val_accuracy |
| --- | --- | --- | --- |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 |
| circles_mlp | rmsprop | 0.0409 | 0.9944 |
| conflicting_batches_classification | sgd_momentum | 0.0643 | 0.9792 |
| direction_reversal_objective | sgd_momentum | 0.0017 | nan |
| harmonic_oscillator_objective | rmsprop | 0.0000 | nan |
| label_noise_breast_cancer | sgd_momentum | 0.2172 | 0.9661 |
| moons_mlp | rmsprop | 0.0489 | 0.9948 |
| narrow_valley_objective | sgd_momentum | 0.0000 | nan |
| noisy_quadratic_objective | sgd_momentum | -0.0487 | nan |
| oscillatory_valley | sgd_momentum | -0.0711 | nan |
| quadratic_bowl_objective | sgd_momentum | 0.0299 | nan |
| rosenbrock_valley | rmsprop | 0.0004 | nan |
| saddle_objective | sgd_momentum | -4.1322 | nan |
| small_batch_instability | sgd_momentum | 0.3698 | 0.8754 |
| wine_mlp | sgd_momentum | 0.0519 | 0.9926 |