# V4Fast And Magneto-Hamiltonian Inventory Report

Optimizer families and companion repositories named below are referenced in [../REFERENCES.md](../REFERENCES.md). The block-direction line mentioned here is a separate companion repository comparison, not part of the public CMO optimizer surface.

## Scope and provenance

This report covers the current accepted mainline of:

- `BlockDirectionOptimizerV4Fast`
- `MagnetoHamiltonianAdam`

The `V4Fast` numbers below come from a fresh isolated rerun of the accepted mainline on `2026-04-29` using:

- branch test file
- smoke run
- tuning run
- benchmark run

The later failed `V4Fast` experiments were moved aside to:

- `reports/block_direction_v4_fast_experimental_backup_20260429`

and are **not** counted here.

The `MagnetoHamiltonianAdam` numbers below come from the existing isolated report suite in:

- `reports/magneto_hamiltonian_adam`

which still matches the current branch code.

## 1. Fast branch inventory

### Core implementation

- `src/optimizers/block_direction_optimizer_v4_fast.py`
- inherited support from:
  - `src/optimizers/block_direction_optimizer_v3.py`
  - `src/optimizers/block_direction_optimizer_v2.py`
  - `src/optimizers/optimizer_utils.py`

### Research harness

- `src/optimizer_research/block_direction_v4_fast_suite.py`
- `src/optimizer_research/baselines.py`
- `src/optimizer_research/benchmarking.py`

### Scripts

- `scripts/run_block_direction_v4_fast_smoke.py`
- `scripts/run_block_direction_v4_fast_tuning.py`
- `scripts/run_block_direction_v4_fast_benchmarks.py`
- `scripts/run_block_direction_v4_fast_ablation.py`
- `scripts/export_block_direction_v4_fast_report.py`

### Configs

- `configs/block_direction_v4_fast_default.yaml`
- `configs/block_direction_v4_fast_tuning.yaml`
- `configs/block_direction_v4_fast_ablation.yaml`
- `configs/block_direction_v4_fast_cnn_probe.yaml`
- `configs/block_direction_v4_fast_cnn_tuning.yaml`
- `configs/block_direction_v4_fast_pinn_probe.yaml`
- `configs/block_direction_v4_fast_pinn_tuning.yaml`
- `configs/block_direction_v4_fast_gpu.yaml`

### Tests

- `tests/test_block_direction_v4_fast.py`

These tests cover:

- initialization and one real parameter-changing step
- diagnostic keys
- convolutional filter path
- mini-suite output generation and schema
- MPS path when available

Latest isolated result:

- `4 passed, 1 warning`

### Report artifacts

- `reports/block_direction_v4_fast/current_state.md`
- `reports/block_direction_v4_fast/math_definition.md`
- `reports/block_direction_v4_fast/literature_scan.md`
- `reports/block_direction_v4_fast/literature_matrix.csv`
- `reports/block_direction_v4_fast/smoke_results.csv`
- `reports/block_direction_v4_fast/tuning_results.csv`
- `reports/block_direction_v4_fast/benchmark_results.csv`
- `reports/block_direction_v4_fast/best_by_task.csv`
- `reports/block_direction_v4_fast/win_flags.csv`

### Extra probe artifacts

- CNN probe: `reports/block_direction_v4_fast_cnn/benchmark_results.csv`
- PINN probe: `reports/block_direction_v4_fast_pinn/benchmark_results.csv`
- MPS probe: `reports/block_direction_v4_fast_mps/benchmark_results.csv`

## 2. What V4Fast does

`BlockDirectionOptimizerV4Fast` is a **blockwise candidate-direction optimizer**, not an Adam-style moment optimizer.

Its novelty claim is not “better scalar rescaling of gradients.” Its core rule is:

1. partition parameters into blocks
2. build a small set of candidate directions for each block
3. score them with block trust signals
4. choose one block direction with winner-take-all
5. apply a bounded, energy-normalized block step

## 3. V4Fast block structure and rules

### Block formation

Default strategy is `smart_v4`:

- vectors and biases: one tensor block
- small 2D matrices: one tensor block
- larger 2D tensors: row blocks
- convolutional tensors: typed conv profile with block views compatible with row/filter structure

### Default candidate set

Per block, `V4Fast` keeps only:

- `gradient`
- `stable_consensus`
- `trusted_direction`
- `low_rank_matrix`

### Candidate meanings

- `gradient`: normalized negative gradient block direction
- `trusted_direction`: normalized trusted-memory direction
- `low_rank_matrix`: structured row/column matrix consensus direction for matrix-shaped parameters
- `stable_consensus`: normalized blend of gradient, trusted memory, smoothed memory, and matrix direction when they align positively

### Trust score

For each candidate `d` in block `i`, the score is:

`T_i(d) = w_d A_i(d) + w_m M_i(d) + w_q Q_i(d) + w_s S_i(d) + w_c C_i(d) - w_o O_i(d) - w_f F_i(d) - w_k cost(d)`

Where:

- `A_i(d)`: descent alignment with the negative gradient
- `M_i(d)`: coherence with trusted and smoothed memory
- `Q_i(d)`: improvement-history quality memory
- `S_i(d)`: block norm stability versus gradient EMA
- `C_i(d)`: consensus support
- `O_i(d)`: oscillation against previous gradient and previous update
- `F_i(d)`: conflict against trusted direction and previous update
- `cost(d)`: fixed per-candidate complexity penalty

### Selection

Default selection mode:

- `winner_take_all`

So:

`d_i* = argmax_d T_i(d)`

### Step magnitude

The direction is **not** scaled by Adam moments. Instead it uses block energy:

`alpha_i = lr * trust_i * ||g_i|| / (sqrt(E_i) + eps)^p`

with:

- `E_i`: EMA of mean squared block gradient energy
- `trust_i`: bounded trust scale
- `p`: energy exponent
- parameter-relative update cap

### Typed conv/dense split

The merged fast branch keeps one optimizer class but two internal profiles:

- dense/vector tensors use the original V4Fast-style defaults
- convolutional tensors use:
  - lower effective learning-rate scale
  - stronger coherence/stability weights
  - smaller conv matrix cutoff
  - conv-safe scaling

### Cheap conv structure support

Conv support is estimated cheaply from:

- channel coherence
- spatial coherence
- filter-bank coherence

These are combined into a block support score in `[0, 1]`.

That support only affects **step safety**, not candidate generation.

### Conv-safe scaling

For conv blocks:

- weak-support conv blocks get a cooler step multiplier
- weak-support conv blocks get a stronger effective energy exponent
- conv blocks also use a lower max update ratio

### Important default exclusions

These are intentionally **not** in the default hot path:

- recoverability gate
- projection
- orthogonal escape
- soft routing
- magneto-style controller stack

## 4. V4Fast accepted benchmark setup

From `configs/block_direction_v4_fast_default.yaml`:

- device: `cpu`
- seeds: `11, 29, 47`
- benchmark epoch scale: `0.8`
- optimizers compared:
  - `block_direction_optimizer_v4_fast`
  - `block_direction_optimizer_v4_fast_legacy`
  - `block_direction_optimizer_v42`
  - `magneto_hamiltonian_adam`
  - `adamw`
  - `rmsprop`
  - `sgd_momentum`

Benchmark tasks:

- `breast_cancer_mlp`
- `moons_mlp`
- `wine_mlp`
- `digits_mlp`
- `digits_cnn`
- `oscillatory_valley`
- `saddle_objective`
- `plateau_escape_objective`
- `direction_reversal_objective`
- `block_structure_classification`
- `low_rank_matrix_objective`

## 5. V4Fast accepted results

### Task summary

| task | mean_best_val_loss | mean_best_val_accuracy | mean_final_val_loss | mean_final_val_accuracy | mean_runtime_per_step_ms | mean_selection_score | mean_optimizer_state_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block_structure_classification | 0.361815 | 0.845486 | 0.690049 | 0.820313 | 40.640766 | 0.807519 | 0.059143 |
| breast_cancer_mlp | 0.061726 | 0.985677 | 0.110304 | 0.971962 | 36.806797 | 0.978523 | 0.063538 |
| digits_cnn | 0.814210 | 0.740294 | 0.878094 | 0.731416 | 50.338680 | 0.640385 | 0.058376 |
| digits_mlp | 0.079455 | 0.979818 | 0.097277 | 0.974886 | 38.533434 | 0.960168 | 0.142410 |
| direction_reversal_objective | 0.001667 | nan | 0.001667 | nan | 4.416484 | -0.084274 | 0.000072 |
| low_rank_matrix_objective | 0.090387 | nan | 0.090387 | nan | 6.226815 | -0.115042 | 0.002239 |
| moons_mlp | 0.182003 | 0.928943 | 0.182003 | 0.928571 | 33.682543 | 0.909829 | 0.019634 |
| oscillatory_valley | -0.071087 | nan | -0.071087 | nan | 4.445604 | 0.069458 | 0.000072 |
| plateau_escape_objective | -1.050024 | nan | -1.050024 | nan | 4.635043 | 1.047797 | 0.000072 |
| saddle_objective | -4.132231 | nan | -4.132231 | nan | 4.420595 | 4.092883 | 0.000072 |
| wine_mlp | 0.058005 | 0.985185 | 0.060464 | 0.970370 | 36.239276 | 0.974221 | 0.050610 |

### Win counts from the accepted run

- vs `block_direction_optimizer_v4_fast_legacy`: `4` wins
- vs `block_direction_optimizer_v42`: `4` wins
- vs `adamw`: `7` wins, `5` tracked `2x` wins
- vs `rmsprop`: `6` wins
- vs `sgd_momentum`: `4` wins, `1` tracked `2x` win
- vs `magneto_hamiltonian_adam`: `10` wins, `1` tracked `2x` win

### Best-by-task reality

`V4Fast` does **not** own the whole suite. In `best_by_task.csv`:

- `rmsprop` wins `breast_cancer_mlp`
- `sgd_momentum` wins `digits_cnn`
- `block_direction_optimizer_v42` wins `digits_mlp`
- `sgd_momentum` wins several direct stress tasks outright

So the honest verdict is:

- `V4Fast` is a credible **specialist/generalist hybrid**
- it is still **not** the strongest overall optimizer on the mixed suite

## 6. V4Fast extra probes

### CNN probe

| task | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_selection_score |
| --- | ---: | ---: | ---: | ---: |
| breast_cancer_mlp | 0.062320 | 0.984375 | 26.732624 | 0.977245 |
| digits_cnn | 0.704691 | 0.757714 | 33.470237 | 0.668201 |
| digits_mlp | 0.077418 | 0.981120 | 25.778616 | 0.962886 |
| moons_mlp | 0.182003 | 0.928943 | 23.529780 | 0.909829 |
| wine_mlp | 0.058005 | 0.985185 | 22.577889 | 0.974221 |

### PINN probe

| task | mean_best_val_loss | mean_runtime_per_step_ms | mean_selection_score |
| --- | ---: | ---: | ---: |
| pinn_harmonic_oscillator | 0.389121 | 24.515694 | -0.422816 |
| pinn_heat_equation | 0.008893 | 34.867125 | -0.020304 |
| pinn_poisson_1d | 0.024617 | 29.468925 | -6.573852 |

This is a usable probe result, but **not** a winning PINN story against stronger PINN baselines like L-BFGS hybrids.

### MPS GPU probe

| task | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_selection_score |
| --- | ---: | ---: | ---: | ---: |
| breast_cancer_mlp | 0.104411 | 0.972656 | 208.903162 | 0.960354 |
| oscillatory_valley | 0.016033 | nan | 20.988896 | -0.023595 |

The MPS path works, but on this machine it is slower than the CPU path for these tasks.

## 7. What actually seems to help in V4Fast

From the accepted line and prior validated branch work, the keepers are:

- blockwise candidate-direction selection
- stable consensus
- trusted direction memory
- low-rank / matrix consensus
- typed conv/dense split
- cheap conv structure support
- conv-safe scaling

What should stay out of the default path:

- recoverability gate
- projection / orthogonal escape
- soft policy routing
- heavier conv trust overlays
- controller stacking

## 8. Magneto branch inventory

### Core implementation

- `src/optimizers/magneto_hamiltonian_adam.py`
- inherited Hamiltonian core:
  - `src/optimizers/hamiltonian_adam.py`
  - specifically `HamiltonianAdamReal`

### Research harness

- `src/optimizer_research/magneto_hamiltonian_suite.py`
- `src/optimizer_research/baselines.py`
- `src/optimizer_research/benchmarking.py`

### Scripts

- `scripts/run_magneto_hamiltonian_adam_smoke.py`
- `scripts/run_magneto_hamiltonian_adam_tuning.py`
- `scripts/run_magneto_hamiltonian_adam_benchmarks.py`
- `scripts/run_magneto_hamiltonian_adam_energy_tests.py`
- `scripts/run_magneto_hamiltonian_adam_ablation.py`
- `scripts/export_magneto_hamiltonian_adam_report.py`

### Configs

- `configs/magneto_hamiltonian_adam_default.yaml`
- `configs/magneto_hamiltonian_adam_tuning.yaml`
- `configs/magneto_hamiltonian_adam_energy.yaml`
- `configs/magneto_hamiltonian_adam_ablation.yaml`

### Tests

- `tests/test_magneto_hamiltonian_adam.py`

These tests cover:

- initialization and parameter update
- neutral-mode closeness to `HamiltonianAdamReal`
- leapfrog closure path and recomputed-gradient flag
- no NaNs
- `state_dict` round-trip
- diagnostics schema
- benchmark schema
- ablation schema
- report generation

Latest isolated result:

- `9 passed`

### Report artifacts

- `reports/magneto_hamiltonian_adam/benchmark_results.csv`
- `reports/magneto_hamiltonian_adam/energy_tests.csv`
- `reports/magneto_hamiltonian_adam/ablation_results.csv`
- `reports/magneto_hamiltonian_adam/best_by_task.csv`
- `reports/magneto_hamiltonian_adam/tuning_results.csv`
- `reports/magneto_hamiltonian_adam/final_report.md`

## 9. What MagnetoHamiltonianAdam does

`MagnetoHamiltonianAdam` is a **directional-coherence controller on top of a real Hamiltonian optimizer**.

It is not a pure Adam variant. Its base state is `HamiltonianAdamReal`, which already has:

- explicit physical momentum `p`
- kinetic/potential/total Hamiltonian tracking
- fixed or adaptive inverse mass
- symplectic Euler or leapfrog-style update path
- optional friction
- optional energy correction

`MagnetoHamiltonianAdam` then adds bounded directional field controls.

## 10. MagnetoHamiltonianAdam modes and state

### Base Hamiltonian modes inherited from `HamiltonianAdamReal`

- `symplectic_euler`
- `leapfrog_with_closure`
- `dissipative_hamiltonian`
- `adam_preconditioned_hamiltonian`
- `hamiltonian_adam_v1_compatibility`

### Mass modes

- `adaptive`
- `fixed`

### Per-parameter state

- `exp_avg`
- `exp_avg_sq`
- `hamiltonian_momentum`
- `prev_update`
- `inverse_mass_ema`
- `last_mass_trust`
- `last_mass_shock`

### Global state

- `prev_total_hamiltonian`
- `energy_ema`
- `last_mode_used`

## 11. Magneto controls and rules

### Inverse mass

If adaptive mass is enabled:

1. update second moment
2. form diagonal inverse mass proposal
3. normalize around a geometric center
4. clamp anisotropy
5. trust-gate it by:
   - warmup
   - momentum/update alignment
   - mass shock penalty
6. smooth and clamp change rate

So Adam-style second moment is used only as **preconditioning/mass**, not as the full optimizer rule.

### Magneto signals

The branch computes:

- gradient-momentum cosine
- force-momentum cosine
- gradient-previous-gradient cosine
- force-previous-update cosine
- rotation score
- coherence score
- conflict score
- conflict gate
- rotation gate
- magneto activation
- stable gate
- field strength

### Field controls

From those signals it derives:

- friction multiplier
- alignment scale
- projection strength back toward force direction

This keeps the branch close to the real Hamiltonian core on stable tasks and lets it intervene more when direction reversal, rotation, or conflict rises.

### Update structure

Per step:

1. compute/update inverse mass
2. compute magneto controls
3. do momentum half-step or symplectic-Euler style step depending on mode
4. form base step `inverse_mass * momentum`
5. optionally blend toward the force direction using bounded projection
6. scale by magneto alignment factor
7. apply parameter step
8. if closure is available in leapfrog-like modes, recompute gradient
9. apply second half-step and optional post-projection
10. compute Hamiltonian diagnostics and optional energy correction

## 12. Magneto accepted benchmark setup

From `configs/magneto_hamiltonian_adam_default.yaml`:

- device: `cpu`
- seeds: `11, 29, 47`
- benchmark epoch scale: `0.85`

Optimizers compared:

- `magneto_hamiltonian_adam`
- `real_hamiltonian_adam`
- `magneto_adam`
- `hamiltonian_adam`
- `adamw`
- `rmsprop`
- `sgd_momentum`
- `lion`
- `topological_adam`

Main benchmark tasks:

- `breast_cancer_mlp`
- `circles_mlp`
- `moons_mlp`
- `wine_mlp`
- `label_noise_breast_cancer`
- `oscillatory_valley`
- `rosenbrock_valley`
- `saddle_objective`
- `small_batch_instability`
- `conflicting_batches_classification`
- `direction_reversal_objective`

Plus separate Hamiltonian energy tests:

- `harmonic_oscillator_objective`
- `narrow_valley_objective`
- `noisy_quadratic_objective`
- `quadratic_bowl_objective`
- and the shared stress/Hamiltonian tasks

## 13. Magneto main benchmark results

### Task summary

| task | mean_best_val_loss | mean_best_val_accuracy | mean_final_val_loss | mean_final_val_accuracy | mean_runtime_per_step_ms | mean_selection_score | mean_optimizer_state_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer_mlp | 0.079857 | 0.980469 | 0.084820 | 0.980469 | 6.526128 | 0.969736 | 0.089012 |
| circles_mlp | 0.476012 | 0.916295 | 0.476012 | 0.916295 | 4.943101 | 0.868456 | 0.041222 |
| conflicting_batches_classification | 0.062339 | 0.979167 | 0.078817 | 0.962240 | 5.844910 | 0.969038 | 0.054039 |
| direction_reversal_objective | 0.008298 | nan | 0.141867 | nan | 1.166873 | -0.134239 | 0.000046 |
| label_noise_breast_cancer | 0.216981 | 0.944010 | 0.223898 | 0.924392 | 5.781989 | 0.921235 | 0.066856 |
| moons_mlp | 0.259942 | 0.892485 | 0.259942 | 0.889881 | 5.697127 | 0.864882 | 0.027122 |
| oscillatory_valley | -0.037761 | nan | 0.011815 | nan | 1.342334 | 0.018847 | 0.000046 |
| rosenbrock_valley | 0.001829 | nan | 0.001829 | nan | 1.134051 | -0.025762 | 0.000046 |
| saddle_objective | -4.107943 | nan | -3.676144 | nan | 1.264579 | 4.053687 | 0.000046 |
| small_batch_instability | 0.380347 | 0.862360 | 0.393899 | 0.849530 | 5.487120 | 0.822428 | 0.058250 |
| wine_mlp | 0.306235 | 0.955556 | 0.306235 | 0.955556 | 6.288108 | 0.921534 | 0.072578 |

### Reported win counts from the validated branch report

- vs `real_hamiltonian_adam`: `13`
- vs `magneto_adam`: `7`
- vs `adamw`: `8`
- vs `rmsprop`: `4`
- vs `topological_adam`: `8`

### Best-by-task reality

From `best_by_task.csv`, `MagnetoHamiltonianAdam` is **not** the overall best optimizer on the suite.

Strongest overall baselines there are still:

- `rmsprop`
- `sgd_momentum`

That said, Magneto remains the strongest **custom specialist** branch overall.

## 14. Magneto energy-test summary

### Magneto Hamiltonian energy results

| task | mean_best_val_loss | mean_relative_energy_drift | mean_total_hamiltonian | mean_kinetic_energy | mean_potential_energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| direction_reversal_objective | 0.007434 | -0.005785 | 1.188210 | 0.415069 | 0.773141 |
| harmonic_oscillator_objective | 0.000713 | -0.001714 | 0.063628 | 0.022317 | 0.041311 |
| narrow_valley_objective | 0.000014 | -0.002511 | 0.142422 | 0.049399 | 0.093023 |
| noisy_quadratic_objective | -0.048246 | -0.002543 | 0.224488 | 0.081950 | 0.142538 |
| oscillatory_valley | -0.070958 | -0.002898 | 0.155957 | 0.068257 | 0.087700 |
| quadratic_bowl_objective | 0.030422 | -0.002278 | 0.266052 | 0.082089 | 0.183963 |
| rosenbrock_valley | 0.000014 | -0.002936 | 0.525988 | 0.182220 | 0.343767 |
| saddle_objective | -4.130424 | -0.004564 | -3.402117 | 0.175527 | -3.577644 |

Compared to `real_hamiltonian_adam`, the magneto branch usually gives up some energy conservatism in exchange for better directional handling.

## 15. Magneto ablation findings

From the validated branch report:

- activation gating: neutral to harmful
- projection: helpful
- conflict damping: harmful
- best ablation row overall: still `rmsprop_baseline`

So the current branch lesson is:

- keep projection
- do not assume more controller logic helps

## 16. Direct comparison and practical use

### Best custom optimizer overall

- `MagnetoHamiltonianAdam`

Why:

- strongest overall custom specialist
- better than the real Hamiltonian branch in the validated comparisons
- better directional-control story than the block branch on conflict/reversal tasks

### Best novel block optimizer mainline

- `BlockDirectionOptimizerV4Fast`

Why:

- cleaner and faster than the later block descendants
- better dense/stress balance than the older block branches
- still clearly non-Adam in its update logic

### Use-case split

Use `BlockDirectionOptimizerV4Fast` when you want:

- the main novel block-direction line
- structured blockwise direction choice
- stress-task and some low-rank/block sensitivity
- one cleaner, consolidated block optimizer instead of several variants

Use `MagnetoHamiltonianAdam` when you want:

- the strongest custom optimizer overall
- oscillatory or reversing gradient regimes
- conflicting-batch or directional-coherence problems
- a physics-motivated optimizer with explicit Hamiltonian diagnostics

## 17. Bottom line

### V4Fast

- real
- tested
- technically distinct
- still not broadly better than `rmsprop` or `sgd_momentum`
- best viewed as a **novel structured specialist/generalist hybrid**

### MagnetoHamiltonianAdam

- real
- tested
- strongest custom branch overall
- still not the universal best optimizer
- best viewed as a **directional-coherence Hamiltonian specialist**

### Most honest ranking right now

1. strongest overall baseline on these suites: `rmsprop`
2. strongest classical competitor: `sgd_momentum`
3. strongest custom optimizer: `MagnetoHamiltonianAdam`
4. strongest novel block mainline: `BlockDirectionOptimizerV4Fast`
