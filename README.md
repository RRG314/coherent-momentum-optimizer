# Coherent Momentum Optimizer

A PyTorch optimizer for directional coherence in unstable gradient regimes.

`CoherentMomentumOptimizer` is the public alias for the current stable `MagnetoHamiltonianAdam` implementation in this repository. The repo keeps the original research class names for backward compatibility, but the public-facing framing is directional coherence under unstable gradient dynamics, not a claim of universal optimizer superiority.

## Overview

This repository focuses on a specialist optimizer family built around directional reliability:

- compare the current gradient to momentum, the previous gradient, and the previous update
- measure when direction becomes unstable, conflicting, or oscillatory
- intervene conservatively through bounded projection, friction scaling, and coherence-aware step control
- keep a real Hamiltonian base optimizer visible so the extra control logic remains inspectable

The current honest conclusion from the included reports is:

- this is **not** a universal replacement for `RMSProp`, `SGD+momentum`, or `AdamW`
- it is best treated as a **specialist optimizer** for oscillatory, reversal, unstable-direction, and related stress regimes
- the improved GPU-safe branch is real, but it is slower than the practical baselines and does **not** close the CNN gap yet

## What Problem This Optimizer Targets

The narrow claim for this repository is documented in `docs/CLAIM.md`:

- Coherent Momentum Optimizer should be evaluated where gradient direction is unreliable
- examples in this repo include oscillation, direction reversal, small-batch instability, nonstationary data regimes, and related conflict-style stress settings

This repository is therefore organized around a **focused proof story**, not a broad “best default optimizer” story.

## Relation to Existing Optimizers

`CoherentMomentumOptimizer` sits in a different design space than the standard magnitude-rescaling families:

- `SGD` follows the raw gradient direction directly.
- `SGD+momentum` smooths the direction across steps.
- `RMSProp` rescales gradient magnitudes adaptively.
- `Adam` / `AdamW` combine momentum with adaptive per-parameter scaling.
- `CoherentMomentumOptimizer` explicitly measures directional coherence, conflict, and rotation, then adjusts the step when the direction becomes unstable.

Internally, the stable alias currently maps to `MagnetoHamiltonianAdam`, which depends on `HamiltonianAdamReal` for the physical base dynamics.

The repo keeps a fuller comparison note in [docs/COMPARISONS.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/COMPARISONS.md). The bibliography for every optimizer family named in this repository is collected in [REFERENCES.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REFERENCES.md).

## Current Implementation

Public optimizer entry points:

- `src/optimizers/coherent_momentum_optimizer.py`
  - `CoherentMomentumOptimizer`
  - `CoherentMomentumOptimizerImproved`
- `src/optimizers/magneto_hamiltonian_adam.py`
  - `MagnetoHamiltonianAdam`
- `src/optimizers/magneto_hamiltonian_adam_improved.py`
  - `MagnetoHamiltonianAdamImproved`
- `src/optimizers/hamiltonian_adam.py`
  - `HamiltonianAdamReal`

The public alias is:

```python
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer
```

The original research names remain valid:

```python
from optimizers import MagnetoHamiltonianAdam, MagnetoHamiltonianAdamImproved, HamiltonianAdamReal
```

### Step-by-step behavior

The current stable mainline follows this high-level flow:

1. Build or update inverse mass in the real Hamiltonian core.
2. Read the current gradient, optimizer momentum, previous gradient, and previous update.
3. Compute directional signals such as:
   - gradient-momentum cosine
   - force-momentum cosine
   - gradient-history cosine
   - update-history cosine
   - rotation score
   - coherence score
   - conflict score
4. Convert those signals into bounded control values:
   - friction multiplier
   - alignment scale
   - projection strength
5. Update momentum with the selected Hamiltonian mode.
6. Form the base step from inverse mass times momentum.
7. Optionally blend the step toward the force direction using bounded projection.
8. Apply the update and record diagnostics.
9. If leapfrog-style recomputation is active and a closure is provided, run the second half-step.
10. Track energy drift and related diagnostics.

The improved branch keeps the same broad structure but moves more control computation onto tensors, adds diagnostics throttling, and evaluates safer presets such as `standard_safe`, `stress_specialist`, and `cnn_safe`.

The method details live in [docs/METHOD.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/METHOD.md), and the reason the repo keeps `HamiltonianAdamReal` visible is explained in [docs/REAL_BASELINE.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/REAL_BASELINE.md).

## Repository Layout

```text
README.md
REPRODUCING.md
pyproject.toml
src/
  optimizers/
  optimizer_research/
tests/
scripts/
configs/
reports/
notebooks/
examples/
docs/
```

Important note:

- this repo still contains some internal comparison modules and legacy research support code copied forward from the wider optimizer workspace
- the public surface for this repo is the coherent momentum / Magneto-Hamiltonian family
- the benchmark harness keeps additional optimizers because they are needed to reproduce the included comparison reports

That broader internal surface is a reproducibility choice. It is not meant to suggest that every inherited module is part of the public optimizer identity.

## Installation

This repo is installable with the included `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[dev,vision]
pip install -e .[dev,modern-baselines]
```

The default dependency set covers the current included scripts:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `pytest`

No external dataset download is required for the default included tasks. Optional torchvision-backed CNN tasks are available when the `vision` extra is installed.

The supporting explanation set for a newcomer is:

- [docs/CLAIM.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/CLAIM.md)
- [docs/COMPARISONS.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/COMPARISONS.md)
- [docs/FAILURE_CASES.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/FAILURE_CASES.md)
- [REFERENCES.md](/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REFERENCES.md)

## Quick Start

```python
import torch
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer

model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1),
)

optimizer = CoherentMomentumOptimizer(
    model.parameters(),
    lr=0.02,
    mode="adam_preconditioned_hamiltonian",
)

criterion = torch.nn.MSELoss()
x = torch.randn(16, 32)
y = torch.randn(16, 1)

optimizer.zero_grad(set_to_none=True)
loss = criterion(model(x), y)
optimizer.set_current_loss(loss.item())
loss.backward()
optimizer.step()

print(optimizer.latest_diagnostics())
```

## Running Tests

Focused repo-ready test commands:

```bash
pytest tests/test_magneto_hamiltonian_adam.py -q
pytest tests/test_magneto_gpu_compatibility.py -q
pytest tests/test_magneto_benchmark_outputs.py -q
```

These are the relevant readiness tests for this repository. Do **not** use unrelated workspace test failures as evidence against this repo.

The focused tests currently cover:

- optimizer import and initialization
- one-step parameter updates
- no-NaN smoke behavior
- state dict save/load
- diagnostics enable/disable and throttling
- GPU-like device compatibility when available
- benchmark output schema and report export outputs

Those are the tests used as repo-readiness evidence in the audit and reproduction notes.

## Running Benchmarks

Stable mainline script names:

```bash
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_magneto_hamiltonian_adam_benchmarks.py --config configs/magneto_hamiltonian_adam_default.yaml
python scripts/run_magneto_hamiltonian_adam_energy_tests.py --config configs/magneto_hamiltonian_adam_energy.yaml
python scripts/run_magneto_hamiltonian_adam_ablation.py --config configs/magneto_hamiltonian_adam_ablation.yaml
python scripts/export_magneto_hamiltonian_adam_report.py
```

If a fresh `reports/magneto_hamiltonian_adam/` benchmark folder does not exist yet, the export wrapper falls back to the accepted historical snapshot in `reports/accepted_magneto_hamiltonian/` instead of failing.

Public-alias wrapper script names:

```bash
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/run_coherent_momentum_optimizer_benchmarks.py --config configs/magneto_hamiltonian_adam_default.yaml
python scripts/run_coherent_momentum_optimizer_energy_tests.py --config configs/magneto_hamiltonian_adam_energy.yaml
python scripts/run_coherent_momentum_optimizer_ablation.py --config configs/magneto_hamiltonian_adam_ablation.yaml
python scripts/export_coherent_momentum_optimizer_report.py
```

GPU compatibility and specialist suite:

```bash
python scripts/run_magneto_gpu_smoke.py
python scripts/run_magneto_gpu_benchmarks.py --config configs/magneto_gpu_default.yaml
python scripts/run_magneto_gpu_cnn_benchmarks.py --config configs/magneto_gpu_cnn.yaml
python scripts/run_magneto_gpu_stress_benchmarks.py --config configs/magneto_gpu_stress.yaml
python scripts/run_magneto_gpu_multitask_benchmarks.py --config configs/magneto_gpu_multitask.yaml
python scripts/run_magneto_gpu_ablation.py --config configs/magneto_gpu_ablation.yaml
python scripts/export_magneto_gpu_report.py
```

Focused newcomer-facing proof benchmark:

```bash
python scripts/run_directional_instability_benchmark.py --config configs/directional_instability_benchmark.yaml
python scripts/export_directional_instability_report.py
python scripts/demo_directional_instability.py
```

CNN credibility check:

```bash
python scripts/run_cnn_credibility_benchmark.py --config configs/cnn_credibility_benchmark.yaml
python scripts/export_cnn_credibility_report.py
```

## Colab Notebook

A full Colab-oriented notebook is included at:

- `notebooks/coherent_momentum_full_eval.ipynb`

It is designed to:

- install the repo in Colab
- run the focused tests
- run the mainline smoke / benchmark / energy / ablation scripts
- run the GPU compatibility and specialist benchmark scripts
- print the result CSVs and markdown reports
- display the stored benchmark figures

The notebook is generated from:

- `scripts/generate_colab_notebook.py`

## What It Compares Against

The repo keeps the standard baselines that matter most for this narrow claim:

- `SGD`
- `SGD+momentum`
- `RMSProp`
- `Adam`
- `AdamW`
- `HamiltonianAdamReal`
- `MagnetoAdam`

Modern baseline status is documented in `docs/MODERN_BASELINES.md`.

The newcomer-facing default configs keep the core baselines. The extra modern baselines are available as documented opt-ins so the default reproduction path stays tractable.

Current status:

- `Lion`: included directly
- `Muon hybrid`: included directly when the local PyTorch build exposes `torch.optim.Muon`
- `AdaBelief`: included directly
- `SAM` / `ASAM`: included directly, but not enabled in every default benchmark because they are substantially slower
- `Schedule-Free AdamW`: optional adapter, skipped if the `schedulefree` package is not installed
- `PCGrad` / `CAGrad`: documented, but not wired into the default scalar-loss harness yet

## Results

The current repo contains two main result families:

1. accepted mainline Magneto-Hamiltonian reports in `reports/accepted_magneto_hamiltonian/`
2. GPU compatibility + improved-branch audit reports in `reports/magneto_gpu/`

### Accepted mainline snapshots

Selected rows from `reports/accepted_magneto_hamiltonian/benchmark_results.csv`:

| task | best val loss | best val acc |
| --- | ---: | ---: |
| breast_cancer_mlp | 0.079857 | 0.980469 |
| conflicting_batches_classification | 0.062339 | 0.979167 |
| direction_reversal_objective | 0.008298 | - |
| rosenbrock_valley | 0.001829 | - |
| saddle_objective | -4.107943 | - |

Validated win counts from the accepted mainline report:

- vs `real_hamiltonian_adam`: `13`
- vs `magneto_adam`: `7`
- vs `adamw`: `8`
- vs `rmsprop`: `4`
- vs `topological_adam`: `8`

### GPU / improved-branch audit summary

From `reports/magneto_gpu/final_magneto_gpu_report.md`:

- improved branch vs current Magneto: `8` meaningful wins, `3` tracked `2x` wins
- improved branch vs `AdamW`: `7` meaningful wins, `3` tracked `2x` wins
- improved branch vs `RMSProp`: `7` meaningful wins
- improved branch vs `SGD+momentum`: `5` meaningful wins

Important limitation:

- those wins are concentrated in directional synthetic stress tasks
- broad task-winner counts are still led by `SGD+momentum` and `RMSProp`
- the improved branch remains slower than `AdamW`, `RMSProp`, and `SGD+momentum`
- CNN performance is still weak compared with the practical baselines

### Focused directional-instability proof benchmark

From `reports/directional_instability/final_report.md`:

- benchmark tasks in the current newcomer-facing default:
  - `oscillatory_valley`
  - `direction_reversal_objective`
  - `small_batch_instability`
- current CMO meaningful wins vs `AdamW`: `2`
- improved CMO meaningful wins vs `AdamW`: `2`
- improved CMO meaningful wins vs `RMSProp`: `0`
- improved CMO meaningful wins vs `SGD+momentum`: `0`

Best-by-task snapshot from the current focused proof benchmark:

- `direction_reversal_objective`: `sgd`
- `oscillatory_valley`: `rmsprop`
- `small_batch_instability`: `sgd_momentum`

Interpretation:

- the focused proof benchmark does show that the coherent-momentum family can beat `AdamW` on the oscillation/reversal slice
- it does **not** show broad superiority over `RMSProp` or `SGD+momentum`
- the narrow claim should therefore stay narrow

### CNN credibility benchmark

From `reports/cnn_credibility/final_report.md`:

- tasks actually run in this environment:
  - `digits_cnn`
  - `digits_cnn_label_noise`
  - `digits_cnn_input_noise`
- optional torchvision tasks were not active in this environment
- improved CMO wins vs `AdamW`: `0`
- improved CMO wins vs `RMSProp`: `0`
- improved CMO wins vs `SGD+momentum`: `0`

Current CNN takeaway:

- the CNN gap is still open
- `RMSProp` is the strongest CNN baseline in the included digits-based benchmark slice
- `AdamW` also beats the coherent-momentum family on the included CNN rows

### Where it is strongest

- oscillatory valley
- direction reversal
- rosenbrock-style directional stress
- saddle-style stress objectives
- some instability / conflict regimes where direction quality matters more than broad default robustness

### Where it is not strongest

- standard dense tabular MLP tasks
- broad CNN accuracy
- runtime efficiency relative to `AdamW`, `RMSProp`, and `SGD+momentum`
- broad “best default optimizer” use cases

## Reports and Outputs

Mainline reports:

- `reports/accepted_magneto_hamiltonian/`

GPU and improved-branch audit:

- `reports/magneto_gpu/`

Focused directional-instability proof:

- `reports/directional_instability/`

CNN credibility:

- `reports/cnn_credibility/`

Directional demo:

- `reports/demo_directional_instability/`

Real Hamiltonian reference:

- `reports/reference_real_hamiltonian/`

Repo readiness audit:

- `reports/repo_readiness_audit.md`

Additional audit docs:

- `reports/magneto_repo_audit/code_audit.md`
- `reports/magneto_repo_audit/improvement_plan.md`

## Limitations

- This is **not** a universal optimizer replacement.
- `RMSProp` and `SGD+momentum` remain stronger on many standard tasks in the current included reports.
- CNN performance is still under development and currently trails the strongest baseline optimizers in this repo.
- The improved GPU-safe branch is slower than the simpler baselines.
- The repo still contains internal comparison modules beyond the public coherent momentum surface because they are used by the reproducibility harness.
- The focused proof benchmark is designed as a representative newcomer-facing slice, not as the largest possible multi-seed instability sweep.

## Known Failure Cases

See `docs/FAILURE_CASES.md`.

Short version:

- on clean standard tasks, `RMSProp` or `SGD+momentum` may be the better choice
- on CNNs, the coherent-momentum family still trails the strongest practical baselines here
- on PINNs, `LBFGS` or `AdamW -> LBFGS` hybrids may still be better
- on stable smooth problems, simpler optimizers are faster and usually sufficient

## Development Roadmap

Current practical next steps:

- simplify the improved branch toward a lighter standard-safe or low-projection default
- keep projection as an optional stress preset rather than forcing it into the default path
- separate any CNN-specific work into a lighter dedicated branch instead of increasing global control complexity
- expand held-out stress evaluation while keeping the benchmark claims honest
- continue reducing diagnostics and control overhead in the hot path
- improve CNN behavior without weakening the directional-instability niche
- make optional modern baseline comparisons easier to reproduce on fresh environments

## Citation

If you use this repository, cite it as software and point readers to the included benchmark reports:

```bibtex
@software{reid_coherent_momentum_optimizer,
  title = {Coherent Momentum Optimizer},
  author = {Reid, Steven},
  year = {2026},
  note = {PyTorch research optimizer repository with Magneto-Hamiltonian and coherent momentum implementations},
}
```
