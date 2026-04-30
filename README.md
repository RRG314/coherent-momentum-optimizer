# Coherent Momentum Optimizer

A PyTorch optimizer for directional coherence in unstable gradient regimes.

`CoherentMomentumOptimizer` is the public class for the stable mainline in this repository. The optimizer is not presented here as a universal replacement for `SGD`, `RMSProp`, or `AdamW`. Its intended use is narrower: training problems where the update direction oscillates, reverses, conflicts across batches, or becomes unreliable under noise.

The current evidence supports Coherent Momentum as a specialist optimizer for directional-instability regimes. The included reports also show clear limitations: `RMSProp` and `SGD+momentum` remain stronger on many standard tasks, and CNN performance is still an open gap.

## Overview

This repository is set up as a clone-and-run research package. It includes the optimizer implementation, focused tests, benchmark scripts, configuration files, report exports, and reproduction notes.

The public surface is:

```python
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer
```

The repository also keeps related internal comparison classes available for reproducibility, including `CoherentMomentumOptimizerImproved` and `CoherentMomentumRealBaseline`.

The current claim is intentionally narrow:

> Coherent Momentum is designed to help when the gradient direction is unreliable.

It should not be read as a claim that this optimizer is the best default choice for ordinary supervised learning, CNN training, or all neural network optimization.

## What Problem This Optimizer Targets

Most optimizers begin from the assumption that the gradient already points in a usable direction and that the optimizer’s main job is to scale, smooth, or precondition that direction. That assumption often works well, but it is not always safe. In narrow valleys, saddle regions, noisy small-batch training, nonstationary data, and conflicting-batch regimes, the local gradient can point in a direction that is technically correct for the current step but unstable over time.

The result is familiar: zig-zagging, repeated reversal, weak momentum accumulation, or updates that look locally sensible but damage progress across steps. Coherent Momentum targets that failure mode directly. Instead of only asking how large the update should be, it also asks whether the update direction is coherent enough to trust.

The focused claim for this repository is documented in [docs/CLAIM.md](docs/CLAIM.md). The benchmark layout follows that claim instead of trying to force a broad “best default optimizer” story.

## Relation to Existing Optimizers

`SGD` follows the raw gradient direction directly. `SGD+momentum` smooths that direction across time, but it does not explicitly test whether the accumulated direction has become unreliable. `RMSProp` rescales update magnitudes using squared-gradient history. It is strong in many small and noisy tasks, but its main control is magnitude scaling rather than direct direction-quality measurement.

`Adam` and `AdamW` combine momentum with adaptive per-parameter scaling. They are strong general baselines, but they still begin from the gradient direction and transform it. `SAM` and `ASAM` also matter here because they test neighborhood stability, but they do so by perturbing the objective rather than by using gradient/momentum coherence as the central control signal. `PCGrad` and `CAGrad` are important conflict-aware references as well, although they operate in explicit multitask settings rather than ordinary single-loss training.

`CoherentMomentumOptimizer` sits in a different design space. It explicitly measures directional coherence, conflict, and rotation, then adjusts the step when the direction becomes unstable. The central question is not only “how large should the update be?” but also “is this direction reliable enough to trust?”

Internally, the stable alias currently maps to `CoherentMomentumOptimizer`, which depends on `CoherentMomentumRealBaseline` for the physical base dynamics.

The repo keeps a fuller comparison note in [docs/COMPARISONS.md](docs/COMPARISONS.md). The bibliography for every optimizer family named in this repository is collected in [REFERENCES.md](REFERENCES.md). The repository-level attribution note is in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## Current Implementation

Public optimizer entry points:

- `src/optimizers/coherent_momentum_optimizer.py`
  - `CoherentMomentumOptimizer`
- `src/optimizers/coherent_momentum_optimizer_improved.py`
  - `CoherentMomentumOptimizerImproved`
- `src/optimizers/coherent_momentum_real_baseline.py`
  - `CoherentMomentumRealBaseline`
  - `CoherentMomentumPhysicalBaseline`

The public alias is:

```python
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer
```

Related internal comparison classes remain importable:

```python
from optimizers import CoherentMomentumOptimizer, CoherentMomentumOptimizerImproved, CoherentMomentumRealBaseline
```

The public name is Coherent Momentum. Older internal naming from the research branch has been kept only where needed for compatibility and reproducibility.

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

The method details live in [docs/METHOD.md](docs/METHOD.md), and the reason the repo keeps `CoherentMomentumRealBaseline` visible is explained in [docs/REAL_BASELINE.md](docs/REAL_BASELINE.md).

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

This repository still contains internal comparison modules and historical research-support code copied forward from the wider optimizer workspace. Those files are kept so the included reports can be reproduced. They are not all part of the public optimizer identity.

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

The default dependency set covers the current included scripts. No external dataset download is required for the default included tasks. Optional torchvision-backed CNN tasks are available when the `vision` extra is installed.

For a newcomer, the shortest explanation path through the repository is:

- [docs/CLAIM.md](docs/CLAIM.md)
- [docs/COMPARISONS.md](docs/COMPARISONS.md)
- [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md)
- [REFERENCES.md](REFERENCES.md)

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
pytest tests/test_coherent_momentum_optimizer.py -q
pytest tests/test_coherent_momentum_gpu_compatibility.py -q
pytest tests/test_coherent_momentum_benchmark_outputs.py -q
```

These are the relevant readiness tests for this repository. They cover import paths, initialization, one-step parameter updates, no-NaN smoke behavior, state dict save/load, diagnostics enable/disable and throttling, GPU-like device compatibility when available, benchmark-output schemas, and report-export outputs.

Use these focused tests as the readiness signal for this repo. Do not treat unrelated workspace failures as evidence against it.

## Running Benchmarks

Stable mainline script names:

```bash
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/run_coherent_momentum_optimizer_benchmarks.py --config configs/coherent_momentum_optimizer_default.yaml
python scripts/run_coherent_momentum_optimizer_energy_tests.py --config configs/coherent_momentum_optimizer_energy.yaml
python scripts/run_coherent_momentum_optimizer_ablation.py --config configs/coherent_momentum_optimizer_ablation.yaml
python scripts/export_coherent_momentum_optimizer_report.py
```

If a fresh `reports/coherent_momentum_mainline/` benchmark folder does not exist yet, the export wrapper falls back to the accepted historical snapshot in `reports/accepted_coherent_momentum/` instead of failing.

GPU compatibility and specialist suite:

```bash
python scripts/run_coherent_momentum_gpu_smoke.py
python scripts/run_coherent_momentum_gpu_benchmarks.py --config configs/coherent_momentum_gpu_default.yaml
python scripts/run_coherent_momentum_gpu_cnn_benchmarks.py --config configs/coherent_momentum_gpu_cnn.yaml
python scripts/run_coherent_momentum_gpu_stress_benchmarks.py --config configs/coherent_momentum_gpu_stress.yaml
python scripts/run_coherent_momentum_gpu_multitask_benchmarks.py --config configs/coherent_momentum_gpu_multitask.yaml
python scripts/run_coherent_momentum_gpu_ablation.py --config configs/coherent_momentum_gpu_ablation.yaml
python scripts/export_coherent_momentum_gpu_report.py
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

A full Colab-oriented notebook is included at `notebooks/coherent_momentum_full_eval.ipynb`.

It installs the repository, runs the focused tests, runs the mainline smoke / benchmark / energy / ablation scripts, runs the GPU compatibility and specialist benchmark scripts when available, prints the result CSVs and markdown reports, and displays the stored benchmark figures.

The notebook is generated from `scripts/generate_colab_notebook.py`.

## What It Compares Against

The default benchmark path keeps the core baselines required to interpret the narrow claim:

- `SGD`
- `SGD+momentum`
- `RMSProp`
- `Adam`
- `AdamW`
- `CoherentMomentumRealBaseline`
- `CoherentDirectionReferenceOptimizer`

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

The repository contains four result groups that matter most for orientation:

1. accepted mainline Coherent Momentum reports in `reports/accepted_coherent_momentum/`
2. GPU compatibility and improved-branch audit reports in `reports/coherent_momentum_gpu/`
3. focused directional-instability proof reports in `reports/directional_instability/`
4. CNN credibility reports in `reports/cnn_credibility/`

### Accepted mainline snapshots

Selected rows from `reports/accepted_coherent_momentum/benchmark_results.csv`:

| task | best val loss | best val acc |
| --- | ---: | ---: |
| breast_cancer_mlp | 0.079857 | 0.980469 |
| conflicting_batches_classification | 0.062339 | 0.979167 |
| direction_reversal_objective | 0.008298 | - |
| rosenbrock_valley | 0.001829 | - |
| saddle_objective | -4.107943 | - |

Validated win counts from the accepted mainline report:

- vs `coherent_momentum_real_baseline`: `13`
- vs `coherent_direction_reference`: `7`
- vs `adamw`: `8`
- vs `rmsprop`: `4`
- vs `topological_adam`: `8`

These rows support Coherent Momentum as a useful custom specialist branch, but not as a broad default optimizer.

### GPU / improved-branch audit summary

From `reports/coherent_momentum_gpu/final_coherent_momentum_gpu_report.md`:

- improved branch vs current Coherent Momentum mainline: `8` meaningful wins, `3` tracked `2x` wins
- improved branch vs `AdamW`: `7` meaningful wins, `3` tracked `2x` wins
- improved branch vs `RMSProp`: `7` meaningful wins
- improved branch vs `SGD+momentum`: `5` meaningful wins

Those wins are concentrated in directional synthetic stress tasks. Broad task-winner counts are still led by `SGD+momentum` and `RMSProp`, the improved branch remains slower than `AdamW`, `RMSProp`, and `SGD+momentum`, and CNN performance is still weak compared with the practical baselines.

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

The interpretation should stay narrow. The focused proof benchmark does show that the coherent-momentum family can beat `AdamW` on part of the oscillation/reversal slice. It does **not** show broad superiority over `RMSProp` or `SGD+momentum`.

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

The CNN takeaway is straightforward: the CNN gap is still open. `RMSProp` is the strongest CNN baseline in the included digits-based benchmark slice, and `AdamW` also beats the coherent-momentum family on the included CNN rows.

### Interpretation

The evidence supports a narrow conclusion. Coherent Momentum is useful when directional instability is the bottleneck. It is not currently better than `RMSProp` or `SGD+momentum` across standard tasks, and the CNN path is not solved.

The most defensible claim is:

> Coherent Momentum is a specialist optimizer for unstable gradient-direction regimes, with evidence of improvement over AdamW in selected instability slices and stronger internal performance in focused stress runs.

The current evidence does not support a universal optimizer claim, a general CNN optimizer claim, a broad state-of-the-art claim, or a replacement claim over `RMSProp` or `SGD+momentum`.

## Reports and Outputs

Mainline reports:

- `reports/accepted_coherent_momentum/`

GPU and improved-branch audit:

- `reports/coherent_momentum_gpu/`

Focused directional-instability proof:

- `reports/directional_instability/`

CNN credibility:

- `reports/cnn_credibility/`

Directional demo:

- `reports/demo_directional_instability/`

Real Hamiltonian reference:

- `reports/reference_real_baseline/`

Repo readiness audit:

- `reports/repo_readiness_audit.md`

Additional audit docs:

- `reports/coherent_momentum_repo_audit/code_audit.md`
- `reports/coherent_momentum_repo_audit/improvement_plan.md`

## Limitations

- This is **not** a universal optimizer replacement.
- `RMSProp` and `SGD+momentum` remain stronger on many standard tasks in the current included reports.
- CNN performance is still under development and currently trails the strongest baseline optimizers in this repo.
- The improved GPU-safe branch is slower than the simpler baselines.
- The repo still contains internal comparison modules beyond the public coherent momentum surface because they are used by the reproducibility harness.
- The focused proof benchmark is designed as a representative newcomer-facing slice, not as the largest possible multi-seed instability sweep.

## Known Failure Cases

See `docs/FAILURE_CASES.md`.

Short version: on clean standard tasks, `RMSProp` or `SGD+momentum` may be the better choice. On CNNs, the coherent-momentum family still trails the strongest practical baselines in this repository. On PINNs, `LBFGS` or `AdamW -> LBFGS` hybrids may still be better. On stable smooth problems, simpler optimizers are faster and usually sufficient.

## Development Roadmap

Current practical next steps are to simplify the improved branch toward a lighter standard-safe or low-projection default, keep projection as an optional stress preset rather than forcing it into the default path, separate any CNN-specific work into a lighter dedicated branch instead of increasing global control complexity, expand held-out stress evaluation while keeping the benchmark claims honest, continue reducing diagnostics and control overhead in the hot path, improve CNN behavior without weakening the directional-instability niche, and make optional modern baseline comparisons easier to reproduce on fresh environments.

## Citation

If you use this repository, cite it as software and point readers to the included benchmark reports:

```bibtex
@software{reid_coherent_momentum_optimizer,
  title = {Coherent Momentum Optimizer},
  author = {Reid, Steven},
  year = {2026},
  note = {PyTorch research optimizer repository for directional coherence in unstable gradient regimes},
}
```
