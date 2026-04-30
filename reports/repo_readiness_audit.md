# Repo Readiness Audit

Repository: `magneto-hamiltonian-adam`  
Audit date: `2026-04-29`

## Summary

This repository is now close to clone-and-run ready for the current Coherent Momentum / Magneto-Hamiltonian optimizer family.

The repo now has:

- a stable public alias: `CoherentMomentumOptimizer`
- backward-compatible research class names
- focused tests for optimizer behavior, GPU compatibility, and benchmark/report outputs
- reproducible script entry points
- a Colab notebook for running the full repo evaluation flow
- explicit reports for accepted mainline results and GPU/improved-branch audit results
- a focused directional-instability benchmark for the narrow claim
- a CNN credibility benchmark for the known weak point
- failure-case documentation and modern-baseline status notes
- a checked bibliography in `REFERENCES.md`
- a direct comparison note in `docs/COMPARISONS.md`

## Inventory

### Optimizer implementation files

- `src/optimizers/coherent_momentum_optimizer.py`
- `src/optimizers/magneto_hamiltonian_adam.py`
- `src/optimizers/magneto_hamiltonian_adam_improved.py`
- `src/optimizers/hamiltonian_adam.py`
- `src/optimizers/magneto_adam.py`

### Benchmark and report scripts

Stable research entry points:

- `scripts/run_magneto_hamiltonian_adam_smoke.py`
- `scripts/run_magneto_hamiltonian_adam_benchmarks.py`
- `scripts/run_magneto_hamiltonian_adam_energy_tests.py`
- `scripts/run_magneto_hamiltonian_adam_ablation.py`
- `scripts/export_magneto_hamiltonian_adam_report.py`
- `scripts/run_magneto_gpu_smoke.py`
- `scripts/run_magneto_gpu_benchmarks.py`
- `scripts/run_magneto_gpu_cnn_benchmarks.py`
- `scripts/run_magneto_gpu_stress_benchmarks.py`
- `scripts/run_magneto_gpu_multitask_benchmarks.py`
- `scripts/run_magneto_gpu_ablation.py`
- `scripts/export_magneto_gpu_report.py`
- `scripts/run_directional_instability_benchmark.py`
- `scripts/export_directional_instability_report.py`
- `scripts/run_cnn_credibility_benchmark.py`
- `scripts/export_cnn_credibility_report.py`
- `scripts/demo_directional_instability.py`

Public-alias wrapper scripts:

- `scripts/run_coherent_momentum_optimizer_smoke.py`
- `scripts/run_coherent_momentum_optimizer_benchmarks.py`
- `scripts/run_coherent_momentum_optimizer_energy_tests.py`
- `scripts/run_coherent_momentum_optimizer_ablation.py`
- `scripts/export_coherent_momentum_optimizer_report.py`

### Config files

- `configs/magneto_hamiltonian_adam_default.yaml`
- `configs/magneto_hamiltonian_adam_tuning.yaml`
- `configs/magneto_hamiltonian_adam_energy.yaml`
- `configs/magneto_hamiltonian_adam_ablation.yaml`
- `configs/magneto_gpu_smoke.yaml`
- `configs/magneto_gpu_default.yaml`
- `configs/magneto_gpu_cnn.yaml`
- `configs/magneto_gpu_stress.yaml`
- `configs/magneto_gpu_multitask.yaml`
- `configs/magneto_gpu_ablation.yaml`
- `configs/directional_instability_benchmark.yaml`
- `configs/cnn_credibility_benchmark.yaml`
- `configs/presets/standard_safe.yaml`
- `configs/presets/stress_specialist.yaml`
- `configs/presets/cnn_safe.yaml`

### Tests

- `tests/test_magneto_hamiltonian_adam.py`
- `tests/test_magneto_gpu_compatibility.py`
- `tests/test_magneto_benchmark_outputs.py`
- `tests/test_real_hamiltonian_adam.py`

### Reports and result CSVs

Mainline accepted run:

- `reports/accepted_magneto_hamiltonian/`

GPU + improved-branch audit:

- `reports/magneto_gpu/`

Focused claim benchmark:

- `reports/directional_instability/`

CNN credibility benchmark:

- `reports/cnn_credibility/`

Directional demo:

- `reports/demo_directional_instability/`

Real Hamiltonian reference:

- `reports/reference_real_hamiltonian/`

### Figures

Figures are present in:

- `reports/accepted_magneto_hamiltonian/figures/`
- `reports/magneto_gpu/figures/`
- `reports/reference_real_hamiltonian/figures/`

### Install and dependency files

- `pyproject.toml`
- `LICENSE`
- `CITATION.cff`
- `README.md`
- `REPRODUCING.md`
- `REFERENCES.md`

### Notebook

- `notebooks/coherent_momentum_full_eval.ipynb`

## Validation Results

The following commands were run after the repo cleanup:

```bash
python -m compileall src
pytest tests/test_magneto_hamiltonian_adam.py -q
pytest tests/test_magneto_gpu_compatibility.py tests/test_magneto_benchmark_outputs.py -q
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/generate_colab_notebook.py
python scripts/run_directional_instability_benchmark.py --config configs/directional_instability_benchmark.yaml
python scripts/export_directional_instability_report.py
python scripts/run_cnn_credibility_benchmark.py --config configs/cnn_credibility_benchmark.yaml
python scripts/export_cnn_credibility_report.py
python scripts/demo_directional_instability.py
```

Observed outcomes:

- `compileall`: passed
- `tests/test_magneto_hamiltonian_adam.py -q`: `11 passed, 1 warning`
- `tests/test_magneto_gpu_compatibility.py tests/test_magneto_benchmark_outputs.py -q`: `8 passed, 1 warning`
- mainline smoke script: passed
- coherent-alias smoke wrapper: passed
- notebook generation: passed
- directional-instability benchmark: ran and exported report output
- cnn-credibility benchmark: ran and exported report output
- directional demo: ran and exported report output

The warning is the existing pytest config warning about `asyncio_mode`, not an optimizer failure.

## Readiness Questions

### Can someone install this repo?

Yes.

The repo is installable with:

```bash
pip install -e .[dev]
```

`pyproject.toml` is sufficient for the current dependency surface.

### Can someone run a smoke test?

Yes.

Working commands:

```bash
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/run_magneto_gpu_smoke.py
```

### Can someone reproduce the benchmark summary?

Yes, within the scope of the scripts and configs currently stored in the repo.

Reproduction commands are documented in:

- `README.md`
- `REPRODUCING.md`
- `notebooks/coherent_momentum_full_eval.ipynb`

### Can someone inspect results?

Yes.

Results are available as:

- markdown reports
- benchmark CSVs
- figure PNGs
- the Colab notebook, which prints the CSV contents and report text
- focused claim documents in `docs/CLAIM.md`, `docs/MODERN_BASELINES.md`, and `docs/FAILURE_CASES.md`
- comparison and bibliography documents in `docs/COMPARISONS.md` and `REFERENCES.md`

### Are GPU / MPS / CUDA paths supported?

Yes, with an important caveat.

- CPU path: verified
- MPS path: verified for compatibility on this machine
- CUDA path: supported by the code and tests, but not exercised on this machine because CUDA hardware was not available

Broad benchmark claims in the current repo were intentionally run on CPU, not MPS, so MPS compatibility should not be read as a performance claim.

### Are there stale or misleading files?

Yes, but they are now documented.

The repo still includes:

- internal comparison optimizers beyond the public coherent momentum surface
- historical report folders
- internal benchmark suites copied forward from the wider research workspace

These are retained for reproducibility and comparison, not because they are all equal public products of this repo. The README now scopes them explicitly.

## Missing Pieces / Remaining Gaps

- The public repo story is now coherent, but the codebase still contains legacy comparison modules that could be split out later.
- The improved Magneto branch is still slower than `AdamW`, `RMSProp`, and `SGD+momentum`.
- CNN performance is still weak relative to the strongest baselines.
- The current default tasks do not yet include stronger torchvision-based datasets in the active benchmark scripts.
- Optional torchvision CNN tasks are supported in code, but not active in the current local environment because `torchvision` is not installed.
- `torchvision` and `schedulefree` remain optional extras rather than default dependencies.
- The default directional-instability benchmark is a newcomer-facing proof sweep, not an exhaustive paper-scale multi-seed run.
- CUDA performance has not been measured on this machine.

## Overall Readiness Verdict

For the current research scope, this repo is now:

- installable
- smoke-testable
- benchmark-script runnable
- report-inspectable
- notebook reproducible

It is ready for clone-and-run evaluation of the current Coherent Momentum / Magneto-Hamiltonian optimizer family, with the current limitations stated directly rather than hidden.
