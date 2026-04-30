# Complete Repository Inventory

Prepared: `2026-04-30`

This document is the concise inventory for the current standalone Coherent Momentum Optimizer repository. It is meant to answer a practical question: if someone clones this repository fresh, what is the public surface, what is the comparison surface, and where do the reproducible reports live?

## Public Identity

The public optimizer name in this repository is `CoherentMomentumOptimizer`. The implementation lives in [../src/optimizers/coherent_momentum_optimizer.py](../src/optimizers/coherent_momentum_optimizer.py), and the improved comparison branch lives in [../src/optimizers/coherent_momentum_optimizer_improved.py](../src/optimizers/coherent_momentum_optimizer_improved.py).

The repository also keeps two closely related internal comparison classes:

- `CoherentMomentumRealBaseline`, implemented in [../src/optimizers/coherent_momentum_real_baseline.py](../src/optimizers/coherent_momentum_real_baseline.py)
- `CoherentDirectionReferenceOptimizer`, implemented in [../src/optimizers/coherent_direction_reference.py](../src/optimizers/coherent_direction_reference.py)

Those classes are kept because the benchmark and report layer compares the public optimizer against a simpler real-dynamics baseline and a lighter directional controller reference.

## Top-Level Files

The files that define the public package and its evaluation story are:

- [../README.md](../README.md)
- [../REPRODUCING.md](../REPRODUCING.md)
- [../REFERENCES.md](../REFERENCES.md)
- [../pyproject.toml](../pyproject.toml)
- [../ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../CHANGELOG.md](../CHANGELOG.md)
- [../CITATION.cff](../CITATION.cff)

## Source Layout

The public optimizer code lives under [../src/optimizers](../src/optimizers).

The files most readers should start with are:

- [../src/optimizers/coherent_momentum_optimizer.py](../src/optimizers/coherent_momentum_optimizer.py)
- [../src/optimizers/coherent_momentum_optimizer_improved.py](../src/optimizers/coherent_momentum_optimizer_improved.py)
- [../src/optimizers/coherent_momentum_real_baseline.py](../src/optimizers/coherent_momentum_real_baseline.py)
- [../src/optimizers/coherent_direction_reference.py](../src/optimizers/coherent_direction_reference.py)
- [../src/optimizers/__init__.py](../src/optimizers/__init__.py)

The benchmark and report harness lives under [../src/optimizer_research](../src/optimizer_research). The main newcomer-facing suites are:

- [../src/optimizer_research/coherent_momentum_suite.py](../src/optimizer_research/coherent_momentum_suite.py)
- [../src/optimizer_research/coherent_momentum_gpu_suite.py](../src/optimizer_research/coherent_momentum_gpu_suite.py)
- [../src/optimizer_research/directional_instability_suite.py](../src/optimizer_research/directional_instability_suite.py)
- [../src/optimizer_research/cnn_credibility_suite.py](../src/optimizer_research/cnn_credibility_suite.py)

The repo still contains broader comparison helpers such as [../src/optimizer_research/baselines.py](../src/optimizer_research/baselines.py), [../src/optimizer_research/benchmarking.py](../src/optimizer_research/benchmarking.py), and [../src/optimizer_research/reporting.py](../src/optimizer_research/reporting.py). Those files are part of the reproducibility story, not a separate public product surface.

## Scripts

The mainline runnable scripts are:

- [../scripts/run_coherent_momentum_optimizer_smoke.py](../scripts/run_coherent_momentum_optimizer_smoke.py)
- [../scripts/run_coherent_momentum_optimizer_tuning.py](../scripts/run_coherent_momentum_optimizer_tuning.py)
- [../scripts/run_coherent_momentum_optimizer_benchmarks.py](../scripts/run_coherent_momentum_optimizer_benchmarks.py)
- [../scripts/run_coherent_momentum_optimizer_energy_tests.py](../scripts/run_coherent_momentum_optimizer_energy_tests.py)
- [../scripts/run_coherent_momentum_optimizer_ablation.py](../scripts/run_coherent_momentum_optimizer_ablation.py)
- [../scripts/export_coherent_momentum_optimizer_report.py](../scripts/export_coherent_momentum_optimizer_report.py)

Focused proof and credibility scripts are:

- [../scripts/run_directional_instability_benchmark.py](../scripts/run_directional_instability_benchmark.py)
- [../scripts/export_directional_instability_report.py](../scripts/export_directional_instability_report.py)
- [../scripts/run_cnn_credibility_benchmark.py](../scripts/run_cnn_credibility_benchmark.py)
- [../scripts/export_cnn_credibility_report.py](../scripts/export_cnn_credibility_report.py)
- [../scripts/demo_directional_instability.py](../scripts/demo_directional_instability.py)

GPU-oriented scripts are:

- [../scripts/run_coherent_momentum_gpu_smoke.py](../scripts/run_coherent_momentum_gpu_smoke.py)
- [../scripts/run_coherent_momentum_gpu_benchmarks.py](../scripts/run_coherent_momentum_gpu_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_cnn_benchmarks.py](../scripts/run_coherent_momentum_gpu_cnn_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_stress_benchmarks.py](../scripts/run_coherent_momentum_gpu_stress_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_multitask_benchmarks.py](../scripts/run_coherent_momentum_gpu_multitask_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_ablation.py](../scripts/run_coherent_momentum_gpu_ablation.py)
- [../scripts/export_coherent_momentum_gpu_report.py](../scripts/export_coherent_momentum_gpu_report.py)

## Configs

Mainline configs:

- [../configs/coherent_momentum_optimizer_default.yaml](../configs/coherent_momentum_optimizer_default.yaml)
- [../configs/coherent_momentum_optimizer_tuning.yaml](../configs/coherent_momentum_optimizer_tuning.yaml)
- [../configs/coherent_momentum_optimizer_energy.yaml](../configs/coherent_momentum_optimizer_energy.yaml)
- [../configs/coherent_momentum_optimizer_ablation.yaml](../configs/coherent_momentum_optimizer_ablation.yaml)

Focused benchmark configs:

- [../configs/directional_instability_benchmark.yaml](../configs/directional_instability_benchmark.yaml)
- [../configs/cnn_credibility_benchmark.yaml](../configs/cnn_credibility_benchmark.yaml)
- [../configs/presets/standard_safe.yaml](../configs/presets/standard_safe.yaml)
- [../configs/presets/stress_specialist.yaml](../configs/presets/stress_specialist.yaml)
- [../configs/presets/cnn_safe.yaml](../configs/presets/cnn_safe.yaml)

GPU configs:

- [../configs/coherent_momentum_gpu_smoke.yaml](../configs/coherent_momentum_gpu_smoke.yaml)
- [../configs/coherent_momentum_gpu_default.yaml](../configs/coherent_momentum_gpu_default.yaml)
- [../configs/coherent_momentum_gpu_cnn.yaml](../configs/coherent_momentum_gpu_cnn.yaml)
- [../configs/coherent_momentum_gpu_stress.yaml](../configs/coherent_momentum_gpu_stress.yaml)
- [../configs/coherent_momentum_gpu_multitask.yaml](../configs/coherent_momentum_gpu_multitask.yaml)
- [../configs/coherent_momentum_gpu_ablation.yaml](../configs/coherent_momentum_gpu_ablation.yaml)

## Tests

Focused repo-readiness tests:

- [../tests/test_coherent_momentum_optimizer.py](../tests/test_coherent_momentum_optimizer.py)
- [../tests/test_coherent_momentum_gpu_compatibility.py](../tests/test_coherent_momentum_gpu_compatibility.py)
- [../tests/test_coherent_momentum_benchmark_outputs.py](../tests/test_coherent_momentum_benchmark_outputs.py)
- [../tests/test_coherent_momentum_real_baseline.py](../tests/test_coherent_momentum_real_baseline.py)

These are the tests used as evidence that the repo is currently clone-and-run ready.

## Notebook and Example

- [../notebooks/coherent_momentum_full_eval.ipynb](../notebooks/coherent_momentum_full_eval.ipynb)
- [../examples/basic_usage.py](../examples/basic_usage.py)

The notebook exists to give a newcomer one place to run the tests, smoke commands, focused benchmarks, and report exports inside Colab or a local Jupyter environment.

## Reports

The report families in this repository are intentionally separated:

- [accepted_coherent_momentum](accepted_coherent_momentum)
  - accepted historical mainline snapshot
- [coherent_momentum_mainline](coherent_momentum_mainline)
  - current runnable mainline output directory
- [coherent_momentum_gpu](coherent_momentum_gpu)
  - GPU-capability and improved-branch report family
- [directional_instability](directional_instability)
  - newcomer-facing narrow-claim benchmark
- [cnn_credibility](cnn_credibility)
  - honest CNN gap report
- [demo_directional_instability](demo_directional_instability)
  - small reproducible niche demo
- [reference_real_baseline](reference_real_baseline)
  - real baseline reference line

The report-index document for those folders is [README.md](README.md).

## Current Limit

This repository is now organized around one public optimizer identity, but it still contains a broader comparison harness because the reports depend on it. That is intentional. The current repo is therefore best understood as a professional research optimizer repository with a focused public surface, not as a minimal single-file package.
