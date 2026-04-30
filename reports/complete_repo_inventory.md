# Complete Repository Inventory

Repository root: `/Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam`  
Prepared: `2026-04-29`

## Purpose

This report is a complete inventory of the current standalone Coherent Momentum / Magneto-Hamiltonian optimizer repository. It is meant to give one place to inspect:

- the public optimizer surface
- the internal implementation files
- scripts and configs
- tests
- reports and result artifacts
- notebook support
- install and reproduction files
- the current README content

This report does **not** claim that every file in the repo is equally public-facing. The repo still contains internal comparison and reference material used for reproducibility.

## Public Repository Identity

Public-facing optimizer name:

- `CoherentMomentumOptimizer`

Backward-compatible research names still exposed:

- `MagnetoHamiltonianAdam`
- `MagnetoHamiltonianAdamImproved`
- `HamiltonianAdamReal`

Public alias file:

- [coherent_momentum_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/coherent_momentum_optimizer.py>)

Primary implementation files:

- [magneto_hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/magneto_hamiltonian_adam.py>)
- [magneto_hamiltonian_adam_improved.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/magneto_hamiltonian_adam_improved.py>)
- [hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/hamiltonian_adam.py>)

## Top-Level Files

- [README.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/README.md>)
- [REPRODUCING.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REPRODUCING.md>)
- [pyproject.toml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/pyproject.toml>)
- [LICENSE](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/LICENSE>)
- [CITATION.cff](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/CITATION.cff>)
- [CONTRIBUTING.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/CONTRIBUTING.md>)
- [CHANGELOG.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/CHANGELOG.md>)
- [LICENSE.parent_workspace.txt](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/LICENSE.parent_workspace.txt>)

## Source Inventory

### Public and closely related optimizer files

- [src/optimizers/coherent_momentum_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/coherent_momentum_optimizer.py>)
- [src/optimizers/magneto_hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/magneto_hamiltonian_adam.py>)
- [src/optimizers/magneto_hamiltonian_adam_improved.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/magneto_hamiltonian_adam_improved.py>)
- [src/optimizers/hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/hamiltonian_adam.py>)
- [src/optimizers/magneto_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/magneto_adam.py>)
- [src/optimizers/base.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/base.py>)
- [src/optimizers/optimizer_utils.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/optimizer_utils.py>)
- [src/optimizers/diagnostics.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/diagnostics.py>)
- [src/optimizers/__init__.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/__init__.py>)

### Internal comparison / legacy research optimizer files still present

- [src/optimizers/topological_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/topological_adam.py>)
- [src/optimizers/sds_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/sds_adam.py>)
- [src/optimizers/unified_physics_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/unified_physics_adam.py>)
- [src/optimizers/thermodynamic_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/thermodynamic_adam.py>)
- [src/optimizers/diffusion_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/diffusion_adam.py>)
- [src/optimizers/uncertainty_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/uncertainty_adam.py>)
- [src/optimizers/direction_recovery_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/direction_recovery_optimizer.py>)
- [src/optimizers/observation_recovery_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/observation_recovery_optimizer.py>)
- [src/optimizers/constraint_consensus_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/constraint_consensus_optimizer.py>)
- block-direction families:
  - [block_direction_optimizer.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer.py>)
  - [block_direction_optimizer_v2.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v2.py>)
  - [block_direction_optimizer_v3.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v3.py>)
  - [block_direction_optimizer_v4_fast.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v4_fast.py>)
  - [block_direction_optimizer_v41.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v41.py>)
  - [block_direction_optimizer_v42.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v42.py>)
  - [block_direction_optimizer_v43.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v43.py>)
  - [block_direction_optimizer_v44.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v44.py>)
  - [block_direction_optimizer_v45.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizers/block_direction_optimizer_v45.py>)

### Research harness modules

- [src/optimizer_research/baselines.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/baselines.py>)
- [src/optimizer_research/benchmarking.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/benchmarking.py>)
- [src/optimizer_research/reporting.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/reporting.py>)
- [src/optimizer_research/tasks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/tasks.py>)
- [src/optimizer_research/config.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/config.py>)
- [src/optimizer_research/magneto_hamiltonian_suite.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/magneto_hamiltonian_suite.py>)
- [src/optimizer_research/magneto_gpu_suite.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/magneto_gpu_suite.py>)
- [src/optimizer_research/real_hamiltonian_suite.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/real_hamiltonian_suite.py>)

Internal historical suite modules still present:

- block-direction suites `v2` through `v45`
- [optimizer_strategy.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/optimizer_strategy.py>)
- [physics_reporting.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/physics_reporting.py>)
- [pinn_optimizer_suite.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/pinn_optimizer_suite.py>)
- [unified_physics_suite.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/src/optimizer_research/unified_physics_suite.py>)

## Script Inventory

### Public mainline and wrapper scripts

- [scripts/run_magneto_hamiltonian_adam_smoke.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_hamiltonian_adam_smoke.py>)
- [scripts/run_magneto_hamiltonian_adam_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_hamiltonian_adam_benchmarks.py>)
- [scripts/run_magneto_hamiltonian_adam_energy_tests.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_hamiltonian_adam_energy_tests.py>)
- [scripts/run_magneto_hamiltonian_adam_ablation.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_hamiltonian_adam_ablation.py>)
- [scripts/run_magneto_hamiltonian_adam_tuning.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_hamiltonian_adam_tuning.py>)
- [scripts/export_magneto_hamiltonian_adam_report.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/export_magneto_hamiltonian_adam_report.py>)

Wrapper names using the public alias:

- [scripts/run_coherent_momentum_optimizer_smoke.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_coherent_momentum_optimizer_smoke.py>)
- [scripts/run_coherent_momentum_optimizer_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_coherent_momentum_optimizer_benchmarks.py>)
- [scripts/run_coherent_momentum_optimizer_energy_tests.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_coherent_momentum_optimizer_energy_tests.py>)
- [scripts/run_coherent_momentum_optimizer_ablation.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_coherent_momentum_optimizer_ablation.py>)
- [scripts/export_coherent_momentum_optimizer_report.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/export_coherent_momentum_optimizer_report.py>)

### GPU and improved-branch scripts

- [scripts/run_magneto_gpu_smoke.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_smoke.py>)
- [scripts/run_magneto_gpu_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_benchmarks.py>)
- [scripts/run_magneto_gpu_cnn_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_cnn_benchmarks.py>)
- [scripts/run_magneto_gpu_stress_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_stress_benchmarks.py>)
- [scripts/run_magneto_gpu_multitask_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_multitask_benchmarks.py>)
- [scripts/run_magneto_gpu_ablation.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_magneto_gpu_ablation.py>)
- [scripts/export_magneto_gpu_report.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/export_magneto_gpu_report.py>)

### Reference real-Hamiltonian scripts retained for comparison

- [scripts/run_real_hamiltonian_adam_smoke.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_real_hamiltonian_adam_smoke.py>)
- [scripts/run_real_hamiltonian_adam_benchmarks.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_real_hamiltonian_adam_benchmarks.py>)
- [scripts/run_real_hamiltonian_adam_energy_tests.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_real_hamiltonian_adam_energy_tests.py>)
- [scripts/run_real_hamiltonian_adam_ablation.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/run_real_hamiltonian_adam_ablation.py>)
- [scripts/export_real_hamiltonian_adam_report.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/export_real_hamiltonian_adam_report.py>)

### Support script

- [scripts/generate_colab_notebook.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/scripts/generate_colab_notebook.py>)

## Config Inventory

Mainline:

- [configs/magneto_hamiltonian_adam_default.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_hamiltonian_adam_default.yaml>)
- [configs/magneto_hamiltonian_adam_tuning.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_hamiltonian_adam_tuning.yaml>)
- [configs/magneto_hamiltonian_adam_energy.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_hamiltonian_adam_energy.yaml>)
- [configs/magneto_hamiltonian_adam_ablation.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_hamiltonian_adam_ablation.yaml>)

GPU / improved branch:

- [configs/magneto_gpu_smoke.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_smoke.yaml>)
- [configs/magneto_gpu_default.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_default.yaml>)
- [configs/magneto_gpu_cnn.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_cnn.yaml>)
- [configs/magneto_gpu_stress.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_stress.yaml>)
- [configs/magneto_gpu_multitask.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_multitask.yaml>)
- [configs/magneto_gpu_ablation.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/magneto_gpu_ablation.yaml>)

Reference real-Hamiltonian configs retained:

- [configs/real_hamiltonian_adam_default.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/real_hamiltonian_adam_default.yaml>)
- [configs/real_hamiltonian_adam_energy.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/real_hamiltonian_adam_energy.yaml>)
- [configs/real_hamiltonian_adam_ablation.yaml](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/configs/real_hamiltonian_adam_ablation.yaml>)

## Test Inventory

- [tests/test_magneto_hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/tests/test_magneto_hamiltonian_adam.py>)
  - imports, initialization, one-step behavior, alias coverage, state dict, diagnostics, report schema
- [tests/test_magneto_gpu_compatibility.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/tests/test_magneto_gpu_compatibility.py>)
  - GPU-like device path, device-local state tensors, CPU↔GPU state_dict load, no-NaN smoke, diagnostics throttling
- [tests/test_magneto_benchmark_outputs.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/tests/test_magneto_benchmark_outputs.py>)
  - smoke / benchmark / cnn / stress / multitask / ablation / export output schema
- [tests/test_real_hamiltonian_adam.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/tests/test_real_hamiltonian_adam.py>)
  - retained reference-branch tests

## Notebook and Example Inventory

- [notebooks/coherent_momentum_full_eval.ipynb](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/notebooks/coherent_momentum_full_eval.ipynb>)
  - installs the repo
  - runs the focused tests
  - runs mainline and GPU script flows
  - prints CSV outputs and markdown reports
  - displays figures
- [examples/basic_usage.py](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/examples/basic_usage.py>)
  - minimal public-alias usage example

## Documentation Inventory

- [README.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/README.md>)
- [REPRODUCING.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REPRODUCING.md>)
- [REFERENCES.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REFERENCES.md>)
- [docs/CLAIM.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/CLAIM.md>)
- [docs/COMPARISONS.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/COMPARISONS.md>)
- [docs/FAILURE_CASES.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/FAILURE_CASES.md>)
- [docs/METHOD.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/METHOD.md>)
- [docs/MODERN_BASELINES.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/MODERN_BASELINES.md>)
- [docs/REAL_BASELINE.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/docs/REAL_BASELINE.md>)
- [reports/README.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/README.md>)
- [reports/repo_readiness_audit.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/repo_readiness_audit.md>)
- [reports/repo_update_report.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/repo_update_report.md>)
- [reports/magneto_repo_audit/code_audit.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_repo_audit/code_audit.md>)
- [reports/magneto_repo_audit/improvement_plan.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_repo_audit/improvement_plan.md>)

## Report and Result Inventory

### Main accepted Magneto run

- [reports/accepted_magneto_hamiltonian/final_report.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/final_report.md>)
- [reports/accepted_magneto_hamiltonian/benchmark_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/benchmark_results.csv>)
- [reports/accepted_magneto_hamiltonian/energy_tests.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/energy_tests.csv>)
- [reports/accepted_magneto_hamiltonian/ablation_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/ablation_results.csv>)
- [reports/accepted_magneto_hamiltonian/tuning_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/tuning_results.csv>)
- [reports/accepted_magneto_hamiltonian/smoke_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/smoke_results.csv>)
- [reports/accepted_magneto_hamiltonian/best_by_task.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/best_by_task.csv>)
- figures in [reports/accepted_magneto_hamiltonian/figures](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/accepted_magneto_hamiltonian/figures>)

### GPU and improved-branch audit

- [reports/magneto_gpu/final_magneto_gpu_report.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/final_magneto_gpu_report.md>)
- [reports/magneto_gpu/gpu_smoke_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_smoke_results.csv>)
- [reports/magneto_gpu/gpu_benchmark_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_benchmark_results.csv>)
- [reports/magneto_gpu/gpu_cnn_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_cnn_results.csv>)
- [reports/magneto_gpu/gpu_stress_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_stress_results.csv>)
- [reports/magneto_gpu/gpu_multitask_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_multitask_results.csv>)
- [reports/magneto_gpu/gpu_ablation_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_ablation_results.csv>)
- [reports/magneto_gpu/runtime_memory_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/runtime_memory_results.csv>)
- [reports/magneto_gpu/best_by_task.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/best_by_task.csv>)
- [reports/magneto_gpu/win_flags.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/win_flags.csv>)
- summary markdowns:
  - [gpu_smoke_summary.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_smoke_summary.md>)
  - [gpu_benchmark_summary.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_benchmark_summary.md>)
  - [gpu_cnn_summary.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_cnn_summary.md>)
  - [gpu_stress_summary.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_stress_summary.md>)
  - [gpu_multitask_summary.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/gpu_multitask_summary.md>)
- figures in [reports/magneto_gpu/figures](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/magneto_gpu/figures>)

### Reference real-Hamiltonian run

- [reports/reference_real_hamiltonian/final_report.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/final_report.md>)
- [reports/reference_real_hamiltonian/benchmark_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/benchmark_results.csv>)
- [reports/reference_real_hamiltonian/energy_tests.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/energy_tests.csv>)
- [reports/reference_real_hamiltonian/ablation_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/ablation_results.csv>)
- [reports/reference_real_hamiltonian/tuning_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/tuning_results.csv>)
- [reports/reference_real_hamiltonian/smoke_results.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/smoke_results.csv>)
- [reports/reference_real_hamiltonian/best_by_task.csv](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/best_by_task.csv>)
- figures in [reports/reference_real_hamiltonian/figures](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/reference_real_hamiltonian/figures>)

### Older carried-forward inventory file

- [reports/fast_and_magneto_inventory_report.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/fast_and_magneto_inventory_report.md>)

This older report was copied from the wider workspace history and is not the primary readiness document for this repo. The current readiness and public-facing inventory docs are:

- [reports/repo_readiness_audit.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/repo_readiness_audit.md>)
- this file

## Install and Run Readiness

Current installation path:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Focused validation commands used in the current repo cleanup:

```bash
python -m compileall src
pytest tests/test_magneto_hamiltonian_adam.py -q
pytest tests/test_magneto_gpu_compatibility.py tests/test_magneto_benchmark_outputs.py -q
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/generate_colab_notebook.py
```

Observed results:

- `compileall`: passed
- `test_magneto_hamiltonian_adam.py`: `11 passed, 1 warning`
- `test_magneto_gpu_compatibility.py` + `test_magneto_benchmark_outputs.py`: `6 passed, 1 warning`
- smoke scripts: passed
- notebook generation: passed

## README Alignment

The current [README.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/README.md>) now matches the actual repository in these ways:

- exposes a real public alias: `CoherentMomentumOptimizer`
- keeps internal names for backward compatibility
- documents the actual install path from `pyproject.toml`
- lists working test commands
- lists working script entry points
- names the Colab notebook that now exists
- points to real report folders and CSVs
- states limitations honestly:
  - not universal
  - slower than the main practical baselines
  - CNN performance still under development

## README Appendix

The current README content is included here for one-document inspection.

---

## README.md Snapshot

```markdown
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

## Relation to Existing Optimizers

`CoherentMomentumOptimizer` sits in a different design space than the standard magnitude-rescaling families:

- `SGD` follows the raw gradient direction directly.
- `SGD+momentum` smooths the direction across steps.
- `RMSProp` rescales gradient magnitudes adaptively.
- `Adam` / `AdamW` combine momentum with adaptive per-parameter scaling.
- `CoherentMomentumOptimizer` explicitly measures directional coherence, conflict, and rotation, then adjusts the step when the direction becomes unstable.

Internally, the stable alias currently maps to `MagnetoHamiltonianAdam`, which depends on `HamiltonianAdamReal` for the physical base dynamics.

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

## Installation

This repo is installable with the included `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

The default dependency set covers the current included scripts:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `pytest`

No external dataset download is required for the default included tasks. The current CNN benchmarks use scikit-learn digits rather than torchvision datasets.

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

## Running Benchmarks

Stable mainline script names:

```bash
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_magneto_hamiltonian_adam_benchmarks.py --config configs/magneto_hamiltonian_adam_default.yaml
python scripts/run_magneto_hamiltonian_adam_energy_tests.py --config configs/magneto_hamiltonian_adam_energy.yaml
python scripts/run_magneto_hamiltonian_adam_ablation.py --config configs/magneto_hamiltonian_adam_ablation.yaml
python scripts/export_magneto_hamiltonian_adam_report.py
```

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

## Results

The current repo contains two main result families:

1. accepted mainline Magneto-Hamiltonian reports in `reports/accepted_magneto_hamiltonian/`
2. GPU compatibility + improved-branch audit reports in `reports/magneto_gpu/`

## Reports and Outputs

Mainline reports:

- `reports/accepted_magneto_hamiltonian/`

GPU and improved-branch audit:

- `reports/magneto_gpu/`

Real Hamiltonian reference:

- `reports/reference_real_hamiltonian/`

Repo readiness audit:

- `reports/repo_readiness_audit.md`

## Limitations

- This is **not** a universal optimizer replacement.
- `RMSProp` and `SGD+momentum` remain stronger on many standard tasks in the current included reports.
- CNN performance is still under development and currently trails the strongest baseline optimizers in this repo.
- The improved GPU-safe branch is slower than the simpler baselines.
- The repo still contains internal comparison modules beyond the public coherent momentum surface because they are used by the reproducibility harness.
```

## Final Note

This report is the complete repo inventory snapshot for the current state of the standalone Coherent Momentum / Magneto-Hamiltonian repository. For install and run readiness, pair it with:

- [README.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/README.md>)
- [REPRODUCING.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/REPRODUCING.md>)
- [reports/repo_readiness_audit.md](</Users/stevenreid/Documents/New project/repos/magneto-hamiltonian-adam/reports/repo_readiness_audit.md>)
