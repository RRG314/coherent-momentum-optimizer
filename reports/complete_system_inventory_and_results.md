# Coherent Momentum Complete System, Inventory, and Results Report

Prepared: `2026-04-30`

This document is the all-in-one reference for the current Coherent Momentum Optimizer repository. It brings together the public system description, the repository inventory, the report and benchmark surface, and the current accepted and focused-result story in one place.

The public optimizer name in this repository is `CoherentMomentumOptimizer`. The repo also keeps the improved branch, the real baseline, and the directional reference branch because they are part of the benchmark story and are needed for reproducibility.

## 1. System Overview

The repository has one public optimizer identity and three closely related comparison branches. The public stable branch is `CoherentMomentumOptimizer`. The improved experimental branch is `CoherentMomentumOptimizerImproved`. The physical reference branch is `CoherentMomentumRealBaseline`. The lighter directional comparator is `CoherentDirectionReferenceOptimizer`.

The repo is organized in six practical layers.

The package layer in [../src/optimizers](../src/optimizers) contains the importable optimizer implementations. The research harness in [../src/optimizer_research](../src/optimizer_research) defines tasks, baselines, reporting, and benchmark suites. The scripts layer under [../scripts](../scripts) exposes smoke, benchmark, ablation, energy-test, GPU, focused-proof, CNN-credibility, and demo runs. The config layer under [../configs](../configs) fixes those evaluation surfaces. The notebook and example layer provides one Colab-oriented walkthrough and one lightweight usage example. The report layer stores accepted artifacts, GPU and improved-branch audits, focused proof reports, failure-case checks, and reference-baseline outputs.

## 2. Public Identity and Code Surface

The public import path is:

```python
from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer
```

The package export layer in [../src/optimizers/__init__.py](../src/optimizers/__init__.py) also exposes:

```python
from optimizers import (
    CoherentMomentumOptimizer,
    CoherentMomentumOptimizerImproved,
    CoherentMomentumRealBaseline,
    CoherentDirectionReferenceOptimizer,
)
```

The main implementation files are:

- [../src/optimizers/coherent_momentum_optimizer.py](../src/optimizers/coherent_momentum_optimizer.py)
- [../src/optimizers/coherent_momentum_optimizer_improved.py](../src/optimizers/coherent_momentum_optimizer_improved.py)
- [../src/optimizers/coherent_momentum_real_baseline.py](../src/optimizers/coherent_momentum_real_baseline.py)
- [../src/optimizers/coherent_direction_reference.py](../src/optimizers/coherent_direction_reference.py)

The common support files most relevant to the optimizer path are:

- [../src/optimizers/base.py](../src/optimizers/base.py)
- [../src/optimizers/diagnostics.py](../src/optimizers/diagnostics.py)
- [../src/optimizers/optimizer_utils.py](../src/optimizers/optimizer_utils.py)

As with the BCDO repo, additional optimizer modules remain in the package tree because the broader research harness and historical comparison runs depend on them. They are part of the reproducibility surface, not the public identity of Coherent Momentum.

## 3. Research Harness Inventory

The main research-harness files are:

- [../src/optimizer_research/coherent_momentum_suite.py](../src/optimizer_research/coherent_momentum_suite.py)
- [../src/optimizer_research/coherent_momentum_gpu_suite.py](../src/optimizer_research/coherent_momentum_gpu_suite.py)
- [../src/optimizer_research/coherent_momentum_real_baseline_suite.py](../src/optimizer_research/coherent_momentum_real_baseline_suite.py)
- [../src/optimizer_research/directional_instability_suite.py](../src/optimizer_research/directional_instability_suite.py)
- [../src/optimizer_research/cnn_credibility_suite.py](../src/optimizer_research/cnn_credibility_suite.py)
- [../src/optimizer_research/tasks.py](../src/optimizer_research/tasks.py)
- [../src/optimizer_research/baselines.py](../src/optimizer_research/baselines.py)
- [../src/optimizer_research/benchmarking.py](../src/optimizer_research/benchmarking.py)
- [../src/optimizer_research/reporting.py](../src/optimizer_research/reporting.py)
- [../src/optimizer_research/physics_reporting.py](../src/optimizer_research/physics_reporting.py)
- [../src/optimizer_research/config.py](../src/optimizer_research/config.py)

This harness supports three different reading modes. The accepted historical mainline shows the stable branch against its baseline set. The focused directional-instability benchmark tests the narrow public claim. The GPU and CNN suites test practical gaps and compatibility claims.

## 4. Script Inventory

The mainline scripts are:

- [../scripts/run_coherent_momentum_optimizer_smoke.py](../scripts/run_coherent_momentum_optimizer_smoke.py)
- [../scripts/run_coherent_momentum_optimizer_tuning.py](../scripts/run_coherent_momentum_optimizer_tuning.py)
- [../scripts/run_coherent_momentum_optimizer_benchmarks.py](../scripts/run_coherent_momentum_optimizer_benchmarks.py)
- [../scripts/run_coherent_momentum_optimizer_energy_tests.py](../scripts/run_coherent_momentum_optimizer_energy_tests.py)
- [../scripts/run_coherent_momentum_optimizer_ablation.py](../scripts/run_coherent_momentum_optimizer_ablation.py)
- [../scripts/export_coherent_momentum_optimizer_report.py](../scripts/export_coherent_momentum_optimizer_report.py)

The real-baseline scripts are:

- [../scripts/run_coherent_momentum_real_baseline_smoke.py](../scripts/run_coherent_momentum_real_baseline_smoke.py)
- [../scripts/run_coherent_momentum_real_baseline_benchmarks.py](../scripts/run_coherent_momentum_real_baseline_benchmarks.py)
- [../scripts/run_coherent_momentum_real_baseline_energy_tests.py](../scripts/run_coherent_momentum_real_baseline_energy_tests.py)
- [../scripts/run_coherent_momentum_real_baseline_ablation.py](../scripts/run_coherent_momentum_real_baseline_ablation.py)
- [../scripts/export_coherent_momentum_real_baseline_report.py](../scripts/export_coherent_momentum_real_baseline_report.py)

The GPU and improved-branch scripts are:

- [../scripts/run_coherent_momentum_gpu_smoke.py](../scripts/run_coherent_momentum_gpu_smoke.py)
- [../scripts/run_coherent_momentum_gpu_benchmarks.py](../scripts/run_coherent_momentum_gpu_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_cnn_benchmarks.py](../scripts/run_coherent_momentum_gpu_cnn_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_stress_benchmarks.py](../scripts/run_coherent_momentum_gpu_stress_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_multitask_benchmarks.py](../scripts/run_coherent_momentum_gpu_multitask_benchmarks.py)
- [../scripts/run_coherent_momentum_gpu_ablation.py](../scripts/run_coherent_momentum_gpu_ablation.py)
- [../scripts/export_coherent_momentum_gpu_report.py](../scripts/export_coherent_momentum_gpu_report.py)

The focused proof and credibility scripts are:

- [../scripts/run_directional_instability_benchmark.py](../scripts/run_directional_instability_benchmark.py)
- [../scripts/export_directional_instability_report.py](../scripts/export_directional_instability_report.py)
- [../scripts/demo_directional_instability.py](../scripts/demo_directional_instability.py)
- [../scripts/run_cnn_credibility_benchmark.py](../scripts/run_cnn_credibility_benchmark.py)
- [../scripts/export_cnn_credibility_report.py](../scripts/export_cnn_credibility_report.py)

The paper-facing helper scripts are:

- [../scripts/build_paper_artifacts.py](../scripts/build_paper_artifacts.py)
- [../scripts/run_paper_smoke.py](../scripts/run_paper_smoke.py)

The notebook generator is:

- [../scripts/generate_colab_notebook.py](../scripts/generate_colab_notebook.py)

## 5. Config Inventory

The mainline configs are:

- [../configs/coherent_momentum_optimizer_default.yaml](../configs/coherent_momentum_optimizer_default.yaml)
- [../configs/coherent_momentum_optimizer_tuning.yaml](../configs/coherent_momentum_optimizer_tuning.yaml)
- [../configs/coherent_momentum_optimizer_energy.yaml](../configs/coherent_momentum_optimizer_energy.yaml)
- [../configs/coherent_momentum_optimizer_ablation.yaml](../configs/coherent_momentum_optimizer_ablation.yaml)

The real-baseline configs are:

- [../configs/coherent_momentum_real_baseline_default.yaml](../configs/coherent_momentum_real_baseline_default.yaml)
- [../configs/coherent_momentum_real_baseline_energy.yaml](../configs/coherent_momentum_real_baseline_energy.yaml)
- [../configs/coherent_momentum_real_baseline_ablation.yaml](../configs/coherent_momentum_real_baseline_ablation.yaml)

The GPU and improved-branch configs are:

- [../configs/coherent_momentum_gpu_smoke.yaml](../configs/coherent_momentum_gpu_smoke.yaml)
- [../configs/coherent_momentum_gpu_default.yaml](../configs/coherent_momentum_gpu_default.yaml)
- [../configs/coherent_momentum_gpu_cnn.yaml](../configs/coherent_momentum_gpu_cnn.yaml)
- [../configs/coherent_momentum_gpu_stress.yaml](../configs/coherent_momentum_gpu_stress.yaml)
- [../configs/coherent_momentum_gpu_multitask.yaml](../configs/coherent_momentum_gpu_multitask.yaml)
- [../configs/coherent_momentum_gpu_ablation.yaml](../configs/coherent_momentum_gpu_ablation.yaml)

The focused benchmark configs are:

- [../configs/directional_instability_benchmark.yaml](../configs/directional_instability_benchmark.yaml)
- [../configs/cnn_credibility_benchmark.yaml](../configs/cnn_credibility_benchmark.yaml)
- [../configs/presets/standard_safe.yaml](../configs/presets/standard_safe.yaml)
- [../configs/presets/stress_specialist.yaml](../configs/presets/stress_specialist.yaml)
- [../configs/presets/cnn_safe.yaml](../configs/presets/cnn_safe.yaml)

## 6. Test Inventory

The focused repo-readiness tests are:

- [../tests/test_coherent_momentum_optimizer.py](../tests/test_coherent_momentum_optimizer.py)
- [../tests/test_coherent_momentum_real_baseline.py](../tests/test_coherent_momentum_real_baseline.py)
- [../tests/test_coherent_momentum_gpu_compatibility.py](../tests/test_coherent_momentum_gpu_compatibility.py)
- [../tests/test_coherent_momentum_benchmark_outputs.py](../tests/test_coherent_momentum_benchmark_outputs.py)

These tests cover imports, initialization, parameter updates, no-NaN smoke behavior, state-dict behavior, diagnostics toggling, GPU-like compatibility when available, benchmark-output schemas, and export behavior.

## 7. Documentation, Notebook, and Example Inventory

The top-level package and documentation files that define the repo surface are:

- [../README.md](../README.md)
- [../REPRODUCING.md](../REPRODUCING.md)
- [../REFERENCES.md](../REFERENCES.md)
- [../ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md)
- [../LICENSE](../LICENSE)
- [../pyproject.toml](../pyproject.toml)
- [../CITATION.cff](../CITATION.cff)
- [../docs/CLAIM.md](../docs/CLAIM.md)
- [../docs/METHOD.md](../docs/METHOD.md)
- [../docs/COMPARISONS.md](../docs/COMPARISONS.md)
- [../docs/FAILURE_CASES.md](../docs/FAILURE_CASES.md)
- [../docs/MODERN_BASELINES.md](../docs/MODERN_BASELINES.md)
- [../docs/REAL_BASELINE.md](../docs/REAL_BASELINE.md)
- [repo_readiness_audit.md](repo_readiness_audit.md)
- [repo_update_report.md](repo_update_report.md)
- [complete_repo_inventory.md](complete_repo_inventory.md)
- [../paper/cmo_draft.md](../paper/cmo_draft.md)
- [../paper/paper_claims_audit.md](../paper/paper_claims_audit.md)

The notebook and example layer is:

- [../notebooks/coherent_momentum_full_eval.ipynb](../notebooks/coherent_momentum_full_eval.ipynb)
- [../examples/basic_usage.py](../examples/basic_usage.py)
- [../examples/cmo_minimal_mlp.py](../examples/cmo_minimal_mlp.py)
- [../examples/cmo_directional_instability_demo.py](../examples/cmo_directional_instability_demo.py)
- [../examples/cmo_compare_against_adamw.py](../examples/cmo_compare_against_adamw.py)

The bibliography and external optimizer references are centralized in [../REFERENCES.md](../REFERENCES.md). That file is the canonical reference index for the repository’s discussions of SGD, momentum, RMSProp, AdamW, Lion, Muon, SAM, ASAM, AdaBelief, Schedule-Free AdamW, PCGrad, CAGrad, and the other named external methods that appear in the docs and reports.

## 8. Report and Artifact Inventory

The checked-in report families are:

- [accepted_coherent_momentum](accepted_coherent_momentum)
- [coherent_momentum_gpu](coherent_momentum_gpu)
- [directional_instability](directional_instability)
- [cnn_credibility](cnn_credibility)
- [demo_directional_instability](demo_directional_instability)
- [reference_real_baseline](reference_real_baseline)
- [../paper](../paper)
- `coherent_momentum_mainline/` and `coherent_momentum_optimizer/` historical smoke/output folders that remain in the repo tree
- [coherent_momentum_repo_audit](coherent_momentum_repo_audit)

Across these folders, the artifact types include:

- `benchmark_results.csv`
- `best_by_task.csv`
- `win_flags.csv`
- `smoke_results.csv`
- `tuning_results.csv`
- `ablation_results.csv` or `gpu_ablation_results.csv`
- `energy_tests.csv`
- `runtime_memory_results.csv`
- `benchmark_metadata.json`
- `summary.csv`
- `final_report.md` or specialized final-report markdown
- `figures/`

The report families serve different purposes and should not be collapsed into one leaderboard. [accepted_coherent_momentum](accepted_coherent_momentum) is the accepted historical stable branch. [directional_instability](directional_instability) is the newcomer-facing narrow-claim proof slice. [coherent_momentum_gpu](coherent_momentum_gpu) is the GPU and improved-branch audit. [cnn_credibility](cnn_credibility) is the deliberate weakness check. [reference_real_baseline](reference_real_baseline) is the physical baseline reference line.

## 9. Accepted Historical Mainline Results

The accepted historical mainline source is [accepted_coherent_momentum](accepted_coherent_momentum), especially:

- [accepted_coherent_momentum/final_report.md](accepted_coherent_momentum/final_report.md)
- [accepted_coherent_momentum/benchmark_results.csv](accepted_coherent_momentum/benchmark_results.csv)
- [accepted_coherent_momentum/best_by_task.csv](accepted_coherent_momentum/best_by_task.csv)
- [accepted_coherent_momentum/ablation_results.csv](accepted_coherent_momentum/ablation_results.csv)
- [accepted_coherent_momentum/energy_tests.csv](accepted_coherent_momentum/energy_tests.csv)

The accepted mainline summary that still matters publicly is:

- meaningful wins vs `CoherentMomentumRealBaseline`: `9`
- meaningful wins vs `CoherentDirectionReferenceOptimizer`: `4`
- meaningful wins vs `AdamW`: `4`
- meaningful wins vs `RMSProp`: `3`
- meaningful wins vs `SGD+momentum`: `3`
- meaningful wins vs `TopologicalAdam`: `4`

The accepted best-row summary is:

- `breast_cancer_mlp`: mean best val loss `0.079857`, mean best val accuracy `0.980469`
- `conflicting_batches_classification`: mean best val loss `0.062339`, mean best val accuracy `0.979167`
- `direction_reversal_objective`: mean best val loss `0.008298`
- `rosenbrock_valley`: mean best val loss `0.001829`
- `saddle_objective`: mean best val loss `-4.107943`

The accepted mainline interpretation is still narrow. It supports Coherent Momentum as a useful custom specialist branch, but not as a broad default replacement for RMSProp or SGD with momentum.

## 10. Focused Directional-Instability Results

The narrow-claim proof source is [directional_instability](directional_instability), especially:

- [directional_instability/final_report.md](directional_instability/final_report.md)
- [directional_instability/benchmark_results.csv](directional_instability/benchmark_results.csv)
- [directional_instability/best_by_task.csv](directional_instability/best_by_task.csv)
- [directional_instability/win_flags.csv](directional_instability/win_flags.csv)

That benchmark is intentionally smaller and more direct than the accepted historical mainline. The tasks currently run in the newcomer-facing config are:

- `oscillatory_valley`
- `direction_reversal_objective`
- `small_batch_instability`

The focused comparison summary is:

- improved CMO meaningful wins vs `AdamW`: `2`
- improved CMO meaningful wins vs `RMSProp`: `0`
- improved CMO meaningful wins vs `SGD+momentum`: `0`

The current focused best-row summary is:

- CMO current best row: `small_batch_instability`, best val loss `0.562360`, best val accuracy `0.792111`
- CMO improved best row: `small_batch_instability`, best val loss `0.625783`, best val accuracy `0.713034`
- AdamW best row: `small_batch_instability`, best val loss `0.552155`, best val accuracy `0.788681`
- RMSProp best row: `small_batch_instability`, best val loss `0.391650`, best val accuracy `0.864329`
- SGD+momentum best row: `small_batch_instability`, best val loss `0.399740`, best val accuracy `0.876524`

This report family is the strongest narrow public proof story for the repo, and it still only supports a specialist claim. It does not support broad superiority over RMSProp or SGD with momentum.

## 11. GPU and Improved-Branch Results

The GPU and improved-branch source is [coherent_momentum_gpu](coherent_momentum_gpu), especially:

- [coherent_momentum_gpu/final_coherent_momentum_gpu_report.md](coherent_momentum_gpu/final_coherent_momentum_gpu_report.md)
- [coherent_momentum_gpu/gpu_benchmark_results.csv](coherent_momentum_gpu/gpu_benchmark_results.csv)
- [coherent_momentum_gpu/runtime_memory_results.csv](coherent_momentum_gpu/runtime_memory_results.csv)
- [coherent_momentum_gpu/gpu_cnn_results.csv](coherent_momentum_gpu/gpu_cnn_results.csv)
- [coherent_momentum_gpu/gpu_stress_results.csv](coherent_momentum_gpu/gpu_stress_results.csv)
- [coherent_momentum_gpu/gpu_multitask_results.csv](coherent_momentum_gpu/gpu_multitask_results.csv)
- [coherent_momentum_gpu/win_flags.csv](coherent_momentum_gpu/win_flags.csv)

The top-line GPU audit conclusion is that the optimizer family is GPU-capable, with compatibility verified on MPS in this environment, but the broad quality and runtime comparisons were intentionally kept CPU-based rather than turned into MPS performance claims.

The improved branch summary from that report is:

- improved branch wins vs current Coherent Momentum mainline: `8`
- improved branch wins vs `AdamW`: `7`
- improved branch wins vs `RMSProp`: `7`
- improved branch wins vs `SGD+momentum`: `5`

Those wins are concentrated in directional synthetic stress tasks. The same report also shows that the improved branch remains slower than AdamW, RMSProp, and SGD with momentum on broad tabular and CNN tasks.

The runtime and memory summary from [runtime_memory_results.csv](coherent_momentum_gpu/runtime_memory_results.csv) is:

- `CoherentMomentumOptimizer`: mean runtime per step `5.1188 ms`, mean optimizer state `0.0405 MB`
- `CoherentMomentumOptimizerImproved`: mean runtime per step `8.2828 ms`, mean optimizer state `0.0405 MB`
- `AdamW`: mean runtime per step `1.5732 ms`
- `RMSProp`: mean runtime per step `1.4352 ms`
- `SGD+momentum`: mean runtime per step `1.3887 ms`

That is a real performance cost and it should be treated as part of the public evaluation story.

## 12. CNN Credibility Results

The CNN weakness check lives in [cnn_credibility](cnn_credibility), especially:

- [cnn_credibility/final_report.md](cnn_credibility/final_report.md)
- [cnn_credibility/benchmark_results.csv](cnn_credibility/benchmark_results.csv)
- [cnn_credibility/best_by_task.csv](cnn_credibility/best_by_task.csv)

The current benchmark scope is a digits-based CNN slice:

- `digits_cnn`
- `digits_cnn_label_noise`
- `digits_cnn_input_noise`

The current outcome is straightforward:

- improved CMO wins vs `AdamW`: `0`
- improved CMO wins vs `RMSProp`: `0`
- improved CMO wins vs `SGD+momentum`: `0`

The best CNN task rows in this repo are still standard baselines:

- `RMSProp` wins `digits_cnn`
- `RMSProp` wins `digits_cnn_input_noise`
- `AdamW` wins `digits_cnn_label_noise`

This report family exists to keep the CNN gap visible. It should not be softened in the public interpretation of the repo.

## 13. Demo and Reference Baseline Reports

The small newcomer-facing demo lives in [demo_directional_instability](demo_directional_instability). It is useful for showing the niche clearly in one short run rather than across the full benchmark tree.

The physical reference baseline lives in [reference_real_baseline](reference_real_baseline). That report family matters because the public Coherent Momentum branch is supposed to be read as a directional-coherence controller layered on top of a visible real Hamiltonian baseline, not as an ungrounded new optimizer with no internal reference point.

## 14. Practical Interpretation

The repo now supports a clean and narrow public statement. Coherent Momentum is a specialist optimizer for unstable gradient-direction regimes. It has evidence of improvement over AdamW on selected instability slices and some stronger internal performance on focused stress runs. It is not a general replacement for RMSProp, SGD with momentum, or AdamW.

The public reading also has to keep the failure cases visible. RMSProp and SGD with momentum still dominate broad CNN and many standard-task rows. The improved branch remains slower than the simple baselines. The directional-instability benchmark is a representative proof slice, not a paper-scale exhaustive instability benchmark. The CNN path remains openly incomplete.

## 15. Reproduction Entry Points

The shortest reproducibility path is:

- install from [../pyproject.toml](../pyproject.toml)
- run the focused tests listed in [../README.md](../README.md)
- run [../scripts/run_coherent_momentum_optimizer_smoke.py](../scripts/run_coherent_momentum_optimizer_smoke.py)
- run [../scripts/run_directional_instability_benchmark.py](../scripts/run_directional_instability_benchmark.py) with [../configs/directional_instability_benchmark.yaml](../configs/directional_instability_benchmark.yaml)
- run [../scripts/export_directional_instability_report.py](../scripts/export_directional_instability_report.py)
- optionally run [../scripts/run_cnn_credibility_benchmark.py](../scripts/run_cnn_credibility_benchmark.py) with [../configs/cnn_credibility_benchmark.yaml](../configs/cnn_credibility_benchmark.yaml)

For the accepted historical line, read [accepted_coherent_momentum/final_report.md](accepted_coherent_momentum/final_report.md). For exact step-by-step reproduction commands, use [../REPRODUCING.md](../REPRODUCING.md).

## 16. Current Repository Status

The repository is clone-and-run ready for its present research scope. It has a stable public import path, a notebook, focused tests, runnable scripts, a narrow-claim document, a modern-baselines status document, a failure-cases document, a checked bibliography, accepted historical artifacts, GPU and improved-branch audits, a focused proof benchmark, and a CNN credibility benchmark.

The main remaining gaps are empirical rather than organizational. The optimizer is still slower than the strongest simple baselines. The CNN gap is still open. Some optional modern baselines remain dependency-gated or only partially integrated. The narrow claim is credible; a broad default-optimizer claim is not.
