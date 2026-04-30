# Reproducing Results

This repository is installable with `pyproject.toml` and can be evaluated without external dataset downloads for the default smoke, directional-instability, tabular, and digits-CNN tasks. Optional torchvision-backed CNN tasks require the `vision` extra.

If you are trying to understand what is being compared and why, read these alongside the commands below:

- [docs/CLAIM.md](docs/CLAIM.md)
- [docs/COMPARISONS.md](docs/COMPARISONS.md)
- [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md)
- [REFERENCES.md](REFERENCES.md)

If you are new to the repo, the shortest useful path is:

1. run the focused tests
2. run `python scripts/run_coherent_momentum_optimizer_smoke.py`
3. run the directional-instability benchmark and export
4. read `reports/directional_instability/final_report.md`

That path shows the narrow public claim first. The accepted historical line and the GPU/improved-branch audit should be read after that, not before.

## Environment

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

## Basic validation

Compile the package:

```bash
python -m compileall src
```

Run the focused optimizer tests:

```bash
pytest tests/test_coherent_momentum_optimizer.py -q
pytest tests/test_coherent_momentum_gpu_compatibility.py -q
pytest tests/test_coherent_momentum_benchmark_outputs.py -q
```

## Focused claim reproduction

Directional instability benchmark:

```bash
python scripts/run_directional_instability_benchmark.py --config configs/directional_instability_benchmark.yaml
python scripts/export_directional_instability_report.py
```

CNN credibility benchmark:

```bash
python scripts/run_cnn_credibility_benchmark.py --config configs/cnn_credibility_benchmark.yaml
python scripts/export_cnn_credibility_report.py
```

Directional demo:

```bash
python scripts/demo_directional_instability.py
```

## Mainline optimizer reproduction

Smoke test:

```bash
python scripts/run_coherent_momentum_optimizer_smoke.py
```

Benchmark summary:

```bash
python scripts/run_coherent_momentum_optimizer_benchmarks.py --config configs/coherent_momentum_optimizer_default.yaml
python scripts/run_coherent_momentum_optimizer_energy_tests.py --config configs/coherent_momentum_optimizer_energy.yaml
python scripts/run_coherent_momentum_optimizer_ablation.py --config configs/coherent_momentum_optimizer_ablation.yaml
python scripts/export_coherent_momentum_optimizer_report.py
```

If a fresh `reports/coherent_momentum_mainline/` benchmark directory has not been generated yet, the export wrapper falls back to `reports/accepted_coherent_momentum/`.

## GPU compatibility and specialist benchmark suite

GPU smoke:

```bash
python scripts/run_coherent_momentum_gpu_smoke.py
```

Full specialist suite:

```bash
python scripts/run_coherent_momentum_gpu_benchmarks.py --config configs/coherent_momentum_gpu_default.yaml
python scripts/run_coherent_momentum_gpu_cnn_benchmarks.py --config configs/coherent_momentum_gpu_cnn.yaml
python scripts/run_coherent_momentum_gpu_stress_benchmarks.py --config configs/coherent_momentum_gpu_stress.yaml
python scripts/run_coherent_momentum_gpu_multitask_benchmarks.py --config configs/coherent_momentum_gpu_multitask.yaml
python scripts/run_coherent_momentum_gpu_ablation.py --config configs/coherent_momentum_gpu_ablation.yaml
python scripts/export_coherent_momentum_gpu_report.py
```

## Output locations

- Mainline reports: `reports/accepted_coherent_momentum/`
- GPU audit and specialist reports: `reports/coherent_momentum_gpu/`
- Focused claim benchmark: `reports/directional_instability/`
- CNN credibility benchmark: `reports/cnn_credibility/`
- Directional demo: `reports/demo_directional_instability/`
- Real Hamiltonian reference run: `reports/reference_real_baseline/`
- Repo readiness audit: `reports/repo_readiness_audit.md`
- Repo update report: `reports/repo_update_report.md`
- Colab notebook: `notebooks/coherent_momentum_full_eval.ipynb`
- Modern baseline notes: `docs/MODERN_BASELINES.md`
- Narrow claim statement: `docs/CLAIM.md`
- Failure-case documentation: `docs/FAILURE_CASES.md`
- Full reference list: `REFERENCES.md`

## Notes

- The focused repo tests are the right readiness signal for this repository. Do not use unrelated workspace test failures as evidence against this repo.
- GPU compatibility is exercised on available accelerator hardware, but CPU is the reference platform for the benchmark claims currently stored in this repository.
- The newcomer-facing directional-instability config is intentionally lighter than a full paper-scale sweep. Increase the seed count and task list if you want a deeper audit.
- The accepted historical mainline report source remains `reports/accepted_coherent_momentum/`. The newer focused proof and GPU reports are complementary, not replacements for that accepted historical line.
