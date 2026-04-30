# Contributing

Thanks for working on the Coherent Momentum optimizer family.

## Scope

This repository is intentionally focused on:

- `CoherentMomentumRealBaseline`
- `CoherentMomentumOptimizer`
- the benchmark and reporting harness required to reproduce the accepted results

It should not become a catch-all optimizer collection.

## Development workflow

1. Create or activate a virtual environment.
2. Install editable dependencies:

```bash
pip install -e .[dev]
```

3. Run branch-local tests:

```bash
pytest tests/test_coherent_momentum_optimizer.py tests/test_coherent_momentum_real_baseline.py -q
```

4. Run benchmark stages serially:

```bash
python scripts/run_coherent_momentum_optimizer_smoke.py
python scripts/run_coherent_momentum_optimizer_tuning.py --config configs/coherent_momentum_optimizer_tuning.yaml
python scripts/run_coherent_momentum_optimizer_benchmarks.py --config configs/coherent_momentum_optimizer_default.yaml
python scripts/run_coherent_momentum_optimizer_energy_tests.py --config configs/coherent_momentum_optimizer_energy.yaml
python scripts/run_coherent_momentum_optimizer_ablation.py --config configs/coherent_momentum_optimizer_ablation.yaml
python scripts/export_coherent_momentum_optimizer_report.py
```

Use the matching `real_baseline` scripts for the base reference branch.

## Rules

- Keep the Hamiltonian core mathematically explicit.
- Keep the coherence layer separate from the Hamiltonian base in both code and reporting.
- Do not cite overloaded or interrupted runs.
- Do not claim broad superiority if the optimizer only wins in its specialist regimes.

## Pull request expectations

- State whether the change affects the real base, the coherence controller, or both
- Include any energy-drift impact
- Include any benchmark deltas and whether they are accepted or experimental
- Call out regressions directly
