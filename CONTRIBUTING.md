# Contributing

Thanks for working on the Magneto-Hamiltonian optimizer family.

## Scope

This repository is intentionally focused on:

- `HamiltonianAdamReal`
- `MagnetoHamiltonianAdam`
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
pytest tests/test_magneto_hamiltonian_adam.py tests/test_real_hamiltonian_adam.py -q
```

4. Run benchmark stages serially:

```bash
python scripts/run_magneto_hamiltonian_adam_smoke.py
python scripts/run_magneto_hamiltonian_adam_tuning.py --config configs/magneto_hamiltonian_adam_tuning.yaml
python scripts/run_magneto_hamiltonian_adam_benchmarks.py --config configs/magneto_hamiltonian_adam_default.yaml
python scripts/run_magneto_hamiltonian_adam_energy_tests.py --config configs/magneto_hamiltonian_adam_energy.yaml
python scripts/run_magneto_hamiltonian_adam_ablation.py --config configs/magneto_hamiltonian_adam_ablation.yaml
python scripts/export_magneto_hamiltonian_adam_report.py
```

Use the matching `real_hamiltonian` scripts for the base reference branch.

## Rules

- Keep the Hamiltonian core mathematically explicit.
- Keep the magneto layer separate from the Hamiltonian base in both code and reporting.
- Do not cite overloaded or interrupted runs.
- Do not claim broad superiority if the optimizer only wins in its specialist regimes.

## Pull request expectations

- State whether the change affects the real base, the magneto controller, or both
- Include any energy-drift impact
- Include any benchmark deltas and whether they are accepted or experimental
- Call out regressions directly
