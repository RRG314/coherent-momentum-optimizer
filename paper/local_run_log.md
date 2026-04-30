# Local Run Log

This log records the local verification commands run for the paper-package hardening pass on 2026-04-30.

## Environment

- machine: local workstation
- execution mode: local only
- networked benchmark services: not used
- external APIs: not used
- accelerator note: existing repo GPU compatibility work targets CPU, MPS, and CUDA when available; this paper-package pass did not add remote or cloud execution

## Commands run

### Focused tests

```bash
../../.venv_research/bin/pytest tests/test_coherent_momentum_optimizer.py -q
```

Result:

- `11 passed, 1 warning in 13.52s`
- warning: existing `PytestConfigWarning` for `asyncio_mode`

```bash
../../.venv_research/bin/pytest tests/test_coherent_momentum_gpu_compatibility.py tests/test_coherent_momentum_benchmark_outputs.py -q
```

Result:

- `9 passed, 1 warning in 103.17s`
- warning: existing `PytestConfigWarning` for `asyncio_mode`

### Examples

```bash
../../.venv_research/bin/python examples/cmo_minimal_mlp.py
```

Result:

- best validation loss: `0.048177`
- best validation accuracy: `0.979021`

```bash
../../.venv_research/bin/python examples/cmo_directional_instability_demo.py
```

Result:

- Coherent Momentum best loss: `-0.015139`
- Coherent Momentum final loss: `8.222876`
- AdamW best/final loss: `0.027358`
- RMSProp best/final loss: `0.025961`
- SGD+momentum best loss: `0.041785`
- SGD+momentum final loss: `0.190435`
- saved plots:
  - `reports/demo_directional_instability/example_loss_curves.png`
  - `reports/demo_directional_instability/example_direction_diagnostics.png`

Note:

- this example was kept because it is honest about the branch behavior on a toy instability objective
- Coherent Momentum reaches the best observed loss in the run but does not stay there, which is exactly the kind of behavior that should remain visible in a paper draft

```bash
../../.venv_research/bin/python examples/cmo_compare_against_adamw.py
```

Result:

- Coherent Momentum best validation loss: `0.152535`
- Coherent Momentum best validation accuracy: `0.988304`
- AdamW best validation loss: `0.185778`
- AdamW best validation accuracy: `0.994152`

### Paper artifacts

```bash
../../.venv_research/bin/python scripts/build_paper_artifacts.py
```

Result:

- generated `paper/paper_results_summary.csv`
- generated `paper/paper_claims_audit.md`
- generated `paper/cmo_draft.md`
- generated `paper/tables/`
- generated `paper/figures/`

```bash
../../.venv_research/bin/python scripts/run_paper_smoke.py
```

Result:

- passed
- verified compile step, focused tests, examples, and paper artifact generation

## Commands not run

None of the required local verification commands were skipped during this pass.

## Notes

- The paper artifact build uses only checked-in local CSVs and markdown artifacts.
- No fresh full benchmark rerun was required for this pass.
- A separate documentation reconciliation pass was needed because some older accepted-historical prose counts did not match the checked-in benchmark CSV aggregation anymore. The paper package now follows the CSV-derived counts.
