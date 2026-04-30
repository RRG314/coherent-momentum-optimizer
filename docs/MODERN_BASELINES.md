# Modern Baselines

This repository keeps the coherent momentum / Magneto-Hamiltonian family as the public subject of study, but it also needs comparison points that reflect current optimizer practice rather than only older default baselines.

References for every baseline named below are collected in [../REFERENCES.md](../REFERENCES.md).

## Included Directly

### Lion

Lion matters because it is a compact modern first-order optimizer that changes update geometry in a way practitioners actually use. It is included locally in `src/optimizer_research/baselines.py` and is most useful here as a modern low-overhead comparison on stress and tabular tasks.

### Muon Hybrid

Muon matters because it is the closest modern structure-aware baseline to the question “should geometry be changed using matrix structure rather than only scalar adaptive moments?” In this repo it is exposed as `muon_hybrid`, and availability depends on whether the local PyTorch build includes `torch.optim.Muon`.

### AdaBelief

AdaBelief matters because it is close enough to Adam-family practice to test whether CMO is doing more than another variant of adaptive scaling. It is included directly as `adabelief`.

### SAM / ASAM

SAM and ASAM matter because they are strong modern stability baselines that improve training by changing the objective neighborhood rather than by only changing the raw direction rule. This repo includes `sam_adamw` and `asam_adamw` wrappers that work with the closure-aware harness, but they are slower and therefore not turned on in every default config.

## Optional

### Schedule-Free AdamW

Schedule-Free AdamW matters because it is a serious modern baseline for algorithmic efficiency. In this repo it is exposed through an optional adapter called `schedulefree_adamw`. If the package is missing, the suites skip it cleanly and record that fact instead of pretending the comparison was run.

## Documented But Not Wired Into The Default Harness

### PCGrad / CAGrad

PCGrad and CAGrad matter because they are the cleanest modern references for explicit gradient-conflict handling. They are documented here but not wired into the default scalar-loss harness. That is a harness limitation, not a claim that they are unimportant. The current repo conflict tasks use alternating or perturbed regimes rather than a true multi-loss interface, so a direct comparison would be misleading unless the harness is extended first.

## How To Enable Optional Baselines

`schedulefree_adamw` requires the external `schedulefree` package. `sam_adamw` and `asam_adamw` already work with the repo’s closure-aware harness and only need to be added to the optimizer list in the relevant config. `muon_hybrid` becomes available automatically when the local PyTorch build exposes `torch.optim.Muon`.

## Practical Interpretation

The modern baselines are here to pressure-test the repo’s narrow claim, not to make the benchmark table look more fashionable. If one of these baselines is unavailable, the suite should say so explicitly. If one of them wins, that is evidence the coherent-momentum story is narrower than the public description might otherwise imply.
