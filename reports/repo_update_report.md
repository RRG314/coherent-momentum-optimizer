# Repository Update Report

Date: `2026-04-30`  
Repository: `coherent-momentum-optimizer`

## What This Pass Changed

This pass was a publication-hardening pass, not a new optimizer-design pass. The goal was to make the repository read consistently as the Coherent Momentum Optimizer repo rather than as a partial rename of older internal work.

The work in this pass focused on four areas:

1. removing stale naming drift from public docs, reports, scripts, and internal benchmark identifiers where practical
2. making the bibliography and cross-optimizer comparison layer easier to trace
3. cleaning the report filenames and report index so the repository reads cleanly on GitHub
4. verifying the repo still compiles and passes its focused tests after the naming cleanup

## Concrete Changes

- Added a top-level acknowledgement file at [../ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md)
- Kept [../REFERENCES.md](../REFERENCES.md) as the canonical bibliography and linked the public docs back to it
- Cleaned [../README.md](../README.md) and [../REPRODUCING.md](../REPRODUCING.md) so they use repo-native paths instead of stale local absolute paths
- Cleaned [../src/optimizers/__init__.py](../src/optimizers/__init__.py) so the package exports are not duplicated
- Renamed the old GPU final-report filename to [coherent_momentum_gpu/final_coherent_momentum_gpu_report.md](coherent_momentum_gpu/final_coherent_momentum_gpu_report.md)
- Rewrote the repo inventory and readiness audit so they describe the current repo rather than deleted or stale files
- Updated the historical internal comparison key to `coherent_momentum_physical_baseline` across the active config and harness surface

## What This Pass Did Not Change

- It did not claim new wins
- It did not replace the accepted historical benchmark line
- It did not remove the real baseline from the repo
- It did not claim the CNN gap is solved
- It did not claim broad superiority over RMSProp, SGD with momentum, or AdamW

## Current Push Status

The local repository can now be validated and prepared for push, but the private GitHub destination still needs to be confirmed from the connected account context. The connected GitHub installation currently exposes a public repository named `RRG314/coherent-momentum-optimizer`, while the requested private target has not yet been identified through the available repo listing.

That means the repo cleanup and validation work can be completed locally, but the final push should only be done after the private repository target is confirmed explicitly.

## Suggested GitHub About text

Suggested repository name:

- `Coherent Momentum Optimizer`

Suggested short description for the GitHub About field:

- `A PyTorch optimizer for directional coherence in unstable gradient regimes, with focused benchmarks on oscillation, reversal, conflict, and noisy-gradient training settings.`
