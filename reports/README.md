# Reports

This repository contains several report families. They serve different purposes.

When a report mentions `AdamW`, `RMSProp`, `SGD+momentum`, `Lion`, `Muon`, `SAM`, `PCGrad`, or other comparison baselines, the canonical bibliography for those names is [../REFERENCES.md](../REFERENCES.md).

## Accepted Mainline Report

- `reports/accepted_coherent_momentum/`

This is the accepted CPU-oriented historical benchmark line for the stable Coherent Momentum branch and its comparison baselines.

## GPU and Improved-Branch Report

- `reports/coherent_momentum_gpu/`

This directory contains the GPU-capability audit, the improved branch comparisons, device/runtime statistics, and the broader specialist benchmark slices.

## Directional Instability Report

- `reports/directional_instability/`

This is the focused proof benchmark for the narrow repository claim: that Coherent Momentum Optimizer helps mainly when gradient direction is unreliable. The default config is a representative newcomer-facing slice rather than the largest possible instability sweep.

## CNN Credibility Report

- `reports/cnn_credibility/`

This report exists to test the known weak point of the optimizer family. If the optimizer still trails AdamW, RMSProp, or SGD with momentum on CNNs, this report should say so clearly.

## Directional Demo

- `reports/demo_directional_instability/`

This is a small reproducible example showing one unstable training regime and the optimizer traces used to interpret it.

## Real Hamiltonian Reference

- `reports/reference_real_baseline/`

This is the reference report for the real Hamiltonian baseline used throughout the comparison suites.

## Repo Readiness and Inventory

- `reports/repo_readiness_audit.md`
- `reports/complete_repo_inventory.md`

These documents describe clone-and-run readiness and the current repository contents.

The current documentation-hardening change log is in `reports/repo_update_report.md`.
