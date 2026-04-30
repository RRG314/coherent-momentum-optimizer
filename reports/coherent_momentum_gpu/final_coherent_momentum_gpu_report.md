# Coherent Momentum Adam GPU Audit

Optimizer families mentioned below are referenced in [../../REFERENCES.md](../../REFERENCES.md). The public comparison notes for this repository are in [../../docs/COMPARISONS.md](../../docs/COMPARISONS.md).

## Related Work
- [AdamW](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
- [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [Deep Learning via Hamiltonian Monte Carlo](https://arxiv.org/abs/1206.1901)
- [PCGrad](https://arxiv.org/abs/2001.06782)
- [CAGrad](https://openreview.net/forum?id=_61Qh8tULj_)
- [Lion](https://arxiv.org/abs/2302.06675)

## Summary
- GPU-capable on available device(s): `yes`
- GPU smoke device: `mps` / `Apple MPS`
- Broad benchmark device(s): `cpu`
- Compatibility was verified on GPU-capable hardware. Broad quality and runtime comparisons were intentionally run on CPU so MPS was not used as a performance-claim platform.
- Current Magneto best row: `conflicting_batches_classification`
- Improved Magneto best row: `breast_cancer_mlp`
- Best AdamW row: `breast_cancer_mlp`
- Best RMSProp row: `breast_cancer_mlp`
- Best SGD momentum row: `wine_mlp`
- Broad task-winner count leader: `sgd_momentum` with `9` task wins

## Code Audit Findings
- The old Magneto path carried too much Python scalar control logic in the hot path, which risks host synchronization overhead on accelerator devices.
- Diagnostics were previously too eager. They now support `enable_step_diagnostics` and `diagnostics_every_n_steps` so high-overhead logging can be throttled or disabled.
- The optimizer was GPU-compatible in basic operation, but there was no dedicated device-transfer test coverage for state tensors, closure execution, or state_dict round-trips before this pass.
- Previous reports already suggested that conflict damping and extra activation gating were harming or neutral. The new audit kept those assumptions explicit and tested them rather than promoting them.

## Improvements Kept
- Tensor-based Magneto control computation on device to reduce per-step host synchronization.
- Diagnostics throttling through `diagnostics_every_n_steps` and `enable_step_diagnostics`.
- Soft conflict correction instead of heavy conflict damping.
- Conservative projection modes with conflict-only default.
- Conv-safe preset and typed conv update cap for CNN tasks.

## Improvements That Hurt Or Stayed Neutral
- Current ablation top row: `rmsprop_baseline`
- `improved_balanced` mean selection score: `1.1683` vs current `1.1549`
- Best improved preset in the ablation slice: `improved_cnn_safe` (`1.1778`)
- CNN-safe and standard-safe presets were the strongest improved presets in the limited ablation slice, but neither displaced RMSProp or SGD momentum overall.
- Conflict damping remained harmful enough that the improved branch uses softer correction rather than strong suppression.

## Direct Win Counts
| optimizer | baseline | wins | two_x |
| --- | --- | --- | --- |
| coherent_momentum_optimizer | adamw | 7 | 2 |
| coherent_momentum_optimizer | rmsprop | 5 | 0 |
| coherent_momentum_optimizer | sgd_momentum | 5 | 0 |
| coherent_momentum_optimizer | coherent_momentum_real_baseline | 8 | 2 |
| coherent_momentum_optimizer | coherent_direction_reference | 7 | 4 |
| coherent_momentum_optimizer | topological_adam | 7 | 3 |
| coherent_momentum_optimizer_improved | adamw | 7 | 3 |
| coherent_momentum_optimizer_improved | rmsprop | 7 | 0 |
| coherent_momentum_optimizer_improved | sgd_momentum | 5 | 0 |
| coherent_momentum_optimizer_improved | coherent_momentum_real_baseline | 9 | 3 |
| coherent_momentum_optimizer_improved | coherent_direction_reference | 7 | 6 |
| coherent_momentum_optimizer_improved | topological_adam | 7 | 3 |
| coherent_momentum_optimizer_improved | coherent_momentum_optimizer | 8 | 3 |

## Best Rows
| optimizer | task | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| coherent_momentum_optimizer_improved | breast_cancer_mlp | 0.1731 | 0.9531 | 10.5769 | nan |
| coherent_momentum_optimizer | conflicting_batches_classification | 0.1057 | 0.9761 | 5.0887 | nan |
| adamw | breast_cancer_mlp | 0.0724 | 0.9766 | 1.1522 | nan |
| rmsprop | breast_cancer_mlp | 0.0567 | 0.9883 | 0.9875 | nan |
| sgd_momentum | wine_mlp | 0.0563 | 0.9852 | 0.9293 | nan |

## CNN Snapshot
| optimizer | task | best_val_loss | best_val_accuracy | runtime_per_step_ms | optimizer_step_time_ms |
| --- | --- | --- | --- | --- | --- |
| coherent_momentum_optimizer | digits_cnn_label_noise | 2.2984 | 0.1723 | 13.7997 | 8.8728 |
| coherent_momentum_optimizer_improved | digits_cnn | 2.2960 | 0.1900 | 19.8271 | 14.9565 |
| adamw | digits_cnn | 0.6538 | 0.7884 | 5.0391 | 0.3944 |
| rmsprop | digits_cnn | 0.2207 | 0.9359 | 4.8711 | 0.1901 |
| sgd_momentum | digits_cnn | 0.6816 | 0.7846 | 4.8358 | 0.1184 |
| coherent_momentum_real_baseline | digits_cnn | 2.2946 | 0.1739 | 12.3951 | 7.5972 |
| coherent_direction_reference | digits_cnn | 0.7626 | 0.7753 | 7.0569 | 2.2672 |

## Stress Snapshot
| optimizer | task | best_val_loss | best_val_accuracy | runtime_per_step_ms | optimizer_step_time_ms |
| --- | --- | --- | --- | --- | --- |
| coherent_momentum_optimizer | saddle_objective | -4.1298 | nan | 1.2678 | 1.0393 |
| coherent_momentum_optimizer_improved | saddle_objective | -4.1322 | nan | 1.8533 | 1.6365 |
| adamw | saddle_objective | -2.6062 | nan | 0.2782 | 0.0908 |
| rmsprop | saddle_objective | -4.1322 | nan | 0.2378 | 0.0525 |
| sgd_momentum | saddle_objective | -4.1322 | nan | 0.2228 | 0.0404 |
| coherent_momentum_real_baseline | saddle_objective | -4.1028 | nan | 1.0033 | 0.7939 |
| coherent_direction_reference | saddle_objective | -3.1715 | nan | 0.5805 | 0.3785 |

## Multitask / Conflict Snapshot
| optimizer | task | best_val_loss | best_val_accuracy | runtime_per_step_ms | optimizer_step_time_ms |
| --- | --- | --- | --- | --- | --- |
| coherent_momentum_optimizer | conflicting_batches_classification | 0.0684 | 0.9844 | 5.0616 | 4.1910 |
| coherent_momentum_optimizer_improved | conflicting_batches_classification | 0.1090 | 0.9766 | 9.4455 | 8.5347 |
| adamw | conflicting_batches_classification | 0.0497 | 0.9883 | 1.0820 | 0.2866 |
| rmsprop | conflicting_batches_classification | 0.0474 | 0.9883 | 0.9056 | 0.1281 |
| sgd_momentum | conflicting_batches_classification | 0.0444 | 0.9883 | 0.8608 | 0.0793 |
| coherent_momentum_real_baseline | conflicting_batches_classification | 0.0748 | 0.9844 | 4.1682 | 3.3040 |
| coherent_direction_reference | conflicting_batches_classification | 0.0508 | 0.9883 | 2.6257 | 1.7209 |

## Best Optimizer Per Task
| task | best_optimizer | mean_best_val_loss | mean_best_val_accuracy |
| --- | --- | --- | --- |
| block_structure_classification | sgd_momentum | 0.3656 | 0.8464 |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 |
| conflicting_batches_classification | sgd_momentum | 0.0643 | 0.9792 |
| digits_cnn | rmsprop | 0.2793 | 0.9124 |
| digits_cnn_input_noise | rmsprop | 0.4588 | 0.8417 |
| digits_cnn_label_noise | rmsprop | 0.5984 | 0.8665 |
| direction_reversal_objective | sgd_momentum | 0.0017 | nan |
| narrow_valley_objective | sgd_momentum | 0.0000 | nan |
| noisy_quadratic_objective | sgd_momentum | -0.0487 | nan |
| oscillatory_valley | sgd_momentum | -0.0711 | nan |
| plateau_escape_objective | sgd_momentum | -1.0500 | nan |
| rosenbrock_valley | coherent_momentum_optimizer_improved | 0.0000 | nan |
| saddle_objective | sgd_momentum | -4.1322 | nan |
| small_batch_instability | rmsprop | 0.3853 | 0.8742 |
| wine_mlp | sgd_momentum | 0.0563 | 0.9852 |

## Runtime / Memory
| optimizer | mean_runtime_per_step_ms | mean_optimizer_step_time_ms | mean_samples_per_second | mean_optimizer_state_mb | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| topological_adam | 0.7873 | 0.4889 | 2637.7672 | 0.0045 | nan |
| sgd_momentum | 1.3887 | 0.0700 | 12238.3890 | 0.0068 | nan |
| rmsprop | 1.4352 | 0.1147 | 11464.4097 | 0.0068 | nan |
| adamw | 1.5732 | 0.2299 | 10027.5154 | 0.0135 | nan |
| adam | 1.7025 | 0.1850 | 7131.9074 | 0.0093 | nan |
| coherent_direction_reference | 1.9195 | 1.0713 | 4531.6902 | 0.0226 | nan |
| coherent_momentum_real_baseline | 4.4366 | 3.0581 | 3045.8466 | 0.0338 | nan |
| coherent_momentum_optimizer | 5.1188 | 3.7364 | 2577.9964 | 0.0405 | nan |
| coherent_momentum_optimizer_improved | 8.2828 | 6.8906 | 1540.6628 | 0.0405 | nan |

## Direct Answers
- Is Coherent Momentum GPU-capable? `yes`
- What code issues were found? Hot-path scalar control overhead, over-eager diagnostics, and missing explicit GPU/state transfer coverage were the main issues addressed.
- Which improvements helped? Device-safe tensor control computation, diagnostics throttling, and safer presets helped the implementation quality. In the ablation slice, simpler presets outperformed the heavier full-controller path.
- Which improvements hurt? Heavy conflict damping, extra activation gating, and globally forcing more controller logic remained neutral-to-harmful.
- Is it faster or slower than baselines? `slower` overall. Improved Magneto remained materially slower than AdamW, RMSProp, and SGD momentum on tabular and CNN tasks.
- Is memory acceptable? `yes` in the narrow sense: optimizer state stayed modest, but runtime overhead is still the practical limiter.
- Does improved Magneto beat AdamW anywhere? `yes`; the wins remain concentrated in directional synthetic stress tasks rather than broad practical tasks.
- Does improved Magneto beat RMSProp anywhere? `yes`; again as a specialist, not broadly.
- Does improved Magneto beat SGD momentum anywhere? `yes`; not broadly across standard MLP/CNN tasks.
- Does it improve CNN performance? `not enough`. The improved branch remained far behind AdamW, RMSProp, and SGD momentum on the CNN slice.
- Does it generalize better than AdamW on noisy/small/conflict tasks? `sometimes on synthetic directional stress tasks`, but the current evidence does not support a broad practical generalization claim.
- Does it remain a specialist or become broader? `specialist`.
- Best preset from this pass: `improved_cnn_safe`

## Honest Positioning
- This optimizer should be treated as a specialist unless the held-out noisy, conflict, and oscillation suites show broad wins against RMSProp or SGD momentum.
- MPS compatibility is verified here, but MPS should not be treated as a performance-claim platform.
- RMSProp and SGD momentum still dominate the broad CNN/tabular results in this repo.
- The improved branch is real and better on directional stress tasks than the current branch, but it is also slower and it does not close the practical CNN gap.
- Improved vs current direct result: `8` meaningful wins and `3` tracked 2x wins.

## Exact Next Step
- Make the improved branch simpler, not more complicated: promote a lighter standard-safe / no-projection default, keep projection as an optional stress preset, and stop pushing global controller complexity upward.
- If CNN strength matters, pursue a separate lighter conv-specific branch instead of trying to force the main Magneto specialist path to become a broad CNN optimizer.
