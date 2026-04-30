# Magneto GPU Stress Benchmarks

- Device: `cpu`
- Device name: `CPU`
- Rows: `216`

## Best Rows
| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| conflicting_batches_classification | sgd_momentum | 0.0643 | 0.9792 | 0.8735 | nan |
| conflicting_batches_classification | coherent_momentum_optimizer | 0.0983 | 0.9779 | 5.2254 | nan |
| conflicting_batches_classification | topological_adam | 0.0639 | 0.9753 | 2.6028 | nan |
| conflicting_batches_classification | adam | 0.0641 | 0.9753 | 1.0647 | nan |
| conflicting_batches_classification | adamw | 0.0641 | 0.9753 | 1.0851 | nan |
| conflicting_batches_classification | coherent_direction_reference | 0.0646 | 0.9753 | 2.5706 | nan |
| conflicting_batches_classification | rmsprop | 0.0805 | 0.9753 | 0.9154 | nan |
| conflicting_batches_classification | coherent_momentum_real_baseline | 0.1052 | 0.9727 | 4.1555 | nan |