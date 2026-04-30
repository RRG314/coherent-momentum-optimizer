# Magneto GPU Multitask Benchmarks

- Device: `cpu`
- Device name: `CPU`
- Rows: `63`

## Best Rows
| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| conflicting_batches_classification | sgd_momentum | 0.0643 | 0.9792 | 0.9901 | nan |
| conflicting_batches_classification | adamw | 0.0641 | 0.9753 | 1.0793 | nan |
| conflicting_batches_classification | coherent_direction_reference | 0.0646 | 0.9753 | 2.6042 | nan |
| conflicting_batches_classification | rmsprop | 0.0805 | 0.9753 | 0.9088 | nan |
| conflicting_batches_classification | coherent_momentum_optimizer | 0.1094 | 0.9753 | 5.0484 | nan |
| conflicting_batches_classification | coherent_momentum_real_baseline | 0.1184 | 0.9701 | 4.0989 | nan |
| conflicting_batches_classification | coherent_momentum_optimizer_improved | 0.1824 | 0.9518 | 9.3672 | nan |
| small_batch_instability | rmsprop | 0.3853 | 0.8742 | 0.9973 | nan |