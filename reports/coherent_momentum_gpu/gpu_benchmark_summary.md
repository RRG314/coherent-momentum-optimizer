# Magneto GPU Benchmarks

- Device: `cpu`
- Device name: `CPU`
- Rows: `105`

## Best Rows
| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 | 0.9875 | nan |
| wine_mlp | rmsprop | 0.0454 | 0.9852 | 0.9894 | nan |
| wine_mlp | sgd_momentum | 0.0563 | 0.9852 | 0.9293 | nan |
| breast_cancer_mlp | sgd_momentum | 0.0645 | 0.9831 | 0.9147 | nan |
| breast_cancer_mlp | coherent_direction_reference | 0.0718 | 0.9818 | 2.7073 | nan |
| conflicting_batches_classification | sgd_momentum | 0.0643 | 0.9792 | 0.8519 | nan |
| breast_cancer_mlp | adamw | 0.0724 | 0.9766 | 1.1522 | nan |
| conflicting_batches_classification | adamw | 0.0641 | 0.9753 | 1.0959 | nan |