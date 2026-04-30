# Magneto GPU Smoke

- Device: `mps`
- Device name: `Apple MPS`
- Rows: `20`

## Best Rows
| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | rmsprop | 0.0842 | 0.9844 | 17.8497 | 26.8750 |
| breast_cancer_mlp | adamw | 0.1426 | 0.9531 | 18.0778 | 26.8750 |
| breast_cancer_mlp | sgd_momentum | 0.1024 | 0.9471 | 16.7031 | 26.8750 |
| breast_cancer_mlp | coherent_momentum_optimizer | 0.5364 | 0.9375 | 121.1847 | 26.8281 |
| breast_cancer_mlp | coherent_momentum_optimizer_improved | 0.5335 | 0.9297 | 160.0305 | 26.8750 |
| digits_cnn | rmsprop | 0.3899 | 0.8741 | 25.9652 | 28.2500 |
| digits_cnn | adamw | 1.1660 | 0.6419 | 25.6466 | 28.2500 |
| digits_cnn | sgd_momentum | 2.1529 | 0.4769 | 24.2329 | 28.2500 |