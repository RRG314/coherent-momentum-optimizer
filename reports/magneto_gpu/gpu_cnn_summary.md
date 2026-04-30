# Magneto GPU CNN Benchmarks

- Device: `cpu`
- Device name: `CPU`
- Rows: `63`

## Best Rows
| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_peak_device_memory_mb |
| --- | --- | --- | --- | --- | --- |
| digits_cnn | rmsprop | 0.2735 | 0.9166 | 4.8591 | nan |
| digits_cnn_label_noise | rmsprop | 0.5984 | 0.8665 | 4.9660 | nan |
| digits_cnn_input_noise | rmsprop | 0.4588 | 0.8417 | 4.9772 | nan |
| digits_cnn | adam | 0.8271 | 0.7429 | 5.2496 | nan |
| digits_cnn | adamw | 0.8265 | 0.7422 | 5.0899 | nan |
| digits_cnn_label_noise | adam | 1.0906 | 0.7298 | 5.1908 | nan |
| digits_cnn_label_noise | adamw | 1.0892 | 0.7260 | 5.1994 | nan |
| digits_cnn_input_noise | adamw | 1.0401 | 0.6786 | 5.3700 | nan |