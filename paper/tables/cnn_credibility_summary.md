# Cnn Credibility Summary

CNN credibility results kept visible as a failure check rather than folded into a broad average.

Source CSVs:
- `reports/cnn_credibility/benchmark_results.csv`

| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | source_csv |
| --- | --- | --- | --- | --- | --- |
| digits_cnn | adamw | 0.9871 | 0.7266 | 30.4979 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn | coherent_momentum_optimizer | 2.3021 | 0.1044 | 170.1648 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn | coherent_momentum_optimizer_improved | 2.3038 | 0.1044 | 229.6838 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn | rmsprop | 0.3899 | 0.8741 | 29.3537 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn | sgd_momentum | 2.1529 | 0.4769 | 28.3649 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_input_noise | adamw | 1.2809 | 0.5945 | 29.4792 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_input_noise | coherent_momentum_optimizer | 2.3024 | 0.1044 | 179.2481 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_input_noise | coherent_momentum_optimizer_improved | 2.3041 | 0.1044 | 230.8428 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_input_noise | rmsprop | 0.5386 | 0.8123 | 29.5133 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_input_noise | sgd_momentum | 2.1890 | 0.3024 | 28.4135 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_label_noise | adamw | 1.2945 | 0.7131 | 27.7910 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_label_noise | coherent_momentum_optimizer | 2.3028 | 0.1416 | 178.6485 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_label_noise | coherent_momentum_optimizer_improved | 2.3043 | 0.1320 | 233.0329 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_label_noise | rmsprop | 1.0668 | 0.6165 | 30.6802 | reports/cnn_credibility/benchmark_results.csv |
| digits_cnn_label_noise | sgd_momentum | 2.2167 | 0.2675 | 28.1065 | reports/cnn_credibility/benchmark_results.csv |