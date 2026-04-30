# Focused And Branch Win Summary

Focused directional-instability, improved/GPU branch, and CNN credibility win summaries.

Source CSVs:
- `reports/directional_instability/benchmark_results.csv`
- `reports/coherent_momentum_gpu/gpu_benchmark_results.csv`
- `reports/cnn_credibility/benchmark_results.csv`

| benchmark_group | optimizer | baseline | meaningful_wins | two_x_wins | source_csv |
| --- | --- | --- | --- | --- | --- |
| directional_instability | coherent_momentum_optimizer | adamw | 2 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| directional_instability | coherent_momentum_optimizer | rmsprop | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| directional_instability | coherent_momentum_optimizer | sgd_momentum | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| directional_instability | coherent_momentum_optimizer_improved | adamw | 2 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| directional_instability | coherent_momentum_optimizer_improved | rmsprop | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| directional_instability | coherent_momentum_optimizer_improved | sgd_momentum | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| gpu_improved_branch | coherent_momentum_optimizer_improved | coherent_momentum_optimizer | 8 | 3 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| gpu_improved_branch | coherent_momentum_optimizer_improved | adamw | 7 | 3 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| gpu_improved_branch | coherent_momentum_optimizer_improved | rmsprop | 7 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| gpu_improved_branch | coherent_momentum_optimizer_improved | sgd_momentum | 5 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| cnn_credibility | coherent_momentum_optimizer_improved | adamw | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| cnn_credibility | coherent_momentum_optimizer_improved | rmsprop | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |
| cnn_credibility | coherent_momentum_optimizer_improved | sgd_momentum | 0 | 0 | reports/directional_instability/benchmark_results.csv | reports/coherent_momentum_gpu/gpu_benchmark_results.csv | reports/cnn_credibility/benchmark_results.csv |