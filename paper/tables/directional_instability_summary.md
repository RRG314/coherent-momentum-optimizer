# Directional Instability Summary

Focused newcomer-facing directional-instability benchmark slice. This is the narrowest public proof benchmark in the repository.

Source CSVs:
- `reports/directional_instability/benchmark_results.csv`

| task | optimizer | task_family | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | source_csv |
| --- | --- | --- | --- | --- | --- | --- |
| direction_reversal_objective | adamw | stress | 2.2133 | nan | 6.2985 | reports/directional_instability/benchmark_results.csv |
| direction_reversal_objective | coherent_momentum_optimizer | stress | 0.0901 | nan | 22.8243 | reports/directional_instability/benchmark_results.csv |
| direction_reversal_objective | coherent_momentum_optimizer_improved | stress | 0.0432 | nan | 34.4516 | reports/directional_instability/benchmark_results.csv |
| direction_reversal_objective | rmsprop | stress | 0.0425 | nan | 6.0673 | reports/directional_instability/benchmark_results.csv |
| direction_reversal_objective | sgd_momentum | stress | 0.0417 | nan | 5.6911 | reports/directional_instability/benchmark_results.csv |
| oscillatory_valley | adamw | stress | 1.1521 | nan | 6.2576 | reports/directional_instability/benchmark_results.csv |
| oscillatory_valley | coherent_momentum_optimizer | stress | 0.0211 | nan | 24.2353 | reports/directional_instability/benchmark_results.csv |
| oscillatory_valley | coherent_momentum_optimizer_improved | stress | 0.0160 | nan | 33.3230 | reports/directional_instability/benchmark_results.csv |
| oscillatory_valley | rmsprop | stress | 0.0160 | nan | 3.9305 | reports/directional_instability/benchmark_results.csv |
| oscillatory_valley | sgd_momentum | stress | 0.0160 | nan | 4.4108 | reports/directional_instability/benchmark_results.csv |
| small_batch_instability | adamw | stability | 0.5522 | 0.7887 | 23.7783 | reports/directional_instability/benchmark_results.csv |
| small_batch_instability | coherent_momentum_optimizer | stability | 0.5624 | 0.7921 | 128.3584 | reports/directional_instability/benchmark_results.csv |
| small_batch_instability | coherent_momentum_optimizer_improved | stability | 0.6258 | 0.7130 | 181.4506 | reports/directional_instability/benchmark_results.csv |
| small_batch_instability | rmsprop | stability | 0.3916 | 0.8643 | 22.7385 | reports/directional_instability/benchmark_results.csv |
| small_batch_instability | sgd_momentum | stability | 0.3997 | 0.8765 | 22.4772 | reports/directional_instability/benchmark_results.csv |