# Accepted Mainline Summary

Accepted historical Coherent Momentum rows aggregated from the checked-in mainline benchmark CSV.

Source CSVs:
- `reports/accepted_coherent_momentum/benchmark_results.csv`

| task | optimizer | task_family | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_optimizer_state_mb | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | adamw | neural | 0.0635 | 0.9844 | 1.1322 | 0.0297 | reports/accepted_coherent_momentum/benchmark_results.csv |
| breast_cancer_mlp | coherent_direction_reference | neural | 0.0635 | 0.9844 | 2.7112 | 0.0593 | reports/accepted_coherent_momentum/benchmark_results.csv |
| breast_cancer_mlp | coherent_momentum_optimizer | neural | 0.0799 | 0.9805 | 6.5261 | 0.0890 | reports/accepted_coherent_momentum/benchmark_results.csv |
| breast_cancer_mlp | coherent_momentum_real_baseline | neural | 0.1197 | 0.9635 | 4.2552 | 0.0742 | reports/accepted_coherent_momentum/benchmark_results.csv |
| breast_cancer_mlp | rmsprop | neural | 0.0567 | 0.9883 | 0.9737 | 0.0149 | reports/accepted_coherent_momentum/benchmark_results.csv |
| breast_cancer_mlp | sgd_momentum | neural | 0.0599 | 0.9857 | 0.9024 | 0.0148 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | adamw | stress | 0.0632 | 0.9766 | 1.0395 | 0.0180 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | coherent_direction_reference | stress | 0.0635 | 0.9753 | 2.4804 | 0.0360 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | coherent_momentum_optimizer | stress | 0.0623 | 0.9792 | 5.8449 | 0.0540 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | coherent_momentum_real_baseline | stress | 0.0857 | 0.9779 | 4.0743 | 0.0450 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | rmsprop | stress | 0.0643 | 0.9753 | 0.9214 | 0.0090 | reports/accepted_coherent_momentum/benchmark_results.csv |
| conflicting_batches_classification | sgd_momentum | stress | 0.0643 | 0.9792 | 0.8238 | 0.0090 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | adamw | stress | 2.4679 | nan | 0.2519 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | coherent_direction_reference | stress | 2.3902 | nan | 0.4014 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | coherent_momentum_optimizer | stress | 0.0083 | nan | 1.1669 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | coherent_momentum_real_baseline | stress | 0.2989 | nan | 1.0049 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | rmsprop | stress | 0.0308 | nan | 0.2184 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| direction_reversal_objective | sgd_momentum | stress | 0.0017 | nan | 0.2098 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | adamw | stress | -0.0646 | nan | 0.2841 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | coherent_direction_reference | stress | -0.0709 | nan | 0.6004 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | coherent_momentum_optimizer | stress | -0.0378 | nan | 1.3423 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | coherent_momentum_real_baseline | stress | 0.0106 | nan | 1.0190 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | rmsprop | stress | -0.0711 | nan | 0.2459 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| oscillatory_valley | sgd_momentum | stress | -0.0711 | nan | 0.2373 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | adamw | stress | 0.2158 | nan | 0.2523 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | coherent_direction_reference | stress | 0.1240 | nan | 0.6520 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | coherent_momentum_optimizer | stress | 0.0018 | nan | 1.1341 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | coherent_momentum_real_baseline | stress | 0.0589 | nan | 0.9844 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | rmsprop | stress | 0.0004 | nan | 0.2125 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| rosenbrock_valley | sgd_momentum | stress | inf | nan | 0.2098 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | adamw | stability | -2.3313 | nan | 0.2650 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | coherent_direction_reference | stability | -3.1017 | nan | 0.5691 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | coherent_momentum_optimizer | stability | -4.1079 | nan | 1.2646 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | coherent_momentum_real_baseline | stability | -3.9802 | nan | 0.9680 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | rmsprop | stability | -4.1322 | nan | 0.2278 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| saddle_objective | sgd_momentum | stability | -4.1322 | nan | 0.2123 | 0.0000 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | adamw | stability | 0.3953 | 0.8676 | 1.0228 | 0.0194 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | coherent_direction_reference | stability | 0.4017 | 0.8661 | 2.5528 | 0.0388 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | coherent_momentum_optimizer | stability | 0.3803 | 0.8624 | 5.4871 | 0.0583 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | coherent_momentum_real_baseline | stability | 0.3815 | 0.8673 | 4.2079 | 0.0485 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | rmsprop | stability | 0.3843 | 0.8742 | 0.8723 | 0.0097 | reports/accepted_coherent_momentum/benchmark_results.csv |
| small_batch_instability | sgd_momentum | stability | 0.3698 | 0.8754 | 0.7915 | 0.0097 | reports/accepted_coherent_momentum/benchmark_results.csv |