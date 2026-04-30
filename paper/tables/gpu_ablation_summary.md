# Gpu Ablation Summary

Improved/GPU branch ablation summary. This is useful engineering context but does not replace the accepted historical line.

Source CSVs:
- `reports/coherent_momentum_gpu/gpu_ablation_results.csv`

| variant_name | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_selection_score | source_csv |
| --- | --- | --- | --- | --- | --- |
| rmsprop_baseline | -0.7057 | 0.9364 | 1.4185 | 1.3631 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| sgd_momentum_baseline | -0.6446 | 0.8918 | 1.3682 | 1.3333 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| no_projection | -0.2563 | 0.7023 | 8.6860 | 1.1813 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| improved_cnn_safe | -0.2598 | 0.6958 | 8.7091 | 1.1778 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| improved_standard_safe | -0.2598 | 0.6958 | 8.7275 | 1.1778 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| coherent_direction_reference_only | -0.4687 | 0.8839 | 2.7940 | 1.1712 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| improved_balanced | -0.2560 | 0.6804 | 8.6363 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| adaptive_mass | -0.2560 | 0.6804 | 8.6802 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| projection_under_conflict_only | -0.2560 | 0.6804 | 8.7052 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| diagnostics_throttled | -0.2560 | 0.6804 | 8.6669 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| diagnostics_full | -0.2560 | 0.6804 | 9.0629 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| conv_safe_on | -0.2560 | 0.6804 | 8.7946 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| conv_safe_off | -0.2560 | 0.6804 | 8.7589 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| fixed_mass | -0.2560 | 0.6804 | 7.9061 | 1.1683 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| no_activation_gating | -0.2697 | 0.6604 | 5.5216 | 1.1566 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| current_mainline_full | -0.2749 | 0.6604 | 5.5108 | 1.1549 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| no_conflict_damping | -0.2749 | 0.6604 | 5.2708 | 1.1548 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| improved_stress_specialist | -0.2520 | 0.6576 | 8.7671 | 1.1543 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| soft_conflict_correction | -0.2516 | 0.6576 | 8.7122 | 1.1542 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| coherent_momentum_real_baseline | -0.2156 | 0.6700 | 4.5166 | 1.0895 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |
| adamw_baseline | -0.3547 | 0.8853 | 1.5947 | 1.0540 | reports/coherent_momentum_gpu/gpu_ablation_results.csv |