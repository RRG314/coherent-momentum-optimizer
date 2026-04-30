# Accepted Ablation Summary

Accepted historical ablation summary for the stable public line.

Source CSVs:
- `reports/accepted_coherent_momentum/ablation_results.csv`

| variant_name | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_selection_score | source_csv |
| --- | --- | --- | --- | --- | --- |
| rmsprop_baseline | -0.6688 | 0.9819 | 0.5789 | 1.1785 | reports/accepted_coherent_momentum/ablation_results.csv |
| no_activation_gating | -0.4602 | 0.9264 | 3.5093 | 1.1246 | reports/accepted_coherent_momentum/ablation_results.csv |
| projection_only_coherence | -0.4624 | 0.9297 | 3.1757 | 1.1242 | reports/accepted_coherent_momentum/ablation_results.csv |
| no_conflict_damping | -0.4640 | 0.9273 | 3.0996 | 1.1224 | reports/accepted_coherent_momentum/ablation_results.csv |
| combined_full | -0.4637 | 0.9273 | 3.1457 | 1.1222 | reports/accepted_coherent_momentum/ablation_results.csv |
| fixed_mass_combined | -0.4637 | 0.9273 | 3.0951 | 1.1222 | reports/accepted_coherent_momentum/ablation_results.csv |
| no_alignment_scaling | -0.4617 | 0.9273 | 3.1400 | 1.1212 | reports/accepted_coherent_momentum/ablation_results.csv |
| no_projection | -0.4441 | 0.9210 | 3.0560 | 1.0920 | reports/accepted_coherent_momentum/ablation_results.csv |
| lr_scale_only_coherence | -0.4312 | 0.9210 | 3.1052 | 1.0800 | reports/accepted_coherent_momentum/ablation_results.csv |
| damping_only_coherence | -0.4264 | 0.9185 | 3.1188 | 1.0681 | reports/accepted_coherent_momentum/ablation_results.csv |
| hamiltonian_real_baseline | -0.4088 | 0.9185 | 2.5504 | 1.0514 | reports/accepted_coherent_momentum/ablation_results.csv |
| direction_reference_baseline | -0.4679 | 0.9794 | 1.6626 | 0.9603 | reports/accepted_coherent_momentum/ablation_results.csv |
| adamw_baseline | -0.3207 | 0.9798 | 0.6855 | 0.8147 | reports/accepted_coherent_momentum/ablation_results.csv |
| topological_baseline | -0.3205 | 0.9798 | 1.4891 | 0.8145 | reports/accepted_coherent_momentum/ablation_results.csv |