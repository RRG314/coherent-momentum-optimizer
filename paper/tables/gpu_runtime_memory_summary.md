# Gpu Runtime Memory Summary

Runtime and memory summary from the checked-in improved/GPU audit.

Source CSVs:
- `reports/coherent_momentum_gpu/runtime_memory_results.csv`

| optimizer | mean_runtime_per_step_ms | mean_optimizer_step_time_ms | mean_samples_per_second | mean_optimizer_state_mb | mean_peak_device_memory_mb | divergence_rate | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| topological_adam | 0.7873 | 0.4889 | 2637.7672 | 0.0045 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| sgd_momentum | 1.3887 | 0.0700 | 12238.3890 | 0.0068 | nan | 0.0667 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| rmsprop | 1.4352 | 0.1147 | 11464.4097 | 0.0068 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| adamw | 1.5732 | 0.2299 | 10027.5154 | 0.0135 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| adam | 1.7025 | 0.1850 | 7131.9074 | 0.0093 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| coherent_direction_reference | 1.9195 | 1.0713 | 4531.6902 | 0.0226 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| coherent_momentum_real_baseline | 4.4366 | 3.0581 | 3045.8466 | 0.0338 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| coherent_momentum_optimizer | 5.1188 | 3.7364 | 2577.9964 | 0.0405 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |
| coherent_momentum_optimizer_improved | 8.2828 | 6.8906 | 1540.6628 | 0.0405 | nan | 0.0000 | reports/coherent_momentum_gpu/runtime_memory_results.csv |