# CNN Credibility Benchmark

Optimizer families mentioned below are referenced in [../../REFERENCES.md](../../REFERENCES.md). The intended conservative reading of these results is documented in [../../docs/FAILURE_CASES.md](../../docs/FAILURE_CASES.md).

## Benchmark Scope
- Tasks run: digits_cnn, digits_cnn_label_noise, digits_cnn_input_noise
- Skipped tasks: mnist_small_cnn, mnist_deeper_cnn, fashion_mnist_small_cnn, fashion_mnist_deeper_cnn
- Optimizers: magneto_hamiltonian_adam, magneto_hamiltonian_adam_improved, sgd_momentum, rmsprop, adamw
- Skipped optional optimizers: {"schedulefree_adamw": "optional dependency `schedulefree` is not installed"}

## Best By Task
| task | best_optimizer | mean_best_val_loss | mean_best_val_accuracy |
| --- | --- | --- | --- |
| digits_cnn | rmsprop | 0.3899 | 0.8741 |
| digits_cnn_input_noise | rmsprop | 0.5386 | 0.8123 |
| digits_cnn_label_noise | adamw | 1.2945 | 0.7131 |

## CMO vs Standard CNN Baselines
- Improved CMO wins vs AdamW: `0`
- Improved CMO wins vs RMSProp: `0`
- Improved CMO wins vs SGD momentum: `0`

## Honest Read
- This benchmark exists to test the known weak point of the optimizer family.
- If AdamW, RMSProp, or SGD momentum still dominate the CNN rows, the gap is still open.
- Digits and optional torchvision tasks should be read as credibility checks, not as a claim of broad vision strength.
