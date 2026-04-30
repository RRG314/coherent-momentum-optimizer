# Directional Instability Benchmark

Optimizer names in this report map to the source list in [../../REFERENCES.md](../../REFERENCES.md). The narrow interpretation of this benchmark is defined in [../../docs/CLAIM.md](../../docs/CLAIM.md).

## Narrow Claim
Coherent Momentum Optimizer is a specialist optimizer for training regimes where gradient direction is unreliable: oscillation, reversal, conflict, noisy small-batch drift, or saddle and narrow-valley dynamics.

## Benchmark Scope
- Tasks: oscillatory_valley, direction_reversal_objective, small_batch_instability
- Optimizers: coherent_momentum_optimizer, coherent_momentum_optimizer_improved, sgd, sgd_momentum, rmsprop, adam, adamw, coherent_momentum_real_baseline, coherent_direction_reference
- Skipped optional optimizers: none

## Best By Task
| task | best_optimizer | mean_best_val_loss | mean_best_val_accuracy |
| --- | --- | --- | --- |
| direction_reversal_objective | sgd | 0.0417 | nan |
| oscillatory_valley | rmsprop | 0.0160 | nan |
| small_batch_instability | sgd_momentum | 0.3997 | 0.8765 |

## Best Rows
- CMO current: `small_batch_instability` | best val loss `0.562360` | best val acc `0.7921112775802612`
- CMO improved: `small_batch_instability` | best val loss `0.625783` | best val acc `0.713033527135849`
- AdamW: `small_batch_instability` | best val loss `0.552155` | best val acc `0.7886813879013062`
- RMSProp: `small_batch_instability` | best val loss `0.391650` | best val acc `0.8643292486667633`
- SGD momentum: `small_batch_instability` | best val loss `0.399740` | best val acc `0.8765243887901306`

## Focused Comparison
- Improved CMO meaningful wins vs AdamW: `2`
- Improved CMO meaningful wins vs RMSProp: `0`
- Improved CMO meaningful wins vs SGD momentum: `0`

## Honest Read
- This benchmark is the strongest narrow proof story for the optimizer family.
- If AdamW, RMSProp, or SGD momentum win more tasks here, that is a real result and should not be hidden.
- The optimizer should be described as a specialist unless it wins across these instability families broadly.
