# Coherent Momentum Claims Audit

This file is the paper-facing claim check for the current repository state. It is intentionally narrower than a README overview.

## Supported claims

The checked-in reports support describing Coherent Momentum as a directional-coherence optimizer for unstable gradient-direction regimes. The strongest evidence comes from the accepted historical line in `reports/accepted_coherent_momentum/` and the focused newcomer-facing slice in `reports/directional_instability/`.

The accepted historical line still shows `4` meaningful wins over AdamW across its broader stress-inclusive suite, and the focused directional-instability slice shows `2` meaningful wins for the improved branch over AdamW on the narrower instability benchmark.

The checked-in GPU audit also supports a more limited engineering claim: the branch can be made device-safe, diagnostics can be throttled, and the improved branch can win more synthetic directional stress comparisons than the stable mainline without changing the public identity of the method.

## Claims the repository does not support

The repository does not support claiming that Coherent Momentum is a general replacement for RMSProp, SGD with momentum, or AdamW. The accepted historical line still shows RMSProp and SGD with momentum as stronger practical baselines on many ordinary tasks, and the focused directional-instability slice does not reverse that conclusion.

The repository also does not support presenting Coherent Momentum as a competitive CNN optimizer yet. The CNN credibility report remains visible precisely because the gap is still open.

## Failure cases that should remain visible

Clean classification, mainstream CNN training, and closure-friendly PINN-style problems are still failure checks for this repository, not public win conditions. The current docs should continue to say that directly.

## Unfair claims

It would be unfair to claim that the improved branch is the new default public optimizer, that the CNN problem is solved, or that the repo shows broad superiority over RMSProp or SGD with momentum. Those claims are not supported by the checked-in CSVs.

## Safe public wording

Coherent Momentum Optimizer is a specialist optimizer for unstable gradient-direction regimes. In the checked-in reports it improves on AdamW in selected instability slices, but it remains slower than simpler baselines and does not replace RMSProp, SGD with momentum, or mainstream CNN optimizers as a default choice.

## Runtime context

The improved branch currently averages `8.2828 ms` per step against `1.5732 ms` for AdamW in the checked-in runtime audit, which is why runtime cost remains part of the honest public story.

## Sources

- `reports/accepted_coherent_momentum/benchmark_results.csv`
- `reports/directional_instability/benchmark_results.csv`
- `reports/coherent_momentum_gpu/runtime_memory_results.csv`
- `reports/cnn_credibility/benchmark_results.csv`