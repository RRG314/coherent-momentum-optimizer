# Coherent Momentum Optimizer: A Directional-Coherence Optimizer for Unstable Gradient Regimes

## Abstract

This draft documents the current narrow claim supported by the repository: Coherent Momentum Optimizer is a specialist optimizer for training regimes where the update direction becomes unreliable. The method builds on a real Hamiltonian momentum baseline and adds bounded directional controls driven by gradient-momentum coherence, force alignment, rotation, and conflict signals. The checked-in evidence supports a limited claim. The accepted historical benchmark line shows improvement over the real Hamiltonian baseline and over AdamW on selected stress-oriented tasks, while the focused directional-instability benchmark shows that the improved branch can beat AdamW on parts of the instability slice. The same report set does not support broad superiority over RMSProp or SGD with momentum, and the CNN credibility report remains negative. This draft therefore presents Coherent Momentum as a targeted optimizer family rather than as a default replacement for standard first-order methods.

## Introduction

Many practical optimizers assume that the gradient already points in a usable direction and focus on smoothing, scaling, or preconditioning that direction. This assumption works well in many settings, but it becomes fragile when optimization trajectories oscillate, reverse, or become inconsistent under noisy or conflicting minibatches. Coherent Momentum targets that failure mode directly. The public method asks whether the direction itself is coherent enough to trust before it applies bounded control to the underlying Hamiltonian step.

The repository is intentionally not making a universal optimizer claim. The accepted historical report still shows strong ordinary-task performance for RMSProp and SGD with momentum. The value of this branch therefore depends on a narrower question: can directional-coherence control help when the direction becomes unreliable?

## Related Work

The comparison burden in this repository is anchored to standard first-order baselines and nearby stability methods. SGD and momentum remain the cleanest raw-direction baselines (Bottou, 2010; Sutskever et al., 2013). RMSProp, Adam, and AdamW are the main adaptive baselines because they change magnitude and smoothing while still committing to one update direction (Hinton, 2012; Kingma and Ba, 2015; Loshchilov and Hutter, 2019). SGHMC, symplectic optimization, and Hamiltonian descent are the closest conceptual references for the real Hamiltonian substrate beneath Coherent Momentum (Chen et al., 2014; Maddox et al., 2018; Wilson et al., 2018). SAM, ASAM, PCGrad, and CAGrad matter as alternative stability or conflict-aware approaches, but they target a different control surface than the one used here.

## Method

Let `g_t` be the current gradient, `p_t` the Hamiltonian momentum, and `f_t = M_t^{-1} p_t` the current force direction under inverse mass `M_t^{-1}`. The real baseline produces a base step from position-momentum dynamics, friction, and optional energy correction. Coherent Momentum augments that base with directional observables:

- `c_t^{gm} = cos(g_t, p_t)`
- `c_t^{fm} = cos(f_t, p_t)`
- `c_t^{gg-1} = cos(g_t, g_{t-1})`
- `c_t^{uu-1} = cos(u_t, u_{t-1})`
- rotation and conflict scores derived from these alignments

These observables drive bounded control values for friction, alignment, and optional projection toward the force direction. The practical update is therefore still a Hamiltonian-momentum update, but with a controller that only intervenes when the direction appears unstable. The exact thresholds and preset balances remain empirical. The repository does not claim a formal proof that the current controller schedule is optimal.

## Experimental Setup

This draft uses only local checked-in artifacts:

- accepted historical mainline: `reports/accepted_coherent_momentum/benchmark_results.csv`
- focused directional-instability slice: `reports/directional_instability/benchmark_results.csv`
- improved/GPU audit: `reports/coherent_momentum_gpu/*.csv`
- CNN credibility check: `reports/cnn_credibility/benchmark_results.csv`

The accepted historical line covers a broader stress-inclusive suite with three seeds. The newcomer-facing directional-instability benchmark is intentionally smaller and easier to rerun. The improved/GPU audit is treated as an engineering and branch-follow-up report rather than as a replacement for the accepted historical line.

## Benchmarks

The accepted historical line includes ordinary tabular neural tasks, label-noise and conflicting-batch cases, and several explicit stress objectives. The focused directional benchmark isolates a smaller instability slice intended to match the narrow public claim. The CNN credibility benchmark remains separate because it is a failure check for the method, not a surface to hide.

## Results

The best accepted Coherent Momentum row in the historical benchmark is `breast_cancer_mlp` with mean best validation loss `0.079857` and mean best validation accuracy `0.980469`. The accepted historical line still shows `4` meaningful wins against AdamW, which is enough to keep the branch interesting. The focused newcomer-facing instability slice is stricter: the improved branch shows `2` meaningful wins against AdamW there, but `0` against RMSProp and `0` against SGD with momentum.

That result pattern matters. The repository supports the claim that directional-coherence control can help on selected instability slices. It does not support a claim that the branch broadly displaces the strongest simple baselines.

## Ablations

The accepted ablation report remains important because it distinguishes the core method from the controller details. Projection survives as a useful optional control. Heavy conflict damping does not. Extra activation gating does not earn default status as a paper-level claim. The improved branch should therefore be described as an engineering refinement, not as proof that every additional controller term matters.

## Runtime and Memory Costs

The improved branch remains materially slower than simpler baselines. In the checked-in runtime audit it averages `8.2828 ms` per step, against `1.4352 ms` for RMSProp and `1.3887 ms` for SGD with momentum. Optimizer-state memory remains modest in the local audit, but runtime overhead is still the main practical cost.

## Failure Cases

The repository should keep its failure cases visible. CNN performance remains weak in the current checked-in credibility benchmark. Ordinary clean supervised tasks often favor RMSProp or SGD with momentum. PINN-style closure-heavy workloads are still better described as failure checks than as win conditions for this method.

## Limitations

This draft is based on checked-in local artifacts rather than on a fresh paper-scale rerun of every suite. The focused directional benchmark is a representative slice, not the largest possible instability sweep. The improved branch is still slower than the practical baselines it is being compared against. The CNN gap remains open.

## Reproducibility Statement

The repository includes focused tests, example scripts, a `build_paper_artifacts.py` script that regenerates these paper tables and figures from local CSVs, and a `run_paper_smoke.py` script that checks imports, examples, focused tests, and artifact generation without rerunning the full benchmark suite.

## Conclusion

The strongest defensible reading of the current repository is narrow. Coherent Momentum is a directional-coherence optimizer for unstable gradient-direction regimes. It remains interesting because it improves on AdamW in selected instability slices and because its real Hamiltonian baseline keeps the design conceptually cleaner than a generic “Adam with more gates” story. It should still be treated as a specialist optimizer rather than as a broad default replacement.

## Figures and Tables

- Accepted historical summary: `paper/tables/accepted_mainline_summary.md`
- Focused directional benchmark: `paper/tables/directional_instability_summary.md`
- GPU runtime and memory summary: `paper/tables/gpu_runtime_memory_summary.md`
- CNN credibility summary: `paper/tables/cnn_credibility_summary.md`
- Ablation summary: `paper/tables/accepted_ablation_summary.md`
- Figures: `paper/figures/`

## References

- Bottou, Léon. “Large-Scale Machine Learning with Stochastic Gradient Descent.” 2010. <https://leon.bottou.org/papers/bottou-2010>
- Sutskever, Ilya, James Martens, George Dahl, and Geoffrey Hinton. “On the Importance of Initialization and Momentum in Deep Learning.” ICML 2013. <https://proceedings.mlr.press/v28/sutskever13.html>
- Hinton, Geoffrey. “Neural Networks for Machine Learning, Lecture 6e.” 2012. <https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf>
- Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” 2015. <https://arxiv.org/abs/1412.6980>
- Loshchilov, Ilya, and Frank Hutter. “Decoupled Weight Decay Regularization.” 2019. <https://arxiv.org/abs/1711.05101>
- Chen, Tianqi, Emily Fox, and Carlos Guestrin. “Stochastic Gradient Hamiltonian Monte Carlo.” 2014. <https://arxiv.org/abs/1402.4102>
- Maddox, William, et al. “On Symplectic Optimization.” 2018. <https://arxiv.org/abs/1802.03653>
- Wilson, Ashia C., et al. “Hamiltonian Descent Methods.” 2018. <https://arxiv.org/abs/1809.05042>
- Foret, Pierre, et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization.” 2021. <https://openreview.net/forum?id=6Tm1mposlrM>
- Yu, Tianhe, et al. “Gradient Surgery for Multi-Task Learning.” 2020. <https://arxiv.org/abs/2001.06782>
