# References

This repository compares Coherent Momentum Optimizer against standard adaptive, momentum-based, structure-aware, and conflict-aware baselines. The goal of this file is to keep those comparisons anchored to real sources instead of leaving optimizer names as uncited shorthand.

All external URLs in this file were checked during the private GitHub preparation pass on 2026-04-30. When a comparison baseline is repository-local rather than literature-defined, that is stated explicitly.

## How To Read This File

- If a document in this repo mentions `SGD`, `RMSProp`, `AdamW`, `Lion`, `Muon`, `SAM`, `PCGrad`, or another named optimizer family, this file is the canonical reference point.
- If a document discusses an internal comparison baseline such as `CoherentMomentumRealBaseline` or `CoherentDirectionReferenceOptimizer`, this file points back to the local implementation and the nearest external literature that gives the idea context.
- If a baseline is implementation-defined rather than paper-defined, the entry says so directly.

## Foundational First-Order Baselines

### SGD

- Léon Bottou, “Large-Scale Machine Learning with Stochastic Gradient Descent,” 2010.  
  URL: [https://leon.bottou.org/papers/bottou-2010](https://leon.bottou.org/papers/bottou-2010)
- Why it matters here: plain SGD is the cleanest raw-direction baseline. It provides the reference point for “follow the current gradient without extra directional control.”

### SGD With Momentum

- Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton, “On the importance of initialization and momentum in deep learning,” ICML 2013.  
  URL: [https://proceedings.mlr.press/v28/sutskever13.html](https://proceedings.mlr.press/v28/sutskever13.html)
- Why it matters here: momentum is the standard way to smooth direction over time. CMO must therefore justify itself relative to a mature and inexpensive directional smoother, not just relative to plain SGD.

### RMSProp

- Geoffrey Hinton, “Neural Networks for Machine Learning, Lecture 6e,” RMSProp lecture notes.  
  URL: [https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)
- Why it matters here: RMSProp is one of the strongest practical baselines in the included reports. It rescales magnitude adaptively without adding the extra directional controller that CMO uses.

### Adam

- Diederik P. Kingma, Jimmy Ba, “Adam: A Method for Stochastic Optimization,” 2014.  
  URL: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Why it matters here: Adam is the default adaptive-momentum baseline that most readers will mentally compare against when they see any new optimizer family.

### AdamW

- Ilya Loshchilov, Frank Hutter, “Decoupled Weight Decay Regularization,” 2017.  
  URL: [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- Why it matters here: AdamW is the repo’s main practical Adam-family baseline. Many claims in this repo reduce to whether CMO improves on AdamW when direction is unreliable.

## Modern Baselines Mentioned In The Repo

### Lion

- Chen et al., “Symbolic Discovery of Optimization Algorithms,” 2023.  
  URL: [https://arxiv.org/abs/2302.06675](https://arxiv.org/abs/2302.06675)
- Why it matters here: Lion is a compact modern optimizer that changes the update geometry through sign-based momentum. It is a useful pressure test for claims about low-overhead alternatives to AdamW.

### AdaBelief

- Juntang Zhuang et al., “AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients,” 2020.  
  URL: [https://arxiv.org/abs/2010.07468](https://arxiv.org/abs/2010.07468)
- Why it matters here: AdaBelief is close enough to Adam-family practice that it helps test whether CMO is doing anything beyond another adaptive-scaling variant.

### SAM

- Pierre Foret et al., “Sharpness-Aware Minimization for Efficiently Improving Generalization,” ICLR 2021.  
  URL: [https://openreview.net/forum?id=6Tm1mposlrM](https://openreview.net/forum?id=6Tm1mposlrM)
- Why it matters here: SAM is a strong stability/generalization comparison because it explicitly changes the training objective around local sharpness, not just the raw update rule.

### ASAM

- Jungmin Kwon et al., “ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks,” 2021.  
  URL: [https://arxiv.org/abs/2102.11600](https://arxiv.org/abs/2102.11600)
- Why it matters here: ASAM matters because it is a stronger scale-aware variant of SAM and a more credible robustness baseline than plain “turn on more damping.”

### Schedule-Free AdamW

- “The Road Less Scheduled,” schedule-free optimization reference used by this repo for optional comparison.  
  URL: [https://arxiv.org/abs/2405.15682](https://arxiv.org/abs/2405.15682)
- Why it matters here: schedule-free methods are a modern algorithmic-efficiency baseline. If they are installed locally, they should be treated as serious comparators rather than optional decoration.

### Muon

- PyTorch optimizer reference for `torch.optim.Muon`.  
  URL: [https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
- Why it matters here: Muon is the relevant modern structure-aware comparator for the repo’s “coherence under unstable direction” story. In this repo it is treated as an optional practical baseline because availability depends on the local PyTorch build.

### Shampoo

- Vineet Gupta et al., “Shampoo: Preconditioned Stochastic Tensor Optimization,” 2018.  
  URL: [https://arxiv.org/abs/1802.09568](https://arxiv.org/abs/1802.09568)
- Why it matters here: Shampoo is a canonical matrix/tensor-structured optimizer reference whenever the repo discusses structure-aware alternatives.

### K-FAC

- James Martens, Roger Grosse, “Optimizing Neural Networks with Kronecker-factored Approximate Curvature,” 2015.  
  URL: [https://arxiv.org/abs/1503.05671](https://arxiv.org/abs/1503.05671)
- Why it matters here: K-FAC is the most recognizable approximate-curvature baseline in this comparison family. It helps distinguish CMO from second-order or curvature-approximation claims that this repo is not making.

## Conflict-Aware and Multitask References

### PCGrad

- Tianhe Yu et al., “Gradient Surgery for Multi-Task Learning,” 2020.  
  URL: [https://arxiv.org/abs/2001.06782](https://arxiv.org/abs/2001.06782)
- Why it matters here: PCGrad is a real conflict-aware reference point whenever the repo talks about conflicting gradients or multitask directional disagreement.

### CAGrad

- Bo Liu et al., “Conflict-Averse Gradient Descent for Multi-task Learning,” NeurIPS 2021.  
  URL: [https://openreview.net/forum?id=_61Qh8tULj_](https://openreview.net/forum?id=_61Qh8tULj_)
- Why it matters here: CAGrad is a stricter conflict-aware comparator than informal damping heuristics. It matters for the repo’s conflict-task framing even though the default harness does not yet expose true multitask gradient interfaces.

## Physics And Dynamics References

### SGHMC

- Tianqi Chen, Emily Fox, Carlos Guestrin, “Stochastic Gradient Hamiltonian Monte Carlo,” 2014.  
  URL: [https://arxiv.org/abs/1402.4102](https://arxiv.org/abs/1402.4102)
- Why it matters here: SGHMC is the nearest canonical reference for Hamiltonian momentum with friction under stochastic gradients.

### Symplectic Optimization

- William Maddox et al., “On Symplectic Optimization,” 2018.  
  URL: [https://arxiv.org/abs/1802.03653](https://arxiv.org/abs/1802.03653)
- Why it matters here: this is the cleanest nearby reference for turning Hamiltonian ideas into discrete optimization updates instead of leaving them as analogy.

### Hamiltonian Descent

- Wilson, Mackey, Wibisono, “Hamiltonian Descent Methods,” 2018.  
  URL: [https://arxiv.org/abs/1809.05042](https://arxiv.org/abs/1809.05042)
- Why it matters here: this is the main theoretical reference for viewing optimization as dissipative Hamiltonian dynamics rather than as heuristic momentum tuning.

### Velocity-Regularized Adam

- “A Physics-Inspired Optimizer: Velocity Regularized Adam,” 2025.  
  URL: [https://arxiv.org/abs/2505.13196](https://arxiv.org/abs/2505.13196)
- Why it matters here: it occupies nearby design space by trying to stabilize adaptive optimization through explicitly physical control ideas.

## Closure-Oriented Baseline Mentioned In Failure Cases

### L-BFGS

- Dong C. Liu, Jorge Nocedal, “On the limited memory BFGS method for large scale optimization,” 1989.  
  URL: [https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf)
- Why it matters here: the repo’s failure-case documentation mentions L-BFGS in PINN-style settings, so it needs to be cited as a real closure-based alternative rather than a casual name-drop.

## Repository-Local Comparison Baselines

These are part of this repository’s internal comparison surface. They are not presented as outside literature claims.

### CoherentMomentumRealBaseline

- Implementation: [src/optimizers/coherent_momentum_real_baseline.py](src/optimizers/coherent_momentum_real_baseline.py)
- Explanation: [docs/REAL_BASELINE.md](docs/REAL_BASELINE.md)
- Context literature: SGHMC, Symplectic Optimization, and Hamiltonian Descent references above.

### CoherentDirectionReferenceOptimizer

- Implementation: [src/optimizers/coherent_direction_reference.py](src/optimizers/coherent_direction_reference.py)
- Why it matters here: it isolates a directional-coherence controller without the full real-Hamiltonian base, making it a useful internal ablation baseline.

### CoherentMomentumOptimizer

- Implementation: [src/optimizers/coherent_momentum_optimizer.py](src/optimizers/coherent_momentum_optimizer.py)
- Public alias surface: [src/optimizers/coherent_momentum_optimizer.py](src/optimizers/coherent_momentum_optimizer.py)
- Main accepted report: [reports/accepted_coherent_momentum/final_report.md](reports/accepted_coherent_momentum/final_report.md)

### CoherentMomentumOptimizerImproved

- Implementation: [src/optimizers/coherent_momentum_optimizer_improved.py](src/optimizers/coherent_momentum_optimizer_improved.py)
- GPU/improved-branch report: [reports/coherent_momentum_gpu/final_coherent_momentum_gpu_report.md](reports/coherent_momentum_gpu/final_coherent_momentum_gpu_report.md)
- Why it matters here: it keeps the same conceptual family while testing device-safe control computation, presets, and reduced-diagnostics behavior.

### TopologicalAdam

- Companion repository baseline: [https://github.com/RRG314/topological-adam](https://github.com/RRG314/topological-adam)
- Why it matters here: this repo uses Topological Adam as an external comparison baseline in several benchmark/report surfaces. It is a repository baseline rather than a separately cited paper in the current documentation set.
