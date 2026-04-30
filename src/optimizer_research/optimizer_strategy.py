from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmarking import run_benchmark_suite, run_stress_suite, run_tuning_suite
from .config import ensure_output_dir
from .reporting import aggregate_results, best_by_task, compute_meaningful_wins


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "optimizer_research"


@dataclass(slots=True)
class RepoAuditTarget:
    name: str
    path: Path
    optimizer_or_idea: str
    implemented: str
    conceptual_only: str
    promising: str
    weak_or_redundant: str
    literature_overlap: str
    contribution: str
    evidence_files: list[str]


@dataclass(slots=True)
class LiteratureEntry:
    family: str
    representative_method: str
    source_title: str
    source_url: str
    core_mechanism: str
    problem_solved: str
    assumptions: str
    where_wins: str
    where_fails: str
    compute_cost: str
    memory_cost: str
    overlap_with_local_ideas: str
    novelty_opening: str


@dataclass(slots=True)
class GapScore:
    gap_name: str
    summary: str
    novelty_potential: int
    implementation_difficulty: int
    benchmarkability: int
    expected_speed: int
    expected_memory_efficiency: int
    chance_vs_classic: int
    chance_vs_recent: int
    connection_to_local_work: int
    redundancy_risk: int


@dataclass(slots=True)
class CandidateDirection:
    candidate_name: str
    plain_idea: str
    mathematical_signal: str
    update_rule: str
    differs_from: str
    best_target_tasks: str
    likely_failure_modes: str
    compute_cost: str
    memory_cost: str
    novelty_potential: int
    implementation_difficulty: int
    benchmarkability: int
    expected_speed: int
    chance_vs_classic: int
    chance_vs_recent: int
    connection_to_local_work: int
    redundancy_risk: int
    prototype_name: str | None = None


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in frame.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("nan")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join("| " + " | ".join(items) + " |" for items in rows)


def _count_matching(path: Path, patterns: list[str]) -> int:
    total = 0
    for pattern in patterns:
        total += len(list(path.glob(pattern)))
    return total


def _repo_targets() -> list[RepoAuditTarget]:
    return [
        RepoAuditTarget(
            name="current_workspace_optimizer_suite",
            path=PROJECT_ROOT,
            optimizer_or_idea="Current Adam-family and physics/recoverability benchmark harness",
            implemented="Real PyTorch optimizers for SDS, Magneto, Thermodynamic, Diffusion, Hamiltonian (V1/V2/real), Magneto-Hamiltonian, UnifiedPhysics; honest benchmark and report tooling under src/optimizers and src/optimizer_research.",
            conceptual_only="A truly non-Adam optimizer direction had not existed yet; most prior work here stayed in controller-augmented Adam territory.",
            promising="Directional coherence consistently carried more signal than full controller stacking; Hamiltonian variants produced niche wins on oscillatory and saddle-style tasks.",
            weak_or_redundant="Unified stacking over-regularized and the full physics bundle failed to outperform strong simple baselines overall.",
            literature_overlap="High overlap with AdamW control variants, SGLD/SGHMC, entropy-based control, PCGrad/CAGrad-style alignment, and stability-aware optimization.",
            contribution="Strong benchmark harness and diagnostics foundation; useful evidence about which signals are worth keeping.",
            evidence_files=[
                "src/optimizers",
                "src/optimizer_research",
                "reports/real_hamiltonian_adam/final_report.md",
                "reports/magneto_hamiltonian_adam/final_report.md",
                "reports/unified_physics_adam/final_report.md",
            ],
        ),
        RepoAuditTarget(
            name="topological_adam",
            path=PROJECT_ROOT / "repos" / "topological-adam",
            optimizer_or_idea="Topology-inspired Adam correction and SDS-stability extensions",
            implemented="TopologicalAdam, TopologicalAdamV2, TopologicalAdamSDS, diagnostics, stopping helper, benchmark harness, and tests.",
            conceptual_only="Strong novelty claims remain unproven; docs explicitly avoid claiming benchmark-level superiority.",
            promising="Useful diagnostics language, bounded auxiliary correction, and careful falsification discipline.",
            weak_or_redundant="Benchmark evidence is modest; much of the mechanism still behaves like Adam plus a small correction field.",
            literature_overlap="High overlap with adaptive-gradient control, stability heuristics, and regularized Adam-family variants.",
            contribution="Good baseline and a source of bounded-correction ideas; not the clearest path to a genuinely new optimizer.",
            evidence_files=[
                "README.md",
                "docs/results.md",
                "TOPOLOGICAL_ADAM_FINAL_REPORT.md",
            ],
        ),
        RepoAuditTarget(
            name="tadam_repo",
            path=PROJECT_ROOT / "repos" / "rdt-spatial-index" / "tadam_repo",
            optimizer_or_idea="Tiered Adam with RDT/RGE transition damping",
            implemented="Torch, NumPy, and PINN variants plus C backend, synthetic/real-data/RNN/adversarial/PINN benchmarks, and documentation.",
            conceptual_only="The spatial-index origin story is broader than the optimizer itself; the optimizer remains an Adam-family specialization.",
            promising="Transition-aware damping on heavy-tail and noisy regimes is concrete and benchmarked.",
            weak_or_redundant="Still very close to Adam with extra per-parameter structure; novelty risk is high if reused directly.",
            literature_overlap="Adaptive damping, gradient clipping/transition control, and noise-robust Adam variants.",
            contribution="Can inform event-gated damping or trust shifts, but is not the cleanest foundation for a novel direction.",
            evidence_files=[
                "README.md",
                "docs/THEORY.md",
                "docs/WHEN_TO_USE.md",
            ],
        ),
        RepoAuditTarget(
            name="rdt_optimizer_search",
            path=PROJECT_ROOT / "rdt-dual-track-research",
            optimizer_or_idea="Recursive divide-score-prune search over continuous, discrete, and hyperparameter spaces",
            implemented="RDTSearch, RDTDiscreteSearch, RDTHyperparameterSearch, RDTOptimizerV2, RDT-BB, reports, and tests.",
            conceptual_only="Not a drop-in gradient optimizer for deep nets; it is primarily a search/controller family.",
            promising="Strongest local signal outside Adam-like work; especially good on discrete, mixed, noisy, and hyperparameter search settings.",
            weak_or_redundant="Continuous first-order optimization remains weak compared with dedicated gradient optimizers.",
            literature_overlap="Black-box optimization, branch-and-bound, tree search, CMA-ES/TPE-style search rather than Adam-style training.",
            contribution="Best path for outer-loop controller search or candidate-direction search, not per-step replacement of SGD/Adam.",
            evidence_files=[
                "README.md",
                "reports/optimizer_v2/final_optimizer_v2_report.md",
                "reports/rdt_bb/final_rdt_bb_report.md",
            ],
        ),
        RepoAuditTarget(
            name="rdt_ml_evaluation",
            path=PROJECT_ROOT / "rdt-ml-evaluation",
            optimizer_or_idea="Cross-domain evaluation of where RDT partitioning helps",
            implemented="Benchmark and reporting repo spanning optimization search and other domains.",
            conceptual_only="It evaluates where RDT matters rather than defining a train-time optimizer itself.",
            promising="Its domain ranking points directly to optimization_search as the strongest RDT application area.",
            weak_or_redundant="Does not by itself furnish a competitive train-time update rule.",
            literature_overlap="Meta-search and problem decomposition studies rather than optimizer-update literature.",
            contribution="Useful evidence for where to apply RDT ideas: outer-loop search, controller tuning, candidate selection.",
            evidence_files=[
                "README.md",
                "reports/final_domain_ranking.md",
            ],
        ),
        RepoAuditTarget(
            name="ocp_research_program",
            path=PROJECT_ROOT / "repos" / "ocp-research-program",
            optimizer_or_idea="Protected-state, recoverability, and constrained observation theory",
            implemented="Formal theorems, experiments, scripts, and reports around recoverability and observation constraints.",
            conceptual_only="No mature deep-learning optimizer lives here yet; the optimizer implication is indirect.",
            promising="Recoverability as a trust criterion under perturbation is a strong conceptual opening and is not just another Adam control knob.",
            weak_or_redundant="Without a practical low-cost scoring rule it can stay too abstract for training-time optimization.",
            literature_overlap="Touches observability, identifiability, robustness, and information-geometry themes more than optimizer mainstream.",
            contribution="Most useful source for a new optimizer principle: choose directions that survive perturbation and partial observation.",
            evidence_files=[
                "README.md",
                "docs/overview/main-contributions.md",
                "docs/restricted-results/strongest-paper-lane.md",
            ],
        ),
        RepoAuditTarget(
            name="arps",
            path=PROJECT_ROOT / "arps",
            optimizer_or_idea="Adaptive recursive partitioning and recoverability-style falsification",
            implemented="Source, scripts, tests, and reports around recursive partition scoring and benchmark evaluation.",
            conceptual_only="Not a gradient optimizer package; the optimizer relevance is via recoverability and robustness scoring.",
            promising="Provides a concrete perturb-and-score mindset that can be reused in a direction-trust optimizer.",
            weak_or_redundant="Too expensive or indirect for naive per-step use if lifted wholesale.",
            literature_overlap="Partition-based model selection and robustness scoring rather than first-order optimizer design.",
            contribution="Can inform a cheap recovery score instead of a full tree-search update rule.",
            evidence_files=[
                "README.md",
                "tests",
                "reports",
            ],
        ),
        RepoAuditTarget(
            name="mhd_toolkit",
            path=PROJECT_ROOT / "mhd-toolkit",
            optimizer_or_idea="Field/topology research with an experimental Topological Adam bridge",
            implemented="Toolkit, experiments, docs, and an explicit experimental bridge note to Topological Adam.",
            conceptual_only="Optimizer bridge remains secondary and explicitly experimental.",
            promising="Good discipline about separating metaphor from evidence; useful as a warning against overclaiming physics language.",
            weak_or_redundant="Not a path to competitive training by itself.",
            literature_overlap="Physics-inspired interpretation rather than new optimizer mechanics.",
            contribution="Helps keep novelty claims honest and grounded.",
            evidence_files=[
                "README.md",
                "docs/discoveries/topological_adam_bridge.md",
            ],
        ),
        RepoAuditTarget(
            name="rdt_kernel",
            path=PROJECT_ROOT / "repos" / "rdt-kernel",
            optimizer_or_idea="RDT diffusion kernel and nonlinear diffusion operators",
            implemented="Kernel package, tests, examples, and build artifacts.",
            conceptual_only="No competitive optimizer lives here; relevance is indirect through diffusion-style field evolution.",
            promising="Could inspire field-smoothing or multi-scale direction evolution.",
            weak_or_redundant="Far from a practical first-order optimizer without a much simpler reduction.",
            literature_overlap="Diffusion and PDE perspectives already overlap with SGLD/heat-flow style optimization ideas.",
            contribution="Low direct contribution right now.",
            evidence_files=["README.md", "tests"],
        ),
        RepoAuditTarget(
            name="rdt_noise",
            path=PROJECT_ROOT / "repos" / "rdt-noise",
            optimizer_or_idea="Noise-robust RDT experiments",
            implemented="Benchmarks, results, and tests around noise handling.",
            conceptual_only="No mainstream optimizer class here.",
            promising="Supports the theme that recursive structure can help in noisy, unstable regimes.",
            weak_or_redundant="Insufficient as a direct deep-learning optimizer direction on its own.",
            literature_overlap="Noise-robust search and partition methods.",
            contribution="Minor supporting evidence for a noise/recovery angle.",
            evidence_files=["benchmarks", "results", "tests"],
        ),
    ]


def generate_repo_audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for target in _repo_targets():
        if not target.path.exists():
            continue
        tests_count = _count_matching(target.path, ["tests/test_*.py", "tests/**/*.py", "tests/**/*.mjs"])
        results_count = _count_matching(target.path, ["reports/**/*.md", "reports/**/*.csv", "results/**/*.csv", "results/**/*.md"])
        rows.append(
            {
                "repo_or_module": target.name,
                "path": str(target.path.relative_to(PROJECT_ROOT)),
                "optimizer_or_idea": target.optimizer_or_idea,
                "implemented": target.implemented,
                "conceptual_only": target.conceptual_only,
                "tests_exist": tests_count,
                "results_exist": results_count,
                "promising": target.promising,
                "weak_or_redundant": target.weak_or_redundant,
                "literature_overlap": target.literature_overlap,
                "contribution_to_new_direction": target.contribution,
                "evidence_files": "; ".join(target.evidence_files),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "repo_audit.csv", index=False)
    lines = [
        "# Optimizer Repo Audit",
        "",
        "This audit focuses on optimizer-adjacent repos and modules in the local workspace. Counts below are filesystem counts, while the prose fields are evidence-based summaries from the listed docs and reports.",
        "",
        _markdown_table(frame[["repo_or_module", "path", "tests_exist", "results_exist", "optimizer_or_idea"]]),
        "",
    ]
    for target in _repo_targets():
        if not target.path.exists():
            continue
        lines.extend(
            [
                f"## {target.name}",
                f"1. Repo/module name: `{target.name}`",
                f"2. What idea it contains: {target.optimizer_or_idea}",
                f"3. What is actually implemented: {target.implemented}",
                f"4. What is only conceptual: {target.conceptual_only}",
                f"5. What tests exist: `{_count_matching(target.path, ['tests/test_*.py', 'tests/**/*.py', 'tests/**/*.mjs'])}` detected test files under `{target.path.relative_to(PROJECT_ROOT) / 'tests'}` where present.",
                f"6. What results exist: `{_count_matching(target.path, ['reports/**/*.md', 'reports/**/*.csv', 'results/**/*.csv', 'results/**/*.md'])}` report/result files detected. Key evidence: {', '.join(target.evidence_files)}.",
                f"7. What looks promising: {target.promising}",
                f"8. What looks weak or redundant: {target.weak_or_redundant}",
                f"9. Whether it overlaps with known literature: {target.literature_overlap}",
                f"10. Whether it can contribute to a new optimizer direction: {target.contribution}",
                "",
            ]
        )
    (output_path / "repo_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def _literature_entries() -> list[LiteratureEntry]:
    return [
        LiteratureEntry(
            family="Momentum / SGD family",
            representative_method="SGD, Polyak momentum, Nesterov",
            source_title="On the importance of initialization and momentum in deep learning",
            source_url="https://proceedings.mlr.press/v28/sutskever13.html",
            core_mechanism="Velocity accumulation in the gradient direction with low state cost.",
            problem_solved="Cheap and strong optimization on many supervised tasks, especially when schedules are tuned.",
            assumptions="Requires decent learning-rate schedules and can be sensitive to curvature and noise.",
            where_wins="Small and medium-scale supervised learning, noisy tasks, and regimes where simplicity matters.",
            where_fails="Poorly tuned schedules, anisotropic curvature, and tasks that benefit from per-parameter adaptivity.",
            compute_cost="Low",
            memory_cost="Low",
            overlap_with_local_ideas="Low direct overlap except with Hamiltonian-style momentum intuition.",
            novelty_opening="Little novelty headroom as a family; mostly a baseline to beat or hybridize.",
        ),
        LiteratureEntry(
            family="RMSProp family",
            representative_method="RMSProp",
            source_title="Neural Networks for Machine Learning Lecture 6e",
            source_url="https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf",
            core_mechanism="EMA of squared gradients to normalize coordinate-wise step sizes.",
            problem_solved="Curvature and scale mismatch without full second-order cost.",
            assumptions="Per-coordinate second-moment estimates are useful and stable.",
            where_wins="Small-data and noisy benchmarks; often stronger than expected on CPU-scale tasks.",
            where_fails="May generalize worse than SGD on some large tasks; still coordinate-wise and memory-heavy versus pure SGD.",
            compute_cost="Low",
            memory_cost="Medium",
            overlap_with_local_ideas="Explains why several local Hamiltonian variants still lost overall: RMSProp already smooths noisy force magnitudes well.",
            novelty_opening="Headroom exists mostly through directionality or structure, not another scalar-damping clone.",
        ),
        LiteratureEntry(
            family="Adam family",
            representative_method="Adam",
            source_title="Adam: A Method for Stochastic Optimization",
            source_url="https://arxiv.org/abs/1412.6980",
            core_mechanism="EMA of first and second gradient moments with coordinate-wise normalization.",
            problem_solved="Fast practical default optimizer with low tuning burden.",
            assumptions="Past gradient mean and variance are informative enough coordinate-wise.",
            where_wins="General-purpose training defaults and fast early optimization.",
            where_fails="Can generalize poorly versus SGD and has well-known instability/failure variants.",
            compute_cost="Low",
            memory_cost="Medium",
            overlap_with_local_ideas="Very high overlap with most local controller-augmented Adam variants.",
            novelty_opening="Low if the proposal still keeps Adam's two-moment logic intact.",
        ),
        LiteratureEntry(
            family="AdamW",
            representative_method="AdamW",
            source_title="Decoupled Weight Decay Regularization",
            source_url="https://arxiv.org/abs/1711.05101",
            core_mechanism="Adam plus decoupled weight decay rather than folding regularization into the moment update.",
            problem_solved="Stronger practical regularization behavior than vanilla Adam.",
            assumptions="Adam-like adaptivity remains desirable; weight decay should be separated from it.",
            where_wins="Practical default baseline across modern training pipelines.",
            where_fails="Still coordinate-wise and memory-heavy; not always best on noisy or small benchmarks.",
            compute_cost="Low",
            memory_cost="Medium",
            overlap_with_local_ideas="This is the main baseline every new adaptive optimizer here must survive.",
            novelty_opening="Novelty only if the new method is not basically AdamW plus extra gates.",
        ),
        LiteratureEntry(
            family="Adam refinements",
            representative_method="NAdam / RAdam / AdaBelief / Yogi",
            source_title="AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients",
            source_url="https://arxiv.org/abs/2010.07468",
            core_mechanism="Keep Adam-style state but change momentum or second-moment behavior to improve stability or generalization.",
            problem_solved="Known Adam pathologies such as variance rectification, surprise sensitivity, and unstable second-moment growth.",
            assumptions="Adam's coordinate-wise scaffold is still the right backbone.",
            where_wins="Drop-in practical variants where pure Adam is close but not ideal.",
            where_fails="Usually still bounded by Adam-family logic and memory cost.",
            compute_cost="Low",
            memory_cost="Medium",
            overlap_with_local_ideas="High overlap with uncertainty, entropy, and stability-control language used locally.",
            novelty_opening="Low unless the proposal departs from coordinate-wise moment adaptation.",
        ),
        LiteratureEntry(
            family="Sign / low-state adaptive methods",
            representative_method="Lion",
            source_title="Symbolic Discovery of Optimization Algorithms",
            source_url="https://arxiv.org/abs/2302.06675",
            core_mechanism="Update by sign of a momentum-like filtered gradient instead of storing full second moments.",
            problem_solved="Reduce memory while preserving strong practical training behavior.",
            assumptions="Sign information and light state can be enough when learning rates are tuned carefully.",
            where_wins="Memory-sensitive modern training and some vision/language workloads.",
            where_fails="Sensitive to tuning and not uniformly better on small noisy tabular tasks.",
            compute_cost="Low",
            memory_cost="Low",
            overlap_with_local_ideas="Supports the case for lighter-than-Adam state and more directional logic.",
            novelty_opening="Some headroom remains if the state is structured rather than per-coordinate.",
        ),
        LiteratureEntry(
            family="Second-order-influenced first-order methods",
            representative_method="Sophia",
            source_title="Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training",
            source_url="https://arxiv.org/abs/2305.14342",
            core_mechanism="Use cheap curvature estimates to scale updates more aggressively than Adam.",
            problem_solved="Improve training efficiency on very large language-model workloads.",
            assumptions="Hessian information can be approximated cheaply enough to matter.",
            where_wins="Large-scale pretraining regimes where curvature estimates are informative.",
            where_fails="More complexity and problem-specific tuning; weaker fit for tiny CPU benchmarks.",
            compute_cost="Medium",
            memory_cost="Medium",
            overlap_with_local_ideas="Low direct overlap; more curvature-centric than local physics-controller experiments.",
            novelty_opening="Less attractive for this workspace unless the goal shifts to large-model pretraining.",
        ),
        LiteratureEntry(
            family="Structured preconditioning",
            representative_method="Shampoo",
            source_title="Shampoo: Preconditioned Stochastic Tensor Optimization",
            source_url="https://arxiv.org/abs/1802.09568",
            core_mechanism="Tensor-wise matrix preconditioning instead of scalar per-coordinate variance scaling.",
            problem_solved="Exploit matrix/tensor structure and curvature more faithfully than Adam/RMSProp.",
            assumptions="Matrix-shaped parameters and extra linear algebra cost are acceptable.",
            where_wins="Large models and structured layers where richer preconditioning pays off.",
            where_fails="Heavy memory/compute for small CPU-scale experiments.",
            compute_cost="High",
            memory_cost="High",
            overlap_with_local_ideas="Strongly overlaps with structure-aware and Muon-like ambitions.",
            novelty_opening="Openings remain if structure is used for direction trust rather than full preconditioning.",
        ),
        LiteratureEntry(
            family="Natural-gradient approximations",
            representative_method="K-FAC",
            source_title="Optimizing Neural Networks with Kronecker-factored Approximate Curvature",
            source_url="https://proceedings.mlr.press/v37/martens15.html",
            core_mechanism="Approximate natural-gradient updates with Kronecker-factored curvature blocks.",
            problem_solved="Inject more faithful geometry than first-order methods.",
            assumptions="Layerwise curvature approximations are useful and affordable enough.",
            where_wins="Large networks and regimes where second-order structure matters.",
            where_fails="Engineering and memory complexity; not a cheap default.",
            compute_cost="High",
            memory_cost="High",
            overlap_with_local_ideas="Overlaps with structure-aware and matrix-aware ambitions but not with recovery-based trust.",
            novelty_opening="Better used as contrast than as a direct target.",
        ),
        LiteratureEntry(
            family="Outer-loop stabilizers",
            representative_method="Lookahead",
            source_title="Lookahead Optimizer: k steps forward, 1 step back",
            source_url="https://arxiv.org/abs/1907.08610",
            core_mechanism="Wrap a base optimizer with fast/slow weights to smooth optimization trajectories.",
            problem_solved="Stability and reduced sensitivity without inventing a full new base optimizer.",
            assumptions="Averaging between optimistic and conservative iterates is beneficial.",
            where_wins="As a low-risk wrapper around strong base optimizers.",
            where_fails="Still inherits the inner optimizer's logic and cost.",
            compute_cost="Low to Medium",
            memory_cost="Medium",
            overlap_with_local_ideas="Overlaps with the urge to stabilize dynamics without changing core gradient logic.",
            novelty_opening="Limited unless paired with a truly distinct inner update.",
        ),
        LiteratureEntry(
            family="Sharpness/stability-aware methods",
            representative_method="SAM / ASAM",
            source_title="Sharpness-Aware Minimization for Efficiently Improving Generalization",
            source_url="https://arxiv.org/abs/2010.01412",
            core_mechanism="Adversarially perturb weights before taking the actual update to prefer flat neighborhoods.",
            problem_solved="Generalization and sharp-minima sensitivity.",
            assumptions="One extra forward/backward pass is acceptable and flatness matters.",
            where_wins="Generalization-sensitive training, especially larger models.",
            where_fails="Higher step cost; less attractive for cheap CPU loops.",
            compute_cost="High",
            memory_cost="Medium",
            overlap_with_local_ideas="Closest overlap with recovery/perturbation ideas, but SAM perturbs parameters rather than scoring direction recoverability.",
            novelty_opening="A recovery-guided optimizer can still be distinct if it uses gradient-space perturbation and blockwise candidate selection instead of adversarial sharpness search.",
        ),
        LiteratureEntry(
            family="Gradient conflict methods",
            representative_method="PCGrad / CAGrad",
            source_title="Conflict-Averse Gradient Descent for Multi-task Learning",
            source_url="https://openreview.net/forum?id=_61Qh8tULj_",
            core_mechanism="Project or combine gradients based on conflict and alignment.",
            problem_solved="Multi-task interference and directional conflict.",
            assumptions="Multiple tasks or multiple conflicting gradient sources are available.",
            where_wins="Explicitly conflicting-gradient settings.",
            where_fails="Single-task settings can see limited upside if conflict signals are weak.",
            compute_cost="Medium",
            memory_cost="Low to Medium",
            overlap_with_local_ideas="Very high overlap with Magneto-style alignment and conflict language.",
            novelty_opening="Still open if the method scores recoverability of blockwise directions rather than only projecting task gradients.",
        ),
        LiteratureEntry(
            family="Stochastic diffusion / sampling-inspired optimizers",
            representative_method="SGLD / SGHMC",
            source_title="Stochastic Gradient Hamiltonian Monte Carlo",
            source_url="https://proceedings.mlr.press/v32/cheni14.html",
            core_mechanism="Inject Langevin/Hamiltonian noise and friction into stochastic updates.",
            problem_solved="Sampling, exploration, and sometimes better escape from poor basins.",
            assumptions="Noise injection is beneficial and extra stochasticity is acceptable.",
            where_wins="Posterior sampling or highly nonconvex exploration-centric tasks.",
            where_fails="Can hurt convergence and final accuracy on ordinary supervised training.",
            compute_cost="Low to Medium",
            memory_cost="Low to Medium",
            overlap_with_local_ideas="Direct overlap with local diffusion and Hamiltonian branches.",
            novelty_opening="Low for pure noise-based exploration; any new contribution must come from a different trust signal.",
        ),
        LiteratureEntry(
            family="Physics-inspired Adam variants",
            representative_method="VRAdam",
            source_title="VRAdam: A velocity-regulated Adam optimizer through the lens of frictionless dynamics",
            source_url="https://openreview.net/forum?id=QQN0aLXEGL",
            core_mechanism="Reinterpret Adam dynamics through velocity regulation and physical analogies.",
            problem_solved="Stability and speed through modified velocity handling.",
            assumptions="Physical reinterpretation yields a practical controller rather than just metaphor.",
            where_wins="Selected tasks where velocity regulation helps.",
            where_fails="Risk of collapsing back to another tuned Adam variant.",
            compute_cost="Low",
            memory_cost="Medium",
            overlap_with_local_ideas="Extremely high overlap with the local Hamiltonian/thermodynamic line.",
            novelty_opening="Low for more physics-named Adam variants unless the update leaves Adam logic behind.",
        ),
        LiteratureEntry(
            family="Orthogonalized / matrix-structured updates",
            representative_method="Muon",
            source_title="Moonlight: Training Smaller Networks for LLMs with Matrix Orthogonalization",
            source_url="https://arxiv.org/abs/2502.16982",
            core_mechanism="Use matrix-structured update orthogonalization / Newton-Schulz-style transforms on 2D parameters.",
            problem_solved="Improve training efficiency of matrix-shaped neural layers.",
            assumptions="2D hidden-layer parameters dominate and orthogonalized updates are affordable.",
            where_wins="Modern matrix-heavy neural training regimes.",
            where_fails="Mixed parameter shapes, small tabular models, and environments where pure 2D assumptions break.",
            compute_cost="Medium",
            memory_cost="Low to Medium",
            overlap_with_local_ideas="Connects strongly to structure-aware and directional-coherence themes, but not to recoverability.",
            novelty_opening="Good space for hybrids, but a simple Muon clone would not be novel.",
        ),
        LiteratureEntry(
            family="Memory-efficient Adam descendants",
            representative_method="Adam-mini",
            source_title="Adam-mini: Use Fewer Learning Rates To Gain More",
            source_url="https://arxiv.org/abs/2406.16793",
            core_mechanism="Reduce Adam's memory footprint by sharing learning-rate resources across parameter blocks.",
            problem_solved="Adam-like performance with lower optimizer-state cost.",
            assumptions="Blockwise learning-rate sharing preserves most of Adam's benefit.",
            where_wins="Memory-constrained large-model training.",
            where_fails="Still fundamentally Adam-like; novelty headroom is low unless the block logic is doing something qualitatively new.",
            compute_cost="Low",
            memory_cost="Low to Medium",
            overlap_with_local_ideas="Supports blockwise state rather than per-parameter state.",
            novelty_opening="Openings remain for blockwise directional state rather than shared Adam rates.",
        ),
        LiteratureEntry(
            family="Schedule-free methods",
            representative_method="Schedule-Free AdamW / SGD",
            source_title="The Road Less Scheduled",
            source_url="https://arxiv.org/abs/2405.15682",
            core_mechanism="Use iterate averaging and update design to avoid explicit learning-rate schedules and warmup.",
            problem_solved="Reduce dependence on hand-tuned schedules.",
            assumptions="Averaging-based control can replace explicit schedules without losing performance.",
            where_wins="Practical training pipelines where schedule tuning is costly.",
            where_fails="May still underperform specialized schedules in some settings.",
            compute_cost="Low",
            memory_cost="Low to Medium",
            overlap_with_local_ideas="Matches the desire for self-regulating optimization better than local SDS-style horizon control did.",
            novelty_opening="Open if schedule-free behavior is paired with a non-Adam directional trust rule.",
        ),
        LiteratureEntry(
            family="Evolutionary / black-box optimizers",
            representative_method="CMA-ES / Differential Evolution",
            source_title="The CMA Evolution Strategy: A Tutorial",
            source_url="https://arxiv.org/abs/1604.00772",
            core_mechanism="Population-based search without relying on gradients.",
            problem_solved="Black-box, rugged, or discrete landscapes.",
            assumptions="Many objective evaluations are acceptable.",
            where_wins="Derivative-free search, hyperparameters, discrete spaces.",
            where_fails="Too slow for per-step deep-learning training.",
            compute_cost="High",
            memory_cost="High",
            overlap_with_local_ideas="Strong overlap with the RDT outer-search direction, not with train-time gradient optimization.",
            novelty_opening="Better as an outer loop or controller tuner than as a replacement for SGD/Adam.",
        ),
        LiteratureEntry(
            family="Hyperparameter search",
            representative_method="TPE / Optuna-style search",
            source_title="Algorithms for Hyper-Parameter Optimization",
            source_url="https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc91544a2af6f4-Paper.pdf",
            core_mechanism="Model the hyperparameter-response surface and search it efficiently.",
            problem_solved="Optimizer and training-config selection rather than inner-loop training.",
            assumptions="Repeated trial evaluation is acceptable.",
            where_wins="Outer-loop tuning and configuration search.",
            where_fails="Not a train-time update rule.",
            compute_cost="High",
            memory_cost="Medium",
            overlap_with_local_ideas="Aligns with RDT-based controller search more than with per-step updates.",
            novelty_opening="Useful supporting tool, not the core novel optimizer.",
        ),
        LiteratureEntry(
            family="Learned optimizers",
            representative_method="VeLO / learned optimization",
            source_title="VeLO: Training Versatile Learned Optimizers by Scaling Up",
            source_url="https://arxiv.org/abs/2211.09760",
            core_mechanism="Learn update rules from meta-training rather than hand-designing them.",
            problem_solved="Potentially outperform hand-designed optimizers on families of tasks.",
            assumptions="Meta-training infrastructure and distribution match are available.",
            where_wins="When training and test optimizer distributions line up.",
            where_fails="Heavy infrastructure, transfer risk, and poor interpretability.",
            compute_cost="Very High",
            memory_cost="High",
            overlap_with_local_ideas="Low direct overlap; could eventually learn a recoverability policy but is overkill here.",
            novelty_opening="Not the right first step for this workspace's CPU-scale research loop.",
        ),
    ]


def generate_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(entry) for entry in _literature_entries()])
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# Optimizer Literature Scan",
        "",
        "This scan is deliberately pragmatic: it focuses on optimizer families that overlap with the local repos and with plausible next-step implementations.",
        "",
        "## Summary",
        "- AdamW remains the strongest universal practical Adam-family baseline for this workspace.",
        "- RMSProp and SGD momentum remain serious baselines on small/noisy CPU benchmarks and should not be treated as weak legacy controls.",
        "- Muon, Adam-mini, and Schedule-Free AdamW point toward three modern openings: structure-aware updates, lower-state adaptation, and less schedule dependence.",
        "- PCGrad/CAGrad, SAM/ASAM, and SGLD/SGHMC already occupy much of the local alignment, perturbation, and physics-control space.",
        "- The most believable opening is not another physics-named Adam controller; it is a lighter direction-selection rule with recoverability or structure trust.",
        "",
        _markdown_table(
            frame[
                [
                    "family",
                    "representative_method",
                    "compute_cost",
                    "memory_cost",
                    "overlap_with_local_ideas",
                    "novelty_opening",
                ]
            ]
        ),
        "",
        "## Source Links",
    ]
    for entry in _literature_entries():
        lines.append(f"- [{entry.source_title}]({entry.source_url})")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def _gap_scores() -> list[GapScore]:
    return [
        GapScore(
            gap_name="direction_aware_optimization",
            summary="Track coherence, rotation, conflict, and persistent useful directions explicitly rather than only scaling coordinates.",
            novelty_potential=4,
            implementation_difficulty=3,
            benchmarkability=5,
            expected_speed=4,
            expected_memory_efficiency=4,
            chance_vs_classic=3,
            chance_vs_recent=3,
            connection_to_local_work=5,
            redundancy_risk=2,
        ),
        GapScore(
            gap_name="recovery_guided_optimization",
            summary="Trust directions that survive perturbation, masking, or observation shifts; downweight fragile directions.",
            novelty_potential=5,
            implementation_difficulty=3,
            benchmarkability=4,
            expected_speed=4,
            expected_memory_efficiency=4,
            chance_vs_classic=3,
            chance_vs_recent=3,
            connection_to_local_work=5,
            redundancy_risk=2,
        ),
        GapScore(
            gap_name="structure_aware_optimization",
            summary="Use block, tensor, or matrix structure instead of purely per-coordinate state.",
            novelty_potential=4,
            implementation_difficulty=4,
            benchmarkability=4,
            expected_speed=3,
            expected_memory_efficiency=3,
            chance_vs_classic=4,
            chance_vs_recent=3,
            connection_to_local_work=4,
            redundancy_risk=3,
        ),
        GapScore(
            gap_name="region_search_hybrid",
            summary="Use RDT-style search over update rules, parameter groups, or direction candidates.",
            novelty_potential=4,
            implementation_difficulty=5,
            benchmarkability=3,
            expected_speed=1,
            expected_memory_efficiency=2,
            chance_vs_classic=2,
            chance_vs_recent=2,
            connection_to_local_work=5,
            redundancy_risk=2,
        ),
        GapScore(
            gap_name="schedule_free_self_regulating",
            summary="Reduce schedule tuning and warmup dependence through self-averaging and bounded update control.",
            novelty_potential=3,
            implementation_difficulty=2,
            benchmarkability=5,
            expected_speed=5,
            expected_memory_efficiency=4,
            chance_vs_classic=3,
            chance_vs_recent=2,
            connection_to_local_work=3,
            redundancy_risk=4,
        ),
        GapScore(
            gap_name="memory_efficient_block_state",
            summary="Blockwise or low-rank state that competes with Adam-mini or Lion on memory.",
            novelty_potential=3,
            implementation_difficulty=3,
            benchmarkability=4,
            expected_speed=4,
            expected_memory_efficiency=5,
            chance_vs_classic=3,
            chance_vs_recent=3,
            connection_to_local_work=4,
            redundancy_risk=3,
        ),
        GapScore(
            gap_name="small_noisy_data_specialization",
            summary="Target noisy, sparse, overfitting-prone, or unstable small-data regimes where classical defaults are inconsistent.",
            novelty_potential=4,
            implementation_difficulty=2,
            benchmarkability=5,
            expected_speed=4,
            expected_memory_efficiency=4,
            chance_vs_classic=4,
            chance_vs_recent=3,
            connection_to_local_work=5,
            redundancy_risk=2,
        ),
        GapScore(
            gap_name="weak_gradient_or_non_gradient_learning",
            summary="Use search, inference, or candidate selection when gradients are weak, conflicting, or unreliable.",
            novelty_potential=5,
            implementation_difficulty=5,
            benchmarkability=2,
            expected_speed=1,
            expected_memory_efficiency=2,
            chance_vs_classic=2,
            chance_vs_recent=2,
            connection_to_local_work=5,
            redundancy_risk=1,
        ),
    ]


def generate_gap_analysis(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for gap in _gap_scores():
        overall = (
            1.4 * gap.novelty_potential
            + 1.2 * gap.benchmarkability
            + 1.0 * gap.expected_speed
            + 1.0 * gap.connection_to_local_work
            + 0.8 * gap.chance_vs_classic
            + 0.8 * gap.chance_vs_recent
            - 1.0 * gap.implementation_difficulty
            - 1.0 * gap.redundancy_risk
        )
        rows.append({**asdict(gap), "overall_priority_score": round(overall, 3)})
    frame = pd.DataFrame(rows).sort_values("overall_priority_score", ascending=False)
    frame.to_csv(output_path / "gap_scores.csv", index=False)
    lines = [
        "# Optimizer Gap Analysis",
        "",
        "Scores below are heuristic 1-5 judgments used to rank which openings are worth prototyping next.",
        "",
        _markdown_table(frame),
    ]
    (output_path / "gap_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def _candidate_directions() -> list[CandidateDirection]:
    return [
        CandidateDirection(
            candidate_name="recovery_direction_optimizer",
            plain_idea="Pick a blockwise update direction only if it is coherent and recoverable under small perturbations; otherwise keep or blend a previously trusted direction.",
            mathematical_signal="Gradient-space recoverability score from perturbed directions, plus coherence with recent trusted and previous-update directions.",
            update_rule="Choose the best candidate among current, memory, blend, and conflict-resolved directions; scale the step by gradient norm and bounded trust.",
            differs_from="AdamW uses moment EMAs; RMSProp uses squared-gradient scaling; Muon orthogonalizes matrices; SAM perturbs parameters rather than scoring direction recoverability.",
            best_target_tasks="Conflicting batches, noisy small-batch learning, oscillatory valleys, sparse/noisy small-data regimes.",
            likely_failure_modes="Can lag on very smooth large-scale problems where coordinate-wise adaptive scaling is already enough.",
            compute_cost="Low to medium: a few extra tensor normalizations and small perturbation probes per block.",
            memory_cost="Low to medium: one trusted direction plus previous gradients/updates per tensor block.",
            novelty_potential=5,
            implementation_difficulty=3,
            benchmarkability=5,
            expected_speed=4,
            chance_vs_classic=4,
            chance_vs_recent=3,
            connection_to_local_work=5,
            redundancy_risk=2,
            prototype_name="direction_recovery_optimizer",
        ),
        CandidateDirection(
            candidate_name="magneto_muon_hybrid",
            plain_idea="Combine directional-coherence gating with Muon-style matrix orthogonalization.",
            mathematical_signal="Alignment/rotation scores plus matrix-orthogonalized updates on 2D parameters.",
            update_rule="Orthogonalize matrix updates, then gate magnitude by coherence/conflict signals.",
            differs_from="Closer to Muon than AdamW; more matrix-structured and less coordinate-wise.",
            best_target_tasks="Matrix-heavy neural nets and structured layers.",
            likely_failure_modes="Mixed parameter shapes and small tabular models; high redundancy risk with Muon literature.",
            compute_cost="Medium",
            memory_cost="Medium",
            novelty_potential=3,
            implementation_difficulty=4,
            benchmarkability=3,
            expected_speed=3,
            chance_vs_classic=3,
            chance_vs_recent=2,
            connection_to_local_work=4,
            redundancy_risk=4,
        ),
        CandidateDirection(
            candidate_name="rdt_direction_controller",
            plain_idea="Use RDT search as an outer loop over candidate directions, parameter groups, or update templates.",
            mathematical_signal="Recursive partition score over direction candidates and group assignments.",
            update_rule="Generate a small candidate set, score recursively, then apply the winning local update.",
            differs_from="Acts like a search controller rather than a standard optimizer state update.",
            best_target_tasks="Very noisy or discrete/mixed optimization settings; controller search.",
            likely_failure_modes="Too slow for dense per-step training and difficult to benchmark fairly against fast first-order methods.",
            compute_cost="High",
            memory_cost="Medium",
            novelty_potential=4,
            implementation_difficulty=5,
            benchmarkability=2,
            expected_speed=1,
            chance_vs_classic=2,
            chance_vs_recent=2,
            connection_to_local_work=5,
            redundancy_risk=2,
        ),
        CandidateDirection(
            candidate_name="recovery_sam_hybrid",
            plain_idea="Prefer directions that are both recoverable under gradient perturbation and robust under local parameter perturbation.",
            mathematical_signal="Recovery score plus SAM-style worst-case neighborhood loss.",
            update_rule="Filter candidate directions by recovery, then run a sharpness-aware correction.",
            differs_from="Extends SAM rather than AdamW and doubles down on perturbation robustness.",
            best_target_tasks="Generalization-heavy settings and noisy small-data classification.",
            likely_failure_modes="Too expensive for CPU-scale research loops and overlaps with existing SAM literature.",
            compute_cost="High",
            memory_cost="Medium",
            novelty_potential=3,
            implementation_difficulty=4,
            benchmarkability=3,
            expected_speed=1,
            chance_vs_classic=3,
            chance_vs_recent=2,
            connection_to_local_work=4,
            redundancy_risk=4,
        ),
        CandidateDirection(
            candidate_name="block_direction_optimizer",
            plain_idea="Store trusted directions blockwise instead of per-parameter moments and update only through block consensus.",
            mathematical_signal="Block coherence, block trust, and block norm ratios.",
            update_rule="Maintain one direction per block and update it via consensus with the current gradient.",
            differs_from="Closer to blockwise SGD/Lion hybrids than to AdamW.",
            best_target_tasks="Sparse features, block-structured problems, and memory-sensitive settings.",
            likely_failure_modes="May underperform on tasks that genuinely need coordinate-wise adaptivity.",
            compute_cost="Low",
            memory_cost="Low",
            novelty_potential=4,
            implementation_difficulty=2,
            benchmarkability=5,
            expected_speed=5,
            chance_vs_classic=3,
            chance_vs_recent=3,
            connection_to_local_work=4,
            redundancy_risk=3,
            prototype_name="block_direction_optimizer",
        ),
        CandidateDirection(
            candidate_name="schedule_free_direction_optimizer",
            plain_idea="Use direction memory and recovery trust inside a schedule-free iterate-averaging shell.",
            mathematical_signal="Direction coherence and trust, combined with iterate averaging instead of explicit schedules.",
            update_rule="Take direction-selected steps and maintain a slow averaged iterate instead of relying on warmup/schedules.",
            differs_from="Closer to schedule-free learning than to AdamW; uses direction selection rather than moment scaling.",
            best_target_tasks="Tasks where schedule sensitivity is a bottleneck.",
            likely_failure_modes="Redundancy risk with schedule-free literature unless the direction rule adds clear value.",
            compute_cost="Low",
            memory_cost="Low to medium",
            novelty_potential=3,
            implementation_difficulty=3,
            benchmarkability=4,
            expected_speed=4,
            chance_vs_classic=3,
            chance_vs_recent=2,
            connection_to_local_work=3,
            redundancy_risk=4,
        ),
        CandidateDirection(
            candidate_name="observation_recovery_optimizer",
            plain_idea="Mask or corrupt parts of the gradient observation and only update along directions reconstructible from partial observations.",
            mathematical_signal="Partial-observation recoverability and reconstruction consistency.",
            update_rule="Estimate trusted direction from multiple masked views, then update along the reconstructible component only.",
            differs_from="More observation-theoretic than AdamW, RMSProp, Muon, or SAM.",
            best_target_tasks="Sparse gradients, noisy partial information, and structured sensing-style problems.",
            likely_failure_modes="Extra compute and possible under-utilization when the gradient is already clean.",
            compute_cost="Medium",
            memory_cost="Medium",
            novelty_potential=5,
            implementation_difficulty=4,
            benchmarkability=3,
            expected_speed=3,
            chance_vs_classic=3,
            chance_vs_recent=3,
            connection_to_local_work=5,
            redundancy_risk=2,
            prototype_name="observation_recovery_optimizer",
        ),
    ]


def generate_candidate_directions(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in _candidate_directions():
        overall = (
            1.5 * candidate.novelty_potential
            + 1.2 * candidate.benchmarkability
            + 1.0 * candidate.expected_speed
            + 1.0 * candidate.connection_to_local_work
            + 1.0 * candidate.chance_vs_classic
            + 0.8 * candidate.chance_vs_recent
            - 1.0 * candidate.implementation_difficulty
            - 1.0 * candidate.redundancy_risk
        )
        rows.append({**asdict(candidate), "overall_candidate_score": round(overall, 3)})
    frame = pd.DataFrame(rows).sort_values("overall_candidate_score", ascending=False)
    frame["selected_for_prototype"] = frame["candidate_name"] == frame.iloc[0]["candidate_name"]
    frame.to_csv(output_path / "candidate_scores.csv", index=False)

    lines = [
        "# Candidate Optimizer Directions",
        "",
        "The strongest candidates are those that leave Adam's coordinate-wise moment logic behind while still being cheap enough to benchmark honestly on CPU.",
        "",
        _markdown_table(
            frame[
                [
                    "candidate_name",
                    "overall_candidate_score",
                    "novelty_potential",
                    "benchmarkability",
                    "expected_speed",
                    "redundancy_risk",
                    "prototype_name",
                ]
            ]
        ),
        "",
    ]
    for candidate in _candidate_directions():
        lines.extend(
            [
                f"## {candidate.candidate_name}",
                f"1. Idea: {candidate.plain_idea}",
                f"2. Mathematical signal: {candidate.mathematical_signal}",
                f"3. Update rule: {candidate.update_rule}",
                f"4. How it differs from AdamW / RMSProp / Muon / SAM: {candidate.differs_from}",
                f"5. Best target tasks: {candidate.best_target_tasks}",
                f"6. Likely failure modes: {candidate.likely_failure_modes}",
                f"7. Compute/memory cost: {candidate.compute_cost}; {candidate.memory_cost}",
                f"8. Existing repo leverage: connection score {candidate.connection_to_local_work}/5",
                f"9. Novelty potential: {candidate.novelty_potential}/5",
                "",
            ]
        )
    (output_path / "candidate_optimizer_directions.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def select_best_candidate(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    candidate_scores = pd.read_csv(Path(output_dir) / "candidate_scores.csv")
    best_row = candidate_scores.sort_values("overall_candidate_score", ascending=False).iloc[0].to_dict()
    return best_row


def implement_best_optimizer_prototype(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best = select_best_candidate(output_path)
    summary = {
        "selected_candidate": best["candidate_name"],
        "prototype_name": best.get("prototype_name"),
        "status": "implemented" if best.get("prototype_name") else "design_only",
        "reason": "Selected for the best balance of novelty, benchmarkability, speed, and connection to local recoverability work.",
    }
    (output_path / "prototype_summary.md").write_text(
        "\n".join(
            [
                "# Prototype Selection",
                "",
                f"- Selected candidate: `{summary['selected_candidate']}`",
                f"- Prototype name: `{summary['prototype_name']}`",
                f"- Status: `{summary['status']}`",
                f"- Reason: {summary['reason']}",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def _combined_benchmark_tasks(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tuning_config = dict(config)
    benchmark_config = dict(config)
    stress_config = dict(config)
    tuning_frame = run_tuning_suite(tuning_config)
    benchmark_frame = run_benchmark_suite(benchmark_config)
    stress_frame = run_stress_suite(stress_config)
    return tuning_frame, benchmark_frame, stress_frame


def novel_optimizer_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 3,
        "search_seed": 2026,
        "tuning_epoch_scale": 0.4,
        "benchmark_epoch_scale": 0.7,
        "stress_epoch_scale": 0.7,
        "optimizers": [
            "sgd",
            "sgd_momentum",
            "rmsprop",
            "adam",
            "adamw",
            "nadam",
            "radam",
            "lion",
            "muon_hybrid",
            "topological_adam",
            "magneto_adam",
            "real_hamiltonian_adam",
            "magneto_hamiltonian_adam",
            "direction_recovery_optimizer",
            "block_direction_optimizer",
            "observation_recovery_optimizer",
        ],
        "tuning_tasks": [
            "moons_mlp",
            "conflicting_batches_classification",
            "small_batch_instability",
            "block_structure_classification",
            "low_rank_matrix_objective",
        ],
        "benchmark_tasks": [
            "linear_regression",
            "logistic_regression",
            "breast_cancer_mlp",
            "wine_mlp",
            "moons_mlp",
            "circles_mlp",
            "digits_mlp",
            "block_structure_classification",
            "low_rank_matrix_objective",
        ],
        "stress_tasks": [
            "noisy_gradients_classification",
            "label_noise_breast_cancer",
            "small_batch_instability",
            "conflicting_batches_classification",
            "overfit_small_wine",
            "saddle_objective",
            "oscillatory_valley",
            "sparse_gradients_linear",
            "nonstationary_moons",
            "plateau_escape_objective",
        ],
        "use_tuning_results": True,
    }


def run_novel_optimizer_benchmarks(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    output_path = ensure_output_dir(config)
    tuning_frame, benchmark_frame, stress_frame = _combined_benchmark_tasks(config)
    combined = pd.concat([benchmark_frame, stress_frame], ignore_index=True)
    combined.to_csv(output_path / "benchmark_results.csv", index=False)
    aggregated = aggregate_results(combined)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)

    baseline_names = ["adamw", "rmsprop", "sgd_momentum", "muon_hybrid", "topological_adam"]
    candidate_names = [
        optimizer_name
        for optimizer_name in ["direction_recovery_optimizer", "block_direction_optimizer", "observation_recovery_optimizer"]
        if optimizer_name in set(aggregated["optimizer"])
    ]
    win_rows = []
    for candidate_name in candidate_names:
        for baseline in baseline_names:
            if baseline not in set(aggregated["optimizer"]):
                continue
            wins = compute_meaningful_wins(aggregated, candidate_name, baseline)
            if wins.empty:
                continue
            win_rows.append(wins)
    win_flags = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    win_flags.to_csv(output_path / "win_flags.csv", index=False)
    return {
        "tuning": tuning_frame,
        "benchmark": combined,
        "aggregated": aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
    }


def _load_trace_frames(result_frame: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for path in result_frame.get("trace_path", pd.Series(dtype=str)).dropna().unique():
        trace_path = Path(path)
        if trace_path.exists():
            frames.append(pd.read_csv(trace_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_metric(trace_frame: pd.DataFrame, output_path: Path, metric: str, tasks: list[str], optimizers: list[str], title: str, event: str = "train") -> None:
    if trace_frame.empty or metric not in trace_frame.columns:
        return
    subset = trace_frame[(trace_frame["task"].isin(tasks)) & (trace_frame["optimizer"].isin(optimizers))]
    if event:
        subset = subset[subset["event"] == event]
    if subset.empty:
        return
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 3.5 * len(tasks)), sharex=False)
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks):
        task_frame = subset[subset["task"] == task]
        for optimizer in optimizers:
            opt_frame = task_frame[task_frame["optimizer"] == optimizer]
            curve = opt_frame.groupby("step")[metric].mean().dropna()
            if not curve.empty:
                axis.plot(curve.index, curve.values, label=optimizer)
        axis.set_title(task)
        axis.set_ylabel(metric)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(aggregated: pd.DataFrame, output_path: Path, baseline_name: str = "adamw") -> None:
    if aggregated.empty:
        return
    tasks = sorted(aggregated["task"].unique())
    optimizers = sorted(aggregated["optimizer"].unique())
    rows = []
    valid_tasks = []
    for task in tasks:
        task_frame = aggregated[aggregated["task"] == task].set_index("optimizer")
        if baseline_name not in task_frame.index:
            continue
        baseline = task_frame.loc[baseline_name]
        row_values = []
        for optimizer in optimizers:
            if optimizer not in task_frame.index:
                row_values.append(np.nan)
                continue
            row = task_frame.loc[optimizer]
            if pd.notna(row["mean_best_val_accuracy"]) and pd.notna(baseline["mean_best_val_accuracy"]):
                row_values.append(float(row["mean_best_val_accuracy"] - baseline["mean_best_val_accuracy"]))
            else:
                row_values.append(float(baseline["mean_best_val_loss"] - row["mean_best_val_loss"]))
        rows.append(row_values)
        valid_tasks.append(task)
    if not rows:
        return
    matrix = np.array(rows)
    plt.figure(figsize=(11, 6))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label=f"relative score vs {baseline_name}")
    plt.xticks(range(len(optimizers)), optimizers, rotation=45, ha="right")
    plt.yticks(range(len(valid_tasks)), valid_tasks)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_optimizer_strategy_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    audit = pd.read_csv(output_path / "repo_audit.csv")
    literature = pd.read_csv(output_path / "literature_matrix.csv")
    gaps = pd.read_csv(output_path / "gap_scores.csv")
    candidates = pd.read_csv(output_path / "candidate_scores.csv")
    benchmark = pd.read_csv(output_path / "benchmark_results.csv")
    aggregated = aggregate_results(benchmark)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags_path = output_path / "win_flags.csv"
    win_flags = pd.read_csv(win_flags_path) if win_flags_path.exists() else pd.DataFrame()

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark)
    comparison_optimizers = [
        "adamw",
        "rmsprop",
        "sgd_momentum",
        "muon_hybrid",
        "direction_recovery_optimizer",
        "block_direction_optimizer",
        "observation_recovery_optimizer",
    ]
    _plot_metric(trace_frame, figure_dir / "loss_curves.png", "train_loss", ["moons_mlp", "conflicting_batches_classification", "low_rank_matrix_objective"], comparison_optimizers, "Loss Curves")
    _plot_metric(trace_frame, figure_dir / "validation_accuracy_curves.png", "val_accuracy", ["moons_mlp", "wine_mlp", "block_structure_classification"], comparison_optimizers, "Validation Accuracy", event="val")
    _plot_metric(trace_frame, figure_dir / "recovery_curves.png", "recovery_score", ["conflicting_batches_classification", "small_batch_instability"], ["direction_recovery_optimizer"], "Recovery Score")
    _plot_metric(trace_frame, figure_dir / "coherence_curves.png", "direction_coherence", ["conflicting_batches_classification", "oscillatory_valley"], ["direction_recovery_optimizer"], "Direction Coherence")
    _plot_metric(trace_frame, figure_dir / "block_coherence_curves.png", "block_coherence", ["block_structure_classification", "low_rank_matrix_objective"], ["block_direction_optimizer"], "Block Coherence")
    _plot_metric(trace_frame, figure_dir / "observation_recoverability_curves.png", "observation_recoverability", ["sparse_gradients_linear", "noisy_gradients_classification"], ["observation_recovery_optimizer"], "Observation Recoverability")
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png", baseline_name="adamw")

    best_candidate = candidates.sort_values("overall_candidate_score", ascending=False).iloc[0]
    prototype_name = str(best_candidate.get("prototype_name") or "")
    baseline_pool = aggregated[aggregated["optimizer"].isin(["sgd", "sgd_momentum", "rmsprop", "adam", "adamw", "nadam", "radam", "lion", "muon_hybrid", "topological_adam", "magneto_adam", "real_hamiltonian_adam", "magneto_hamiltonian_adam"])]
    strongest_baseline = baseline_pool.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
    prototype_rows = aggregated[aggregated["optimizer"] == prototype_name]
    prototype_best = prototype_rows.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0] if not prototype_rows.empty else None
    implemented_prototypes = [str(name) for name in candidates["prototype_name"].dropna().tolist() if str(name) in set(aggregated["optimizer"])]
    implemented_summary_rows = []
    for optimizer_name in implemented_prototypes:
        frame = aggregated[aggregated["optimizer"] == optimizer_name]
        if frame.empty:
            continue
        best_row = frame.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
        implemented_summary_rows.append(
            {
                "optimizer": optimizer_name,
                "best_task": best_row["task"],
                "mean_best_val_loss": best_row["mean_best_val_loss"],
                "mean_best_val_accuracy": best_row["mean_best_val_accuracy"],
                "mean_runtime_seconds": best_row["mean_runtime_seconds"],
            }
        )
    implemented_summary = pd.DataFrame(implemented_summary_rows)
    implemented_leader_name = None
    implemented_leader = None
    if implemented_prototypes:
        implemented_run_summary = (
            benchmark[benchmark["optimizer"].isin(implemented_prototypes)]
            .groupby("optimizer", as_index=False)
            .agg(
                mean_selection_score=("selection_score", "mean"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
                divergence_rate=("diverged", "mean"),
            )
            .sort_values(
                ["mean_selection_score", "divergence_rate", "mean_runtime_seconds"],
                ascending=[False, True, True],
            )
        )
        if not implemented_run_summary.empty:
            implemented_leader = implemented_run_summary.iloc[0]
            implemented_leader_name = str(implemented_leader["optimizer"])

    win_counts = {}
    for baseline_name in ["adamw", "rmsprop", "sgd_momentum", "muon_hybrid", "topological_adam"]:
        wins = compute_meaningful_wins(aggregated, prototype_name, baseline_name) if prototype_name in set(aggregated["optimizer"]) and baseline_name in set(aggregated["optimizer"]) else pd.DataFrame()
        win_counts[baseline_name] = wins
    total_two_x = int(sum(int(frame["two_x"].sum()) for frame in win_counts.values() if not frame.empty))
    credible_novelty = bool(
        best_candidate["novelty_potential"] >= 4
        and best_candidate["redundancy_risk"] <= 2
        and any((not frame.empty and bool(frame["win"].astype(bool).any())) for frame in win_counts.values())
    )

    redundant_ideas = [
        "Thermodynamic / diffusion / uncertainty controller stacking on top of AdamW",
        "Pure Hamiltonian or Langevin naming without leaving Adam-family logic",
        "Another small bounded correction on top of Adam moments without a different update principle",
    ]
    promising_ideas = [
        "Directional coherence and conflict signals",
        "Recoverability / perturbation survival as a trust score",
        "Blockwise or matrix-aware state that is lighter than Adam but more structured than SGD",
        "RDT as an outer-loop controller rather than an inner-loop replacement",
    ]

    report_lines = [
        "# Final Optimizer Strategy",
        "",
        "## 1. What do my repos already contain?",
        f"- Audited {len(audit)} optimizer-adjacent repos/modules. The strongest concrete assets are the current benchmark harness, Topological Adam as a bounded-control baseline, TAdam as a serious Adam-family specialization, and RDT search as an outer-loop idea.",
        "",
        "## 2. What is redundant with existing literature?",
        "- " + "; ".join(redundant_ideas),
        "",
        "## 3. What looks genuinely promising?",
        "- " + "; ".join(promising_ideas),
        "",
        "## 4. What does current optimizer literature suggest?",
        "- AdamW remains the default adaptive baseline; RMSProp and SGD momentum are still strong on small/noisy tasks; Muon, Adam-mini, and Schedule-Free AdamW highlight structure, memory efficiency, and schedule removal as modern openings.",
        "",
        "## 5. What is the best candidate optimizer direction?",
        f"- `{best_candidate['candidate_name']}` with prototype `{prototype_name}`.",
        f"- Strongest implemented candidate so far: `{implemented_leader_name}`." if implemented_leader_name else "- No implemented candidate benchmarked yet beyond the original prototype.",
        "",
        "## 6. What is the proposed mathematical principle?",
        f"- {best_candidate['mathematical_signal']}",
        "",
        "## 7. Is it gradient-based, gradient-light, or non-gradient?",
        "- Gradient-based, but not Adam-like: it uses normalized block directions, perturbation-recoverability scoring, and trust-weighted candidate selection rather than first/second-moment adaptation.",
        "",
        "## 8. How does it differ from AdamW, RMSProp, SGD momentum, Muon, Schedule-Free AdamW, SAM, etc.?",
        f"- {best_candidate['differs_from']}",
        "",
        "## 9. Was a prototype implemented?",
        f"- {'Yes' if prototype_name else 'No'}.",
        "",
        "## 10. Did it beat any strong baselines?",
        f"- Wins vs AdamW: {int(win_counts['adamw']['win'].sum()) if not win_counts['adamw'].empty else 0}",
        f"- Wins vs RMSProp: {int(win_counts['rmsprop']['win'].sum()) if not win_counts['rmsprop'].empty else 0}",
        f"- Wins vs SGD momentum: {int(win_counts['sgd_momentum']['win'].sum()) if not win_counts['sgd_momentum'].empty else 0}",
        f"- Wins vs practical Muon baseline: {int(win_counts['muon_hybrid']['win'].sum()) if not win_counts['muon_hybrid'].empty else 0}",
        f"- Wins vs Topological Adam: {int(win_counts['topological_adam']['win'].sum()) if not win_counts['topological_adam'].empty else 0}",
        "",
        "## 11. Is it fast enough?",
        f"- Best prototype row: {prototype_best['task']} with mean runtime {prototype_best['mean_runtime_seconds']:.4f}s." if prototype_best is not None else "- Not evaluated.",
        "",
        "## 12. Is there any credible novelty?",
        f"- {'Tentatively yes, but still claim-light.' if credible_novelty else 'Not yet strong enough for a novelty claim.'}",
        "",
        "## 13. What exact next experiment should be run?",
        (
            f"- Promote `{implemented_leader_name}` to the next iteration path and rerun it on matrix-heavy MLPs, conflicting batches, sparse gradients, and noisy small-data tasks against RMSProp, SGD momentum, AdamW, Muon hybrid, and the best Hamiltonian baseline."
            if implemented_leader_name
            else "- Run a larger held-out test of the recovery-direction optimizer on matrix-heavy MLPs and transformer-style blocks, with a practical Muon hybrid baseline and a schedule-free wrapper."
        ),
        "",
        "## 14. What should be abandoned?",
        "- Full physics-controller stacking as a default optimizer direction; more physics naming without a distinct update rule; any path that remains AdamW plus extra gates.",
        "",
        "## Repo Audit Snapshot",
        _markdown_table(audit[["repo_or_module", "optimizer_or_idea", "contribution_to_new_direction"]]),
        "",
        "## Literature Matrix Snapshot",
        _markdown_table(literature[["family", "representative_method", "novelty_opening"]].head(12)),
        "",
        "## Gap Ranking Snapshot",
        _markdown_table(gaps[["gap_name", "overall_priority_score"]]),
        "",
        "## Candidate Ranking Snapshot",
        _markdown_table(candidates[["candidate_name", "overall_candidate_score", "prototype_name"]]),
        "",
        "## Implemented Candidate Benchmark Summary",
        _markdown_table(implemented_summary) if not implemented_summary.empty else "_No implemented candidates benchmarked._",
        "",
        "## Implemented Candidate Leader",
        (
            f"- `{implemented_leader_name}` led the implemented candidates with mean selection score `{implemented_leader['mean_selection_score']:.4f}`, mean runtime `{implemented_leader['mean_runtime_seconds']:.4f}s`, and divergence rate `{implemented_leader['divergence_rate']:.4f}`."
            if implemented_leader_name and implemented_leader is not None
            else "_No implemented candidate leader available._"
        ),
        "",
        "## Best Optimizer Per Task",
        _markdown_table(best_frame[["task", "best_optimizer", "mean_best_val_loss", "mean_best_val_accuracy"]]),
        "",
        "## Prototype Benchmark Summary",
        _markdown_table(prototype_rows[["task", "mean_best_val_loss", "mean_best_val_accuracy", "mean_runtime_seconds"]]) if prototype_best is not None else "_Prototype not benchmarked._",
        "",
        "## Strongest Baseline Observed",
        f"- `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{strongest_baseline['mean_best_val_loss']:.4f}` and mean best val accuracy `{strongest_baseline['mean_best_val_accuracy']:.4f}`.",
        "",
        "## Two-X Events",
        f"- Total 2x events for the prototype across tracked baselines: `{total_two_x}`",
        "",
        "## Source Links",
    ]
    for entry in _literature_entries():
        report_lines.append(f"- [{entry.source_title}]({entry.source_url})")

    final_report = "\n".join(report_lines)
    (output_path / "final_optimizer_strategy.md").write_text(final_report, encoding="utf-8")
    return {
        "audit": audit,
        "literature": literature,
        "gaps": gaps,
        "candidates": candidates,
        "aggregated": aggregated,
        "best_by_task": best_frame,
        "win_flags": win_flags,
        "credible_novelty": credible_novelty,
    }
