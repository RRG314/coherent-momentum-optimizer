from .baselines import (
    available_optimizer_names,
    benchmark_optimizer_names,
    build_optimizer_registry,
    instantiate_optimizer,
    optimizer_availability,
    sample_search_configs,
)
from .benchmarking import run_ablation_suite, run_benchmark_suite, run_smoke_suite, run_tuning_suite
from .cnn_credibility_suite import export_cnn_credibility_report, run_cnn_credibility_benchmark
from .config import ensure_output_dir, load_yaml_config
from .directional_instability_suite import export_directional_instability_report, run_directional_instability_benchmark
from .coherent_momentum_suite import (
    export_coherent_momentum_report,
    run_coherent_momentum_ablation,
    run_coherent_momentum_benchmarks,
    run_coherent_momentum_energy_tests,
    run_coherent_momentum_smoke,
)
from .coherent_momentum_gpu_suite import (
    export_coherent_momentum_gpu_report,
    run_coherent_momentum_gpu_ablation,
    run_coherent_momentum_gpu_benchmarks,
    run_coherent_momentum_gpu_cnn_benchmarks,
    run_coherent_momentum_gpu_multitask_benchmarks,
    run_coherent_momentum_gpu_smoke,
    run_coherent_momentum_gpu_stress_benchmarks,
)
from .coherent_momentum_real_baseline_suite import (
    export_coherent_momentum_real_baseline_report,
    run_coherent_momentum_real_baseline_ablation,
    run_coherent_momentum_real_baseline_benchmarks,
    run_coherent_momentum_real_baseline_energy_tests,
    run_coherent_momentum_real_baseline_smoke,
)
from .reporting import export_report
from .tasks import build_task_registry

__all__ = [
    "available_optimizer_names",
    "benchmark_optimizer_names",
    "build_optimizer_registry",
    "build_task_registry",
    "export_cnn_credibility_report",
    "export_directional_instability_report",
    "ensure_output_dir",
    "export_coherent_momentum_report",
    "export_coherent_momentum_gpu_report",
    "export_coherent_momentum_real_baseline_report",
    "export_report",
    "instantiate_optimizer",
    "load_yaml_config",
    "optimizer_availability",
    "run_ablation_suite",
    "run_benchmark_suite",
    "run_cnn_credibility_benchmark",
    "run_directional_instability_benchmark",
    "run_coherent_momentum_ablation",
    "run_coherent_momentum_benchmarks",
    "run_coherent_momentum_energy_tests",
    "run_coherent_momentum_smoke",
    "run_coherent_momentum_gpu_ablation",
    "run_coherent_momentum_gpu_benchmarks",
    "run_coherent_momentum_gpu_cnn_benchmarks",
    "run_coherent_momentum_gpu_multitask_benchmarks",
    "run_coherent_momentum_gpu_smoke",
    "run_coherent_momentum_gpu_stress_benchmarks",
    "run_coherent_momentum_real_baseline_ablation",
    "run_coherent_momentum_real_baseline_benchmarks",
    "run_coherent_momentum_real_baseline_energy_tests",
    "run_coherent_momentum_real_baseline_smoke",
    "run_smoke_suite",
    "run_tuning_suite",
    "sample_search_configs",
]
