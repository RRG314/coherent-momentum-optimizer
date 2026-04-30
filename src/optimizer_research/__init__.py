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
from .magneto_hamiltonian_suite import (
    export_magneto_hamiltonian_report,
    run_magneto_hamiltonian_ablation,
    run_magneto_hamiltonian_benchmarks,
    run_magneto_hamiltonian_energy_tests,
    run_magneto_hamiltonian_smoke,
)
from .magneto_gpu_suite import (
    export_magneto_gpu_report,
    run_magneto_gpu_ablation,
    run_magneto_gpu_benchmarks,
    run_magneto_gpu_cnn_benchmarks,
    run_magneto_gpu_multitask_benchmarks,
    run_magneto_gpu_smoke,
    run_magneto_gpu_stress_benchmarks,
)
from .real_hamiltonian_suite import (
    export_real_hamiltonian_report,
    run_real_hamiltonian_ablation,
    run_real_hamiltonian_benchmarks,
    run_real_hamiltonian_energy_tests,
    run_real_hamiltonian_smoke,
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
    "export_magneto_hamiltonian_report",
    "export_magneto_gpu_report",
    "export_real_hamiltonian_report",
    "export_report",
    "instantiate_optimizer",
    "load_yaml_config",
    "optimizer_availability",
    "run_ablation_suite",
    "run_benchmark_suite",
    "run_cnn_credibility_benchmark",
    "run_directional_instability_benchmark",
    "run_magneto_hamiltonian_ablation",
    "run_magneto_hamiltonian_benchmarks",
    "run_magneto_hamiltonian_energy_tests",
    "run_magneto_hamiltonian_smoke",
    "run_magneto_gpu_ablation",
    "run_magneto_gpu_benchmarks",
    "run_magneto_gpu_cnn_benchmarks",
    "run_magneto_gpu_multitask_benchmarks",
    "run_magneto_gpu_smoke",
    "run_magneto_gpu_stress_benchmarks",
    "run_real_hamiltonian_ablation",
    "run_real_hamiltonian_benchmarks",
    "run_real_hamiltonian_energy_tests",
    "run_real_hamiltonian_smoke",
    "run_smoke_suite",
    "run_tuning_suite",
    "sample_search_configs",
]
