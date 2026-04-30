from __future__ import annotations

import json
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from optimizers.topological_adam import topological_metrics
from optimizers.optimizer_utils import gradient_norm, loss_is_finite, parameter_norm, resolve_device, set_global_seed

from .baselines import benchmark_optimizer_names, build_optimizer_registry, instantiate_optimizer, sample_search_configs
from .config import ensure_output_dir
from .tasks import TaskContext, TrainingPhase, build_task_registry


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _reset_peak_device_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_device_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "driver_allocated_memory"):
            return float(torch.mps.driver_allocated_memory() / (1024.0 * 1024.0))
        if hasattr(torch.mps, "current_allocated_memory"):
            return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
    return float("nan")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, (list, tuple)) and batch:
        first = batch[0]
        if isinstance(first, torch.Tensor) and first.ndim > 0:
            return int(first.shape[0])
    if isinstance(batch, torch.Tensor) and batch.ndim > 0:
        return int(batch.shape[0])
    return 1


def _choose_phase(phases: list[TrainingPhase], epoch: int, epochs: int) -> TrainingPhase:
    fraction = epoch / max(1, epochs)
    selected = phases[0]
    for phase in phases:
        if phase.start_fraction <= fraction:
            selected = phase
    return selected


def _clone_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def _update_norm(model: torch.nn.Module, before: list[torch.Tensor]) -> float:
    total = 0.0
    index = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        diff = param.detach() - before[index]
        total += float(diff.pow(2).sum().item())
        index += 1
    return math.sqrt(max(total, 0.0))


def _extract_optimizer_metrics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if hasattr(optimizer, "latest_diagnostics"):
        latest = optimizer.latest_diagnostics()
        metrics.update({k: v for k, v in latest.items() if k != "optimizer"})
    if optimizer.__class__.__name__.lower().startswith("topological") or hasattr(optimizer, "field_metrics"):
        metrics.update(topological_metrics(optimizer))
    return metrics


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _optimizer_state_mb(optimizer: torch.optim.Optimizer) -> float:
    state = getattr(optimizer, "state", {})
    if isinstance(state, dict):
        total_bytes = sum(_tensor_bytes(item) for item in state.values())
        return total_bytes / (1024.0 * 1024.0)
    return 0.0


def _apply_gradient_noise(model: torch.nn.Module, std: float) -> None:
    if std <= 0.0:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.add_(torch.randn_like(param.grad) * std)


def _selection_score(row: dict[str, Any]) -> float:
    accuracy = row.get("best_val_accuracy")
    variance = float(row.get("loss_variance", 0.0) or 0.0)
    diverged = float(row.get("diverged", 0.0) or 0.0)
    best_loss = float(row.get("best_val_loss", row.get("final_val_loss", 0.0)) or 0.0)
    if accuracy is not None and accuracy == accuracy:
        return float(accuracy) - 0.1 * best_loss - 0.05 * variance - diverged
    return -best_loss - 0.05 * variance - diverged


def _mean_trace_metric(step_rows: list[dict[str, Any]], metric: str) -> float:
    values: list[float] = []
    for row in step_rows:
        if row.get("event") != "train" or metric not in row:
            continue
        value = row.get(metric)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _train_single_run(
    *,
    suite_name: str,
    task_name: str,
    optimizer_name: str,
    hyperparameters: dict[str, Any],
    seed: int,
    device: torch.device,
    output_dir: Path,
    save_trace: bool,
    epoch_scale: float = 1.0,
) -> dict[str, Any]:
    task_registry = build_task_registry()
    set_global_seed(seed)
    context: TaskContext = task_registry[task_name](seed, device)
    context.epochs = max(4, int(math.ceil(context.epochs * max(0.1, epoch_scale))))
    model = context.model
    optimizer, resolved_hparams = instantiate_optimizer(optimizer_name, model.parameters(), hyperparameters)

    train_losses: list[float] = []
    grad_norms: list[float] = []
    update_norms: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    steps_to_target_loss: int | None = None
    steps_to_target_accuracy: int | None = None
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    best_val_step = 0
    step_rows: list[dict[str, Any]] = []
    diverged = False
    global_step = 0
    total_train_examples = 0
    step_durations_ms: list[float] = []

    tracemalloc.start()
    _reset_peak_device_memory(device)
    start_time = time.perf_counter()

    for epoch in range(context.epochs):
        phase = _choose_phase(context.train_phases, epoch, context.epochs)
        model.train()
        for batch in phase.loader:
            global_step += 1
            total_train_examples += _infer_batch_size(batch)
            before = _clone_parameters(model)
            component_builder = context.metadata.get("component_closure_builder")
            if callable(component_builder) and hasattr(optimizer, "set_component_closures"):
                component_payload = component_builder(model, batch, epoch, global_step, context)
                if isinstance(component_payload, dict) and "closures" in component_payload:
                    optimizer.set_component_closures(component_payload["closures"], component_payload.get("metadata", {}))
                else:
                    optimizer.set_component_closures(component_payload, {})
            optimizer.zero_grad(set_to_none=True)
            loss = context.training_step(model, batch, epoch, global_step, context)
            loss_value = float(loss.detach().item())
            if hasattr(optimizer, "set_current_loss"):
                optimizer.set_current_loss(loss_value)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()

            noise_std = float(context.metadata.get("gradient_noise_std", 0.0) or 0.0)
            _apply_gradient_noise(model, noise_std)
            grad_norm_value = gradient_norm(model.parameters())

            def step_closure():
                optimizer.zero_grad(set_to_none=True)
                closure_loss = context.training_step(model, batch, epoch, global_step, context)
                if hasattr(optimizer, "set_current_loss"):
                    optimizer.set_current_loss(float(closure_loss.detach().item()))
                if torch.isfinite(closure_loss):
                    closure_loss.backward()
                    _apply_gradient_noise(model, noise_std)
                return closure_loss

            _synchronize_device(device)
            step_start = time.perf_counter()
            optimizer.step(step_closure if bool(getattr(optimizer, "wants_step_closure", False)) else None)
            _synchronize_device(device)
            step_durations_ms.append((time.perf_counter() - step_start) * 1000.0)
            update_norm_value = _update_norm(model, before)
            optimizer_metrics = _extract_optimizer_metrics(optimizer)

            train_losses.append(loss_value)
            grad_norms.append(grad_norm_value)
            update_norms.append(update_norm_value)

            step_row = {
                "suite": suite_name,
                "task": task_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "event": "train",
                "epoch": epoch,
                "step": global_step,
                "phase": phase.label,
                "train_loss": loss_value,
                "grad_norm": grad_norm_value,
                "update_norm": update_norm_value,
            }
            step_row.update(optimizer_metrics)
            step_rows.append(step_row)

            params_finite = math.isfinite(parameter_norm(model.parameters()))
            if not loss_is_finite(loss_value) or not math.isfinite(update_norm_value) or not params_finite:
                diverged = True
                break
        if diverged:
            break

        eval_metrics = context.evaluate(model, context)
        val_loss = float(eval_metrics.get("val_loss", float("nan")))
        val_acc = float(eval_metrics["val_accuracy"]) if "val_accuracy" in eval_metrics else float("nan")
        val_losses.append(val_loss)
        if not math.isnan(val_acc):
            val_accuracies.append(val_acc)
        if hasattr(optimizer, "set_external_metrics"):
            latest_train_loss = train_losses[-1] if train_losses else float("nan")
            optimizer.set_external_metrics(
                validation_loss=val_loss,
                validation_accuracy=None if math.isnan(val_acc) else val_acc,
                validation_gap=val_loss - latest_train_loss if math.isfinite(val_loss) and math.isfinite(latest_train_loss) else 0.0,
            )

        best_update = False
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_update = True
        if not math.isnan(val_acc) and val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_update = True
        if best_update:
            best_val_step = global_step

        if steps_to_target_loss is None and context.target_loss is not None and val_loss <= context.target_loss:
            steps_to_target_loss = global_step
        if (
            steps_to_target_accuracy is None
            and context.target_accuracy is not None
            and not math.isnan(val_acc)
            and val_acc >= context.target_accuracy
        ):
            steps_to_target_accuracy = global_step

        step_rows.append(
            {
                "suite": suite_name,
                "task": task_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "event": "val",
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
                "val_accuracy": None if math.isnan(val_acc) else val_acc,
            }
        )

    runtime_seconds = time.perf_counter() - start_time
    _synchronize_device(device)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_device_memory_mb = _peak_device_memory_mb(device)

    final_val_loss = val_losses[-1] if val_losses else float("inf")
    final_val_accuracy = val_accuracies[-1] if val_accuracies else float("nan")
    final_train_loss = train_losses[-1] if train_losses else float("inf")
    loss_variance = float(np.var(train_losses)) if train_losses else float("nan")
    grad_norm_stability = float(np.std(grad_norms) / (np.mean(grad_norms) + 1e-12)) if grad_norms else float("nan")
    update_norm_stability = float(np.std(update_norms) / (np.mean(update_norms) + 1e-12)) if update_norms else float("nan")
    training_stability = 1.0 / (
        1.0
        + max(0.0, loss_variance if math.isfinite(loss_variance) else 0.0)
        + max(0.0, grad_norm_stability if math.isfinite(grad_norm_stability) else 0.0)
        + max(0.0, update_norm_stability if math.isfinite(update_norm_stability) else 0.0)
    )

    trace_path = None
    if save_trace:
        trace_dir = output_dir / "traces" / suite_name
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{task_name}__{optimizer_name}__seed{seed}.csv"
        pd.DataFrame(step_rows).to_csv(trace_path, index=False)

    row = {
        "suite": suite_name,
        "task": task_name,
        "task_family": context.family,
        "problem_type": context.problem_type,
        "optimizer": optimizer_name,
        "seed": seed,
        "device": str(device),
        "device_name": _device_name(device),
        "hyperparameters": json.dumps(resolved_hparams, sort_keys=True, default=str),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": None if math.isnan(final_val_accuracy) else final_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": None if best_val_accuracy == float("-inf") else best_val_accuracy,
        "steps_to_target_loss": steps_to_target_loss,
        "steps_to_target_accuracy": steps_to_target_accuracy,
        "convergence_speed": best_val_step,
        "loss_variance": loss_variance,
        "gradient_norm_variance": float(np.var(grad_norms)) if grad_norms else float("nan"),
        "update_norm_variance": float(np.var(update_norms)) if update_norms else float("nan"),
        "gradient_norm_stability": grad_norm_stability,
        "update_norm_stability": update_norm_stability,
        "training_stability": training_stability,
        "generalization_gap": final_val_loss - final_train_loss,
        "runtime_seconds": runtime_seconds,
        "runtime_per_step_ms": (runtime_seconds * 1000.0) / max(1, len(train_losses)),
        "optimizer_step_time_ms": float(np.mean(step_durations_ms)) if step_durations_ms else float("nan"),
        "samples_per_second": float(total_train_examples / max(runtime_seconds, 1e-12)),
        "peak_memory_mb": peak_memory / (1024.0 * 1024.0),
        "peak_device_memory_mb": peak_device_memory_mb,
        "optimizer_state_mb": _optimizer_state_mb(optimizer),
        "diverged": int(diverged),
        "trace_path": None if trace_path is None else str(trace_path),
        "num_parameters": sum(param.numel() for param in model.parameters()),
        "diagnostics_every_n_steps": int(getattr(optimizer, "diagnostics_every_n_steps", 1)),
        "diagnostics_enabled": int(bool(getattr(optimizer, "enable_step_diagnostics", True))),
        "diagnostics_rows": int(len(optimizer.diagnostics_dataframe())) if hasattr(optimizer, "diagnostics_dataframe") else 0,
        "mean_oscillation_score": _mean_trace_metric(step_rows, "oscillation_score"),
        "mean_energy_drift": _mean_trace_metric(step_rows, "energy_drift"),
        "mean_relative_energy_drift": _mean_trace_metric(step_rows, "relative_energy_drift"),
        "mean_normalized_total_energy": _mean_trace_metric(step_rows, "normalized_total_energy"),
        "mean_kinetic_energy": _mean_trace_metric(step_rows, "kinetic_energy"),
        "mean_potential_energy": _mean_trace_metric(step_rows, "potential_energy"),
        "mean_total_hamiltonian": _mean_trace_metric(step_rows, "total_hamiltonian"),
        "mean_momentum_norm": _mean_trace_metric(step_rows, "momentum_norm"),
        "mean_parameter_step_norm": _mean_trace_metric(step_rows, "parameter_step_norm"),
        "mean_inverse_mass_mean": _mean_trace_metric(step_rows, "inverse_mass_mean"),
        "mean_inverse_mass_std": _mean_trace_metric(step_rows, "inverse_mass_std"),
        "mean_effective_damping": _mean_trace_metric(step_rows, "effective_damping"),
        "mean_alignment_scale": _mean_trace_metric(step_rows, "alignment_scale"),
        "mean_effective_lr_scale": _mean_trace_metric(step_rows, "effective_lr_scale"),
        "mean_grad_momentum_cosine": _mean_trace_metric(step_rows, "grad_momentum_cosine"),
        "mean_force_momentum_cosine": _mean_trace_metric(step_rows, "force_momentum_cosine"),
        "mean_grad_previous_grad_cosine": _mean_trace_metric(step_rows, "grad_previous_grad_cosine"),
        "mean_update_previous_update_cosine": _mean_trace_metric(step_rows, "update_previous_update_cosine"),
        "mean_rotation_score": _mean_trace_metric(step_rows, "rotation_score"),
        "mean_rotation_gate": _mean_trace_metric(step_rows, "rotation_gate"),
        "mean_coherence_score": _mean_trace_metric(step_rows, "coherence_score"),
        "mean_conflict_score": _mean_trace_metric(step_rows, "conflict_score"),
        "mean_conflict_gate": _mean_trace_metric(step_rows, "conflict_gate"),
        "mean_coherence_activation": _mean_trace_metric(step_rows, "coherence_activation"),
        "mean_stable_gate": _mean_trace_metric(step_rows, "stable_gate"),
        "mean_field_strength": _mean_trace_metric(step_rows, "field_strength"),
        "mean_coherence_projection_strength": _mean_trace_metric(step_rows, "coherence_projection_strength"),
        "mean_coherence_friction_multiplier": _mean_trace_metric(step_rows, "coherence_friction_multiplier"),
        "mean_soft_conflict_correction": _mean_trace_metric(step_rows, "soft_conflict_correction"),
        "mean_recovery_score": _mean_trace_metric(step_rows, "recovery_score"),
        "mean_direction_coherence": _mean_trace_metric(step_rows, "direction_coherence"),
        "mean_rotation_score_structural": _mean_trace_metric(step_rows, "rotation_score"),
        "mean_trust_scale": _mean_trace_metric(step_rows, "trust_scale"),
        "mean_stress_gate": _mean_trace_metric(step_rows, "stress_gate"),
        "mean_relative_gradient_scale": _mean_trace_metric(step_rows, "relative_gradient_scale"),
        "mean_constraint_agreement": _mean_trace_metric(step_rows, "constraint_agreement"),
        "mean_recoverability_score": _mean_trace_metric(step_rows, "recoverability_score"),
        "mean_support_balance": _mean_trace_metric(step_rows, "support_balance"),
        "mean_consensus_strength": _mean_trace_metric(step_rows, "consensus_strength"),
        "mean_component_conflict": _mean_trace_metric(step_rows, "component_conflict"),
        "mean_memory_alignment": _mean_trace_metric(step_rows, "memory_alignment"),
        "mean_residual_alignment": _mean_trace_metric(step_rows, "residual_alignment"),
        "mean_block_coherence": _mean_trace_metric(step_rows, "block_coherence"),
        "mean_block_trust_scale": _mean_trace_metric(step_rows, "block_trust_scale"),
        "mean_block_norm_ratio": _mean_trace_metric(step_rows, "block_norm_ratio"),
        "mean_memory_ratio": _mean_trace_metric(step_rows, "memory_ratio"),
        "mean_block_conflict": _mean_trace_metric(step_rows, "block_conflict"),
        "mean_filter_support": _mean_trace_metric(step_rows, "filter_support"),
        "mean_conv_trust_bonus": _mean_trace_metric(step_rows, "conv_trust_bonus"),
        "mean_conv_step_multiplier": _mean_trace_metric(step_rows, "conv_step_multiplier"),
        "mean_route_dense_stable": _mean_trace_metric(step_rows, "route_dense_stable"),
        "mean_route_conv_structured": _mean_trace_metric(step_rows, "route_conv_structured"),
        "mean_route_stress_response": _mean_trace_metric(step_rows, "route_stress_response"),
        "mean_route_entropy": _mean_trace_metric(step_rows, "route_entropy"),
        "mean_route_structure_support": _mean_trace_metric(step_rows, "route_structure_support"),
        "mean_observation_recoverability": _mean_trace_metric(step_rows, "observation_recoverability"),
        "mean_observation_disagreement": _mean_trace_metric(step_rows, "observation_disagreement"),
        "mean_view_support": _mean_trace_metric(step_rows, "view_support"),
    }
    row["selection_score"] = _selection_score(row)
    return row


def _load_best_tuning_map(tuning_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not tuning_path.exists():
        return {}
    frame = pd.read_csv(tuning_path)
    if frame.empty:
        return {}
    best_rows = (
        frame.sort_values(["task", "optimizer", "selection_score"], ascending=[True, True, False])
        .groupby(["task", "optimizer"], as_index=False)
        .head(1)
    )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in best_rows.iterrows():
        result[(row["task"], row["optimizer"])] = json.loads(row["hyperparameters"])
    return result


def _aggregate_suite_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame


def run_smoke_suite(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("smoke_seeds", [11]))
    task_names = list(config.get("smoke_tasks", ["linear_regression", "moons_mlp"]))
    optimizers = list(
        config.get(
            "smoke_optimizers",
            ["adamw", "topological_adam", "sds_adam", "coherent_direction_reference", "thermodynamic_adam", "diffusion_adam", "coherent_momentum_physical_baseline", "uncertainty_adam"],
        )
    )

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for optimizer_name in optimizers:
            for seed in seeds:
                rows.append(
                    _train_single_run(
                        suite_name="smoke",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters={},
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=bool(config.get("save_traces", True)),
                        epoch_scale=float(config.get("smoke_epoch_scale", 0.5)),
                    )
                )
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "smoke_results.csv", index=False)
    return frame


def run_tuning_suite(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    seeds = list(config.get("tuning_seeds", config.get("seeds", [11, 29, 47])))
    task_names = list(config.get("tuning_tasks", ["linear_regression", "moons_mlp", "digits_cnn", "pinn_harmonic_oscillator"]))
    optimizers = list(config.get("optimizers", benchmark_optimizer_names()))
    budget = int(config.get("search_budget", 4))
    search_seed = int(config.get("search_seed", 2026))
    registry = build_optimizer_registry()

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for optimizer_name in optimizers:
            spec = registry[optimizer_name]
            for trial_index, params in enumerate(sample_search_configs(spec, budget, search_seed), start=1):
                seed_rows = [
                    _train_single_run(
                        suite_name="tuning",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters=params,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=False,
                        epoch_scale=float(config.get("tuning_epoch_scale", 0.6)),
                    )
                    for seed in seeds
                ]
                trial_frame = pd.DataFrame(seed_rows)
                summary = {
                    "suite": "tuning",
                    "task": task_name,
                    "optimizer": optimizer_name,
                    "trial_index": trial_index,
                    "hyperparameters": json.dumps(params, sort_keys=True, default=str),
                    "mean_final_val_loss": float(trial_frame["final_val_loss"].mean()),
                    "mean_best_val_loss": float(trial_frame["best_val_loss"].mean()),
                    "mean_final_val_accuracy": float(trial_frame["final_val_accuracy"].dropna().mean()) if trial_frame["final_val_accuracy"].notna().any() else np.nan,
                    "mean_best_val_accuracy": float(trial_frame["best_val_accuracy"].dropna().mean()) if trial_frame["best_val_accuracy"].notna().any() else np.nan,
                    "mean_steps_to_target_loss": float(trial_frame["steps_to_target_loss"].dropna().mean()) if trial_frame["steps_to_target_loss"].notna().any() else np.nan,
                    "mean_steps_to_target_accuracy": float(trial_frame["steps_to_target_accuracy"].dropna().mean()) if trial_frame["steps_to_target_accuracy"].notna().any() else np.nan,
                    "mean_runtime_seconds": float(trial_frame["runtime_seconds"].mean()),
                    "loss_variance": float(trial_frame["loss_variance"].mean()),
                    "divergence_rate": float(trial_frame["diverged"].mean()),
                    "selection_score": float(trial_frame["selection_score"].mean()),
                }
                rows.append(summary)
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "tuning_results.csv", index=False)
    return frame


def run_benchmark_suite(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "benchmark_tasks",
            [
                "linear_regression",
                "noisy_regression",
                "logistic_regression",
                "synthetic_classification",
                "moons_mlp",
                "circles_mlp",
                "digits_logistic",
                "digits_cnn",
                "digits_autoencoder",
                "breast_cancer_mlp",
                "wine_mlp",
                "pinn_harmonic_oscillator",
            ],
        )
    )
    optimizers = list(config.get("optimizers", benchmark_optimizer_names()))
    tuning_map: dict[tuple[str, str], dict[str, Any]] = {}
    tuning_path = Path(output_dir / "tuning_results.csv")
    if bool(config.get("use_tuning_results", True)):
        tuning_map = _load_best_tuning_map(tuning_path)

    rows: list[dict[str, Any]] = []
    save_traces = bool(config.get("save_traces", True))
    for task_name in task_names:
        for optimizer_name in optimizers:
            overrides = tuning_map.get((task_name, optimizer_name), {})
            for seed in seeds:
                rows.append(
                    _train_single_run(
                        suite_name="benchmark",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters=overrides,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=save_traces,
                        epoch_scale=float(config.get("benchmark_epoch_scale", 0.85)),
                    )
                )
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "benchmark_results.csv", index=False)
    return frame


def run_stress_suite(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "stress_tasks",
            [
                "stagnating_regression",
                "unstable_deep_mlp",
                "noisy_gradients_classification",
                "loss_shock_classification",
                "sparse_gradients_linear",
                "high_curvature_regression",
                "nonstationary_moons",
                "direction_reversal_objective",
                "plateau_escape_objective",
                "rosenbrock_valley",
                "oscillatory_valley",
                "label_noise_breast_cancer",
                "small_batch_instability",
                "overfit_small_wine",
            ],
        )
    )
    optimizers = list(config.get("optimizers", benchmark_optimizer_names()))
    tuning_map = _load_best_tuning_map(output_dir / "tuning_results.csv")

    rows: list[dict[str, Any]] = []
    save_traces = bool(config.get("save_traces", True))
    for task_name in task_names:
        for optimizer_name in optimizers:
            overrides = tuning_map.get((task_name, optimizer_name), {})
            for seed in seeds:
                rows.append(
                    _train_single_run(
                        suite_name="stress",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters=overrides,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=save_traces,
                        epoch_scale=float(config.get("stress_epoch_scale", 0.85)),
                    )
                )
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "stress_test_results.csv", index=False)
    return frame


def run_stability_suite(config: dict[str, Any]) -> pd.DataFrame:
    compatibility_config = dict(config)
    if "stability_tasks" in compatibility_config and "stress_tasks" not in compatibility_config:
        compatibility_config["stress_tasks"] = compatibility_config["stability_tasks"]
    if "stability_epoch_scale" in compatibility_config and "stress_epoch_scale" not in compatibility_config:
        compatibility_config["stress_epoch_scale"] = compatibility_config["stability_epoch_scale"]
    return run_stress_suite(compatibility_config)


def _build_ablation_variants() -> dict[str, dict[str, dict[str, Any]]]:
    registry = build_optimizer_registry()
    neutral = {name: dict(spec.neutral_params or {}) for name, spec in registry.items()}
    return {
        "sds_adam": {
            "base": {},
            "neutral_adamw_equivalent": neutral["sds_adam"],
            "no_horizon_control": {"cooling_strength": 0.0, "reheating_strength": 0.0},
            "no_entropy_coupling": {"entropy_weight": 0.0},
            "global_only": {},
        },
        "coherent_direction_reference": {
            "base": {},
            "neutral_adamw_equivalent": neutral["coherent_direction_reference"],
            "no_alignment_control": {"alignment_strength": 0.0},
            "no_rotation_penalty": {"rotation_penalty": 0.0},
            "global_only": {"layerwise_mode": False, "global_mode": True},
            "layerwise_only": {"layerwise_mode": True, "global_mode": False},
        },
        "thermodynamic_adam": {
            "base": {},
            "neutral_adamw_equivalent": neutral["thermodynamic_adam"],
            "no_energy_control": {"energy_weight": 0.0},
            "no_entropy_control": {"entropy_weight": 0.0},
            "no_cooling": {"cooling_strength": 0.0},
            "global_only": {},
        },
        "diffusion_adam": {
            "base": {},
            "neutral_adamw_equivalent": neutral["diffusion_adam"],
            "no_diffusion_signal": {"diffusion_strength": 0.0},
            "no_entropy_scaling": {"entropy_scaled_noise": False},
            "no_aligned_noise": {"aligned_noise_weight": 0.0},
            "global_only": {},
        },
        "coherent_momentum_physical_baseline": {
            "base": {},
            "neutral_adamw_equivalent": neutral["coherent_momentum_physical_baseline"],
            "no_energy_correction": {"energy_correction_strength": 0.0},
            "no_oscillation_damping": {"oscillation_damping": 0.0},
            "no_friction": {"friction": 0.0},
            "global_only": {},
        },
        "uncertainty_adam": {
            "base": {},
            "neutral_adamw_equivalent": neutral["uncertainty_adam"],
            "no_uncertainty_control": {"uncertainty_weight": 0.0},
            "no_interference_control": {"interference_weight": 0.0},
            "no_reliability_boost": {"reliability_strength": 0.0},
            "global_only": {},
        },
    }


def run_ablation_suite(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    device = resolve_device(str(config.get("device", "auto")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(
        config.get(
            "ablation_tasks",
            [
                "moons_mlp",
                "noisy_gradients_classification",
                "loss_shock_classification",
                "nonstationary_moons",
                "plateau_escape_objective",
                "rosenbrock_valley",
                "label_noise_breast_cancer",
            ],
        )
    )
    tuning_map = _load_best_tuning_map(output_dir / "tuning_results.csv")
    ablations = _build_ablation_variants()

    rows: list[dict[str, Any]] = []
    for optimizer_name, variants in ablations.items():
        for task_name in task_names:
            tuned = tuning_map.get((task_name, optimizer_name), {})
            for variant_name, override in variants.items():
                merged = dict(tuned)
                merged.update(override)
                for seed in seeds:
                    row = _train_single_run(
                        suite_name="ablation",
                        task_name=task_name,
                        optimizer_name=optimizer_name,
                        hyperparameters=merged,
                        seed=seed,
                        device=device,
                        output_dir=output_dir,
                        save_trace=False,
                        epoch_scale=float(config.get("ablation_epoch_scale", 0.75)),
                    )
                    row["variant_name"] = variant_name
                    row["base_optimizer"] = optimizer_name
                    row["variant_overrides"] = json.dumps(override, sort_keys=True, default=str)
                    rows.append(row)
    frame = _aggregate_suite_rows(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame
