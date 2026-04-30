from __future__ import annotations

import json
from pathlib import Path


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        _markdown_cell(
            "# Coherent Momentum Optimizer / Coherent Momentum Adam\n\n"
            "This Colab notebook installs the repository, runs the focused tests, executes the smoke and benchmark scripts that currently exist in the repo, and prints the resulting reports and CSV outputs.\n\n"
            "The default repo tasks rely on synthetic data and built-in scikit-learn datasets, so no manual dataset download is required for the included commands."
        ),
        _markdown_cell(
            "## How to use this notebook\n\n"
            "1. Clone the repo into Colab, or upload the repo into `/content`.\n"
            "2. Run the setup cells.\n"
            "3. Leave `RUN_LONG_BENCHMARKS = True` if you want the full repo evaluation, or set it to `False` to stop after tests and smoke checks."
        ),
        _code_cell(
            "from pathlib import Path\n"
            "import os\n"
            "import platform\n"
            "import subprocess\n"
            "import sys\n"
            "\n"
            "REPO_URL = os.environ.get('COHERENT_MOMENTUM_REPO_URL', '').strip()\n"
            "DEFAULT_ROOT = Path('/content/coherent-momentum-optimizer')\n"
            "if REPO_URL and not DEFAULT_ROOT.exists():\n"
            "    subprocess.run(['git', 'clone', REPO_URL, str(DEFAULT_ROOT)], check=True)\n"
            "\n"
            "candidates = [DEFAULT_ROOT, Path('/content/CoherentMomentumOptimizer'), Path.cwd()]\n"
            "ROOT = next((path for path in candidates if (path / 'pyproject.toml').exists()), None)\n"
            "if ROOT is None:\n"
            "    raise RuntimeError('Could not find the repo root. Clone the repo into /content/coherent-momentum-optimizer or run this notebook from inside the repo.')\n"
            "os.chdir(ROOT)\n"
            "print('Repo root:', ROOT)\n"
            "print('Python:', sys.version)\n"
            "print('Platform:', platform.platform())\n"
        ),
        _code_cell(
            "def run(cmd: str, check: bool = True):\n"
            "    print(f'\\\\n$ {cmd}')\n"
            "    result = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n"
            "    print(result.stdout)\n"
            "    if check and result.returncode != 0:\n"
            "        raise RuntimeError(f'Command failed with code {result.returncode}: {cmd}')\n"
            "    return result\n"
            "\n"
            "run(f'{sys.executable} -m pip install -U pip')\n"
            "run(f'{sys.executable} -m pip install -e \".[dev]\"')\n"
        ),
        _code_cell(
            "import torch\n"
            "\n"
            "device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'\n"
            "device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'Apple MPS' if device == 'mps' else 'CPU'\n"
            "print({'device': device, 'device_name': device_name, 'cuda_available': torch.cuda.is_available(), 'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()})\n"
        ),
        _code_cell("run(f'{sys.executable} -m compileall src')"),
        _code_cell(
            "focused_test_commands = [\n"
            "    f'{sys.executable} -m pytest tests/test_coherent_momentum_optimizer.py -q',\n"
            "    f'{sys.executable} -m pytest tests/test_coherent_momentum_gpu_compatibility.py -q',\n"
            "    f'{sys.executable} -m pytest tests/test_coherent_momentum_benchmark_outputs.py -q',\n"
            "]\n"
            "for command in focused_test_commands:\n"
            "    run(command)\n"
        ),
        _code_cell(
            "RUN_LONG_BENCHMARKS = True\n"
            "\n"
            "mainline_commands = [\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_optimizer_smoke.py',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_optimizer_benchmarks.py --config configs/coherent_momentum_optimizer_default.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_optimizer_energy_tests.py --config configs/coherent_momentum_optimizer_energy.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_optimizer_ablation.py --config configs/coherent_momentum_optimizer_ablation.yaml',\n"
            "    f'{sys.executable} scripts/export_coherent_momentum_optimizer_report.py',\n"
            "]\n"
            "gpu_commands = [\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_smoke.py',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_benchmarks.py --config configs/coherent_momentum_gpu_default.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_cnn_benchmarks.py --config configs/coherent_momentum_gpu_cnn.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_stress_benchmarks.py --config configs/coherent_momentum_gpu_stress.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_multitask_benchmarks.py --config configs/coherent_momentum_gpu_multitask.yaml',\n"
            "    f'{sys.executable} scripts/run_coherent_momentum_gpu_ablation.py --config configs/coherent_momentum_gpu_ablation.yaml',\n"
            "    f'{sys.executable} scripts/export_coherent_momentum_gpu_report.py',\n"
            "]\n"
            "if RUN_LONG_BENCHMARKS:\n"
            "    for command in mainline_commands + gpu_commands:\n"
            "        run(command)\n"
            "else:\n"
            "    print('Skipping long benchmark suite. Set RUN_LONG_BENCHMARKS = True to execute every repo script listed in the README and REPRODUCING.md.')\n"
        ),
        _markdown_cell("## Load and print result tables"),
        _code_cell(
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n"
            "\n"
            "pd.set_option('display.max_rows', 400)\n"
            "pd.set_option('display.max_columns', 200)\n"
            "pd.set_option('display.width', 200)\n"
            "\n"
            "result_paths = [\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'benchmark_results.csv',\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'energy_tests.csv',\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'ablation_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_smoke_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_benchmark_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_cnn_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_stress_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_multitask_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'gpu_ablation_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'runtime_memory_results.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'best_by_task.csv',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'win_flags.csv',\n"
            "]\n"
            "for path in result_paths:\n"
            "    if path.exists():\n"
            "        display(Markdown(f'### `{path.relative_to(ROOT)}`'))\n"
            "        display(pd.read_csv(path))\n"
            "    else:\n"
            "        display(Markdown(f'### Missing: `{path.relative_to(ROOT)}`'))\n"
        ),
        _markdown_cell("## Print markdown reports"),
        _code_cell(
            "report_paths = [\n"
            "    ROOT / 'reports' / 'repo_readiness_audit.md',\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'final_report.md',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'final_coherent_momentum_gpu_report.md',\n"
            "]\n"
            "for path in report_paths:\n"
            "    if path.exists():\n"
            "        print('\\n' + '=' * 100)\n"
            "        print(path.relative_to(ROOT))\n"
            "        print('=' * 100)\n"
            "        print(path.read_text())\n"
        ),
        _markdown_cell("## Figures"),
        _code_cell(
            "from IPython.display import Image, display\n"
            "\n"
            "figure_paths = [\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'figures' / 'win_loss_heatmap.png',\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'figures' / 'loss_curves.png',\n"
            "    ROOT / 'reports' / 'accepted_coherent_momentum' / 'figures' / 'energy_drift_curves.png',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'figures' / 'validation_loss_curves.png',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'figures' / 'validation_accuracy_curves.png',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'figures' / 'runtime_comparison.png',\n"
            "    ROOT / 'reports' / 'coherent_momentum_gpu' / 'figures' / 'win_loss_heatmap.png',\n"
            "]\n"
            "for path in figure_paths:\n"
            "    if path.exists():\n"
            "        print(path.relative_to(ROOT))\n"
            "        display(Image(filename=str(path)))\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
            "colab": {
                "name": "coherent_momentum_full_eval.ipynb",
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "coherent_momentum_full_eval.ipynb"
    notebook_path.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(notebook_path)
