from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizers.coherent_momentum_optimizer import CoherentMomentumOptimizer  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    dataset = load_breast_cancer()
    x_train, x_val, y_train, y_val = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=7,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train_t.shape[1], 48),
        torch.nn.Tanh(),
        torch.nn.Linear(48, 1),
    )
    optimizer = CoherentMomentumOptimizer(
        model.parameters(),
        lr=0.02,
        mode="adam_preconditioned_hamiltonian",
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    batch_size = 32

    for _epoch in range(20):
        indices = torch.randperm(x_train_t.shape[0])
        for start in range(0, x_train_t.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.set_current_loss(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = float((preds.eq(y_val_t)).float().mean().item())
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

    diagnostics = optimizer.latest_diagnostics()
    print("Coherent Momentum minimal MLP example")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"best_val_accuracy={best_val_acc:.6f}")
    print("diagnostic_subset=")
    for key in ["rotation_score", "coherence_score", "conflict_score", "grad_momentum_cosine", "force_momentum_cosine"]:
        if key in diagnostics:
            print(f"  {key}={diagnostics[key]}")


if __name__ == "__main__":
    main()
