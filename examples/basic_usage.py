from __future__ import annotations

import torch

from optimizers import CoherentMomentumOptimizer


def main() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    optimizer = CoherentMomentumOptimizer(model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()

    x = torch.randn(16, 32)
    y = torch.randn(16, 1)

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    optimizer.set_current_loss(loss.item())
    loss.backward()
    optimizer.step()

    print("loss:", float(loss.item()))
    print("diagnostics:", optimizer.latest_diagnostics())


if __name__ == "__main__":
    main()
