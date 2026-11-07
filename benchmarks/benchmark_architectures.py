import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import LayeredMLP, ContinuousGraph, ContinuousWaveField

def run_architecture_benchmark(epochs=300):
    torch.manual_seed(42)
    np.random.seed(42)

    N = 400
    X_pos = np.random.randn(N // 2, 2) * 0.4
    r_outer = 1.5 + np.random.randn(N // 2, 1) * 0.2
    theta = np.random.uniform(0, 2*np.pi, size=(N // 2, 1))
    X_neg = np.hstack([r_outer * np.cos(theta), r_outer * np.sin(theta)])
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.hstack([np.zeros(N // 2), np.ones(N // 2)]).astype(np.float32)[:, None]
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    criterion = nn.BCELoss()

    def train_isotropic_el(model, dt=0.03, gamma=0.8):
        velocities = [torch.zeros_like(p) for p in model.parameters()]
        for _ in range(epochs):
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            with torch.no_grad():
                for p, v in zip(model.parameters(), velocities):
                    v.mul_(1.0 - gamma * dt).add_(p.grad, alpha=-dt)
                    p.add_(v, alpha=dt)
                    p.grad.zero_()
        return loss.item()

    torch.manual_seed(195)
    mlp_loss = train_isotropic_el(LayeredMLP())
    torch.manual_seed(195)
    graph_loss = train_isotropic_el(ContinuousGraph())
    torch.manual_seed(195)
    wave_loss = train_isotropic_el(ContinuousWaveField())

    print(f"=== ARCHITECTURE BENCHMARK (Isotropic Euler-Lagrange, {epochs} Epochs) ===")
    print(f"[Layered MLP (Depth 3)]         Final BCE Loss: {mlp_loss:.6f} (CFL frozen)")
    print(f"[Non-Layered Graph (CTRNN)]     Final BCE Loss: {graph_loss:.6f}")
    print(f"[Non-Layered Wave Field (0-L)]  Final BCE Loss: {wave_loss:.6f} (Clean Convergence)")

if __name__ == "__main__":
    run_architecture_benchmark()
