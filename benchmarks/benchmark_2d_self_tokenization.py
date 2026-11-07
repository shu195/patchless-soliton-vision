import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Set seeds
torch.manual_seed(195)
np.random.seed(195)

# 1. Create a 2D Synthetic Benchmark Scene (64x64)
# Contains:
# - High-frequency circular fringe in center
# - Sharp diagonal edge
# - Smooth background gradient
H, W = 64, 64
u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
U, V = np.meshgrid(u, v)

# Ground truth 2D scene
R = np.sqrt(U**2 + V**2)
fringe = np.where(R < 0.45, np.cos(18.0 * R), 0.0)
diagonal = np.where(U + V > 0.3, 0.8, 0.0)
background = 0.3 * (U + 1.0)
scene = np.clip(fringe + diagonal + background, -1.0, 1.0).astype(np.float32)

coords = np.stack([U.flatten(), V.flatten()], axis=-1) # (4096, 2)
targets = scene.flatten()[:, None]                     # (4096, 1)

coords_t = torch.tensor(coords)
targets_t = torch.tensor(targets)

# 2. Self-Tokenizing 2D Continuous Wave Packet Field
class ContinuousSelfTokenizing2D(nn.Module):
    """
    Patchless 2D Continuous Wavelet Soliton Representation:
        psi_k(u, v) = exp( - ((u - mu_u)^2 / (2 sigma_u^2) + (v - mu_v)^2 / (2 sigma_v^2)) ) * cos(k_u * u + k_v * v + phi)
        I_hat(u, v) = sum_{k=1}^K A_k psi_k(u, v) + b
    Zero discrete grid. Tokens are localized dynamic wavepackets in C^inf space.
    """
    def __init__(self, K=64):
        super().__init__()
        self.K = K
        # Centers initialized on a loose grid with small perturbation
        grid_pts = np.linspace(-0.8, 0.8, int(np.sqrt(K)))
        gu, gv = np.meshgrid(grid_pts, grid_pts)
        init_mu = np.stack([gu.flatten(), gv.flatten()], axis=-1)
        if len(init_mu) < K:
            extra = np.random.uniform(-0.8, 0.8, size=(K - len(init_mu), 2))
            init_mu = np.vstack([init_mu, extra])
        init_mu = init_mu[:K]

        self.mu = nn.Parameter(torch.tensor(init_mu, dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.zeros(K, 2) - 1.2) # initial sigma ~ 0.3
        self.k_vec = nn.Parameter(torch.randn(K, 2) * 5.0)
        self.phi = nn.Parameter(torch.zeros(K))
        self.weights = nn.Parameter(torch.randn(K, 1) * 0.2)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, uv):
        # uv: (N, 2) -> (N, 1, 2)
        # mu: (K, 2) -> (1, K, 2)
        # diff: (N, K, 2)
        uv_exp = uv.unsqueeze(1)
        mu_exp = self.mu.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-4

        # 2D Gaussian envelope (spatial token localization)
        scaled_diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(scaled_diff**2, dim=-1) # (N, K)
        envelope = torch.exp(-0.5 * dist_sq)

        # 2D Carrier wave (internal spectral direction and frequency)
        # k_vec: (K, 2), uv: (N, 2) -> (N, K)
        phase = torch.einsum('nd,kd->nk', uv, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        # Soliton wave packet
        solitons = envelope * carrier # (N, K)
        return solitons @ self.weights + self.bias

model = ContinuousSelfTokenizing2D(K=64)
criterion = nn.MSELoss()

# Train with Relativistic / Euler-Lagrange momentum
dt = 0.04
gamma = 0.5
c_light = 4.0
velocities = [torch.zeros_like(p) for p in model.parameters()]

print("=== TRAINING 2D SELF-TOKENIZING WAVE FIELD (Patchless Vision) ===")
loss_history = []
for epoch in range(600):
    pred = model(coords_t)
    loss = criterion(pred, targets_t)
    loss.backward()
    
    with torch.no_grad():
        for p, v in zip(model.parameters(), velocities):
            # Relativistic momentum update
            # dv/dt = -gamma * v - grad
            v.mul_(1.0 - gamma * dt).add_(p.grad, alpha=-dt)
            # Lorentz speed clamping
            v_norm = torch.norm(v)
            if v_norm > c_light:
                v.mul_(c_light / v_norm)
            p.add_(v, alpha=dt)
            p.grad.zero_()
            
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:3d} | MSE Loss: {loss.item():.6f}")

final_pred = model(coords_t).detach().numpy().reshape(H, W)
learned_mu = model.mu.detach().numpy()
learned_sigma = torch.exp(model.log_sigma).detach().numpy()

# 3. Generate Publication Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Panel A: Ground truth 2D scene
im0 = axes[0].imshow(scene, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
axes[0].set_title("Ground Truth 2D Scene\n(Fringe Core + Diagonal + Gradient)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("u coordinate")
axes[0].set_ylabel("v coordinate")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Panel B: Reconstructed Continuous Wave Field
im1 = axes[1].imshow(final_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
axes[1].set_title(f"Reconstructed Wave Field\n(MSE Loss: {loss_history[-1]:.5f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("u coordinate")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Panel C: Autonomous Continuous Token Topology
axes[2].imshow(scene, extent=[-1, 1, -1, 1], origin='lower', cmap='gray', alpha=0.45)
# Draw each token as an ellipse showing its learned center mu and bandwidth sigma
from matplotlib.patches import Ellipse
for i in range(model.K):
    mu_x, mu_y = learned_mu[i]
    sig_x, sig_y = learned_sigma[i]
    ellipse = Ellipse((mu_x, mu_y), width=2*sig_x, height=2*sig_y,
                      edgecolor='cyan', facecolor='none', linewidth=1.2, alpha=0.8)
    axes[2].add_patch(ellipse)
    axes[2].plot(mu_x, mu_y, 'ro', markersize=3.0)

axes[2].set_xlim(-1.1, 1.1)
axes[2].set_ylim(-1.1, 1.1)
axes[2].set_title("Autonomous Token Manifold\n(Centers $\\mu$ + Bandwidths $\\sigma$)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("u coordinate")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig_path = "figures/figure4_2d_self_tokenizing_vision.png"
plt.savefig(fig_path)
plt.close()

print(f"[SUCCESS] Saved 2D self-tokenizing analysis figure to {fig_path}")
print(f"Final MSE Loss: {loss_history[-1]:.6f}")
print(f"Min Token Bandwidth (Fine Feature Zoom): {learned_sigma.min():.4f}")
print(f"Max Token Bandwidth (Global Context): {learned_sigma.max():.4f}")
