import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import shutil

# Set deterministic seeds
torch.manual_seed(195)
np.random.seed(195)

# --- 1. MULTI-ASPECT HIGH-RES 2D TEST BENCHMARK SCENE (128x128 = 16,384 coords) ---
H, W = 128, 128
u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
U, V = np.meshgrid(u, v)

# Quadrant 1 (u > 0, v > 0): Quadratic Chirp / Fresnel Zone (frequency scales with r^2)
R_q1 = np.sqrt(np.maximum(0, U)**2 + np.maximum(0, V)**2)
chirp = np.where((U > 0) & (V > 0), np.cos(30.0 * R_q1**2), 0.0)

# Quadrant 2 (u < 0, v > 0): Sharp Multi-Orientation Polygons (Star/Diamond)
star_dist = np.abs(U) + np.abs(V - 0.5)
star = np.where((U < 0) & (V > 0) & (star_dist < 0.4), 0.9, 0.0)

# Quadrant 3 (u < 0, v < 0): Multi-Scale Interference Texture (Interfering Waves)
texture = np.where((U < 0) & (V < 0), 0.5 * (np.sin(15.0 * U + 10.0 * V) + np.cos(25.0 * U - 8.0 * V)), 0.0)

# Quadrant 4 (u > 0, v < 0): Smooth Gaussian Basin & Gradient
basin = np.where((U > 0) & (V < 0), -0.8 * np.exp(-4.0 * ((U - 0.5)**2 + (V + 0.5)**2)), 0.0)
smooth_grad = 0.25 * U - 0.25 * V

# Composite multi-aspect scene
scene_128 = np.clip(chirp + star + texture + basin + smooth_grad, -1.0, 1.0).astype(np.float32)

coords_128 = np.stack([U.flatten(), V.flatten()], axis=-1)
targets_128 = scene_128.flatten()[:, None]

coords_t = torch.tensor(coords_128)
targets_t = torch.tensor(targets_128)

# --- 2. SCALED 2D SELF-TOKENIZING GABOR-MORLET SOLITON FIELD ---
class ScaledContinuousSelfTokenizing2D(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        # Quasi-random uniform initialization covering [-0.9, 0.9]^2
        init_mu = np.random.uniform(-0.85, 0.85, size=(K, 2)).astype(np.float32)
        self.mu = nn.Parameter(torch.tensor(init_mu))
        # Initial sigma ~ 0.20
        self.log_sigma = nn.Parameter(torch.zeros(K, 2) - 1.6)
        # Random initial frequency wavevectors with rich orientation diversity
        angles = np.random.uniform(0, 2 * np.pi, size=(K, 1))
        freqs = np.random.uniform(3.0, 25.0, size=(K, 1))
        init_k = np.hstack([freqs * np.cos(angles), freqs * np.sin(angles)]).astype(np.float32)
        self.k_vec = nn.Parameter(torch.tensor(init_k))
        self.phi = nn.Parameter(torch.zeros(K))
        self.weights = nn.Parameter(torch.randn(K, 1) * 0.15)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, uv):
        # uv: (N, 2), mu: (K, 2) -> diff: (N, K, 2)
        uv_exp = uv.unsqueeze(1)
        mu_exp = self.mu.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-4

        # Anisotropic 2D Gaussian envelope
        scaled_diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(scaled_diff**2, dim=-1)
        envelope = torch.exp(-0.5 * dist_sq)

        # 2D Directional Carrier Wave
        phase = torch.einsum('nd,kd->nk', uv, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        solitons = envelope * carrier
        return solitons @ self.weights + self.bias

model = ScaledContinuousSelfTokenizing2D(K=128)
criterion = nn.MSELoss()

# --- 3. TRAIN WITH RELATIVISTIC EULER-LAGRANGE OPTIMIZER ---
dt = 0.035
gamma = 0.4
c_light = 5.0
velocities = [torch.zeros_like(p) for p in model.parameters()]

print("=== SCALED 2D SELF-TOKENIZING BENCHMARK (128x128 Coordinates, K=128 Solitons) ===")
epochs = 800
loss_curve = []

for epoch in range(epochs):
    pred = model(coords_t)
    loss = criterion(pred, targets_t)
    loss.backward()

    with torch.no_grad():
        for p, v in zip(model.parameters(), velocities):
            v.mul_(1.0 - gamma * dt).add_(p.grad, alpha=-dt)
            v_norm = torch.norm(v)
            if v_norm > c_light:
                v.mul_(c_light / v_norm)
            p.add_(v, alpha=dt)
            p.grad.zero_()

    loss_val = loss.item()
    loss_curve.append(loss_val)
    if (epoch + 1) % 200 == 0 or epoch == 0:
        psnr = -10.0 * np.log10(loss_val)
        print(f"Epoch {epoch+1:3d}/{epochs} | MSE: {loss_val:.6f} | PSNR: {psnr:.2f} dB")

final_pred_128 = model(coords_t).detach().numpy().reshape(H, W)
learned_mu = model.mu.detach().numpy()
learned_sigma = torch.exp(model.log_sigma).detach().numpy()
learned_k = model.k_vec.detach().numpy()
learned_weights = model.weights.detach().numpy().flatten()

# --- 4. CONTINUOUS SUPER-RESOLUTION TEST (Zero-Shot 256x256 Extrapolation) ---
H_hi, W_hi = 256, 256
u_hi = np.linspace(-1.0, 1.0, W_hi, dtype=np.float32)
v_hi = np.linspace(-1.0, 1.0, H_hi, dtype=np.float32)
U_hi, V_hi = np.meshgrid(u_hi, v_hi)
coords_hi = torch.tensor(np.stack([U_hi.flatten(), V_hi.flatten()], axis=-1))

with torch.no_grad():
    pred_256 = model(coords_hi).numpy().reshape(H_hi, W_hi)

# --- 5. COMPREHENSIVE MULTI-ASPECT PUBLICATION VISUALS (300 DPI) ---
fig = plt.figure(figsize=(18, 12), dpi=300)
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)

# Panel 1: Ground Truth 128x128 Multi-Aspect Scene
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(scene_128, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
ax1.set_title("A. Ground Truth Multi-Aspect Scene (128x128)\n[Chirp, Star Polygon, Waves, Basin]", fontsize=11, fontweight='bold')
ax1.set_xlabel("u coordinate")
ax1.set_ylabel("v coordinate")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Panel 2: Reconstructed Wave Field at 128x128
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(final_pred_128, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
final_psnr = -10.0 * np.log10(loss_curve[-1])
ax2.set_title(f"B. Relativistic E-L Soliton Field (K=128)\n[MSE: {loss_curve[-1]:.5f} | PSNR: {final_psnr:.2f} dB]", fontsize=11, fontweight='bold')
ax2.set_xlabel("u coordinate")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Panel 3: Zero-Shot Super-Resolution (256x256 Continuous Extrapolation)
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(pred_256, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
ax3.set_title("C. Infinite C-Inf Super-Resolution (256x256)\n[Zero-Shot Continuous Manifold Evaluation]", fontsize=11, fontweight='bold')
ax3.set_xlabel("u coordinate")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# Panel 4: Autonomous Token Foveation Topology
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(scene_128, extent=[-1, 1, -1, 1], origin='lower', cmap='gray', alpha=0.4)
for i in range(model.K):
    mx, my = learned_mu[i]
    sx, sy = learned_sigma[i]
    # Highlight tokens based on weight magnitude
    w_mag = np.abs(learned_weights[i])
    col = plt.cm.cool(np.clip(w_mag / 0.5, 0.0, 1.0))
    el = Ellipse((mx, my), width=2*sx, height=2*sy, edgecolor=col, facecolor='none', linewidth=1.1, alpha=0.75)
    ax4.add_patch(el)
    ax4.plot(mx, my, 'o', color=col, markersize=2.2)

ax4.set_xlim(-1.05, 1.05)
ax4.set_ylim(-1.05, 1.05)
ax4.set_title("D. Autonomous Token Foveation\n[Centers $\\mu$ + Adaptive Bandwidths $\\sigma$]", fontsize=11, fontweight='bold')
ax4.set_xlabel("u coordinate")
ax4.set_ylabel("v coordinate")

# Panel 5: Carrier Frequency Orientation Polar Distribution
ax5 = fig.add_subplot(gs[1, 1], projection='polar')
angles = np.arctan2(learned_k[:, 1], learned_k[:, 0])
freqs = np.sqrt(learned_k[:, 0]**2 + learned_k[:, 1]**2)
scatter = ax5.scatter(angles, freqs, c=learned_sigma.mean(axis=1), cmap='viridis_r', s=np.abs(learned_weights)*80 + 15, alpha=0.85)
ax5.set_title("E. Carrier Wavevector Spectrum\n[Angle $\\theta$ vs Spatial Frequency $\\omega$ (rad/s)]", fontsize=11, fontweight='bold', pad=15)
cbar5 = plt.colorbar(scatter, ax=ax5, fraction=0.046, pad=0.1)
cbar5.set_label("Mean Bandwidth $\\bar{\\sigma}$", fontsize=9)

# Panel 6: Optimization Convergence Curve
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(loss_curve, color='#00e5ff', linewidth=1.8, label="Relativistic E-L ($c=5.0$)")
ax6.set_yscale('log')
ax6.set_title("F. Relativistic Phase-Flow Convergence\n[Strict Subluminal Momentum Clamping]", fontsize=11, fontweight='bold')
ax6.set_xlabel("Epoch")
ax6.set_ylabel("MSE Loss (Log Scale)")
ax6.grid(True, linestyle='--', alpha=0.3)
ax6.legend(frameon=True, facecolor='#111111', edgecolor='#333333', labelcolor='white')

# Save figures
os.makedirs("figures", exist_ok=True)
out_fig_repo = "figures/figure5_scaled_2d_self_tokenizing_vision.png"
out_fig_brain = r"C:\Users\Admin\.gemini\antigravity\brain\3d9c4b3c-34f6-4b4e-8d0d-c028ff61e620\figure5_scaled_2d_self_tokenizing_vision.png"
plt.savefig(out_fig_repo, bbox_inches='tight')
plt.savefig(out_fig_brain, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] Saved 300 DPI Multi-Aspect Figure to {out_fig_repo} and {out_fig_brain}")
print(f"Final 128x128 MSE Loss: {loss_curve[-1]:.6f} | PSNR: {final_psnr:.2f} dB")
print(f"Learned Min Bandwidth (Zoom): {learned_sigma.min():.4f} | Max Bandwidth (Global): {learned_sigma.max():.4f}")
print(f"Max Carrier Frequency: {freqs.max():.2f} rad/s | Min Carrier Frequency: {freqs.min():.2f} rad/s")
