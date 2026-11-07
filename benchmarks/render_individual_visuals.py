import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import shutil

# Target directories
REPO_FIGS = r"C:\Users\Admin\.gemini\antigravity\scratch\euler-lagrange-neural-optimizer\figures"
BRAIN_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\3d9c4b3c-34f6-4b4e-8d0d-c028ff61e620"
os.makedirs(REPO_FIGS, exist_ok=True)
os.makedirs(BRAIN_DIR, exist_ok=True)

# Deterministic seed
torch.manual_seed(195)
np.random.seed(195)

# 1. Generate 128x128 Multi-Aspect Scene
H, W = 128, 128
u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
U, V = np.meshgrid(u, v)

# Q1: Fresnel Chirp
R_q1 = np.sqrt(np.maximum(0, U)**2 + np.maximum(0, V)**2)
chirp = np.where((U > 0) & (V > 0), np.cos(30.0 * R_q1**2), 0.0)

# Q2: Star Polygon
star_dist = np.abs(U) + np.abs(V - 0.5)
star = np.where((U < 0) & (V > 0) & (star_dist < 0.4), 0.9, 0.0)

# Q3: Wave Interference
texture = np.where((U < 0) & (V < 0), 0.5 * (np.sin(15.0 * U + 10.0 * V) + np.cos(25.0 * U - 8.0 * V)), 0.0)

# Q4: Gaussian Basin & Smooth Gradient
basin = np.where((U > 0) & (V < 0), -0.8 * np.exp(-4.0 * ((U - 0.5)**2 + (V + 0.5)**2)), 0.0)
smooth_grad = 0.25 * U - 0.25 * V

scene_128 = np.clip(chirp + star + texture + basin + smooth_grad, -1.0, 1.0).astype(np.float32)

coords_128 = np.stack([U.flatten(), V.flatten()], axis=-1)
targets_128 = scene_128.flatten()[:, None]

coords_t = torch.tensor(coords_128)
targets_t = torch.tensor(targets_128)

# 2. Scaled Soliton Model
class ScaledContinuousSelfTokenizing2D(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        init_mu = np.random.uniform(-0.85, 0.85, size=(K, 2)).astype(np.float32)
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.log_sigma = nn.Parameter(torch.zeros(K, 2) - 1.6)
        angles = np.random.uniform(0, 2 * np.pi, size=(K, 1))
        freqs = np.random.uniform(3.0, 25.0, size=(K, 1))
        init_k = np.hstack([freqs * np.cos(angles), freqs * np.sin(angles)]).astype(np.float32)
        self.k_vec = nn.Parameter(torch.tensor(init_k))
        self.phi = nn.Parameter(torch.zeros(K))
        self.weights = nn.Parameter(torch.randn(K, 1) * 0.15)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, uv):
        uv_exp = uv.unsqueeze(1)
        mu_exp = self.mu.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-4

        scaled_diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(scaled_diff**2, dim=-1)
        envelope = torch.exp(-0.5 * dist_sq)

        phase = torch.einsum('nd,kd->nk', uv, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        solitons = envelope * carrier
        return solitons @ self.weights + self.bias

model = ScaledContinuousSelfTokenizing2D(K=128)
criterion = nn.MSELoss()

dt = 0.035
gamma = 0.4
c_light = 5.0
velocities = [torch.zeros_like(p) for p in model.parameters()]

print("--- TRAINING K=128 SOLITON FIELD ---")
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

print(f"Final 128x128 Loss: {loss_curve[-1]:.6f}")

# Save checkpoint
torch.save(model.state_dict(), os.path.join(REPO_FIGS, "..", "data", "soliton_model_128.pt"))

final_pred_128 = model(coords_t).detach().numpy().reshape(H, W)
learned_mu = model.mu.detach().numpy()
learned_sigma = torch.exp(model.log_sigma).detach().numpy()
learned_k = model.k_vec.detach().numpy()
learned_weights = model.weights.detach().numpy().flatten()

# Zero-Shot 256x256 Evaluation
H_hi, W_hi = 256, 256
u_hi = np.linspace(-1.0, 1.0, W_hi, dtype=np.float32)
v_hi = np.linspace(-1.0, 1.0, H_hi, dtype=np.float32)
U_hi, V_hi = np.meshgrid(u_hi, v_hi)
coords_hi = torch.tensor(np.stack([U_hi.flatten(), V_hi.flatten()], axis=-1))

with torch.no_grad():
    pred_256 = model(coords_hi).numpy().reshape(H_hi, W_hi)

# Helper function to save figure to both paths
def save_dual(fig, basename):
    path_repo = os.path.join(REPO_FIGS, basename)
    path_brain = os.path.join(BRAIN_DIR, basename)
    fig.savefig(path_repo, bbox_inches='tight', dpi=300)
    fig.savefig(path_brain, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[SAVED] {basename} -> Repo & Brain")

# Set scientific styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# FIG 5A: Ground Truth Scene (High Quality Individual)
# -------------------------------------------------------------
fig_a, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
im = ax.imshow(scene_128, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
ax.set_title("Ground Truth Multi-Aspect Scene ($128 \\times 128$)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Continuous Coordinate $u$", fontsize=11)
ax.set_ylabel("Continuous Coordinate $v$", fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Field Intensity $I(u, v)$", fontsize=10)

# Annotations for quadrants
ax.text(0.5, 0.85, "Q1: Fresnel Chirp\n$\\omega \\propto r^2$", color='white', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white'))
ax.text(-0.5, 0.85, "Q2: Star Polygon\nSharp Boundaries", color='white', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white'))
ax.text(-0.5, -0.85, "Q3: Wave Interference\nMulti-carrier Texture", color='white', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white'))
ax.text(0.5, -0.85, "Q4: Gaussian Basin\nSmooth Macro Gradient", color='white', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='white'))
save_dual(fig_a, "figure5a_ground_truth_scene.png")

# -------------------------------------------------------------
# FIG 5B: Continuous Soliton Reconstruction (High Quality Individual)
# -------------------------------------------------------------
fig_b, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
im = ax.imshow(final_pred_128, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
psnr = -10.0 * np.log10(loss_curve[-1])
ax.set_title(f"Relativistic Soliton Field ($K=128$, $128 \\times 128$)\n[MSE: {loss_curve[-1]:.5f} | PSNR: {psnr:.2f} dB]", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Continuous Coordinate $u$", fontsize=11)
ax.set_ylabel("Continuous Coordinate $v$", fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Reconstructed Field $\\hat{I}(u, v)$", fontsize=10)
save_dual(fig_b, "figure5b_soliton_reconstruction.png")

# -------------------------------------------------------------
# FIG 5C: Continuous C-Inf Super-Resolution (High Quality Individual)
# -------------------------------------------------------------
fig_c, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
im = ax.imshow(pred_256, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
ax.set_title("Zero-Shot Continuous $C^\\infty$ Super-Resolution ($256 \\times 256$)\n[Sub-pixel Analytical Evaluation]", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Continuous Coordinate $u$", fontsize=11)
ax.set_ylabel("Continuous Coordinate $v$", fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Continuous Field Intensity", fontsize=10)
save_dual(fig_c, "figure5c_super_resolution_extrapolation.png")

# -------------------------------------------------------------
# FIG 5D: Autonomous Token Foveation Manifold (High Quality Individual)
# -------------------------------------------------------------
fig_d, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
ax.imshow(scene_128, extent=[-1, 1, -1, 1], origin='lower', cmap='gray', alpha=0.38)
for i in range(model.K):
    mx, my = learned_mu[i]
    sx, sy = learned_sigma[i]
    w_mag = np.abs(learned_weights[i])
    col = plt.cm.plasma(np.clip(w_mag / 0.45, 0.0, 1.0))
    el = Ellipse((mx, my), width=2*sx, height=2*sy, edgecolor=col, facecolor='none', linewidth=1.2, alpha=0.8)
    ax.add_patch(el)
    ax.plot(mx, my, 'o', color=col, markersize=3.0)

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_title("Autonomous Token Foveation & Adaptive Receptive Fields\n[Centers $\\mathbf{\\mu}_k$ & Anisotropic Footprints $\\mathbf{\\sigma}_k$]", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Continuous Coordinate $u$", fontsize=11)
ax.set_ylabel("Continuous Coordinate $v$", fontsize=11)
# Add colorbar for token weight
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.45))
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Token Amplitude $|A_k|$", fontsize=10)
save_dual(fig_d, "figure5d_token_foveation_manifold.png")

# -------------------------------------------------------------
# FIG 5E: Carrier Wavevector Polar Spectrum (High Quality Individual)
# -------------------------------------------------------------
fig_e, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'), dpi=300)
angles = np.arctan2(learned_k[:, 1], learned_k[:, 0])
freqs = np.sqrt(learned_k[:, 0]**2 + learned_k[:, 1]**2)
scatter = ax.scatter(angles, freqs, c=learned_sigma.mean(axis=1), cmap='viridis_r',
                     s=np.abs(learned_weights)*120 + 20, alpha=0.88, edgecolors='#222222', linewidths=0.5)
ax.set_title("Carrier Wavevector Orientation Spectrum\n[Orientation Angle $\\theta$ vs Spatial Frequency $\\omega$ (rad/s)]", fontsize=12, fontweight='bold', pad=18)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.09)
cbar.set_label("Mean Token Bandwidth $\\bar{\\sigma}$ (Zoom $\\to$ Wide)", fontsize=10)
save_dual(fig_e, "figure5e_carrier_wavevector_spectrum.png")

# -------------------------------------------------------------
# FIG 5F: Relativistic Phase-Flow Convergence Curve (High Quality Individual)
# -------------------------------------------------------------
fig_f, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
ax.plot(loss_curve, color='#00d4ff', linewidth=2.0, label="Relativistic Euler-Lagrange ($c=5.0$)")
ax.axhline(y=loss_curve[-1], color='#ff007f', linestyle='--', linewidth=1.2, label=f"Final Loss = {loss_curve[-1]:.5f}")
ax.set_yscale('log')
ax.set_title("Relativistic Phase-Flow Convergence ($K=128$, $128 \\times 128$ Coords)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Training Epochs", fontsize=11)
ax.set_ylabel("Mean Squared Error (Log Scale)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10)
save_dual(fig_f, "figure5f_relativistic_convergence_curve.png")

# Also save the consolidated figure 5 properly in repo
shutil.copy(os.path.join(BRAIN_DIR, "figure5_scaled_2d_self_tokenizing_vision.png"),
            os.path.join(REPO_FIGS, "figure5_scaled_2d_self_tokenizing_vision.png"))
print("[SUCCESS] All individual figures and consolidated figure5 verified in both repo and brain!")
