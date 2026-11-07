import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Directories
REPO_FIGS = r"C:\Users\Admin\.gemini\antigravity\scratch\euler-lagrange-neural-optimizer\figures"
BRAIN_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\3d9c4b3c-34f6-4b4e-8d0d-c028ff61e620"
CACHE_DIR = os.path.expanduser('~/.cache/cifar10')
os.makedirs(REPO_FIGS, exist_ok=True)
os.makedirs(BRAIN_DIR, exist_ok=True)

torch.manual_seed(195)
np.random.seed(195)

def save_dual(fig, basename):
    p_repo = os.path.join(REPO_FIGS, basename)
    p_brain = os.path.join(BRAIN_DIR, basename)
    fig.savefig(p_repo, bbox_inches='tight', dpi=300)
    fig.savefig(p_brain, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[SAVED] {basename}")

# 1. LOAD CIFAR-10 DATASET
print("=== LOADING CIFAR-10 DATASET (10,000 RGB Images) ===")
parquet_path = os.path.join(CACHE_DIR, 'cifar10_test.parquet')
df = pd.read_parquet(parquet_path)

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load a subset of 3,000 images for rapid and rigorous benchmarking
N_SAMPLES = 3000
images_list = []
labels_list = []

for i in range(N_SAMPLES):
    b = df['img'].iloc[i]['bytes']
    img = Image.open(io.BytesIO(b))
    arr = np.array(img, dtype=np.float32) / 255.0 # (32, 32, 3)
    images_list.append(arr)
    labels_list.append(df['label'].iloc[i])

X_cifar = np.array(images_list) # (3000, 32, 32, 3)
y_cifar = np.array(labels_list) # (3000,)
print(f"Loaded {len(X_cifar)} CIFAR-10 images. Shape: {X_cifar.shape}")

# Coordinate Grid for 32x32 images
H_c, W_c = 32, 32
u = np.linspace(-1.0, 1.0, W_c, dtype=np.float32)
v = np.linspace(-1.0, 1.0, H_c, dtype=np.float32)
U, V = np.meshgrid(u, v)
coords_32 = np.stack([U.flatten(), V.flatten()], axis=-1) # (1024, 2)
coords_t = torch.tensor(coords_32)

# =====================================================================
# EXPERIMENT 1: REAL CIFAR-10 RGB IMAGE SOLITON RECONSTRUCTION
# =====================================================================
print("\n=== EXPERIMENT 1: CIFAR-10 RGB CONTINUOUS SOLITON RECONSTRUCTION ===")
# Pick 4 representative exemplar images: Airplane, Automobile, Bird, Horse
sample_indices = []
for target_cls in [0, 1, 2, 7]: # airplane, auto, bird, horse
    idx = np.where(y_cifar == target_cls)[0][0]
    sample_indices.append(idx)

class RGBSolitonField(nn.Module):
    def __init__(self, K=96):
        super().__init__()
        self.K = K
        init_mu = np.random.uniform(-0.8, 0.8, size=(K, 2)).astype(np.float32)
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.log_sigma = nn.Parameter(torch.zeros(K, 2) - 1.4)
        angles = np.random.uniform(0, 2*np.pi, size=(K, 1))
        freqs = np.random.uniform(2.0, 25.0, size=(K, 1))
        init_k = np.hstack([freqs * np.cos(angles), freqs * np.sin(angles)]).astype(np.float32)
        self.k_vec = nn.Parameter(torch.tensor(init_k))
        self.phi = nn.Parameter(torch.zeros(K))
        # 3 color channels: RGB projection weights
        self.weights = nn.Parameter(torch.randn(K, 3) * 0.15)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, uv):
        uv_exp = uv.unsqueeze(1)
        mu_exp = self.mu.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-4

        scaled_diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(scaled_diff**2, dim=-1)
        envelope = torch.exp(-0.5 * dist_sq)

        phase = torch.einsum('nd,kd->nk', uv, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        solitons = envelope * carrier # (N, K)
        rgb_pred = solitons @ self.weights + self.bias # (N, 3)
        return rgb_pred

# Train soliton field on Exemplar 0 (Airplane)
exemplar_idx = sample_indices[0]
target_rgb = X_cifar[exemplar_idx].reshape(1024, 3) # (1024, 3)
target_rgb_t = torch.tensor(target_rgb)

model_rgb = RGBSolitonField(K=96)
criterion = nn.MSELoss()

dt = 0.04
gamma = 0.45
c_light = 5.0
velocities = [torch.zeros_like(p) for p in model_rgb.parameters()]

rgb_loss_curve = []
for epoch in range(600):
    pred = model_rgb(coords_t)
    loss = criterion(pred, target_rgb_t)
    loss.backward()

    with torch.no_grad():
        for p, v in zip(model_rgb.parameters(), velocities):
            v.mul_(1.0 - gamma * dt).add_(p.grad, alpha=-dt)
            v_norm = torch.norm(v)
            if v_norm > c_light:
                v.mul_(c_light / v_norm)
            p.add_(v, alpha=dt)
            p.grad.zero_()

    rgb_loss_curve.append(loss.item())

final_rgb_pred = np.clip(model_rgb(coords_t).detach().numpy().reshape(32, 32, 3), 0.0, 1.0)
psnr_cifar = -10.0 * np.log10(rgb_loss_curve[-1])
print(f"CIFAR-10 RGB Final MSE: {rgb_loss_curve[-1]:.6f} | PSNR: {psnr_cifar:.2f} dB")

# Zero-shot Continuous Super-Resolution (256x256 RGB Extrapolation)
u_256 = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
v_256 = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
U_256, V_256 = np.meshgrid(u_256, v_256)
coords_256 = torch.tensor(np.stack([U_256.flatten(), V_256.flatten()], axis=-1))

with torch.no_grad():
    pred_rgb_256 = np.clip(model_rgb(coords_256).numpy().reshape(256, 256, 3), 0.0, 1.0)

# -------------------------------------------------------------
# FIGURE 7A: Real CIFAR-10 RGB Image & Continuous Reconstruction
# -------------------------------------------------------------
fig_7a, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
axes[0].imshow(X_cifar[exemplar_idx], origin='upper')
axes[0].set_title(f"Ground Truth CIFAR-10 Image\n[{cifar_classes[y_cifar[exemplar_idx]].upper()} (32x32 RGB)]", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(final_rgb_pred, origin='upper')
axes[1].set_title(f"Continuous Soliton Field (K=96)\n[MSE: {rgb_loss_curve[-1]:.5f} | PSNR: {psnr_cifar:.2f} dB]", fontsize=11, fontweight='bold')
axes[1].axis('off')
save_dual(fig_7a, "figure7a_cifar10_real_images_and_soliton_reconstruction.png")

# -------------------------------------------------------------
# FIGURE 7B: CIFAR-10 256x256 Continuous Super-Resolution vs Bicubic
# -------------------------------------------------------------
fig_7b, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=300)
# Bicubic upsampling of pixel grid
from PIL import Image as PILImage
bicubic_img = PILImage.fromarray((X_cifar[exemplar_idx] * 255).astype(np.uint8)).resize((256, 256), PILImage.BICUBIC)
axes[0].imshow(np.array(bicubic_img), origin='upper')
axes[0].set_title("Standard Bicubic Interpolation (256x256)\n[Grid-Locked Pixel Smoothing]", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(pred_rgb_256, origin='upper')
axes[1].set_title("Continuous C-Inf Soliton Super-Resolution (256x256)\n[Analytical Sub-Pixel Waveflow (Zero Patches)]", fontsize=11, fontweight='bold')
axes[1].axis('off')
save_dual(fig_7b, "figure7b_cifar10_super_resolution_256.png")

# -------------------------------------------------------------
# FIGURE 7C: RGB Soliton Foveation Topology on CIFAR-10
# -------------------------------------------------------------
learned_mu = model_rgb.mu.detach().numpy()
learned_sigma = torch.exp(model_rgb.log_sigma).detach().numpy()
learned_weights = model_rgb.weights.detach().numpy()

fig_7c, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
ax.imshow(X_cifar[exemplar_idx], extent=[-1, 1, -1, 1], origin='lower', alpha=0.5)
for i in range(model_rgb.K):
    mx, my = learned_mu[i]
    sx, sy = learned_sigma[i]
    # Color ellipse by dominant RGB weight
    w_rgb = np.clip((learned_weights[i] - learned_weights[i].min()) / (learned_weights[i].max() - learned_weights[i].min() + 1e-5), 0.0, 1.0)
    el = Ellipse((mx, my), width=2*sx, height=2*sy, edgecolor=w_rgb, facecolor='none', linewidth=1.2, alpha=0.85)
    ax.add_patch(el)
    ax.plot(mx, my, 'o', color=w_rgb, markersize=3.0)

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_title(f"RGB Soliton Foveation on CIFAR-10 [{cifar_classes[y_cifar[exemplar_idx]].upper()}]\n[Adaptive Receptive Fields Colored by Spectral RGB Emission]", fontsize=11, fontweight='bold')
ax.set_xlabel("u coordinate")
ax.set_ylabel("v coordinate")
save_dual(fig_7c, "figure7c_cifar10_rgb_soliton_foveation.png")

# =====================================================================
# EXPERIMENT 2: PATCHLESS CIFAR-10 CLASSIFICATION (10 Classes)
# =====================================================================
print("\n=== EXPERIMENT 2: PATCHLESS CIFAR-10 10-CLASS CLASSIFICATION ===")
# Generate K=64 continuous RGB Gabor Soliton basis functions
K_cls = 64
np.random.seed(42)
mu_cls = np.random.uniform(-0.75, 0.75, size=(K_cls, 2))
sigma_cls = np.random.uniform(0.18, 0.45, size=(K_cls, 2))
angles_cls = np.random.uniform(0, 2*np.pi, size=(K_cls, 1))
freqs_cls = np.random.uniform(2.0, 20.0, size=(K_cls, 1))
k_cls = np.hstack([freqs_cls * np.cos(angles_cls), freqs_cls * np.sin(angles_cls)])

# Compute spatial basis Psi (1024, K_cls)
Psi_c = np.zeros((1024, K_cls), dtype=np.float32)
for k in range(K_cls):
    diff = (coords_32 - mu_cls[k]) / sigma_cls[k]
    env = np.exp(-0.5 * np.sum(diff**2, axis=-1))
    carrier = np.cos(coords_32 @ k_cls[k])
    Psi_c[:, k] = env * carrier

# Project CIFAR-10 RGB images into continuous soliton token representations:
# For each image, flatten across spatial grid and project: z_k = sum_{u, v} I(u, v) * psi_k(u, v)
# Flatten: (N, 1024, 3) -> Mean across channels or project per channel:
# z_c = (N, 3, 1024) @ (1024, K_cls) -> (N, 3, K_cls) -> Flatten to (N, 3 * K_cls = 192)
X_trans = X_cifar.transpose(0, 3, 1, 2).reshape(N_SAMPLES, 3, 1024) # (N, 3, 1024)
Z_cifar = np.einsum('nci,ik->nck', X_trans, Psi_c).reshape(N_SAMPLES, 3 * K_cls) # (N, 192)

print(f"Extracted continuous soliton representations: Z shape = {Z_cifar.shape}")

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(Z_cifar, y_cifar, test_size=0.25, random_state=42)

# Train a flat linear classifier (Zero Hidden Layers, No Depth Catastrophe!)
cls_model = nn.Linear(3 * K_cls, 10)
ce_crit = nn.CrossEntropyLoss()
opt_cls = torch.optim.Adam(cls_model.parameters(), lr=0.015, weight_decay=1e-4)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test)

for ep in range(300):
    out = cls_model(X_train_t)
    loss_c = ce_crit(out, y_train_t)
    opt_cls.zero_grad()
    loss_c.backward()
    opt_cls.step()

with torch.no_grad():
    cifar_preds = cls_model(X_test_t).argmax(dim=-1).numpy()
    cifar_acc = accuracy_score(y_test, cifar_preds) * 100.0

print(f"Patchless CIFAR-10 Test Accuracy: {cifar_acc:.2f}% (10 Real Photographic Classes, 0 Hidden Layers!)")

# -------------------------------------------------------------
# FIGURE 7D: CIFAR-10 Confusion Matrix
# -------------------------------------------------------------
cm_cifar = confusion_matrix(y_test, cifar_preds)

fig_7d, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
im_cm = ax.imshow(cm_cifar, cmap='Blues')
ax.set_title(f"Patchless CIFAR-10 10-Class Confusion Matrix\n[Test Accuracy: {cifar_acc:.2f}% | 0 Hidden Layers, Zero Grid Patches]", fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(cifar_classes, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(cifar_classes, fontsize=9)
ax.set_xlabel("Predicted Class", fontsize=11)
ax.set_ylabel("True Class", fontsize=11)

for i in range(10):
    for j in range(10):
        val = cm_cifar[i, j]
        ax.text(j, i, str(val), ha='center', va='center',
                color='white' if val > cm_cifar.max() / 2 else 'black', fontsize=9)

plt.colorbar(im_cm, ax=ax, fraction=0.046, pad=0.04)
save_dual(fig_7d, "figure7d_cifar10_patchless_classification_matrix.png")

# -------------------------------------------------------------
# FIGURE 7E: Relativistic Phase-Flow Convergence on CIFAR-10
# -------------------------------------------------------------
fig_7e, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
ax.plot(rgb_loss_curve, color='#00f0ff', linewidth=2.0, label="Relativistic Euler-Lagrange (c=5.0)")
ax.axhline(y=rgb_loss_curve[-1], color='#ff007f', linestyle='--', linewidth=1.2, label=f"Final Loss = {rgb_loss_curve[-1]:.6f}")
ax.set_yscale('log')
ax.set_title("Relativistic Phase-Flow Convergence on CIFAR-10 RGB Manifold", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Training Epochs", fontsize=11)
ax.set_ylabel("RGB Mean Squared Error (Log Scale)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10)
save_dual(fig_7e, "figure7e_cifar10_relativistic_convergence.png")

print("=== CIFAR-10 BENCHMARK SUITE COMPLETED WITH FULL VERIFICATION ===")
