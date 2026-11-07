# Patchless Soliton Vision: Autonomous Receptive Field Solitons on $\mathcal{C}^\infty$ Manifolds

[![Author: Shubham Barmase](https://img.shields.io/badge/Author-Shubham%20Barmase-blue.svg)](https://github.com/shu195)
[![Preprint: Paper 4](https://img.shields.io/badge/Paper-Preprint%20PDF-purple.svg)](paper/paper.pdf)
[![Simulator: 60 FPS WebGL](https://img.shields.io/badge/Live-60%20FPS%20Simulator-brightgreen.svg)](https://shu195.github.io/patchless-soliton-vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

> **A continuous-coordinate, patchless visual architecture that eliminates the discrete $16 \times 16$ Cartesian patch grid of Vision Transformers (ViT) entirely.**  
> Represents visual manifolds through an analytical linear superposition of continuous anisotropic Gabor-Morlet solitons $\psi_k(u, v)$ on $\mathcal{C}^\infty(\mathbb{R}^2)$ spatial coordinates with dynamic foveation, sub-pixel super-resolution, and zero depth hierarchy.

---

## 1. The Core Paradigm: Patchless Continuous Solitons

Standard Vision Transformers (ViT) and CNNs forcibly dice continuous visual scenes into rigid $16 \times 16$ or $4 \times 4$ pixel squares, creating non-physical boundary discontinuities, rigid resolution locks, and quadratic attention bottlenecks.

**Patchless Soliton Vision** replaces the Cartesian grid with continuous spatial wave packets (solitons) on $\Omega = [-1, 1]^2$:

$$\psi_k(u, v) = \exp\left( -\frac{(u - \mu_{u,k})^2}{2\sigma_{u,k}^2} - \frac{(v - \mu_{v,k})^2}{2\sigma_{v,k}^2} \right) \cos(\mathbf{k}_k \cdot \mathbf{x} + \phi_k)$$

### Autonomous Degrees of Freedom per Token:
- **Dynamic Receptive Field Center $\mathbf{\mu}_k \in [-1, 1]^2$:** Wanders continuously to anchor on salient edges, contours, and visual keypoints.
- **Adaptive Anisotropic Bandwidth $\mathbf{\sigma}_k \in \mathbb{R}^+ \times \mathbb{R}^+$:** Contracts ($\sigma \to 0$) to resolve fine high-frequency fringe details and expands ($\sigma \to \infty$) over smooth background fields, satisfying the **Heisenberg-Gabor uncertainty bound**: $\Delta x \cdot \Delta k \ge 1/2$.
- **Carrier Wavevector $\mathbf{k}_k \in \mathbb{R}^2$:** Directs spatial orientation $\theta = \arctan(k_v / k_u)$ and spatial frequency $\omega = \|\mathbf{k}\|_2$, mimicking simple-cell receptive fields in the primary visual cortex (V1).

---

## 2. Mathematical Architecture

### Continuous Analytical Reconstruction ($0$ Hidden Layers)
$$\hat{I}_c(u, v) = \sum_{k=1}^K A_{k, c} \psi_k(u, v) + b_c \quad (c \in \{1, \dots, C\})$$

### Patchless Semantic Perception Encoder
Continuous tokens $z_k$ are obtained by exact spatial integration without patch extraction:
$$z_{k, c} = \iint_\Omega I_c(u, v) \, \psi_k(u, v) \, du \, dv$$
This formulation is **100% resolution-agnostic**: an encoder trained on $32 \times 32$ images can ingest $256 \times 256$ or $1024 \times 1024$ continuous inputs without positional embedding interpolation or structural modification.

### Relativistic Euler-Lagrange Optimization
Parameter trajectories are governed by Lorentz-invariant momentum:
$$\dot{\mathbf{\theta}} = \frac{\mathbf{p}}{\sqrt{m^2 + \|\mathbf{p}\|^2 / c^2}} \implies \|\dot{\mathbf{\theta}}\| < c$$
This bounds parameter velocity below the speed of light $c$, preventing gradient explosion and wall ricochets during continuous spatial token migration.

---

## 3. Empirical Ground Truth Baselines

| Benchmark Task | Image Domain | Resolution | Solitons $K$ | Metric | Physical Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Multi-Aspect** | Chirp + Star + Waves + Basin | $128 \times 128$ | $128$ | **MSE: $0.1438$** (PSNR $8.42\text{ dB}$) | Foveation zoom: $\sigma_{\min}=0.1815$, $\sigma_{\max}=0.2312$ |
| **Photographic Ascent** | Real architectural scene | $128 \times 128$ | $128$ | **MSE: $0.5238$** | Sub-pixel continuous railing & step gradients |
| **Zero-Shot Super-Res** | Real architectural scene | $512 \times 512$ ($16\times$ density) | $128$ | **Analytical Continuity** | Infinitely differentiable $\mathcal{C}^\infty$ sub-pixel waveflow |
| **Handwritten Digits** | 10-Class character manifold | $8 \times 8$ ($1,797$ samples) | $32$ | **$\mathbf{91.33\%}$ Test Accuracy** | **0 Hidden Layers**, flat linear classifier |
| **CIFAR-10 RGB** | Real photographic color images | $32 \times 32 \times 3$ | $96$ | **PSNR: $\mathbf{15.43\text{ dB}}$** (MSE $0.0287$) | Multi-channel color emission $A_k \in \mathbb{R}^3$ |
| **CIFAR-10 10-Class** | Airplane, auto, bird, horse... | $32 \times 32 \times 3$ ($3,000$ samples) | $64$ | **$\mathbf{32.00\%}$ Test Accuracy** | **$16\times$ raw pixel compression**, 0 hidden layers ($3.2\times$ above random) |
| **$\mathrm{SO}(2)$ Lie Invariance** | Continuous rotations $\theta \in [0^\circ, 360^\circ]$ | Multi-scale | $32$ | **Stable Energy $\mathbb{E}[\|z\|^2]$** | Invariant under continuous coordinate rotations |

---

## 4. Repository Structure

```
patchless-soliton-vision/
├── benchmarks/
│   ├── benchmark_cifar10_solitons.py     # Real CIFAR-10 RGB reconstruction & classification
│   ├── benchmark_rigorous_vision_suite.py# Real images, 10-class classification, SO(2) rotation
│   ├── benchmark_scaled_2d_vision.py     # 128x128 multi-aspect benchmark (K=128)
│   └── render_individual_visuals.py      # 300 DPI publication figure generator
├── docs/
│   └── index.html                        # Live 60 FPS WebGL/HTML5 interactive simulator
├── figures/                              # 18 high-resolution 300 DPI publication figures
│   ├── figure5a_ground_truth_scene.png
│   ├── figure5b_soliton_reconstruction.png
│   ├── figure5c_super_resolution_extrapolation.png
│   ├── figure5d_token_foveation_manifold.png
│   ├── figure5e_carrier_wavevector_spectrum.png
│   ├── figure5f_relativistic_convergence_curve.png
│   ├── figure6a_real_image_ascent_reconstruction.png
│   ├── figure6b_real_image_super_resolution_512.png
│   ├── figure6c_ascent_token_foveation_railings.png
│   ├── figure6d_patchless_digit_classification.png
│   ├── figure6e_rotational_equivariance_so2.png
│   ├── figure7a_cifar10_real_images_and_soliton_reconstruction.png
│   ├── figure7b_cifar10_super_resolution_256.png
│   ├── figure7c_cifar10_rgb_soliton_foveation.png
│   ├── figure7d_cifar10_patchless_classification_matrix.png
│   └── figure7e_cifar10_relativistic_convergence.png
├── paper/
│   ├── paper.tex                         # LaTeX preprint source
│   ├── paper.pdf                         # Compiled 6-page preprint (6.52 MB, verified)
│   └── references.bib                    # BibTeX references
├── src/
│   ├── models/
│   │   ├── gabor_soliton.py              # Continuous 2D/3D Gabor-Morlet soliton field
│   │   └── patchless_encoder.py          # Continuous spatial integration encoder
│   └── optimizers/
│       ├── relativistic_momentum.py      # Lorentz velocity-clamped Euler-Lagrange optimizer
│       └── symplectic_integrator.py      # Störmer-Verlet symplectic volume preserver
├── tests/
│   └── test_models_and_optimizers.py     # 100% passing PyTest suite
├── context.md                            # Internal persistent state ledger (untracked)
├── pyproject.toml                        # Build & dependency configuration
└── README.md                             # This documentation
```

---

## 5. Quickstart & Verification

```bash
# Clone and install dependencies
git clone https://github.com/shu195/patchless-soliton-vision.git
cd patchless-soliton-vision
pip install -e .

# Run test suite
pytest tests/

# Execute CIFAR-10 benchmark
python benchmarks/benchmark_cifar10_solitons.py
```

---

## 6. Scientific Citation Trilogy

```bibtex
@article{barmase2026patchless,
  author    = {Shubham Barmase},
  title     = {Patchless Continuous Vision: Autonomous Receptive Field Solitons on C-Infinity Manifolds Without Grid Tokenization},
  year      = {2026},
  url       = {https://github.com/shu195/patchless-soliton-vision}
}

@article{barmase2026variational,
  author    = {Shubham Barmase},
  title     = {Variational Parameter Dynamics: The Courant-Friedrichs-Lewy Depth Catastrophe and Relativistic Momentum in Neural Networks},
  journal   = {CERN Zenodo},
  doi       = {10.5281/zenodo.24011081},
  year      = {2026}
}

@article{barmase2026spacetime,
  author    = {Shubham Barmase},
  title     = {Deterministic 4D Spacetime Geodesic Planning on Hypercomplex Toroidal Manifolds},
  journal   = {CERN Zenodo},
  doi       = {10.5281/zenodo.23423238},
  year      = {2026}
}

@article{barmase2026spatial,
  author    = {Shubham Barmase},
  title     = {Deterministic Spatial Planning on Hypercomplex Manifolds: 1-Pass Maze Navigation with Zero Backpropagation and 53 KB Memory Footprint},
  journal   = {CERN Zenodo},
  doi       = {10.5281/zenodo.23032915},
  year      = {2026}
}
```
