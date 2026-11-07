import torch
import torch.nn as nn
import numpy as np

class GaborSolitonField2D(nn.Module):
    """
    2D Continuous Gabor-Morlet Soliton Field.
    Represents visual continuous manifolds in C^inf space with zero discrete grid patches:
        psi_k(u, v) = exp( - ((u - mu_{u,k})^2 / (2 sigma_{u,k}^2) + (v - mu_{v,k})^2 / (2 sigma_{v,k}^2)) ) * cos(k_{u,k} u + k_{v,k} v + phi_k)
        I_hat_c(u, v) = sum_{k=1}^K A_{k, c} psi_k(u, v) + b_c
    """
    def __init__(self, K=128, out_channels=3, init_sigma=0.25):
        super().__init__()
        self.K = K
        self.out_channels = out_channels

        # Dynamic Receptive Field Centers mu in [-0.85, 0.85]^2
        init_mu = np.random.uniform(-0.85, 0.85, size=(K, 2)).astype(np.float32)
        self.mu = nn.Parameter(torch.tensor(init_mu))

        # Anisotropic Spatial Bandwidths sigma
        init_log_sigma = np.log(init_sigma)
        self.log_sigma = nn.Parameter(torch.full((K, 2), init_log_sigma, dtype=torch.float32))

        # Carrier Wavevectors k (direction & frequency)
        angles = np.random.uniform(0, 2 * np.pi, size=(K, 1))
        freqs = np.random.uniform(2.0, 30.0, size=(K, 1))
        init_k = np.hstack([freqs * np.cos(angles), freqs * np.sin(angles)]).astype(np.float32)
        self.k_vec = nn.Parameter(torch.tensor(init_k))

        # Phase offsets
        self.phi = nn.Parameter(torch.zeros(K))

        # Color emission amplitudes A_k in R^{K x out_channels}
        self.weights = nn.Parameter(torch.randn(K, out_channels) * 0.15)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def evaluate_solitons(self, uv):
        """
        Evaluate raw soliton basis activations psi_k(u, v) at coordinates uv (N, 2).
        Returns: (N, K)
        """
        uv_exp = uv.unsqueeze(1) # (N, 1, 2)
        mu_exp = self.mu.unsqueeze(0) # (1, K, 2)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-5 # (1, K, 2)

        # Anisotropic Gaussian envelope
        scaled_diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(scaled_diff**2, dim=-1) # (N, K)
        envelope = torch.exp(-0.5 * dist_sq)

        # Directional carrier wave
        phase = torch.einsum('nd,kd->nk', uv, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        return envelope * carrier # (N, K)

    def forward(self, uv):
        """
        Continuous analytical reconstruction at coordinates uv (N, 2).
        Returns: (N, out_channels)
        """
        solitons = self.evaluate_solitons(uv) # (N, K)
        return solitons @ self.weights + self.bias

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def carrier_frequencies(self):
        return torch.norm(self.k_vec, dim=-1)

    @property
    def carrier_angles(self):
        return torch.atan2(self.k_vec[:, 1], self.k_vec[:, 0])
