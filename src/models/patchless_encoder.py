import torch
import torch.nn as nn
import numpy as np

class PatchlessSolitonEncoder(nn.Module):
    """
    Patchless Continuous Vision Encoder.
    Replaces discrete patch tokenization (ViT 16x16) with analytical spatial integration:
        z_{b, c, k} = iint I_c(u, v) * psi_k(u, v) du dv
    Resolution-independent, zero grid boundary bias, zero positional embedding interpolation.
    """
    def __init__(self, K=64, in_channels=3, out_dim=None):
        super().__init__()
        self.K = K
        self.in_channels = in_channels

        # Learnable Continuous Gabor Soliton Dictionary
        init_mu = np.random.uniform(-0.8, 0.8, size=(K, 2)).astype(np.float32)
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.log_sigma = nn.Parameter(torch.zeros(K, 2) - 1.4)

        angles = np.random.uniform(0, 2*np.pi, size=(K, 1))
        freqs = np.random.uniform(2.0, 25.0, size=(K, 1))
        init_k = np.hstack([freqs * np.cos(angles), freqs * np.sin(angles)]).astype(np.float32)
        self.k_vec = nn.Parameter(torch.tensor(init_k))
        self.phi = nn.Parameter(torch.zeros(K))

        # Optional projection head
        token_dim = in_channels * K
        if out_dim is not None:
            self.head = nn.Linear(token_dim, out_dim)
        else:
            self.head = nn.Identity()

    def get_coordinate_grid(self, H, W, device):
        u = torch.linspace(-1.0, 1.0, W, device=device)
        v = torch.linspace(-1.0, 1.0, H, device=device)
        V, U = torch.meshgrid(v, u, indexing='ij')
        return torch.stack([U.flatten(), V.flatten()], dim=-1) # (H*W, 2)

    def forward(self, x):
        """
        x: (B, C, H, W) where (H, W) can be arbitrary at runtime!
        Returns: (B, token_dim) or (B, out_dim)
        """
        B, C, H, W = x.shape
        coords = self.get_coordinate_grid(H, W, x.device) # (N, 2), N = H*W

        # Evaluate continuous solitons at (N, 2)
        uv_exp = coords.unsqueeze(1) # (N, 1, 2)
        mu_exp = self.mu.unsqueeze(0) # (1, K, 2)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-5

        diff = (uv_exp - mu_exp) / sigma
        dist_sq = torch.sum(diff**2, dim=-1) # (N, K)
        envelope = torch.exp(-0.5 * dist_sq)

        phase = torch.einsum('nd,kd->nk', coords, self.k_vec) + self.phi
        carrier = torch.cos(phase)

        Psi = envelope * carrier # (N, K)

        # Reshape images: (B, C, N)
        x_flat = x.view(B, C, H * W)

        # Continuous spatial integration: z = x_flat @ Psi -> (B, C, K)
        # Normalization by spatial area 2.0 * 2.0 / N
        dA = 4.0 / (H * W)
        z = torch.einsum('bcn,nk->bck', x_flat, Psi) * dA

        # Flatten channel tokens: (B, C * K)
        z_flat = z.view(B, C * self.K)
        return self.head(z_flat)
