import torch
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import GaborSolitonField2D, PatchlessSolitonEncoder
from src.optimizers import RelativisticEulerLagrange, StormerVerletSymplectic

def test_gabor_soliton_field_forward():
    model = GaborSolitonField2D(K=32, out_channels=3)
    coords = torch.randn(100, 2)
    out = model(coords)
    assert out.shape == (100, 3), f"Expected shape (100, 3), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_continuous_super_resolution():
    model = GaborSolitonField2D(K=16, out_channels=1)
    # Low res
    coords_low = torch.randn(64, 2)
    out_low = model(coords_low)
    # 4x higher res
    coords_hi = torch.randn(256, 2)
    out_hi = model(coords_hi)
    assert out_low.shape == (64, 1)
    assert out_hi.shape == (256, 1)

def test_patchless_encoder_variable_resolution():
    encoder = PatchlessSolitonEncoder(K=32, in_channels=3, out_dim=10)
    # Test on 32x32 CIFAR size
    img32 = torch.randn(4, 3, 32, 32)
    out32 = encoder(img32)
    assert out32.shape == (4, 10), f"Expected (4, 10), got {out32.shape}"

    # Test on 64x64 resolution without retraining or interpolation!
    img64 = torch.randn(4, 3, 64, 64)
    out64 = encoder(img64)
    assert out64.shape == (4, 10), f"Expected (4, 10), got {out64.shape}"

def test_relativistic_optimizer_speed_clamping():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = RelativisticEulerLagrange([param], lr=0.1, c_light=2.0)
    # Simulate massive gradient
    param.grad = torch.tensor([1000.0, 1000.0])
    opt.step()
    # Velocity should be clamped to <= c_light
    v = opt.state[param]['velocity']
    assert torch.norm(v).item() <= 2.0001, f"Velocity norm {torch.norm(v)} exceeded c_light 2.0"
