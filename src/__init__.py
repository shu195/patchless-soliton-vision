"""
Patchless Soliton Vision
Continuous Receptive Field Solitons on C-Infinity Manifolds Without Grid Tokenization.
Author: Shubham Barmase (shu195)
"""

__version__ = "1.0.0"
__author__ = "Shubham Barmase"

from .models.gabor_soliton import GaborSolitonField2D
from .models.patchless_encoder import PatchlessSolitonEncoder
from .optimizers.relativistic_momentum import RelativisticEulerLagrange
