"""
Nets package.

Provides various neural network architectures for 3D image segmentation.
"""

# from .detect import UNet
from .detect import UNet
from .kiunet_sep import KiUNet
from .detect_sep import SepUNet

__all__ = ['UNet', 'KiUNet', 'SepUNet']