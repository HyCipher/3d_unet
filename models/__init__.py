"""
Nets package.

Provides various neural network architectures for 3D image segmentation.
"""

# from .detect import UNet
from .detect_sep import UNet
from .kiunet_sep import KiUNet

__all__ = ['UNet', 'KiUNet']