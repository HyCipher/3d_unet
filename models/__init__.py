"""
Nets package.

Provides various neural network architectures for 3D image segmentation.
"""

# from .detect import UNet
from .detect import UNet
from .kiunet_sep import KiUNet
from .detect_sep import SepUNet
from .model_3d_origin import UNet3D
from .resnet_detect_sep import ResSepUNet

__all__ = ['UNet', 'KiUNet', 'SepUNet', 'UNet3D', 'ResSepUNet']