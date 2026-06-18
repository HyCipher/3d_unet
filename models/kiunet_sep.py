import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Cascaded convolution used by the current segmentation models.
def conv1x3x3_xy(in_channels, out_channels):

	return nn.Conv3d(
		in_channels,
		out_channels,
		kernel_size=(1, 3, 3),
		stride=1,
		padding=(0, 1, 1),
		bias=True,
	)


def conv3x1x1_z(in_channels, out_channels):

	return nn.Conv3d(
		in_channels,
		out_channels,
		kernel_size=(3, 1, 1),
		stride=1,
		padding=(1, 0, 0),
		bias=True,
	)


def conv1x1x1(in_channels, out_channels):

	return nn.Conv3d(
		in_channels,
		out_channels,
		kernel_size=(1, 1, 1),
		stride=1,
		padding=(0, 0, 0),
		bias=True,
	)


def maxpool1x2x2():

	return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)


class CascadeConv3d(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(CascadeConv3d, self).__init__()
		self.conv_xy = conv1x3x3_xy(in_channels, out_channels)
		self.conv_z = conv3x1x1_z(out_channels, out_channels)
		self.fusion_conv = conv1x1x1(out_channels * 2, out_channels)
		self.norm = nn.InstanceNorm3d(out_channels, affine=True)

	def forward(self, x):
		feat_xy = self.conv_xy(x)
		feat_z = self.conv_z(feat_xy)
		feat = torch.cat([feat_xy, feat_z], dim=1)
		x = self.fusion_conv(feat)
		x = F.relu(self.norm(x))

		return x


class ConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(ConvBlock3, self).__init__()
		self.conv1 = CascadeConv3d(in_channels, out_channels)
		self.conv2 = CascadeConv3d(out_channels, out_channels)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)

		return x


class DownConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(DownConvBlock3, self).__init__()
		self.downsample = maxpool1x2x2()
		self.convblock = ConvBlock3(in_channels, out_channels)

	def forward(self, x):

		x = self.downsample(x)
		x = self.convblock(x)

		return x


class UpConv1x2x2(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(UpConv1x2x2, self).__init__()
		self.out_channels = out_channels

	def forward(self, x):
		x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
		return x


class UpConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(UpConvBlock3, self).__init__()
		self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
		self.reduce = conv1x1x1(in_channels, in_channels // 2)
		self.convblock = ConvBlock3(in_channels // 2 + out_channels, out_channels)

	def forward(self, xh, xv):

		xv = self.upsample(xv)
		xv = self.reduce(xv)
		x = torch.cat([xh, xv], dim=1)
		x = self.convblock(x)

		return x


class KiUpBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(KiUpBlock3, self).__init__()
		self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
		self.convblock = ConvBlock3(in_channels, out_channels)

	def forward(self, x):
		x = self.upsample(x)
		x = self.convblock(x)
		return x


class CrossResidualFusion(nn.Module):

	def __init__(self, src_channels, tgt_channels):

		super(CrossResidualFusion, self).__init__()
		self.proj = conv1x1x1(src_channels, tgt_channels)

	def forward(self, src, tgt):
		src = self.proj(src)
		src = F.interpolate(src, size=tgt.shape[2:], mode="trilinear", align_corners=False)
		return tgt + src


class KiUNet(nn.Module):
	"""3D Ki-UNet with cascaded conv units for synapse-oriented segmentation."""

	def __init__(self):

		super(KiUNet, self).__init__()
		fs = [16, 32, 64, 128]

		# U-Net encoder path.
		self.u1 = ConvBlock3(1, fs[0])
		self.u2 = DownConvBlock3(fs[0], fs[1])
		self.u3 = DownConvBlock3(fs[1], fs[2])
		self.u4 = DownConvBlock3(fs[2], fs[3])

		# Ki-Net encoder path (inverse scale progression).
		self.k1 = ConvBlock3(1, fs[0])
		self.k2 = KiUpBlock3(fs[0], fs[1])
		self.k3 = KiUpBlock3(fs[1], fs[2])
		self.k4 = KiUpBlock3(fs[2], fs[3])

		# Cross residual feature fusion between two paths.
		self.k1_to_u1 = CrossResidualFusion(fs[0], fs[0])
		self.u1_to_k1 = CrossResidualFusion(fs[0], fs[0])
		self.k2_to_u2 = CrossResidualFusion(fs[1], fs[1])
		self.u2_to_k2 = CrossResidualFusion(fs[1], fs[1])
		self.k3_to_u3 = CrossResidualFusion(fs[2], fs[2])
		self.u3_to_k3 = CrossResidualFusion(fs[2], fs[2])
		self.k4_to_u4 = CrossResidualFusion(fs[3], fs[3])
		self.u4_to_k4 = CrossResidualFusion(fs[3], fs[3])

		# U-Net decoder.
		self.ud3 = UpConvBlock3(fs[3], fs[2])
		self.ud2 = UpConvBlock3(fs[2], fs[1])
		self.ud1 = UpConvBlock3(fs[1], fs[0])

		# Ki branch decoder (inverse: downsample back to input scale).
		self.kd3 = DownConvBlock3(fs[3], fs[2])
		self.kd2 = DownConvBlock3(fs[2], fs[1])
		self.kd1 = DownConvBlock3(fs[1], fs[0])

		self.out_conv = conv1x1x1(fs[0] * 2, 1)

		self._initialize_weights()

	def forward(self, x):
		u1 = self.u1(x)
		k1 = self.k1(x)
		u1_base, k1_base = u1, k1
		u1 = self.k1_to_u1(k1_base, u1_base)
		k1 = self.u1_to_k1(u1_base, k1_base)

		u2 = self.u2(u1)
		k2 = self.k2(k1)
		u2_base, k2_base = u2, k2
		u2 = self.k2_to_u2(k2_base, u2_base)
		k2 = self.u2_to_k2(u2_base, k2_base)

		u3 = self.u3(u2)
		k3 = self.k3(k2)
		u3_base, k3_base = u3, k3
		u3 = self.k3_to_u3(k3_base, u3_base)
		k3 = self.u3_to_k3(u3_base, k3_base)

		u4 = self.u4(u3)
		k4 = self.k4(k3)
		u4_base, k4_base = u4, k4
		u4 = self.k4_to_u4(k4_base, u4_base)
		k4 = self.u4_to_k4(u4_base, k4_base)

		u_dec3 = self.ud3(u3, u4)
		u_dec2 = self.ud2(u2, u_dec3)
		u_dec1 = self.ud1(u1, u_dec2)

		k_dec3 = self.kd3(k4)
		k_dec2 = self.kd2(k_dec3)
		k_dec1 = self.kd1(k_dec2)

		k_dec1 = F.interpolate(k_dec1, size=u_dec1.shape[2:], mode="trilinear", align_corners=False)
		out = torch.cat([u_dec1, k_dec1], dim=1)
		out = self.out_conv(out)

		return out

	def _initialize_weights(self):

		conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv3d)]
		for m in conv_modules:
			n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3] * m.weight.shape[4]
			m.weight.data.normal_(0, np.sqrt(2.0 / n))
			if m.bias is not None:
				m.bias.data.zero_()
