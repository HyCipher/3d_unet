import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 级联卷积
def conv1x3x3_xy(in_channels, out_channels):

	return nn.Conv3d(in_channels, out_channels,
									kernel_size=(1,3,3), stride=1,
									padding=(0,1,1), bias=True)


def conv3x1x1_z(in_channels, out_channels):

	return nn.Conv3d(in_channels, out_channels,
									kernel_size=(3,1,1), stride=1,
									padding=(1,0,0), bias=True)


def conv1x1x1(in_channels, out_channels):

	return nn.Conv3d(in_channels, out_channels,
									kernel_size=(1,1,1), stride=1,
									padding=(0,0,0), bias=True)
 

def maxpool1x2x2():

	return nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2),
											padding=0)

# 分离卷积的下采样
class CascadeConv3d(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(CascadeConv3d, self).__init__()
		self.conv_xy = conv1x3x3_xy(in_channels, out_channels)
		self.conv_z = conv3x1x1_z(out_channels, out_channels)
		self.fusion_conv = conv1x1x1(out_channels*2, out_channels)
	
		self.norm_xy = nn.GroupNorm(num_groups=min(8, out_channels // 2), num_channels=out_channels, affine=True)
		self.norm_z = nn.GroupNorm(num_groups=min(8, out_channels // 2), num_channels=out_channels, affine=True)
		self.norm_fusion = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels, affine=True)

	def forward(self, x):
		feat_xy = self.conv_xy(x)
		feat_xy = self.norm_xy(feat_xy)
		feat_xy = F.relu(feat_xy)
    
		feat_z = self.conv_z(x)
		feat_z = self.norm_z(feat_z)
		feat_z = F.relu(feat_z)
	
		feat = torch.cat([feat_xy, feat_z], dim=1)
		x = self.fusion_conv(feat)
		x = self.norm_fusion(x)
		x = F.relu(x)

		return x


class UpConv1x2x2(nn.Module):

	def __init__(self, channels):

		super(UpConv1x2x2, self).__init__()
		self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
		self.conv = conv1x1x1(channels, channels//2)


	def forward(self, x):

		x = self.upsample(x)
		x = self.conv(x)

		return x


def concat(xh, xv):

	return torch.cat([xh, xv], dim=1)


# Convolution block
class ConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):
	
		super(ConvBlock3, self).__init__()
		self.conv1 = CascadeConv3d(in_channels, out_channels)
		self.conv2 = CascadeConv3d(out_channels, out_channels)
		self.conv3 = CascadeConv3d(out_channels, out_channels)
  
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
  
		return x


# Downconvolution block
class DownConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(DownConvBlock3, self).__init__()
		self.maxpool = maxpool1x2x2()
		self.convblock = ConvBlock3(in_channels, out_channels)

	def forward(self, x):

		x = self.maxpool(x)
		x = self.convblock(x)

		return x


# Upconvolution block
class UpConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(UpConvBlock3, self).__init__()
		self.upsample = UpConv1x2x2(in_channels)
		self.convblock = ConvBlock3(in_channels//2 + out_channels, out_channels)

	def forward(self, xh, xv):

		xv = self.upsample(xv)
		x = concat(xh, xv)
		x = self.convblock(x)

		return x


# Network architecture
class SepUNet(nn.Module):

	def __init__(self):

		super(SepUNet, self).__init__()
		fs = [16,32,64,128,256]
		self.conv_in = ConvBlock3(1, fs[0])
		self.dconv1 = DownConvBlock3(fs[0], fs[1])
		self.dconv2 = DownConvBlock3(fs[1], fs[2])
		self.dconv3 = DownConvBlock3(fs[2], fs[3])
		self.dconv4 = DownConvBlock3(fs[3], fs[4])
		
		self.uconv1 = UpConvBlock3(fs[4], fs[3])
		self.uconv2 = UpConvBlock3(fs[3], fs[2])
		self.uconv3 = UpConvBlock3(fs[2], fs[1])
		self.uconv4 = UpConvBlock3(fs[1], fs[0])
		self.conv_out = conv1x1x1(fs[0], 1)
  
		self._initialize_weights()

	def forward(self, x):

		x1 = self.conv_in(x)
		x2 = self.dconv1(x1)
		x3 = self.dconv2(x2)
		x4 = self.dconv3(x3)
		x5 = self.dconv4(x4)
		x6 = self.uconv1(x4, x5)
		x7 = self.uconv2(x3, x6)
		x8 = self.uconv3(x2, x7)
		x9 = self.uconv4(x1, x8)
		x10 = self.conv_out(x9)

		return x10


	def _initialize_weights(self):

		conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv3d)]
		for m in conv_modules:

			n = m.weight.shape[1]*m.weight.shape[2]*m.weight.shape[3]*m.weight.shape[4]
			m.weight.data.normal_(0, np.sqrt(2./n))
			if m.bias is not None:
				m.bias.data.zero_()