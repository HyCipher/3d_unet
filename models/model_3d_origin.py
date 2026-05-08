import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3x3(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=True)

def maxpool2x2x2():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

class UpConv2x2x2(nn.Module):
    def __init__(self, channels):
        super(UpConv2x2x2, self).__init__()
        self.conv = nn.Conv3d(channels, channels // 2,
                              kernel_size=2, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = F.pad(x, (0,1,0,1,0,1))  # 3D padding
        x = self.conv(x)
        return x

# 3D卷积块
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.conv3 = conv3x3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x

# 3D下采样块
class DownConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock3D, self).__init__()
        self.maxpool = maxpool2x2x2()
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.conv3 = conv3x3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x

# 3D上采样块
class UpConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock3D, self).__init__()
        self.upconv = UpConv2x2x2(in_channels)
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.conv3 = conv3x3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, xh, xv):
        xv = self.upconv(xv)
        x = torch.cat([xh, xv], dim=1)
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x

# 3D UNet模型
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        fs = [16, 32, 64, 128, 256]
        
        # 编码器
        self.conv_in = ConvBlock3D(in_channels, fs[0])
        self.dconv1 = DownConvBlock3D(fs[0], fs[1])
        self.dconv2 = DownConvBlock3D(fs[1], fs[2])
        self.dconv3 = DownConvBlock3D(fs[2], fs[3])
        self.dconv4 = DownConvBlock3D(fs[3], fs[4])
        
        # 解码器
        self.uconv1 = UpConvBlock3D(fs[4], fs[3])
        self.uconv2 = UpConvBlock3D(fs[3], fs[2])
        self.uconv3 = UpConvBlock3D(fs[2], fs[1])
        self.uconv4 = UpConvBlock3D(fs[1], fs[0])
        
        # 输出层
        self.conv_out = nn.Conv3d(fs[0], out_channels, kernel_size=1)
        
        self._initialize_weights()
    
    def forward(self, x):
        # 编码器路径
        x1 = self.conv_in(x)      # 原始分辨率
        x2 = self.dconv1(x1)      # 1/2分辨率
        x3 = self.dconv2(x2)      # 1/4分辨率
        x4 = self.dconv3(x3)      # 1/8分辨率
        x5 = self.dconv4(x4)      # 1/16分辨率（最底层）
        
        # 解码器路径
        x6 = self.uconv1(x4, x5)  # 上采样到1/8分辨率
        x7 = self.uconv2(x3, x6)  # 上采样到1/4分辨率
        x8 = self.uconv3(x2, x7)  # 上采样到1/2分辨率
        x9 = self.uconv4(x1, x8)  # 上采样到原始分辨率
        
        # 输出
        x10 = self.conv_out(x9)
        
        return x10
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()