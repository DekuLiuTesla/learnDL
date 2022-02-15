import torch
import math
from torch.nn import functional as F
from torch import nn


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x + x * weight
        x = torch.fft.irfft2(x, s=(H, W), norm='ortho')
        return x


class Residual_Freq(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0):
        super(Residual_Freq, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            if h == 0 and w == 0:
                self.gfilter = None
            else:
                self.gfilter = GlobalFilter(num_channels, h, w)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
            if self.gfilter:
                X = self.gfilter(X)
        Y += X

        return F.relu(Y)


class Residual_cat(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0):
        super(Residual_cat, self).__init__()
        if use_1x1conv:
            space_channels = num_channels // 2
            freq_channels = num_channels - space_channels
            self.conv1 = nn.Conv2d(input_channels, space_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(space_channels, space_channels,
                                   kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(input_channels, freq_channels,
                                   kernel_size=2, stride=strides)
            if h == 0 and w == 0:
                self.gfilter = None
            else:
                self.gfilter = GlobalFilter(freq_channels, h, w)
            self.bn1 = nn.BatchNorm2d(space_channels)
            self.bn2 = nn.BatchNorm2d(space_channels)
        else:
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
            Y += X
            if self.gfilter:
                X = self.gfilter(X)
            Y = torch.cat((Y, X), 1)
        else:
            Y += X

        return F.relu(Y)
