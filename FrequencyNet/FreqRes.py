import torch
import math
from torch.nn import functional as F
from torch import nn

'''
想要让频域滤波拜托固定尺寸的限制，有几种可能的方法：
1. 固定频域滤波器尺寸，然后降采样到图像大小做滤波: 尺寸不同时性能骤降，行不通
2. 固定滤波器尺寸，然后在空间域先把长边放缩到一样的大小，再对剩余区域补零，随后转换到频域做滤波
3. 通过一定的变换从图像直接生成滤波器
'''


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


class GlobalFilter_free_size(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.taylor_dim_in = 5
        # self.taylor_mlp = nn.Conv2d(self.taylor_dim_in, 2 * dim, kernel_size=1, bias=True)
        self.taylor_mlp = nn.Sequential(
            nn.Conv2d(self.taylor_dim_in, dim, kernel_size=1, bias=True),
            nn.InstanceNorm2d(num_features=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.InstanceNorm2d(num_features=dim)
        )

    def forward(self, input):
        B, C, H, W = input.shape

        # 生成频域滤波器
        xx = torch.arange(0, H, dtype=torch.float32, device=input.device)
        yy = torch.arange(0, W // 2 + 1, dtype=torch.float32, device=input.device)
        grid_x, grid_y = torch.meshgrid(xx, yy)
        grid = torch.stack((grid_x, grid_y, grid_x * grid_x, grid_y * grid_y,
                            grid_y * grid_x), dim=-1)
        grid_dim = grid.reshape(H, W // 2 + 1, self.taylor_dim_in).permute(2, 0, 1).unsqueeze(0)
        complex_weight_dim = self.taylor_mlp(grid_dim).squeeze(0).permute(1, 2, 0)
        complex_weight = complex_weight_dim.reshape(H, W // 2 + 1, 2, C).permute(3, 0, 1, 2).contiguous() * 0.02

        # 频域滤波
        input_fft = torch.fft.rfft2(input, norm='ortho')
        weight = torch.view_as_complex(complex_weight)
        output_fft = input_fft + input_fft * weight
        output = torch.fft.irfft2(output_fft, s=(H, W), norm='ortho')
        return output


class Freq_Attn(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.complex_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        # self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, norm='ortho')
        x = self.complex_conv(x)
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


# 先进行频域滤波再做卷积
class ResFreq_gf(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0):
        super(ResFreq_gf, self).__init__()
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
                self.gfilter = GlobalFilter(input_channels, h, w)
        else:
            self.conv3 = None
            self.gfilter = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.gfilter:
            X = self.gfilter(X)
        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(Y)


# 进一步摆脱了对图像大小的限制
class ResFreq_gf_free(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0, mode='nearest'):
        super(ResFreq_gf_free, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.gfilter = GlobalFilter_free_size(input_channels)
        else:
            self.conv3 = None
            self.gfilter = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.gfilter:
            X = self.gfilter(X)
        if self.conv3:
            X = self.conv3(X)

        Y += X

        return F.relu(Y)
