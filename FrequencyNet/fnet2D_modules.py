import torch
import math
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch import nn
from d2l import torch as d2l



class AmplitudeMLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, dropout_rate=0.2):
        super(AmplitudeMLP, self).__init__()
        self.AMLP = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_inputs)
        )

    def forward(self, x):
        return self.AMLP(x)


class PhaseMLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, dropout_rate=0.2, range_type=None):
        super(PhaseMLP, self).__init__()
        # 控制相位范围的几种可能方法：
        # 1. tanh*pi
        # 2. 线性放缩
        # 3. 不做改变，在复原的时候指数的周期性会解决问题
        self.range_type = range_type
        assert self.range_type in ['tanh', 'linear', None]
        self.PMLP = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_inputs)
        )

    def forward(self, x):
        x_P = self.PMLP(x)
        if self.range_type == 'tanh':
            x_P = torch.tanh(x_P) * math.pi
        elif self.range_type == 'linear':
            x_P_min = torch.unsqueeze(torch.min(x_P, dim=1).values, dim=-1)
            x_P_max = torch.unsqueeze(torch.max(x_P, dim=1).values, dim=-1)
            x_P = ((x_P - x_P_min) / (x_P_max - x_P_min) - 0.5) * math.pi

        return x_P


class ComplexMLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=64, dropout_rate=0.2):
        super(ComplexMLP, self).__init__()
        self.CMLP = nn.Sequential(
            nn.Linear(num_inputs * 2, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_outputs)
        )

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x_h = torch.cat((x_real, x_imag), -1)
        output = self.CMLP(x_h)
        return output


class AP_Attention(nn.Module):
    def __init__(self):
        super(AP_Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)

    def forward(self, x_query, x_key):
        return self.attention(x_query, x_key, x_query, need_weights=False)[0]


class FLock(nn.Module):
    # num_hidden三个维度分别对应amplitudeMLP, phaseMLP, complexMLP的num_hidden
    def __init__(self, num_inputs, num_outputs, num_hidden=[16, 16, 16]):
        super(FLock, self).__init__()
        self.amplitudeMLP = AmplitudeMLP(num_inputs, num_hidden[0])
        self.phaseMLP = PhaseMLP(num_inputs, num_hidden[1])
        self.apAttention = AP_Attention()
        self.paAttention = AP_Attention()
        self.complexMLP = ComplexMLP(num_inputs, num_outputs, num_hidden[2])

    def forward(self, x):
        x_fft = torch.fft.fft(x)
        x_fft_in = torch.cat((x, x_fft.real, x_fft.imag), 1)

        x_fft_out = self.complexMLP(x_fft_in)

        if self.freq_major:
            return x_fft_out
        else:
            x_fft_real = x_fft_out[:, :x_fft_out.shape[1] // 2, :, :]
            x_fft_imag = x_fft_out[:, x_fft_out.shape[1] // 2:, :, :]
            x_fft = torch.complex(x_fft_real, x_fft_imag)
            return torch.fft.ifft(x_fft)


class GlobalFilter_free_size_f(nn.Module):
    def __init__(self, dim, h=48, w=56, mode='nearest'):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, norm='ortho')
        self.complex_weight = F.interpolate(self.complex_weight, x.shape[2:], mode=self.mode)
        weight = torch.view_as_complex(self.complex_weight)
        x = x + x * weight
        x = torch.fft.irfft2(x, s=(H, W), norm='ortho')
        return x


# 在每一步Residual模块都加入频域滤波结果。性能不佳，舍弃
class ResFreq_gf_full(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0):
        super(ResFreq_gf_full, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if h == 0 and w == 0:
            self.gfilter = None
        else:
            self.gfilter = GlobalFilter(input_channels, h, w)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
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


# 用级联的方式融合空域滤波与频域滤波的结果，但性能不佳
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


class GlobalFilter_free_size_s(nn.Module):
    def __init__(self, dim, size=48, mode='nearest'):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, size, size, 2, dtype=torch.float32) * 0.02)
        self.size_filter = size
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        max_wh = max(H, W)
        if not H == W:
            p_right, p_bottom = [(max_wh - s) for s in [W, H]]
            padding = (0, 0, p_right, p_bottom)
            x = F.pad(x, padding, 0, 'constant')
        x = F.interpolate(x, size=self.size_filter, mode=self.mode)
        x = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x + x * weight
        x = torch.fft.irfft2(x, s=(H, W), norm='ortho')
        x = F.interpolate(x, size=max_wh, mode=self.mode)
        x = x[:, :, :H, :W]

        return x
