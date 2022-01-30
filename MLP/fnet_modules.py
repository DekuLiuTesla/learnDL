import torch
import math
import torchvision.transforms as transforms
import torch.functional as F
from torch import nn


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
            x_P = ((x_P - x_P_min)/(x_P_max - x_P_min) - 0.5)*math.pi

        return x_P


class ComplexMLP(nn.Module):
    """MLP computed on complex inputs.

            Args:
                num_inputs (int): Input channel.
                num_outputs (int): Output channel.
                num_hidden (int): Channel of hidden layers.
                dropout_rate (float): Dropout rate applied in dropout layer.

    """  # noqa: W605

    def __init__(self, num_inputs, num_outputs, num_hidden=64, dropout_rate=0.2):
        super(ComplexMLP, self).__init__()
        self.CMLP = nn.Sequential(
            nn.Linear(num_inputs*2, num_hidden),
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
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.amplitudeMLP = AmplitudeMLP(num_inputs, num_hidden[0])
        self.phaseMLP = PhaseMLP(num_inputs, num_hidden[1])
        self.apAttention = AP_Attention()
        self.paAttention = AP_Attention()
        self.complexMLP = ComplexMLP(num_inputs, num_outputs, num_hidden[2])
        self.CMLP = nn.Sequential(
            nn.Linear(num_inputs*2, num_hidden[2]*4),
            nn.ReLU(),
            nn.Linear(num_hidden[2]*4, num_hidden[2]*4),
            nn.ReLU(),
            nn.Linear(num_hidden[2]*4, num_inputs*2)
        )

    def forward(self, x):
        x_fft = torch.fft.fft(x)
        x_fft_amplitude = x_fft.abs()
        x_fft_phase = torch.atan(x_fft.imag / x_fft.real)

        # x_fft_amplitude = torch.unsqueeze(self.amplitudeMLP(x_fft_amplitude), -1)
        # x_fft_phase = torch.unsqueeze(self.phaseMLP(x_fft_phase), -1)
        x_fft_amplitude = self.amplitudeMLP(x_fft_amplitude)
        x_fft_phase = self.phaseMLP(x_fft_phase)

        # xa_fft_amplitude = torch.squeeze(self.apAttention(x_fft_amplitude, x_fft_phase), -1)
        # xa_fft_phase = torch.squeeze(self.paAttention(x_fft_phase, x_fft_amplitude), -1)

        xa_fft = self.CMLP(torch.cat((x_fft_amplitude, x_fft_phase), -1))
        xa_fft_amplitude = xa_fft[:, :self.num_inputs]
        xa_fft_phase = xa_fft[:, self.num_inputs:]

        x_fft_real = xa_fft_amplitude * torch.cos(xa_fft_phase)
        x_fft_img = xa_fft_amplitude * torch.sin(xa_fft_phase)

        x_fft = torch.complex(x_fft_real, x_fft_img)
        x_h = torch.fft.ifft(x_fft)

        return self.complexMLP(x_h)
