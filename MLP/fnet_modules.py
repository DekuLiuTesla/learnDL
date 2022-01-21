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
        assert self.upsample in ['tanh', 'linear', None]
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
            x_P_min = torch.min(x_P, dim=1)
            x_P_max = torch.max(x_P, dim=1)
            x_P = ((x_P - x_P_min)/(x_P_max - x_P_min) - 0.5)*math.pi

        return x_P


class AP_Attention(nn.Module):
    def __init__(self):
        super(AP_Attention, self).__init__()

