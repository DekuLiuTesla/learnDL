import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        # nn.Parameter()将其参数转变为可训练的类型parameter，并且绑定在module中，
        # 从而可以在训练中进行参数优化
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
        # 以下两行与以上两行有同样的效果
        # self.weight = nn.init.normal_(torch.empty(in_units, units))
        # self.bias = nn.init.normal_(torch.empty(units,))

    def forward(self, X):
        linear = torch.mm(X, self.weight) + self.bias
        return F.relu(linear)


layer1 = CenteredLayer()
print(layer1(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)

Y = net(torch.rand(4, 8))
print(Y.mean())

linear = MyLinear(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))

myNet = nn.Sequential(
    MyLinear(64, 8),
    MyLinear(8, 1)
)
print(myNet(torch.rand(2, 64)))

"""作业题解答"""


# 1. 代码如下
class DimensionReduction(nn.Module):
    def __init__(self, i, j, k):
        super(DimensionReduction, self).__init__()
        self.net = nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(i, j))

    def forward(self, X, Y):
        # 先用X和Y做矩阵乘法构成i*j矩阵，
        # 再用卷积层快捷地实现计算功能
        matrix = torch.bmm(x, torch.transpose(y, 1, 2))
        matrix = matrix.unsqueeze(1)  # B*1*i*j
        return self.net(matrix)  # B*5*i*j


myNet1 = DimensionReduction(2, 3, 5)
x = torch.ones(1, 2, 1)  # B*i*1
y = torch.rand(1, 3, 1)  # B*j*1
print(myNet1(x, y))


# 2. 代码如下
class HalfFFT(nn.Module):
    def __init__(self):
        super(HalfFFT, self).__init__()

    def forward(self, X):
        """
        Compute FFT and return half of it
        :param X: size = B*L
        :return: size = B*round(L/2)
        """
        half_len = round(X.shape[1]/2)
        X_f = torch.fft.fft(X)
        return X_f[:, :half_len]


myNet2 = HalfFFT()
print(myNet2(torch.rand(2, 3)))
