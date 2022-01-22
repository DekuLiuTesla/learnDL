import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1,  stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1),  stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)

"""作业题解答"""
# 1. 参看如下代码.可知计算结果与实验结果一致
X = torch.arange(0, 9, dtype=torch.float).reshape(3, 3)
conv2d = nn.Conv2d(1, 1, kernel_size=(2, 2), padding=1,  stride=2)
print(comp_conv2d(conv2d, X).shape)

# 3. 相当于对音频做二倍下采样

# 4. 能够大大减少运算量，提高计算速度
