import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    n_h, n_w = X.shape
    k_h, k_w = K.shape
    Y = torch.zeros(n_h-k_h+1, n_w-k_w+1)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            Y[i, j] = (X[i:i+k_h, j:j+k_w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight)+self.bias


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)

print(corr2d(X.t(), K))

# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2

for i in range(20):
    Y_hat = conv2d(X)
    l = (Y - Y_hat) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代更新权值
    conv2d.weight.data -= lr * conv2d.weight.grad
    if (i+1)%2 == 0:
        print(f"Epoch{i+1}: loss={l.sum()}")

print(conv2d.weight.data)

"""作业题解答"""
# 1. 参看如下代码.可以看出将K应用于X会在斜边处产生负的响应；转置X和转置K
# 效果相近，都会在边缘处得到正的响应，但结果的尺寸会略有不同
X = torch.ones(3, 3)
X = torch.triu(X)  # 下三角部分置零
print(X)
print(corr2d(X, K))  # K应用于X
print(X.t())
print(corr2d(X.t(), K))  # K应用于X的转置
print(corr2d(X, K.t()))  # K的转置应用于X

# 2. 参看如下代码。可以看出处理二维张量不会出错，
# 但对多通道和多个batch的情况无法处理
Conv2d = Conv2D(kernel_size=(1, 2))
Y = corr2d(X, K)
Y_hat = Conv2d(X)
l = (Y - Y_hat) ** 2
l = l.sum()
l.backward()
print(Conv2d.weight.grad)

# 3. 用矩阵乘法计算X与K之间的互相关，首先需要将卷积核zero-padding到图像大小，然后将二者从左到右、自上而下展平并通过循环移位构建矩阵，
# 如6.2.1中的运算，可以转化为以下两个矩阵的相乘：
# X = (0 1 2 3 4 5 6 7 8
#      3 4 5 6 7 8 0 1 2)
# K = (0 1 0 2 3 0 0 0 0
#      0 0 1 0 2 3 0 0 0).t()
# 记图像元素数目为N，则X与K对应的矩阵尺寸分别为：m_out*N 与 N*n_out，且前者每向下一行循环移位图像列的数目，后者每向右一列循环移位步进长度

# 4. 图像二阶导数g(x,y) = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)，即
# K = (0, 1, 0
#      1, -4, 1
#      0, 1, 0)
# 得到d次积分的结果应该至少需要(d+1)大小的核才能运算
# 积分核的形式应该是torch.ones(m, n)/(m*n)
