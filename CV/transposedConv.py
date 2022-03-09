import torch
from torch import nn
from d2l import torch as d2l


def trans_conv(X, K):
    h, w = X.shape[-2:]
    k_h, k_w = K.shape[-2:]
    Y = torch.zeros((h+k_h-1, w+k_w-1))
    for i in range(w):
        for j in range(h):
            Y[i:i+k_w, j:j+k_h] += X[i, j] * K
    return Y


def kernel2matrix(K, X):
    h, w = X.shape[-2:]
    k_h, k_w = K.shape[-2:]
    out_h, out_w = h-k_h+1, w-k_w+1
    Y = torch.zeros(out_h*out_w, h*w)
    off_set = 0
    for i in range(out_h*out_w):
        for j in range(k_h):
            Y[i][off_set+j*w:off_set+j*w+k_w] = K[j, :]
        if (i+1) % out_w == 0:
            off_set = int(((i+1) / out_w) * w)
        else:
            off_set += 1
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))

X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)

W = kernel2matrix(K, X)
print(W)
Y_mat = torch.matmul(W, X.reshape(-1)).reshape(2, 2)
print(Y_mat == Y)
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))


"""作业题解答"""
# 1. 不同，对应同一矩阵W的卷积(W@x)与反置卷积(W.T@x)并非互逆运算

# 2. 效率可能会低一些，因为等价于把卷积核补零到图像大小(h*w)再做卷积，每个位置都需要h*w次乘法和h*w-1次加法，
#   总的运算量反而加大
