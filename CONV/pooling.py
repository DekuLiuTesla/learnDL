import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    assert mode in ['max', 'avg', None]
    n_h, n_w = X.shape
    p_h, p_w = pool_size
    Y = torch.zeros(n_h - p_h + 1, n_w - p_w + 1)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if mode == 'max':
                Y[i, j] = torch.max(X[i:i + p_h, j:j + p_w])
            elif mode == 'avg':
                Y[i, j] = torch.mean(X[i:i + p_h, j:j + p_w])
    return Y


X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
pooling_layer = nn.MaxPool2d(3)
print(pooling_layer(X))
pooling_layer = nn.MaxPool2d(3, padding=1, stride=2)
print(pooling_layer(X))
pooling_layer = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pooling_layer(X))

X = torch.cat((X, X+1), dim=1)
pooling_layer = nn.MaxPool2d(3, padding=1, stride=2)
print(pooling_layer(X))

"""作业题解答"""
# 1. 可以的，实现的代码如下：
X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
K = torch.ones(2, 2)/4
print(d2l.corr2d(X, K))
print(pool2d(X, (2, 2), 'avg'))

# 2. MaxPooling涉及统计排序，可能无法用二维卷积层实现

# 3. 设输出的尺寸为h_out x w_out(具体值的计算参见6.3.2)，则总共的计算量为h_out x w_out x p_h x p_w x c_in个FLOPs

# 4. 因为MaxPooling涉及统计排序，而AveragePooling只是加权平均，在原理上有显著的不同，这也导致了不同的反传机理，MaxPooling反传时梯度保存在
# 最大值对应所在的位置，其他位置梯度为0；而AveragePooling反传时梯度在窗口内各个位置平摊。

# 5. 实际上很少需要，因为总是希望前向传播时能够保留特征图上最强的响应，而MinPooling则缺少这种功效，
# 并且MinPooling接在常用的ReLU后面会消除很多非负响应，经过几次MinPooling之后激活值几乎全是0.有用信息都被消除了，也无法训练；
# 似乎并不存在已知函数用于替代MinPooling

# 6. 有，但MaxPooling和AveragePooling在前向和反向传播过程都更加有利于计算，能够以较小的算力达到降维的目的

