import numpy as np
import math
import torch
from torch import nn
from d2l import torch as d2l


############# Define Training and Testing Function #############
def evaluate_loss(net, data_iter, loss):
    if isinstance(net, nn.Module):
        net.eval()  # 主要是固定BN层和dropout层，保证测试阶段网络稳定
    metric = d2l.Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), y.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-4, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, optimizer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


############# Data Generation #############
max_degree = 20  # 多项式最大阶数
n_train, n_test = 100, 100  # 训练集与测试集样本数目
true_w = np.zeros(max_degree)  # 预分配足够的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape((1, -1)))  # 形状不同，因此运算前会broadcast到同一大小
for i in range(max_degree):
    poly_features[:, i] = poly_features[:, i] / math.factorial(i)  # math.factorial(i)
labels = np.dot(poly_features, true_w)  # 沿着最后一个维度做内积运算，运算后该维度坍缩
labels += np.random.normal(size=labels.shape, scale=0.1)

# NumPy ndarray转换为tensor
# 列表中的每个元素是tensor化后的[true_w, features, poly_features, labels]构成的子列表
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
d2l.float32) for x in [true_w, features, poly_features, labels]]

# # 正常拟合的情况
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
# # 欠拟合情况
# train(poly_features[:n_train, :2], poly_features[n_train:, :2],
#       labels[:n_train], labels[n_train:])
# 过拟合情况
# train(poly_features[:n_train, :], poly_features[n_train:, :],
#       labels[:n_train], labels[n_train:], num_epochs=1500)

d2l.plt.show()
