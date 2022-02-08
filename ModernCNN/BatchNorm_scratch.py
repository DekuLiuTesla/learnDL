import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gama, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):
    # 用is_grad_enabled判断当前处于训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 预测模式下,直接使用传入的移动平均和所得的均值与方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式下
        assert X.dim() == 2 or X.dim() == 4, "Only output from FC layer or CONV layer can be batch normalized!"
        if X.dim() == 2:
            # 保持维度便于进行广播运算
            mean = torch.mean(X, dim=0, keepdim=True)
            var = torch.mean((X - mean)**2, dim=0, keepdim=True)
        else:
            # 保持维度便于进行广播运算
            mean = torch.mean(X, dim=(0, 2, 3), keepdim=True)
            var = torch.mean((X - mean) ** 2, dim=(0, 2, 3), keepdim=True)
        # 训练模式下，使用当前batch的均值与方差进行批量规范化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新滑动平均的均值与方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 进行缩放和偏移
    Y = X_hat * gama + beta

    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if X.device != self.moving_mean.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = \
            batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var)
        return Y


def train_two_net(net1, net2, train_iter, test_iter, num_epochs, lr1, lr2, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net1.apply(init_weights)
    net2.apply(init_weights)
    print('training on', device)
    net1.to(device)
    net2.to(device)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=lr1)
    loss1 = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=lr2)
    loss2 = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net1.train()
        net2.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer1.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat1 = net1(X)
            l1 = loss1(y_hat1, y)
            l1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat2 = net2(X)
            l2 = loss2(y_hat2, y)
            l2.backward()
            optimizer2.step()
            with torch.no_grad():
                metric.add((l1 - l2) * X.shape[0], d2l.accuracy(y_hat1, y) - d2l.accuracy(y_hat2, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net1, test_iter) - d2l.evaluate_accuracy_gpu(net2, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


LeNet_BN = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, 4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, 4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, 2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, 2), nn.Sigmoid(),
    nn.Linear(84, 10)
)

'''数据准备'''
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''开启训练'''
d2l.train_ch6(LeNet_BN, train_iter, test_iter, num_epochs, lr, device=d2l.try_gpu())
d2l.plt.show()


"""作业题解答"""
# 1. 我认为可以，因为是否使用偏置不改变样本方差，而使用偏置b则数据x和均值u会分别变成x+b和u+b，做差的结果仍然是(x-u)，
#   和没有偏置的情况是一样的，因此偏置可以被去掉

# 2. 不使用Batch Norm则学习率为0.9，使用则变成1.0，同时训练精度从81.1%提高到90.3%，而测试精度则从78.1%提高到78.3%。运行
# 下面的代码可以看到训练与测试精度提高量及loss减小量的曲线，最终使用Batch Norm比能够将train acc提高0.068, test acc提高0.066
# loss降低0.177，可见确实能够提高网络性能
'''第二题代码
LeNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

train_two_net(LeNet_BN, LeNet, train_iter, test_iter, num_epochs, 0.9, 1.0, device=d2l.try_gpu())
d2l.plt.show()
'''

# 3. 不删除BN层，则loss 0.268, train acc 0.901, test acc 0.861
#  只保留第二个和第四个BN层，则loss 0.289, train acc 0.894, test acc 0.787
#  只保留第一个和第三个BN层，则loss 0.286, train acc 0.893, test acc 0.820
#  只删除第一个BN层，则loss 0.272, train acc 0.899, test acc 0.769
#  只删除第二个BN层，则loss 0.269, train acc 0.901, test acc 0.774
#  只删除第三个BN层，则loss 0.272, train acc 0.901, test acc 0.857
#  只删除第四个BN层，则loss 0.283, train acc 0.895, test acc 0.857
#  可以看出删除全连接层后面的批量规范化对结果影响不大，因此实际上是可以删除的

# 4. 似乎是可以的，BN层兼具稳定训练过程和正则化的效果，将全连接阶段的BN层全部替换为弃置概率为0.5的Dropout层，
#   则loss 0.486, train acc 0.827, test acc 0.763，对比而言精度不如使用BN层的情况，而且train acc与test acc
#   的差距也更大，说明BN反而能够起到比dropout更好更稳定的正则化效果

# 5. 运行以下代码观察模型中gamma和beta的情况。从结果来看，随着网络加深，beta的值逐渐稳定到0附近，
# 而gamma的值则逐步稳定到1到2之间，说明BN层确实有助于稳定中间结果的数据分布
'''
for i, layer in enumerate(LeNet_BN):
    if isinstance(layer, BatchNorm):
        print(f'Layer {i}:')
        print(layer.gamma.reshape((-1,)))
        print(layer.beta.reshape((-1,)))
'''