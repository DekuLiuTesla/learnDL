import torch
from torch import nn
from d2l import torch as d2l

LeNet = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=7, padding=3),
    # nn.Sigmoid(),
    nn.ReLU(),
    # nn.AvgPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(8, 20, kernel_size=5),
    # nn.Sigmoid(),
    nn.ReLU(),
    # nn.AvgPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(20 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
    # nn.Sigmoid(),
    # nn.Linear(36, 10)
)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # 主要是固定BN层和dropout层，保证测试阶段网络稳定
        if device is None:
            device = next(net.parameters()).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter:  # 一次一个batch
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  # @save
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    # timer为计数器，num_batches为一个epoch内总共的batch数
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 三个值分别记录总损失，正确预测的样本数目以及样本总数
        net.train()
        for i, (X, y) in enumerate(train_iter):  # 一次一个batch
            timer.start()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            #  如果使用Pytorch内置的优化器和损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # 每5个samples记录一次train_l与train_acc
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        # 在每个epoch结束时记录一次test_acc
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1,  (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


X = torch.rand((1, 1, 28, 28), dtype=torch.float32)
for layer in LeNet:
    X = layer(X)
    # .__class__.__name__用于获取类名
    print(layer.__class__.__name__, 'out shape: \t', X.shape)

'''数据准备'''
batch_size = 128
lr, num_epochs = 0.9, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''开启训练'''
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_ch6(LeNet, train_iter, test_iter, num_epochs, lr, device=device)
d2l.plt.show()


"""作业题解答"""
# 1. 换用MaxPooling之后能够进一步提高模型的表达能力，test acc=0.838，提高4个点左右

# 2&3. 在1的基础上对网络结构进行改进，得到了如下结果
#   a. 使用ReLU代替Sigmoid，能够进一步提高性能，test acc=0.874，但也让网络更容易过拟合
#   b. 调整第一层卷积核大小为3，padding为1，则test acc=0.818
#      调整第一层卷积核大小为7，padding为3，则test acc=0.879
#      可见采用更大的卷积核能够提高模型表达能力
#   c. 调整两个卷积层的输出通道数分别为8和20，则test acc=0.882
#      调整两个卷积层的输出通道数分别为4和10，则test acc=0.815
#   d. 删除最后一个全连接层，则test acc=0.886
#      在最后两层中间增加一个神经元数目为36的全连接层，则test acc=0.871
#      可见删除最后一个全连接层能够加强模型的表达能力
#   e. 在末尾增加一个核大小为3，输出通道数64的卷积层，则test acc=0.875，反而加重了过拟合的情况
#   f. 调整了lr和num_epoch，发现反而会引起性能下降，说明教程中的配置已经接近最优；此外将batch size调整
#      为128，能够更好地缓解过拟合现象，同时不损失精度

# 4. 参考以下的代码
with torch.no_grad():
    for X, y in test_iter:
        X = X.to(device)
        y = y.to(device)
        for i, layer in enumerate(LeNet):
            X = layer(X)
            if isinstance(layer, nn.ReLU):
                layer = 1
                s = X[:2, layer, :, :].to('cpu')
                titles = d2l.get_fashion_mnist_labels(y[:2])
                d2l.show_images(
                    d2l.reshape(s, (2, s.shape[1], s.shape[2])), 1, 2, titles=titles)
                d2l.plt.show()
        break


