import torch
from torch import nn
from IPython import display
from d2l import torch as d2l


"""超参数设置与数据加载"""
batch_size = 256
num_inputs = 28 * 28
num_outputs = 10
num_epochs = 5
lr = 0.1
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n  # 列表[0.0]重复n次构成一个新的list

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 利用列表解析机制，在中括号内使用for循环从而高效创建列表

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        d2l.plt.figure()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def softmax(X):
    exp_input = torch.exp(X)
    exp_sum = exp_input.sum(1, keepdim=True)
    return exp_input / exp_sum


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y.long()])  # 每行代表一个样本，选择标签所标志的正确样本的预测值参与运算


def accuracy(y_hat, y):
    y_pred = torch.argmax(y_hat, dim=1)
    if len(y.shape) > 1 and y.shape[1] > 1:  # 如果是one-hot表示的标签
        y = torch.argmax(y, dim=1)
    match = (y_pred == y)
    return match.sum()


def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()  # 主要是固定BN层和dropout层，保证测试阶段网络稳定
    metric = Accumulator(2)
    for X, y in data_iter:  # 一次一个batch
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)  # 三个值分别记录总损失，正确预测的样本数目以及样本总数
    for X, y in train_iter:  # 一次一个batch
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            #  如果使用Pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())  # l记录的是一个batch上loss的均值
        else:
            #  如果使用自定义的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"Loss = {train_metrics[0]:>3f}")
        print(f"Training Accuracy = {train_metrics[1]:>3f}")
        print(f"Testing Accuracy = {test_acc:>3f}\n")
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(torch.argmax(net(X), dim=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.tight_layout()  # 防止显示不全
    d2l.plt.show()  # 显示图像


"""自定义函数测试"""
# y = torch.zeros((2, 3))
# y[0][0] = 1
# y[1][0] = 1
# y_hat = torch.tensor([[0.3, 0.4, 0.3], [0.7, 0.1, 0.2]])
# print(cross_entropy(y_hat, y))
# print(accuracy(y_hat, y) / y_hat.shape[0])
# print(evaluate_accuracy(net, test_iter))

"""作业题解答"""
# 1. 从下面打印出的值可以看出，指数函数会导致数值爆炸问题，这些由于不正确的初始化或者数据噪声带来的
# 过大特征值会占据几乎所有预测概率甚至超过数据类型表示范围，从而影响预测结果
print(torch.exp(torch.tensor(50)))
# 2. 直接使用定义则输入的特征向量，那么如果某一行的预测值太接近0，同样会引起数值爆炸
# 3. 在计算softmax时需要进行均值迁移，对每一个特征向量，让所有分量减去最大分量的值，这样就能在不影响结果的情况下
# 避免数值爆炸的问题；但这样也可能使得迁移后的数据最小值变得非常小，进而使得概率趋于0导致交叉熵的数值爆炸（下溢），
# 就需要进行修正，直接添加补偿项p->p+1e-9。参考https://zhuanlan.zhihu.com/p/92714192
# 4. 并不总是，有时可能同时存在多个概率大的标签，选哪一个都有道理，此时就需要人工干预。在医疗诊断场景中比较常见
# 比如胃癌，肺癌，皮肤癌，肝癌的概率都在0.2左右，那么选取最大值的方案会带来很大误诊风险
# 5. 会导致需要的变换矩阵W太大，带来计算压力；另外也可能使得概率值被均摊，难以有效判别


"""训练与测试"""
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
predict_ch3(net, test_iter)



