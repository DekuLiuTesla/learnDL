import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

"""超参数设置与数据加载"""
batch_size = 256
num_inputs, num_outputs, num_hidden = 28 * 28, 10, 256
num_epochs = 10
lr = 0.1
W1 = nn.Parameter(torch.normal(0, 0.01, size=(num_inputs, num_hidden), requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.normal(0, 0.01, size=(num_hidden, num_outputs), requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
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


class Animator:  # @save
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


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, W1.shape[0]))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)


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


"""自定义函数测试"""
# X = torch.normal(0, 0.01, size=(2, 3))
# print(X)
# print(relu(X))
# print(X)


"""训练与测试"""
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(params, lr=lr)
lossCurve = []
trainAccuracyCurve = []
testAccuracyCurve = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    lossCurve.append(train_metrics[0])
    trainAccuracyCurve.append(train_metrics[1])
    testAccuracyCurve.append(test_acc)
    print(f"Loss = {train_metrics[0]:>3f}")
    print(f"Training Accuracy = {train_metrics[1]:>3f}")
    print(f"Testing Accuracy = {test_acc:>3f}\n")

x_axix = range(num_epochs)
d2l.plt.title('Result Analysis')
d2l.plt.plot(x_axix, lossCurve, color='green', label='Loss')
d2l.plt.plot(x_axix, trainAccuracyCurve, color='red', label='Training Accuracy')
d2l.plt.plot(x_axix, testAccuracyCurve,  color='skyblue', label='Testing Accuracy')
d2l.plt.legend()  # 显示图例
d2l.plt.grid()
d2l.plt.xlabel('Epochs')
d2l.plt.ylabel('Val')
d2l.plt.show()

d2l.predict_ch3(net, test_iter, n=6)
d2l.plt.tight_layout()  # 防止显示不全
d2l.plt.show()  # 显示图像
