import torch
from torch import nn
from d2l import torch as d2l

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 28 * 28, 10, 256, 256
num_epochs, lr, batch_size = 10, 0.5, 256
dropout1, dropout2 = 0, 0
wd = 1e-4
varList_H1, varList_H2 = [], []


def dropout_layer(X, dropout):
    # dropout参数反映随机丢弃单元连接的概率
    assert 0 <= dropout <= 1  # 输入的范围不正确则会报错
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return X * mask / (1 - dropout)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, y):
    y_pred = torch.argmax(y_hat, dim=1)
    if len(y.shape) > 1 and y.shape[1] > 1:  # 如果是one-hot表示的标签
        y = torch.argmax(y, dim=1)
    match = (y_pred == y)
    return match.sum()


def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()  # 主要是固定BN层和dropout层，保证测试阶段网络稳定
    metric = d2l.Accumulator(2)
    for X, y in data_iter:  # 一次一个batch
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, nn.Module):
        net.train()
    metric = d2l.Accumulator(3)  # 三个值分别记录总损失，正确预测的样本数目以及样本总数
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


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）。"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 0.9],
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


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hidden2s,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        H1 += torch.normal(0, 0.01, size=H1.shape)  # 噪声注入
        # varList_H1.append(H1.std().pow(2).detach().numpy())  # 记录第一个隐藏层激活值的方差
        # 只在训练过程中加dropout
        if self.training:
            # 在第一个隐藏层后添加第一个dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        H2 += torch.normal(0, 0.01, size=H2.shape)  # 噪声注入
        # varList_H2.append(H2.std().pow(2).detach().numpy())  # 记录第二个隐藏层激活值的方差
        if self.training:
            # 在第二个隐藏层后添加第二个dropout
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)  # from scratch
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs)
)  # concise version
net.apply(init_weights)  # 会依次对每个层的参数应用init_weights函数做初始化
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 不使用权重衰减
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
d2l.plt.title("dropout1=" + str(dropout1) + ", dropout2=" + str(dropout2))
d2l.plt.show()


"""作业题解答"""
# 1. 保持教材的概率配置，训练10次，对最终的损失函数值、训练及测试精度进行统计和平均，得到交换前的性能结果，
# 随后交换两个层的dropout概率，类似的做10次训练并对结果统计平均，可以观察到交换后网络的性能出现显著退化，
# 测试与训练精度下降约0.01，损失函数值上升约0.02；因此放在前面的层dropout概率应该设小，后面的dropout概率可以设大。
# 此外，论文及博客推荐的概率在0.2-0.5之间，太小则效果不明显，太大则造成过拟合

# 2. 延长训练周期到25个epoch，不使用dropout时最终结果为：
# Loss = 0.253865
# Training Accuracy = 0.905067
# Testing Accuracy = 0.876300
# 而使用dropout，并遵循教材的配置，则最终结果为：
# Loss = 0.261569
# Training Accuracy = 0.901600
# Testing Accuracy = 0.850100
# 从曲线和数据上可以观察到，虽然加入dropout使得同样周期里的训练效果略有下降，但曲线更加平稳，
# 训练精度和测试精度的变化曲线更加接近。这说明dropout能够有效地抑制过拟合现象，但也需要更多的epoch来达到同样的训练效果

# 3. 使用scratch版本的代码，在激活之后，dropout之前对每个隐藏层的方差进行记录，并运行下面的代码在训练完成后显示
# 方差随时间变化的曲线和后8个epoch内的方差均值，可以发现在不使用dropout时，方差均值分别为：
# H1: 0.1766, H2: 0.1630
# 而在使用dropout时，方差均值分别为：
# H1: 0.1408, H2: 0.1664
# 可以看到dropout使得隐藏层的方差显著下降，权重趋于均匀、统一因而简单，有效降低了模型复杂度
# xAxis1 = torch.arange(0, len(varList_H1)) + 1
# xAxis2 = torch.arange(0, len(varList_H2)) + 1
# d2l.plt.plot(xAxis1, varList_H1, label='Var of H1')
# d2l.plt.plot(xAxis2, varList_H2, label='Var of H2')
# d2l.plt.title("dropout1=" + str(dropout1) + ", dropout2=" + str(dropout2))
# d2l.plt.legend()
# d2l.plt.grid()
# d2l.plt.show()
# print("Mean of varList_H1[512:]", sum(varList_H1[512:])/len(varList_H1[512:]))
# print("Mean of varList_H2[512:]", sum(varList_H2[512:])/len(varList_H2[512:]))

# 4. dropout测量本身是为了防止过拟合，希望每个神经元都能够学到有效的特征，避免相互依赖，得到类似训练并集成多个子网络的效果。
# 如果在测试的时候也使用dropout，被抑制掉的神经元就无法发挥作用，相当于只使用了一个子网络，
# 网络整体的特征学习能力反而会下降甚至引起欠拟合，因此需要在测试时关闭dropout

# 5. 使用concise版本的代码，训练10个epoch，仅使用dropout时的结果：
# Loss = 0.337873
# Training Accuracy = 0.876317
# Testing Accuracy = 0.865900
# 进一步使用weight decay=1e-4时的结果：
# Loss = 0.355407
# Training Accuracy = 0.869083
# Testing Accuracy = 0.858800
# 而如果关闭dropout，仅使用weight decay，则
# Loss = 0.313233
# Training Accuracy = 0.881733
# Testing Accuracy = 0.859400
# 可以看出，如果同时使用dropout与weight decay，训练效果反而劣化，主要原因在于两种方法都意在降低算法复杂度，
# 同时使用可能导致算法复杂度过分下降，进而产生欠拟合的趋势，导致网络性能下降

# 6. 如果对权重进行dropout，那么结果只会断开隐藏层单元之间的部分连接，但每一个隐藏层的单元数量及位置不会发生变化，
# 因此在训练过程中单元之间仍然会存在较强的依赖关系，对抗过拟合的效果不如对激活值进行dropout

# 7.不使用dropout，改为每个隐藏层激活值注入均值为0，方差为0.01的高斯噪声，训练10个epoch，得到如下结果：
# Loss = 0.286945
# Training Accuracy = 0.893150
# Testing Accuracy = 0.874100
# 可以发现得到了比dropout更好的训练效果

