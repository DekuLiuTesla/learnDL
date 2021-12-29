import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, 1 + self.rand_weight))
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.randn(2, 20)
print(X)
y = net(X)
# print(y)

net_module = MLP()
y_module = net_module(X)
print(y_module)


myNet = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
y_myNet = myNet(X)
print(y_myNet)

fixedMLP = FixedHiddenMLP()
y_fixedMLP = fixedMLP(X)
print(y_fixedMLP)

nestMLP = nn.Sequential(
    NestMLP(),
    nn.Linear(16, 20),
    FixedHiddenMLP()
)
y_nestMLP = nestMLP(X)
print(y_nestMLP)

"""作业题解答"""
# 1. 以Python列表存储将无法记录块的顺序，因而将无法正确按顺序调用块执行功能
# 2. 代码实现如下：


class ParallelMLP(nn.Module):
    def __init__(self, net1, net2):
        super(ParallelMLP, self).__init__()
        self.net_1st = net1
        self.net_2nd = net2

    def forward(self, X):
        y1 = self.net_1st(X).unsqueeze(-1)  # 网络前向传播增加一个维度
        y2 = self.net_2nd(X).unsqueeze(-1)  # 网络前向传播增加一个维度
        return torch.cat((y1, y2), dim=-1)  # 在最后一个维度进行拼接


parallelNet = ParallelMLP(
    NestMLP(),
    NestMLP()
)
print(parallelNet(X))

# 3. 代码实现如下：


class RepeatedModule(nn.Module):
    def __init__(self, module, n):
        super(RepeatedModule, self).__init__()
        for i in range(n):
            self._modules[str(i)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 20)
)

repeatedModule = RepeatedModule(net, 5)
print(repeatedModule(X))
