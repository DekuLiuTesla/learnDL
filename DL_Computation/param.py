import torch
from torch import nn
from layerModule import NestMLP


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


def init_normal(m):
    if type(m) == nn.Linear:  # type()返回对象类型
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 0.5)
        nn.init.zeros_(m.bias)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 对于用Sequential定义的模型，才可以用索引进行层的访问
print(net[2].state_dict())

# 每个参数都是参数类的一个实例，是一个复合的对象，包括值、梯度和额外信息
# 因此可以用.data属性单独提取数值信息
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(net[2].bias.grad is None)

# 此处的*表示不定量参数，*arg本质是一个tuple
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net[2].state_dict()['bias'].data)  # 与net.state_dict()['2.bias'].data效果一致

regnet = nn.Sequential(
    block2(),
    nn.Linear(4, 1)
)
print(regnet(X))
print(regnet)
print(regnet[0][1][0].bias.data)  # 与regnet[0][1][0].state_dict()['bias'].data效果一致

net.apply(init_normal)  # 函数名作为参数，那么就可以用这个参数存放函数，并通过该参数调用相应函数
print(net[0].weight.data[0])
print(net[0].bias.data)

net.apply(init_constant)
print(net[0].weight.data[0])
print(net[0].bias.data)

net.apply(my_init)
print(net[0].weight[:2])

net[0].weight.data[:] += 1
net[0].weight.data[0][0] = 12
print(net[0].weight.data[0])

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象(绑定)，而不只是有相同的值
# 更新时共享层的梯度是各位置共享层梯度的总和
print(net[2].weight.data[0] == net[4].weight.data[0])

"""作业题解答"""
# 1. 选择5.1中的NestMLP，执行以下代码
nestMLP = NestMLP()
print(nestMLP)
print(nestMLP.net[0].weight.data[0])  # 长度为in_feature的长度，说明对应矩阵尺寸为out_feature*in_feature
print(nestMLP.linear.weight.data[2])
# 2. 参考https://pytorch.org/docs/stable/nn.init.html
# 3. 参考如下代码:
y = net(X).sum()
print("module: \n", net)
y.backward()
for name, params in net.named_parameters():
    print('-->name:', name, '-->grad_requirs:', params.requires_grad)
    print('  -->grad_value:', params.grad)
    print('  -->params:', params.data)

# 4. 共享参数包含以下几个优点：
#   · 节省内存，保持参数量不变的情况下扩大模型尺寸
#   · 对于图像识别中的CNN，共享参数使网络处理同一目标时，无论位于图像哪一个位置都能够
#     得到相近的处理结果（平移不变性），也就是卷积的处理方式
#   · 对于RNN，它在序列的各个时间步之间共享参数，因此可以很好地推广到不同序列长度的示例
#   · 对于自动编码器，编码器和解码器共享参数。 在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交
