import torch
from layerModule import MLP


x = torch.arange(0, 4)
print(x)
torch.save(x, 'x-file')

x_resume = torch.load('x-file')
print(x_resume)

y = torch.ones(4)
torch.save((x, y), 'xy-file')
x_r, y_r = torch.load('xy-file')
print(x_r, y_r)

my_dict = {'x_r': x_r, 'y_r': y_r}
torch.save(my_dict, 'dict-file')
my_dict_r = torch.load('dict-file')
print(my_dict_r)

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')

net_r = MLP()
net_r.load_state_dict(torch.load('mlp.params'))
net_r.eval()
Y_r = net_r(X)
print(Y == Y_r)

"""作业题解答"""
# 1. 能够恢复因意外而中断的训练过程，也有利于对实验的中间结果进行复现和检验
# 2. 在__init__()中就实例化网络架构对应类的对象，并载入保存的参数
# 3. 保证架构每一层都能和保存的参数字典一致
