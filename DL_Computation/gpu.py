import torch
import time
from torch import nn


def try_gpu(i=0):  # @save
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device('cpu')


def my_init(m):
    if type(m) == nn.Linear:
        nn.init.ones_(m.weight)


print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:0'))
print('Number of available devices: ', torch.cuda.device_count())

print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

x = torch.randn(3, 4)
print(x.device)

# 在gpu上创建张量
X = torch.randn(2, 3, device=try_gpu())
print(X)
Y = torch.randn(2, 3, device=try_gpu(10))
print(Y)

# 张量从cpu转移到gpu
Z = Y.cuda(0)
print(Z)

print(X+Z)
print(Z.cuda(0) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))

# 首先指定网络层，然后指定参数，用.data提取数据，并用.device查看设备
print(net[0].weight.data)
# net[0].weight.data = torch.ones(net[0].weight.data.shape, device=try_gpu())
net.apply(my_init)
print(net[0].weight.data)

"""作业题解答"""
# 1. 参看下面的代码,可以看到CPU运行时间是GPU的10倍以上
A = torch.randn(100, 100, device=try_gpu())
B = torch.randn(100, 100, device=try_gpu())
start = time.perf_counter()
C = torch.mm(A, B)
end = time.perf_counter()
print(f'Time on GPU: {end-start:5f} s')
A = torch.randn(100, 100)
B = torch.randn(100, 100)
start = time.perf_counter()
C = torch.mm(A, B)
end = time.perf_counter()
print(f'Time on CPU: {end-start:5f} s')

# 2. 用赋值的方法写参数时，必须先在gpu上创建要赋予的值，再做赋值操作，否则参数会转移到cpu上；
# 另一种写参数的方法时借助nn.Module.apply()进行赋值，可以保证参数写在正确的设备上；
# 读操作就如5.6.3中所示，直接读取即可

# 3. 参看如下代码。可以看到涉及数据在设备间的转移时，多次小操作确实比一次大操作更慢
# 全局解释器锁使得多进程在多核的情况下也只能一个一个进行，这就导致小操作情况下每次都必须等待数据传输的进程
# 结束之后才能继续运算，加上数据传输本身较慢，就降低了算法总体的并行度和速度
start = time.perf_counter()
for i in range(1000):
    A = torch.randn(100, 100, device=try_gpu())
    B = torch.randn(100, 100, device=try_gpu())
    C = torch.mm(A, B).to(device='cpu')
    print(f"result {i}: {torch.norm(C)}")
end = time.perf_counter()
print(f'Total time: {end-start:5f} s')

record = torch.empty(1000, 1, device=try_gpu())
start = time.perf_counter()
for i in range(1000):
    A = torch.randn(100, 100, device=try_gpu())
    B = torch.randn(100, 100, device=try_gpu())
    record[i] = torch.norm(torch.mm(A, B))
record.to(device='cpu')
print(record)
end = time.perf_counter()
print(f'Total time: {end-start:5f} s')
