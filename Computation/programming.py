import torch
from torch import nn
from d2l import torch as d2l

# 命令式编程
"""
def add(a, b):
    return a + b


def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g


print(fancy_func(1, 2, 3, 4))
"""

# 符号式编程
"""
def add_():
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''


def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'


prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
"""


# 混合式编程
class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description=description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')


def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return net


x = torch.randn(size=(1, 512))
net = get_net()
with Benchmark('无torchscript'):
    print(net(x))
net = torch.jit.script(net)
with Benchmark('有torchscript'):
    print(net(x))
net.save('my_mlp')
