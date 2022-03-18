import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l


device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)

with d2l.Benchmark('torch_sync'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

"""作业题解答"""
# 1. 参看如下代码.可以看出在cpu上是否异步性能差异很小，难以观察

with d2l.Benchmark('torch_cpu'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000))
        b = torch.mm(a, a)

with d2l.Benchmark('torch_cpu_sync'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000))
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

