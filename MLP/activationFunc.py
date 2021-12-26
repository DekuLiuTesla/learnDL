import torch
from torch import nn
import numpy as np
from d2l import torch as d2l

model = nn.PReLU()
if isinstance(model, nn.ReLU):
    func = "ReLU"
elif isinstance(model, nn.PReLU):
    func = "PReLU"
elif isinstance(model, nn.Sigmoid):
    func = "Sigmoid"
elif isinstance(model, nn.Tanh):
    func = "Tanh"

x = torch.arange(-8, 8, 0.1, requires_grad=True)
y = model(x)
d2l.plt.figure()
d2l.plot(x.detach(), y.detach(), 'x', func + '(x)', figsize=(5, 2.5))
d2l.plt.show()
y.sum().backward()
d2l.plt.figure()
d2l.plot(x.detach(), x.grad, 'x', 'Grad of '+func+'(x)', figsize=(5, 2.5))
d2l.plt.show()
