import torch
from d2l import torch as d2l

activate_func = torch.nn.Sigmoid()
x = torch.arange(-8.0, 8.0, 0.05, requires_grad=True)
y = activate_func(x)
d2l.plt.plot(x.detach().numpy(), y.detach().numpy(), label='sigmoid')
y.sum().backward()
d2l.plt.plot(x.detach().numpy(), x.grad, linestyle="--", label='gradient')
d2l.plt.legend()
d2l.plt.grid()
d2l.plt.show()

M = torch.normal(0, 1, size=(4, 4))
print("一个矩阵：\n", M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
print("相乘100次后：\n", M)

