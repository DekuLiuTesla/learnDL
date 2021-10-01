import torch
import math
from d2l import torch as d2l

x = torch.arange(4.0)
x.requires_grad_(True)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 计算非标量梯度
x.grad.zero_()
t = x.reshape(1, 4)
y = torch.mm(t.T, t)
print(y)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)

a = torch.randn(size=(), requires_grad=True)
b = a * 2
while b.norm() < 1000:
    b *= 2
if b.sum() > 0:
    c = b
else:
    c = 100 * b

c.backward()
print(a.grad)

############# Practice #############
a = torch.randn(size=(3, 4), requires_grad=True)
b = a * a
while b.norm() < 1000:
    b *= 2
if b.sum() > 0:
    c = b
else:
    c = 100 * b

c.sum().backward()  # 只能对标量进行梯度反传
print(a.grad)

x = torch.arange(0, 2*math.pi, 1e-3)
x.requires_grad = True
f = torch.sin(x)
f.sum().backward()

d2l.set_figsize((12, 9))
d2l.plt.plot(f.detach().numpy(), label="Sin")
d2l.plt.plot(x.grad.numpy(), label="Cos")
d2l.plt.legend()
d2l.plt.show()
