import torch
from d2l import torch as d2l


def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)


sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2.1, 0.1)
d2l.set_figsize((8, 6))

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)

d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)')
d2l.plt.grid()
d2l.plt.legend()
d2l.plt.show()
