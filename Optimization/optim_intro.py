import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l


# 定义风险函数，即整个数据群上的预期损失
def f(x):
    return x * torch.cos(np.pi * x)


# 定义经验风险函数，即训练集上的平均损失
def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)


def annotate(text, xy, xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))


x = torch.range(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\n empirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
d2l.plt.show()

x = torch.range(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
d2l.plt.show()

x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
d2l.plt.show()

# 生成数据
x, y = torch.meshgrid([torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101)])
z = x ** 2 - y ** 2

# 绘制网状曲线
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
# 标记鞍点
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.show()

x = torch.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2.0, 0.0))
d2l.plt.show()


"""作业题解答"""
# 1. 略
# 2. 略
# 3. 如梯度爆炸问题，以及由于不适当的初始化带来的梯度消失
# 4. (1) 因为x和y方向的的梯度变化完全想法，一旦受到y方向的侧向扰动，球马上就会沿着这个方向滑下去
#    (2) 鉴于此，可以在鞍点的位置施加随机扰动从而迅速摆脱鞍点
