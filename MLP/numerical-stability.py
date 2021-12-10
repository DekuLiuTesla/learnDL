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

"""作业题解答"""
# 1. 比如多个不含激活函数的卷积层，由于卷积的可交换性，易换这些层的位置不会改变最后的结果，而激活层的加入可以打破这种对
# 对称性，提高网络的特征表达能力

# 2. 不能，将导致模型简并，表达能力显著下降，且无法通过小批量训练解除

# 3. 两对称正定矩阵的乘积的特征值范围由两矩阵对应最小特征值的乘积和对应最大特征值的乘积所限定，一旦发现特征值超过这个范围，
# 也许就可以认为会出现梯度爆炸的问题

# 4. 能够进行修正。这篇论文主要探讨了在多个GPU运算单元的情况下用大的Batch Size进行训练时遇到的问题，由于Batch Size增大，训练
# 相同数目的epoch网络更新次数就会减慢，希望达到更好的结果就必须增大每次更新的步长，也就是让学习率增大，但这样使得梯度爆炸的问题更加
# 严重，特别是在训练的初期阶段，这对初始化方式以及学习率的选取提出了更加严苛的需求。因此文章提出了LARS算法，在每一层自适应地调整学习率，
# 使之正比于该层权重的范数与对应梯度的范数的比值，从而有效缓解了梯度太大和学习率设置不合理带来的梯度爆炸问题，使得网络能够在几乎不损失精度
# 的条件下用大的Batch Size进行训练。但其实仔细想想还可以进一步改进，让每个权重对应的学习率正比于权重大小与梯度范数的比值，从而避免大的
# 梯度主导学习率值，而使得不同权重更新不均衡

