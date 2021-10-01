import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_prob = torch.ones(6) / 6
num = 1000
print(multinomial.Multinomial(num, fair_prob).sample() / num)

num = 10
exprNum = 1000
counts = multinomial.Multinomial(num, fair_prob).sample((exprNum,))
cum_counts = counts.cumsum(dim=0)
estimate = cum_counts / cum_counts.sum(dim=1, keepdim=True)

d2l.set_figsize((12, 9))
for i in range(6):
    d2l.plt.plot(estimate[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
d2l.plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
d2l.plt.gca().set_xlabel('实验次数')
d2l.plt.gca().set_ylabel('估算概率')
d2l.plt.legend()
d2l.plt.show()
