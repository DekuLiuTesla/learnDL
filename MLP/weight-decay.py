import torch
import numpy as np
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w = torch.ones(num_inputs).reshape(-1, 1) * 0.01
true_b = 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
n_val = 150
val_data = d2l.synthetic_data(true_w, true_b, n_val)
val_iter = d2l.load_array(test_data, batch_size, is_train=False)


def data_generator(weight, bias, num_data):
    X = torch.normal(0, 1, (num_data, len(weight)))
    y = torch.matmul(X, weight) + bias
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    num_epochs, lr = 100, 0.003
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())  # 默认使用Frobenius范数，且该范数只适合二维张量（矩阵）
    d2l.plt.show()


def train_concise(wd):
    num_epochs, lr = 100, 0.003
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([{"params": net[0].weight, "weight_decay": wd},  # 在优化器中提供权重衰减
                                 {"params": net[0].bias}], lr=lr)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
    #                         xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # if (epoch + 1) % 5 == 0:
        #     animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
        #                              d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', net[0].weight.norm().item())
    # d2l.plt.show()
    return [d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)]


def validate(wd):
    num_epochs, lr = 100, 0.003
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([{"params": net[0].weight, "weight_decay": wd},  # 在优化器中提供权重衰减
                                 {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['validate'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, d2l.evaluate_loss(net, val_iter, loss))
    print('w的L2范数是：', net[0].weight.norm().item())
    d2l.plt.show()
    return d2l.evaluate_loss(net, val_iter, loss)


# train(3)
train_concise(3)


"""作业题解答"""
# 1. 运行下面的代码可以观察到，训练集的损失始终很小，而lambda较小时测试集损失很大，存在明显的过拟合线性，
# 而随着lambda的增加，测试集的损失迅速下降并逼近测试集的结果，说明通过设置较大的lambda惩罚模型复杂度能够有效解决过拟合问题
# 但当lambda变得比较大时，测试集的损失也会伴随一定的震荡现象
# lambda_max = 50
# loss = np.zeros((lambda_max, 2))
# for i in range(lambda_max):
#     loss[i] = train_concise(i)
# x_Axis = np.arange(lambda_max)
# d2l.plt.figure()
# d2l.plt.plot(x_Axis[2:], loss[2:, 0], color='green', label='training loss')
# d2l.plt.plot(x_Axis[2:], loss[2:, 1], color='red', label='testing loss')
# d2l.plt.legend()  # 显示图例
# ax = d2l.plt.gca()
# x_major_locator = d2l.plt.MultipleLocator(5)
# ax.xaxis.set_major_locator(x_major_locator)
# d2l.plt.show()

# 2. 运行下面的代码可以观察到，最佳的lambda大约在3-4之间，但实际上lambda的值超过一定程度，
# 验证集上的准确度也不再明显变化了，说明不必真的找到最优的取值，lambda足够大就能够起到控制模型复杂度的效果
loss = np.zeros(10)
for i in range(10):
    loss[i] = validate(i)
x_Axis = np.arange(10)
d2l.plt.figure()
d2l.plt.plot(x_Axis, loss, color='green', label='validation loss')
d2l.plt.legend()  # 显示图例
ax = d2l.plt.gca()
x_major_locator = d2l.plt.MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
d2l.plt.show()

# 3. 反传时|w|^2的导数变成|w|的导数：u(w)-u(-w)，u(x)为阶跃函数，
# 最终每个权重除了更新损失函数的梯度外，还会以加减固定常数的形式实现模型复杂度的控制

# 5. 其他处理过拟合的方法包括：选择更加合适的超参数；剔除数据噪声，保证训练和测试集服从独立同分布；减小特征数量；提前结束训练

# 6. 正则化希望使得模型尽量简单，对于先验概率分布，最简单的分布情况就是均匀分布，此时正则化项应该尽可能小，
# 因此自然而然可以想到用信息熵的负数作为正则化项，越接近均匀分布熵越大，相对的负熵就越小，因此可以作为一种正则化的方式

