import random
import torch
from d2l import torch as d2l


def data_generator(weight, bias, num_data):
    X = torch.normal(0, 1, (num_data, len(weight)))
    y = torch.matmul(X, weight) + bias
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def linreg(X, w, b):
    return torch.matmul(X, w)+b


def data_iter(batch_size, features, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i+batch_size-1, num_examples)]
        yield features[batch_indices, :], labels[batch_indices]


def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2/2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size  # 原地操作，执行后的结果仍然赋给全局变量param
            param.grad.zero_()


true_w = torch.tensor([2, -3.4]).reshape(-1, 1)
true_b = 4.2
num = 1000
features, labels = data_generator(true_w, true_b, num)

print("features: ", features[0], "\nlabel: ", labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
d2l.plt.show()

num_epochs = 5
batch_size = 10
lr = 0.03
w = torch.zeros(true_w.shape, requires_grad=True)  #torch.normal(0, 0.01, true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for t in range(num_epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b)
        loss = squared_loss(y_hat, y).sum()
        loss.backward()
        sgd([w, b], lr, batch_size)
    print(f"Loss = {loss:>3f}\n")

print(f"w的估计误差：{true_w-w.reshape(true_w.shape)}")
print(print(f"b的估计误差：{true_b-b}"))
