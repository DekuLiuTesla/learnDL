import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

"""超参数设置与数据加载"""
batch_size = 256
num_inputs, num_outputs, num_hidden = 28 * 28, 10, 64
num_epochs = 10
lr = 0.1
# W1 = nn.Parameter(torch.normal(0, 0.01, size=(num_inputs, num_hidden), requires_grad=True))
W1 = nn.Parameter(torch.zeros(size=(num_inputs, num_hidden), requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
# W2 = nn.Parameter(torch.normal(0, 0.01, size=(num_hidden, num_outputs), requires_grad=True))
W2 = nn.Parameter(torch.zeros(size=(num_hidden, num_outputs), requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, W1.shape[0]))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)




"""自定义函数测试"""
# X = torch.normal(0, 0.01, size=(2, 3))
# print(X)
# print(relu(X))
# print(X)


"""训练与测试"""
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(params, lr=lr)
lossCurve = []
trainAccuracyCurve = []
testAccuracyCurve = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    lossCurve.append(train_metrics[0])
    trainAccuracyCurve.append(train_metrics[1])
    testAccuracyCurve.append(test_acc)
    print(f"Loss = {train_metrics[0]:>3f}")
    print(f"Training Accuracy = {train_metrics[1]:>3f}")
    print(f"Testing Accuracy = {test_acc:>3f}\n")

x_axix = range(num_epochs)
d2l.plt.title('Result Analysis')
d2l.plt.plot(x_axix, lossCurve, '-', label='Loss')
d2l.plt.plot(x_axix, trainAccuracyCurve, 'm--', label='Training Accuracy')
d2l.plt.plot(x_axix, testAccuracyCurve,  'g-.', label='Testing Accuracy')
d2l.plt.legend()  # 显示图例
d2l.plt.grid()
d2l.plt.xlabel('Epochs')
d2l.plt.ylabel('Val')
d2l.plt.show()

d2l.predict_ch3(net, test_iter, n=6)
d2l.plt.tight_layout()  # 防止显示不全
d2l.plt.show()  # 显示图像

"""习题解答"""
# 1. 调整num_hidden, 固定初始化权重，在以下几个值时分别得到如下的结果。可以看到随着值的增加loss不断减小
#
# num_hidden |     Loss    |  Training Accuracy | Testing Accuracy
#   64       |   0.395612  |      0.861183      |     0.846200
#   128      |   0.387358  |      0.862667      |     0.825600
#   256      |   0.385958  |      0.863450      |     0.842400
#   512      |   0.377460  |      0.866817      |     0.841200
#   1024     |   0.370638  |      0.868633      |     0.858100
