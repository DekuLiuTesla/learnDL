import torch
from torch import nn
from d2l import torch as d2l

"""训练与测试"""
batch_size = 256
num_inputs, num_outputs, num_hidden = 28 * 28, 10, 64
num_epochs = 10
lr = 0.1
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hidden, bias=True),
    nn.ReLU(),
    nn.Linear(num_hidden, num_hidden, bias=True),
    nn.ReLU(),
    nn.Linear(num_hidden, num_outputs, bias=True)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        # nn.init.eye_(m.weight)
        # nn.init.zeros_(m.weight)


net.apply(init_weights)

loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=lr)
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
d2l.plt.plot(x_axix, testAccuracyCurve, 'g-.', label='Testing Accuracy')
d2l.plt.legend()  # 显示图例
d2l.plt.grid()
d2l.plt.xlabel('Epochs')
d2l.plt.ylabel('Val')
d2l.plt.show()

d2l.predict_ch3(net, test_iter, n=6)
d2l.plt.tight_layout()  # 防止显示不全
d2l.plt.show()  # 显示图像

"""习题解答"""
# 1. 在教材的提供的超参数配置下，调整num_hidden个数，分别得到如下的结果。结合曲线，可以看到
# 随着隐藏层增加，模型变得更加复杂，需要更多周期达到同样的训练效果
#   hidden layers   |     Loss    |  Training Accuracy  | Testing Accuracy
#         1         |   0.396305  |      0.860450       |     0.845500
#         2         |   0.431440  |      0.844883       |     0.809700
#         3         |   1.052140  |      0.583833       |     0.584700


# 2. 在教材的提供的超参数配置下，将所有全连接层权重初始化为0，使用不同的激活函数，得到如下结果，
# 可以看到Tanh作为激活函数效果最好
#  Activation  |     Loss    |  Training Accuracy  | Testing Accuracy
#     ReLU     |   0.470808  |      0.833617       |     0.826500
#     PReLU    |   0.446589  |      0.842550       |     0.812000
#     Sigmoid  |   0.622742  |      0.773400       |     0.771500
#     Tanh     |   0.408482  |      0.854017       |     0.837300

# 3. 在教材的提供的超参数配置下，使用ReLU作为激活函数，采用不同的权重初始化方案，得到如下结果，
# 可以看到，将参数初始为均值0，方差0.01的正态分布效果最好
#    Initializtion     |     Loss    |  Training Accuracy  | Testing Accuracy
#        zeros_        |   0.456823  |      0.842833       |     0.838000
#        ones_         |   2.592320  |      0.098733       |     0.100000
#   normal_, std=0.001 |   0.409731  |      0.855667       |     0.842000
#   normal_, std=0.01  |   0.396305  |      0.860450       |     0.845500
#   normal_, std=0.1   |   0.386055  |      0.862667       |     0.842800
#        eye_          |   0.399188  |      0.859100       |     0.825700

