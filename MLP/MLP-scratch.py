import torch
from torch import nn
from IPython import display
from d2l import torch as d2l

"""超参数设置与数据加载"""
batch_size = 256
num_inputs, num_outputs, num_hidden = 28 * 28, 10, 256
num_epochs = 10
lr = 0.1
SEED = 0  # 控制normal按照同一个seed初始化参数，从而方便比较超参数配置带来的差别
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
W1 = nn.Parameter(torch.normal(0, 0.01, size=(num_inputs, num_hidden), requires_grad=True))
# W1 = nn.Parameter(torch.ones(size=(num_inputs, num_hidden), requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W0 = nn.Parameter(torch.normal(0, 0.01, size=(num_hidden, num_hidden), requires_grad=True))
# W2 = nn.Parameter(torch.ones(size=(num_hidden, num_outputs), requires_grad=True))
b0 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.normal(0, 0.01, size=(num_hidden, num_outputs), requires_grad=True))
# W2 = nn.Parameter(torch.ones(size=(num_hidden, num_outputs), requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W0, b0, W2, b2]
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, W1.shape[0]))
    H = relu(X@W1 + b1)
    M = relu(H @ W0 + b0)
    return M @ W2 + b2


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
# 1. 固定初始化权重，固定lr为0.1, 在以下几个调整num_hidden时分别得到如下的结果。结合曲线，可以看到
# 随着调整num_hidden的增加Loss和Training Accuracy不断减小，Testing Accuracy的变化越来越平滑
# 但综合考虑参数量和测试精度，选取num_hidden=256最为合适
# num_hidden |     Loss    |  Training Accuracy  | Testing Accuracy
#   64       |   0.401015  |      0.858500       |     0.820600
#   128      |   0.391482  |      0.861683       |     0.850100
#   256      |   0.384172  |      0.865150       |     0.851000
#   512      |   0.378326  |      0.867633       |     0.837900
#   1024     |   0.374431  |      0.867567       |     0.841800

# 2. 固定初始化权重，固定lr为0.1, num_hidden为256, 在中间添加一层输入、输出尺寸均为num_hidden的层
# 从如下结果可以看出，加入更多隐藏层反而使得性能下降，主要原因在于参数量增加，需要更多周期来达到同样的精度效果
# 但可以预见，给予更多周期，最终收敛时层数更多的感知机将有更高的性能
#   hidden layers   |     Loss    |  Training Accuracy  | Testing Accuracy
#         1         |   0.384172  |      0.865150       |     0.851000
#         2         |   0.391916  |      0.858600       |     0.829700

# 3. 固定初始化权重，固定num_hidden为256，在以下几个lr时分别得到如下的结果。可以看到
# 随着调整lr增加，性能一开始会提高，但超过一定限度就会出现过拟合，导致测试精度的下降，因此最佳的lr大约为0.2
#   lr       |     Loss    |  Training Accuracy  | Testing Accuracy
#   0.05     |   0.443581  |      0.845183       |     0.834400
#   0.10     |   0.384172  |      0.865150       |     0.851000
#   0.20     |   0.335258  |      0.879800       |     0.858600
#   0.30     |   0.317339  |      0.884750       |     0.854500
#   0.50     |   0.306073  |      0.887050       |     0.849500

# 4. 通过对所有超参数(学习率、迭代周期数、隐藏层数、每层的隐藏单元数)进行联合优化，可以得到的最佳结果是
#                |     Loss    |  Training Accuracy  | Testing Accuracy
#      优化前     |   0.384172  |      0.865150       |     0.851000
#      优化后     |   0.335014  |      0.877150       |     0.836700

# 5. 从4的结果可以看到，优化后性能并非全面占优，说明训练结果受到多个超参数影响时，
# 控制变量法得到的各个最优参数的组合并不保证全局最优解。实际上，在多维参数空间寻求全局最优是很困难的，也因此给参数设置带来挑战

# 6. 在参数空间随机均匀选择多个点，选择其中最好的作为参数，类似蒙特卡罗算法
