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
    nn.Linear(num_hidden, num_outputs, bias=True)
)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
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
d2l.plt.plot(x_axix, testAccuracyCurve,  'g-.', label='Testing Accuracy')
d2l.plt.legend()  # 显示图例
d2l.plt.grid()
d2l.plt.xlabel('Epochs')
d2l.plt.ylabel('Val')
d2l.plt.show()

d2l.predict_ch3(net, test_iter, n=6)
d2l.plt.tight_layout()  # 防止显示不全
d2l.plt.show()  # 显示图像