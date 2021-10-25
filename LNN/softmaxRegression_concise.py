import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data


"""超参数设置与数据加载"""
batch_size = 256
num_inputs = 28 * 28
num_outputs = 10
num_epochs = 10
lr = 0.1
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""模型设置"""
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_outputs, bias=True),
)
loss_fn = nn.CrossEntropyLoss()  # 包含了softmax+log+NLLLoss全过程
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


"""模型训练"""
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_metrics = d2l.train_epoch_ch3(model, train_iter, loss_fn, optimizer)
    test_acc = d2l.evaluate_accuracy(model, test_iter)
    print(f"Loss = {train_metrics[0]:>3f}")
    print(f"Training Accuracy = {train_metrics[1]:>3f}")
    print(f"Testing Accuracy = {test_acc:>3f}\n")

d2l.predict_ch3(model, test_iter, n=6)
d2l.plt.tight_layout()  # 防止显示不全
d2l.plt.show()  # 显示图像

