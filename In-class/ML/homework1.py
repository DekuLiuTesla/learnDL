import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# 将类别转换为数字
data[4] = pd.Categorical(data[4]).codes

# 将数据集分为训练集和测试集
np.random.seed(0)
train_data, test_data = train_test_split(data, test_size=0.3)

train_X = train_data.iloc[:, :4].values.astype(np.float32)
train_y = train_data.iloc[:, 4].values.astype(np.int64)
test_X = test_data.iloc[:, :4].values.astype(np.float32)
test_y = test_data.iloc[:, 4].values.astype(np.int64)

# 对输入特征进行归一化处理
mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
train_X = (train_X - mean) / std
test_X = (test_X - mean) / std


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# 设置超参数
input_size = 4
hidden_size = 10
output_size = 3
lr = 0.1
num_epochs = 1000

# 初始化神经网络模型
model = NeuralNetwork(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 训练神经网络模型
for epoch in range(num_epochs):
    # 将数据转换为PyTorch张量
    inputs = torch.from_numpy(train_X)
    targets = torch.from_numpy(train_y)

    # 前向传播
    outputs = model(inputs)

    # 计算损失函数
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 输出每个epoch的损失函数值
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 在测试集上计算准确率
with torch.no_grad():
    inputs = torch.from_numpy(test_X)
    targets = torch.from_numpy(test_y)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == targets).sum().item() / len(targets)
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
