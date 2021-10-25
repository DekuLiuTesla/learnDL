import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)  # 将Tensor按照第一个维度进行打包，封装成一个数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


############# Hyper-parameter Setting #############
epochs = 5
batch_size = 10
lr = 0.03

############# Data Generation #############
true_w = torch.tensor([2, -3.4]).reshape(-1, 1)
true_b = 4.2
num = 1000
features, labels = d2l.synthetic_data(true_w, true_b, num)
train_dataLoader = load_array((features, labels), batch_size)
# print(next(iter(train_dataLoader)))  # Iterator Testing

print("features: ", features[0], "\nlabel: ", labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
d2l.plt.show()

############# Preparation for Training #############
model = nn.Sequential(nn.Linear(2, 1))
loss_fn = nn.MSELoss()  # 若reduction='sum', 则调整lr=0.03/batch_size来达到与调整前相同的效果
# loss_fn = nn.HuberLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

############# Training #############
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    for batch, (X, y) in enumerate(train_dataLoader):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        # print(model[0].weight.grad)  # 在此处使用则会每个batch输出一次梯度，共计输出[num/batch_size]次
        optimizer.step()
    print(f"Loss = {loss:>3f}\n")

############# Result Evaluation #############
w = model[0].weight.data
b = model[0].bias.data
print(f"w的估计误差：{true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差：{true_b - b}")
print("w的梯度：", model[0].weight.grad)

