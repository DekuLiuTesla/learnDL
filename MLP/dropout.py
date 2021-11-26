import torch
from torch import nn
from d2l import torch as d2l

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 28 * 28, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5


def dropout_layer(X, dropout):
    # dropout参数反映随机丢弃单元连接的概率
    assert 0 <= dropout <= 1  # 输入的范围不正确则会报错
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return X * mask / (1 - dropout)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hidden2s,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只在训练过程中加dropout
        if self.training:
            # 在第一个隐藏层后添加第一个dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二个隐藏层后添加第二个dropout
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs)
)
net.apply(init_weights)  # 会依次对每个层的参数应用init_weights函数做初始化
num_epochs, lr, batch_size = 10, 0.5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()
