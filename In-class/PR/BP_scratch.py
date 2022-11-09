import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import d2l.torch as d2l
from torch import nn


class TrainDataset(Dataset):
    def __init__(self, data, labels):
        assert labels.shape[0] == data.shape[0], "Data and Labels have different number of samples."
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class TestDataset(Dataset):
    def __init__(self, data, labels):
        assert labels.shape[0] == data.shape[0], "Data and Labels have different number of samples."
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], d2l.argmax(self.labels[index], axis=0)

    def __len__(self):
        return self.data.shape[0]


class MLP(nn.Module):
    def __init__(self, hidden_dim, input_dim=3, output_dim=3):
        super(MLP, self).__init__()
        # Define Parameters
        self.f = nn.Tanh()
        self.g = nn.Sigmoid()
        self.W1 = torch.randn(input_dim, hidden_dim)
        self.W2 = torch.randn(hidden_dim, output_dim)
        # Define intermediate results
        self.net_h = None
        self.Y = None
        self.net_j = None
        self.Z = None

    def forward(self, X):
        self.X = X
        self.net_h = X @ self.W1  # B x hidden_dim
        self.Y = self.f(self.net_h)  # B x hidden_dim
        self.net_j = self.Y @ self.W2  # B x output_dim
        self.Z = self.g(self.net_j)
        return self.Z

    def backward(self, lr, label):
        num_batch = label.shape[0]
        assert num_batch == self.X.shape[0], "Batch Sizes are not compatible."
        X = self.X.unsqueeze(-1)
        Y = self.Y.unsqueeze(-1)
        Z = self.Z.unsqueeze(1)
        e = (label - self.Z).unsqueeze(1)
        self.W2 += lr * torch.bmm(Y, (Z * (1 - Z) * e)).sum(dim=0)/num_batch
        self.W1 += lr * torch.bmm(X, (torch.bmm(1 - Y * Y, (Z * (1 - Z) * e)) *
                                      self.W2).sum(dim=2).unsqueeze(1)).sum(dim=0)/num_batch


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    loss = nn.MSELoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) * 0.5
            net.backward(lr, y)
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, None))
        train_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, train_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    return metric


data_cls1 = torch.tensor([
    [1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73],
    [1.39, 3.16, 2.87], [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38], [-0.76, 0.84, -1.96]
])
data_cls2 = torch.tensor([
    [0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39],
    [0.74, 0.96, -1.16], [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14], [0.46, 1.49, 0.68]
])
data_cls3 = torch.tensor([
    [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69], [1.86, 3.19, 1.51], [1.68, 1.79, -0.87],
    [3.51, -0.22, -1.39], [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99], [0.66, -0.45, 0.08]
])

# Define data
lr, batch_size, num_epochs = 0.01, 5, 20
data = torch.cat([data_cls1, data_cls2, data_cls3])
labels = torch.zeros_like(data)
labels[:10, 0] = 1
labels[10:20, 1] = 1
labels[20:, 2] = 1
train_iter = DataLoader(TrainDataset(data, labels), batch_size, shuffle=True)
test_iter = DataLoader(TestDataset(data, labels), 30, shuffle=True)

# Define Model
model = MLP(hidden_dim=16)
metric = train(model, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
