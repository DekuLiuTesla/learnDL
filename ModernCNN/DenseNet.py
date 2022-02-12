import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(input_channels+i*num_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

trans_blk = transition_block(23, 10)
print(trans_blk(Y).shape)

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

blks = []
num_channels, growth_rate = 64, 32
num_convs = [4, 4, 4, 4]

for i in range(len(num_convs)):
    blks.append(DenseBlock(num_convs[i], num_channels, growth_rate))
    num_channels += growth_rate*num_convs[i]
    if i != len(num_convs)-1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

DenseNet = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(num_channels, 10)
)

DenseNet.to(d2l.try_gpu())
summary(DenseNet, input_size=(1, 224, 224))

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(DenseNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

'''作业题解答'''
# 1. 主要原因在于过渡层中Pooling之前涉及维度压缩，为了更好完整地保留和传递信息因而采用了AvgPooling
# AvgPooling: loss 0.143, train acc 0.948, test acc 0.867
# MaxPooling: loss 0.109, train acc 0.961, test acc 0.905
#
# 2. 教材中DenseNet于ResNet的卷积层层数及全连接层层数都是一样的，而网络的参数量也主要来自这两个部分。造成差距的主要原因在于卷积层和全连接层
# 在通道数目上的差异。DenseNet通过过渡层不断控制通道数量，每个卷积的输入、输出通道数都没有超过256；反观ResNet，block3中5个卷积层输出通道
# 均为256，block4中5个卷积层输出通道均为512，导致卷积层的参数量大幅增加；此外，在单层参数量最大的全连接层中，DenseNet输入通道数为248，
# 远小于ResNet的512，因此在这一部分也获得了巨大的优势，最终使得DenseNet总体参数量比ResNet有了显著的下降
#
# 3. 将输入大小同为96的情况下DenseNet的显存占用为 3161MiB / 4096MiB，ResNet为 2725MiB / 4096MiB，可见确实占用更大
# 可以通过引入bottleneck结构来降低显存占用

# 4、5：略
