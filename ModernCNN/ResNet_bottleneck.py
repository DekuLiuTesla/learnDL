import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


class ResBottleNeck(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, use_1x1conv=False, strides=1):
        super(ResBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.conv4:
            X = self.conv4(X)
        Y += X

        return F.relu(Y)


def resnet_block_bottleneck(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResBottleNeck(input_channels, num_channels // 4, num_channels,
                                     use_1x1conv=True, strides=2))
        else:
            blk.append(ResBottleNeck(num_channels, num_channels // 4, num_channels))
    return blk


'''架构构建'''
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block_bottleneck(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block_bottleneck(64, 128, 2))
b4 = nn.Sequential(*resnet_block_bottleneck(128, 256, 2))
b5 = nn.Sequential(*resnet_block_bottleneck(256, 512, 2))

ResNet18 = nn.Sequential(b1, b2, b3, b4, b5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(),
                         nn.Linear(512, 10))

model_name = 'ResNet18'
flops, params = get_model_complexity_info(ResNet18, (1, 224, 224), as_strings=True, print_per_layer_stat=True)

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(ResNet18, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()