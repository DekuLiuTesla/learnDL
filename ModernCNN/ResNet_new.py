import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


class Residual_new(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual_new, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = self.conv1(F.relu(self.bn1(X)))
        Y = self.conv2(F.relu(self.bn2(Y)))

        if self.conv3:
            X = self.conv3(X)
        Y += X

        return Y


def resnet_block_new(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_new(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
        else:
            blk.append(Residual_new(num_channels, num_channels))
    return blk


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block_new(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block_new(64, 128, 2))
b4 = nn.Sequential(*resnet_block_new(128, 256, 2))
b5 = nn.Sequential(*resnet_block_new(256, 512, 2))

ResNet18 = nn.Sequential(b1, b2, b3, b4, b5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(),
                         nn.Linear(512, 10))

model_name = 'ResNet18'
flops, params = get_model_complexity_info(ResNet18, (1, 224, 224), as_strings=True, print_per_layer_stat=True)

lr, num_epochs, batch_size = 0.005, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(ResNet18, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
