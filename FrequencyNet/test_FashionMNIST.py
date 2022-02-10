import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x + x * weight
        x = torch.fft.irfft2(x, s=(H, W), norm='ortho')
        return x


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, h=0, w=0):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            assert h > 0 and w > 0
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.gfilter = GlobalFilter(num_channels, h, w)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
            X = self.gfilter(X)
        Y += X

        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False, h=0, w=0):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2, h=h, w=w))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


'''架构构建'''
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2, h=12, w=7))
b4 = nn.Sequential(*resnet_block(128, 256, 2, h=6, w=4))
b5 = nn.Sequential(*resnet_block(256, 512, 2, h=3, w=2))

ResNet18 = nn.Sequential(b1, b2, b3, b4, b5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(),
                         nn.Linear(512, 10))

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

d2l.train_ch6(ResNet18, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()


# no Residual in Frequency Domain: loss 0.028, train acc 0.991, test acc 0.911
# with Residual in Frequency Domain: loss 0.014, train acc 0.996, test acc 0.919

