import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=2, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        Y += X * self.alpha + self.beta

        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


'''算法测试'''
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)

'''架构构建'''
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

ResNet18 = nn.Sequential(b1, b2, b3, b4, b5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(),
                         nn.Linear(512, 10))

model_name = 'ResNet18'
flops, params = get_model_complexity_info(ResNet18, (1, 224, 224), as_strings=True, print_per_layer_stat=True)

lr, num_epochs, batch_size = 0.05, 20, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(ResNet18, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

'''作业题解答'''
# 1. 对比Inception Block和Residual Block，可以看到主要区别在于前者的多出的MaxPooling分支和5x5 Conv分支，
# 删除这两者，再将3x3 Conv分支中的1x1卷积换成3x3卷积，就可以得到与Residual Block相似的结构了。

# 2.略

# 3. 具体实现参见ResNet_bottleneck.py
# 使用前参数量11.178 M，计算量1.744 GMac，性能loss 0.009, train acc 0.998, test acc 0.859
# 使用后参数量0.887 M，计算量0.185 GMac，性能loss 0.114, train acc 0.959, test acc 0.899
# 可以看到引入bottleneck不仅降低了运算量和参数量，也有效提高了性能

# 4. 具体实现参见ResNet_new.py，
# 使用前参数量11.178 M，计算量1.744 GMac，性能loss 0.009, train acc 0.998, test acc 0.859
# 使用后参数量11.177 M，计算量1.744 GMac，性能loss 0.185, train acc 0.937, test acc 0.898
# 可以看到引入bottleneck不仅降低了运算量和参数量，也有效提高了性能

# 5. 因为过于复杂的函数极易引起过拟合，调试更加困难，并且也带来计算量和参数量上的负担，
