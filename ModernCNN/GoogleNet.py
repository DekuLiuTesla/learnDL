import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from ptflops import get_model_complexity_info
from d2l import torch as d2l


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(192, 64, [96, 128], [16, 32], 32),
    Inception(256, 128, [128, 192], [32, 96], 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(480, 192, [96, 208], [16, 48], 64),
    Inception(512, 160, [112, 224], [24, 64], 64),
    Inception(512, 128, [128, 256], [24, 64], 64),
    Inception(512, 112, [144, 288], [32, 64], 64),
    Inception(528, 256, [112, 320], [24, 128], 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    Inception(832, 256, [160, 320], [32, 128], 128),
    Inception(832, 384, [192, 384], [48, 128], 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

GoogleNet = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

GoogleNet.to(d2l.try_gpu())
summary(GoogleNet, input_size=(1, 96, 96))

# 按教材配置，得到test acc=0.847
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(GoogleNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

"""作业题解答"""
# 1. Original: loss 0.237, train acc 0.909, test acc 0.887
#   Batch normalization: loss 0.085, train acc 0.967, test acc 0.918, ，参见GoogleNet_BN.py
#   Inception-v3: loss 0.092, train acc 0.965, test acc 0.893，参见GoogleNet_Inceptionv3.py
#   Label smoothing: loss 0.702, train acc 0.907, test acc 0.897，参见GoogleNet_label_smoothing.py

# 2. b1使得特征图长宽变成原来1/4，b2~b5每个模块都让特征图长宽减半，因此输出结果长款是输入图像的1/32。由于最终AdaptiveAvgPool2d最小
#   输入尺寸为1*1，对应最小输入图像尺寸就是32*32

# 3. 四个网络的参数数量分别为：AlexNet为46.77M，VGG为128.81M，NiN为1.99M，GoogleNet为5.78M。
# 后两个网络用全局平均池化替代全连接层（其参数量远大于卷积层），同时也不再使用11*11这样大的卷积核，从而显著降低了模型的参数量
