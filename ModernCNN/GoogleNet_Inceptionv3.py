import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from ptflops import get_model_complexity_info
from d2l import torch as d2l


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class InceptionA(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(InceptionA, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = BasicConv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = BasicConv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = BasicConv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接两个3x3卷积层
        self.p3_1 = BasicConv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = BasicConv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_3 = BasicConv2d(c3[1], c3[1], kernel_size=3, padding=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = BasicConv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1_1(x)
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_3(self.p3_2(self.p3_1(x)))
        p4 = self.p4_2(self.p4_1(x))

        return torch.cat((p1, p2, p3, p4), dim=1)


class InceptionB(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, c3, **kwargs):
        super(InceptionB, self).__init__(**kwargs)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2 = BasicConv2d(in_channels, c2, kernel_size=3, stride=2)
        # 线路3，1x1卷积层后接两个3x3卷积层
        self.p3_1 = BasicConv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = BasicConv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_3 = BasicConv2d(c3[1], c3[1], kernel_size=3, stride=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2 = self.p2(x)
        p3 = self.p3_3(self.p3_2(self.p3_1(x)))
        p4 = self.p4(x)

        return torch.cat((p2, p3, p4), dim=1)


class InceptionC(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(InceptionC, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = BasicConv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = BasicConv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = BasicConv2d(c2[0], c2[0], kernel_size=(1, 7), padding=(0, 3))
        self.p2_3 = BasicConv2d(c2[0], c2[1], kernel_size=(7, 1), padding=(3, 0))
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = BasicConv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = BasicConv2d(c3[0], c3[0], kernel_size=(7, 1), padding=(3, 0))
        self.p3_3 = BasicConv2d(c3[0], c3[0], kernel_size=(1, 7), padding=(0, 3))
        self.p3_4 = BasicConv2d(c3[0], c3[0], kernel_size=(7, 1), padding=(3, 0))
        self.p3_5 = BasicConv2d(c3[0], c3[1], kernel_size=(1, 7), padding=(0, 3))
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = BasicConv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1_1(x)
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_5(self.p3_4(self.p3_3(self.p3_2(self.p3_1(x)))))
        p4 = self.p4_2(self.p4_1(x))

        return torch.cat((p1, p2, p3, p4), dim=1)


class InceptionD(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, c3, **kwargs):
        super(InceptionD, self).__init__(**kwargs)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2 = BasicConv2d(in_channels, c2[0], kernel_size=1)
        self.p2 = BasicConv2d(c2[0], c2[1], kernel_size=3, stride=2)
        # 线路3，1x1卷积层后接两个3x3卷积层
        self.p3_1 = BasicConv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = BasicConv2d(c3[0], c3[0], kernel_size=(1, 7), padding=(0, 3))
        self.p3_3 = BasicConv2d(c3[0], c3[0], kernel_size=(7, 1), padding=(3, 0))
        self.p3_4 = BasicConv2d(c3[0], c3[1], kernel_size=3, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2 = self.p2(x)
        p3 = self.p3_4(self.p3_3(self.p3_2(self.p3_1(x))))
        p4 = self.p4(x)

        return torch.cat((p2, p3, p4), dim=1)


class InceptionE(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(InceptionE, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = BasicConv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = BasicConv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2a = BasicConv2d(c2[0], c2[0], kernel_size=(1, 3), padding=(0, 1))
        self.p2_2b = BasicConv2d(c2[0], c2[1], kernel_size=(3, 1), padding=(1, 0))
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = BasicConv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = BasicConv2d(c3[0], c3[1], kernel_size=3, padding=1)
        self.p3_3a = BasicConv2d(c3[1], c3[1], kernel_size=(1, 3), padding=(0, 1))
        self.p3_3b = BasicConv2d(c3[1], c3[1], kernel_size=(3, 1), padding=(1, 0))
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = BasicConv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1_1(x)
        # 每层输出都要经过ReLU之后才进行级联
        # 只有中间两条线路需要对中间结果用ReLU进行激活
        p2_1 = self.p2_1(x)
        p2 = [self.p2_2a(p2_1), self.p2_2b(p2_1)]
        p3_1 = self.p3_1(x)
        p3_2 = self.p3_2(p3_1)
        p3 = [self.p3_3a(p3_2), self.p3_3b(p3_2)]
        p4 = self.p4_2(self.p4_1(x))

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
    InceptionA(192, 64, [96, 96], [48, 64], 32),
    InceptionA(256, 64, [96, 96], [48, 64], 64),
    InceptionA(288, 64, [96, 96], [48, 64], 64),
)

b4 = nn.Sequential(
    InceptionB(288, 384, [64, 96]),
    InceptionC(768, 192, [128, 192], [128, 192], 192),
    InceptionC(768, 192, [160, 192], [128, 192], 192),
    InceptionC(768, 192, [160, 192], [128, 192], 192),
    InceptionC(768, 192, [192, 192], [192, 192], 192),
)

b5 = nn.Sequential(
    InceptionD(768, [192, 320], [192, 192]),
    InceptionE(1280, 320, [384, 384], [448, 384], 192),
    InceptionE(2048, 320, [384, 384], [448, 384], 192),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

GoogleNet = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(2048, 10))

GoogleNet.to(d2l.try_gpu())
summary(GoogleNet, input_size=(1, 96, 96))

# 按教材配置，得到test acc=0.847
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(GoogleNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
