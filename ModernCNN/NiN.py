import torch
from torch import nn
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


NiN = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten()
)

# 以下两行可视化模型
# NiN.to(d2l.try_gpu())
# summary(NiN, input_size=(1, 224, 224))
# 以下两行统计参数量和计算量
model_name = 'NiN'
flops, params = get_model_complexity_info(NiN, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
print("%s |%s |%s" % (model_name, flops, params))

# 按教材配置，得到test acc=0.847
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(NiN, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

"""作业题解答"""
# 1. 略

# 2. 注释掉最后一个卷积层和相应的ReLU，得到test acc = 0.883，精度比之前更高。观察loss曲线，也可以发现
# loss也收敛在一个更低的区间内。因此删除一个NiN反而得到更好的效果，这可能是因为任务本身非常简单.不需要很复杂的网络。

# 3. 参数量总计 1.99M，运算量 0.77GMac，训练时占用显存 2527MiB，测试时占用显存 1624MiB

# 4. 只通过一次卷积就将384个通道缩减为10个通道，感受野会比较有限，输出结果的每个位置都只能捕获到输入特征图对应位置3*3领域的信息，
#   通过多级缩减，能够更好地扩大感受野，提高特征提取和表征的能力

