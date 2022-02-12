import torch
from torch import nn
from d2l import torch as d2l
from torchsummary import summary
from ptflops import get_model_complexity_info


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.Linear(out_channels*7*7, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )


size = 224
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
VGGNet = vgg(conv_arch)

# 用torchsummary.summary做可视化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = VGGNet.to(device)
summary(model, (1, size, size))

# 用教材方法做可视化
'''
X = torch.randn(1, 1, 224, 224)
for layer in VGGNet:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
'''

ratio = 8
small_conv_arch = ((pair[0], pair[1]//ratio) for pair in conv_arch)
VGGs = vgg(small_conv_arch)

# ratio=8, # test acc = 0.912
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=size)
d2l.train_ch6(VGGs, train_iter, test_iter, num_epochs, lr, device)
d2l.plt.show()


"""作业题解答"""
# 1. 剩余的三层CONV隐藏在后三个Sequential当中，用torchsummary.summary可以看到

# 2. 因为VGG更深，有更多计算密集型的卷积层。AlexNet有5个卷积层3个最大汇聚层，而VGG11则有足足8个卷积层5个
# 最大汇聚层，导致显存占用增加，计算量则大幅提高，对算力有了更高的要求

# 3. 能够减少运算量，提高运算速度。图像大小为224时模型大小为616.92MB，而改为96之后，第一个全连接层的输入需要相应改成
# out_channels*3*3，此时模型大小为194.57MB，但性能会相应下降，得到结果test acc = 0.893

# 4. VGG16和VGG19的架构如下
conv_arch_vgg16 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
conv_arch_vgg19 = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))


