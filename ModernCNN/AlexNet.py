import torch
from torch import nn
from torchsummary import summary
from thop import profile, clever_format
from ptflops import get_model_complexity_info
from d2l import torch as d2l

AlexNet = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10),
)

X = torch.randn(1, 1, 224, 224)
for layer in AlexNet:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(AlexNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

"""作业题解答"""
# 1. 将迭代次数增加到20，可以看到AlexNet的过拟合现象要弱得多，随着迭代次数的增加，泛化误差依然能够跟随训练误差有效降低

# 3. 改变batch size大小，得到实验结果如下：
#   batch_size=128, test acc = 0.892,  显存 2741MiB /  4096MiB
#   batch_size= 64, test acc = 0.904,  显存 2185MiB /  4096MiB
#   batch_size= 256, test acc = 0.867,  显存 3501MiB /  4096MiB
#   可以看到过高的batch size反而导致了性能的下降和显存占用的显著提高，因此在当前任务下，可以调低batch size来达到更好的测试精度

# 4 运行以下代码，可以看到第一个全连接层占用的显存（参数量）最大，第二个Conv2d的运算量最大，可见卷积层属于低参数量、高运算量的计算密集型操作，
# 而全连接层在同样的参数量下运算开销小很多
'''
model_name = 'AlexNet'
flops, params = get_model_complexity_info(AlexNet, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
print("%s |%s |%s" % (model_name, flops, params))
'''

# 5. 改进后的网络如下，需要将lr下调才能有效收敛，最终选取的lr为0.5，得到test acc = 0.866，相对于改进前提高8个点
#   再加入一个不改变通道数和图像大小的3*3卷积层做预处理，需要进一步下调lr才能收敛，最终选取的lr为0.1，
#   得到test acc = 0.875，又在无预处理的基础上提高了1个点的精度
'''
LeNet = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.ReLU(),  # 预处理
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(84, 10))

batch_size = 128
lr, num_epochs = 0.1, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch6(LeNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
'''
