import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from FreqRes import Residual_Freq, Residual_cat, Residual_Freq_Linear


def load_cifar10(is_train, aug, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=aug, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                             num_workers=d2l.get_dataloader_workers())
    return dataloader


def train_with_data_aug(train_aug, test_aug, net, batch_size, num_epochs, lr=0.001, devices=d2l.try_all_gpus()):
    train_iter = load_cifar10(True, train_aug, batch_size)
    test_iter = load_cifar10(False, test_aug, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False, h=0, w=0):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_Freq_Linear(input_channels, num_channels,
                                    use_1x1conv=True, strides=2, h=h, w=w))
        else:
            blk.append(Residual_Freq_Linear(num_channels, num_channels))
    return blk


'''架构构建'''
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
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

ResNet18.to(d2l.try_gpu())
summary(ResNet18, input_size=(3, 96, 96))

resize = 96
lr, num_epochs, batch_size = 0.001, 10, 256
train_aug = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_aug = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize),
    torchvision.transforms.ToTensor()
])
train_iter = load_cifar10(True, train_aug, batch_size)
test_iter = load_cifar10(False, test_aug, batch_size)

train_with_data_aug(train_aug, test_aug, ResNet18, batch_size, num_epochs, lr)
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {lr}")
print(f"Number of Epochs: {num_epochs}")
d2l.plt.show()

# 使用d2l.resnet18，得到loss 0.168, train acc 0.942, test acc 0.845
# Residual_cat，得到loss 0.238, train acc 0.918, test acc 0.772
# ResNet，得到loss 0.221, train acc 0.924, test acc 0.791
# Residual_Freq，得到loss 0.218, train acc 0.926, test acc 0.840
