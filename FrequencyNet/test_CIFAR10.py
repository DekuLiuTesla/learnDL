import wandb
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchsummary import summary
from FreqRes import Residual_Freq, ResFreq_gf, ResFreq_gf_new


def load_cifar10(is_train, aug, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=aug, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                             num_workers=d2l.get_dataloader_workers())
    return dataloader


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
                wandb.log({"epoch": epoch + (i + 1) / num_batches, "train loss": metric[0] / metric[2],
                           "train acc": metric[1] / metric[3]})
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        wandb.log({"test acc": test_acc})
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


def train_with_data_aug(train_aug, test_aug, net, batch_size, num_epochs, lr=0.001, devices=d2l.try_all_gpus()):
    train_iter = load_cifar10(True, train_aug, batch_size)
    test_iter = load_cifar10(False, test_aug, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False, h=0, w=0):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            stride = 2 if not first_block else 1
            blk.append(ResFreq_gf_new(input_channels, num_channels,
                                      use_1x1conv=True, strides=stride, h=h, w=w))
        else:
            blk.append(ResFreq_gf_new(num_channels, num_channels))
    return blk


total_runs = 2
for run in range(total_runs):
    '''架构构建'''
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2, h=32, w=17))
    b4 = nn.Sequential(*resnet_block(128, 256, 2, h=16, w=9))
    b5 = nn.Sequential(*resnet_block(256, 512, 2, h=8, w=5))
    ResNet18 = nn.Sequential(b1, b2, b3, b4, b5,
                             nn.AdaptiveAvgPool2d((1, 1)),
                             nn.Flatten(),
                             nn.Linear(512, 10))
    # net = d2l.resnet18(10, 3)
    ResNet18.to(d2l.try_gpu())
    summary(ResNet18, input_size=(3, 32, 32))

    with wandb.init(
            # Set the project where this run will be logged
            project="FreqRes",
            # Track hyper-parameters and run metadata
            config={
                "learning_rate": 0.001,
                "batch_size": 256,
                "num_epochs": 10,
                "resize": 96,
                "architecture": "Res_Freq",
                "dataset": "CIFAR-10",
            }):
        config = wandb.config

        train_aug = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(config.resize),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])
        test_aug = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(config.resize),
            torchvision.transforms.ToTensor()
        ])

        train_iter = load_cifar10(True, train_aug, config.batch_size)
        test_iter = load_cifar10(False, test_aug, config.batch_size)

        wandb.watch(ResNet18)
        train_with_data_aug(train_aug, test_aug, ResNet18, config.batch_size, config.num_epochs, config.learning_rate)
        d2l.plt.show()
        wandb.finish()

# 使用d2l.resnet18，得到loss 0.168, train acc 0.942, test acc 0.845
# Residual_cat，得到loss 0.238, train acc 0.918, test acc 0.772
# ResNet，得到loss 0.221, train acc 0.924, test acc 0.791
# Residual_Freq，得到loss 0.218, train acc 0.926, test acc 0.840
