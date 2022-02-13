import torch
import torchvision
from torch import nn
from d2l import torch as d2l


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    results = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(results, num_rows, num_cols, scale=scale)


def load_cifar10(is_train, aug, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=aug, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                             num_workers=d2l.get_dataloader_workers())
    return dataloader


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


# @save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    pred = net(X)
    trainer.zero_grad()
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


# @save
def train_ch13(net, train_iter, test_iter, loss, trainer,
               num_epochs, devices=d2l.try_all_gpus()):
    """用多个GPU训练模型(在第十三章定义)"""
    # timer为计数器，num_batches为一个epoch内总共的batch数
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 三个值分别记录总损失，正确预测的样本数目以及样本总数
        for i, (features, labels) in enumerate(train_iter):  # 一次一个batch
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            with torch.no_grad():
                metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # 每5个samples记录一次train_l与train_acc
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        # 在每个epoch结束时记录一次test_acc
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')


def train_with_data_aug(train_aug, test_aug, net, batch_size, num_epochs, lr=0.001, devices=d2l.try_all_gpus()):
    train_iter = load_cifar10(True, train_aug, batch_size)
    test_iter = load_cifar10(False, test_aug, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)

# 随机左右反转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 随机上下反转
apply(img, torchvision.transforms.RandomVerticalFlip())
# 随机裁切
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)
# 随机改变亮度
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
# 随机改变色调
apply(img, torchvision.transforms.ColorJitter(hue=0.5))
# 随机改变图像色彩，对比度采用均方根对比度的定义进行调整
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
# 结合多种增广仿佛
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), shape_aug, color_aug])
apply(img, augs)

'''在增广后的CIFAR10上做训练'''
all_images = torchvision.datasets.CIFAR10(train=True, root='../data', download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
transforms = torch.nn.Sequential(
    torchvision.transforms.CenterCrop(10),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
scripted_transforms = torch.jit.script(transforms)

train_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

batch_size, num_epochs, net = 256, 10, d2l.resnet18(10, 3)
net.apply(init_weights)
train_with_data_aug(train_aug, test_aug, net, batch_size, num_epochs)
d2l.plt.show()

"""作业题解答"""
# 1. 使用图像增广，则loss 0.172, train acc 0.940, test acc 0.821
# 不使用图像增广(即只应用了transforms.ToTensor())，则loss 0.078, train acc 0.973, test acc 0.832
# 可以看出增广后算法性能有明显的提升，但训练和测试精度的差异变得更大了，无法论证图像增广能够减轻过拟合这一论点

# 2. 加入对颜色的随机变换从而进行图像增广，则loss 0.241, train acc 0.917, test acc 0.823
# 可见能够提高性能，缩小训练和测试精度的差异，缓解过拟合的现象

# 3. 还包括仿射变换、padding、灰度化、透视变换、高斯平滑、反色、直方图均衡、过曝、线性投影、标准化、部分擦除、数据类型变换等可选的方法

# 4. 为什么数据增广看起来没有改变数据集的大小呢？实际上pytorch的处理方式和传统上对数据增广用变换后的数据扩充数据集的理解并不一样，
# pytorch实际上保持了原数据集不变，但在每个epoch对每个batch中的每个图片都应用一次transforms中定义的随机变换，最后每个epoch
# 实际用的都是原始数据集随机变换后的不同的版本，本质上依然达到了图片增广、增强数据多样性的目的，但实现上更为简单了。
