import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')


def train_fine_tuning(net, train_aug, test_aug, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                                              transform=train_aug),
                                             batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                                                             transform=test_aug),
                                            batch_size=batch_size, shuffle=False)
    devices = d2l.try_all_gpus()
    # 为什么设置为none???
    # 和train_batch_ch13中运算相匹配
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        # 根据参数名称筛选出需要微调的部分
        param_1x = [param for name, param in net.named_parameters()
                    if name not in ["fc.weight", "fc.bias"]]
        # 如何对不同部分的参数设置不同的训练方案???
        # 参考https://pytorch.org/docs/0.3.0/optim.html#per-parameter-options
        trainer = torch.optim.SGD([{"params": param_1x}, {"params": net.fc.parameters(), "lr": learning_rate * 10}],
                                  lr=learning_rate, weight_decay=1e-3)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-3)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# train_imgs[i]是一个元组，结构为(img_data,class_id)
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# normalize中的参数来自ImageNet上实验的经验结果
normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))
train_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(96),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])
test_aug = torchvision.transforms.Compose([
    torchvision.transforms.Resize(128),
    torchvision.transforms.CenterCrop(96),
    torchvision.transforms.ToTensor(),
    normalize
])

pretrained_net = torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc)

lr = 5e-5
# 预训练，loss 0.192, train acc 0.932, test acc 0.920
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
train_fine_tuning(finetune_net, train_aug, test_aug, lr)

# 从头训练，loss 0.359, train acc 0.851, test acc 0.836
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
nn.init.xavier_uniform_(scratch_net.fc.weight)
train_fine_tuning(scratch_net, train_aug, test_aug, lr, param_group=False)

d2l.plt.show()

"""作业题解答"""
# 1. 运行如下代码，得到以下数据。可以看到随着学习率上升，测试精度仍有小幅提高，但超过一定限度时，收敛曲线就会震荡明显，性能出现严重下降
# lr = 2.5e-5, test acc = 0.924
# lr = 5.0e-5, test acc = 0.925
# lr = 10.0e-5, test acc = 0.930
# lr = 20.0e-5, test acc = 0.899
'''
total_runs = 3
lr = 2e-4
for run in range(total_runs):
    finetune_net = torchvision.models.resnet18(pretrained=True)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)
    train_fine_tuning(finetune_net, train_aug, test_aug, lr)
    lr *= 2
    d2l.plt.show()
'''

# 2. 略

# 3. 冻结输出层外的其他的参数，如以下代码，得到 loss 0.676, train acc 0.821, test acc 0.890
# 性能会有一定的退化，说明输出层以外参数的微调对于模型的迁移学习是非常重要的
'''
finetune_net = torchvision.models.resnet18(pretrained=True)
for param in finetune_net.parameters():
    param.requires_grad = False
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
train_fine_tuning(finetune_net, train_aug, test_aug, 5e-5)
d2l.plt.show()
'''

# 4. 代码如下，预训练结果为loss 0.261, train acc 0.910, test acc 0.919
'''
finetune_net = torchvision.models.resnet18(pretrained=True)
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
print("Size of hotdog_w: ", hotdog_w.shape)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
finetune_net.fc.weight.data[0] = hotdog_w.data
print(finetune_net.fc.weight)
train_fine_tuning(finetune_net, train_aug, test_aug, 5e-5)
d2l.plt.show()
'''

