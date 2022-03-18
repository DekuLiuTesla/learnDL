import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from Focal_Loss import focal_loss


def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)


# def focal_loss(gamma, x):
#     return -(1.0 - x) ** gamma * torch.log(x)


def cls_predictor(num_inputs, num_anchors, num_classes):
    # 在每个位置生成所属锚框归于各个类别的概率
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    # 在每个位置生成所属锚框的相对偏移量
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


def flatten_preds(pred):
    return torch.flatten((torch.permute(pred, (0, 2, 3, 1))), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_preds(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blks = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blks.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blks)


def get_block(i):
    if i == 0:
        return base_net()
    elif i == 1:
        return down_sample_blk(64, 128)
    elif i == 4:
        return nn.AdaptiveMaxPool2d((1, 1))
    else:
        return down_sample_blk(128, 128)


def blk_forward(X, blk, sizes, ratios, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=sizes, ratios=ratios)
    class_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)

    return Y, anchors, class_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, num_anchors, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_cls = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        self.num_blks = len(idx_to_in_channels)
        for i in range(self.num_blks):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_block(i))
            setattr(self, f'class_predictor{i}', cls_predictor(
                idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_predictor{i}', bbox_predictor(
                idx_to_in_channels[i], num_anchors))

    def forward(self, X, sizes, ratios):
        anchors, cls_preds, bbox_preds = [None] * self.num_blks, [None] * self.num_blks, [None] * self.num_blks
        for i in range(self.num_blks):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                                                                     getattr(self, f'class_predictor{i}'),
                                                                     getattr(self, f'bbox_predictor{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_cls + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    # bbox_loss = nn.L1Loss(reduction='none')
    # cls_loss = focal_loss(size_average=False)
    bbox_loss = nn.SmoothL1Loss(reduction='none')
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[-1]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    # cls_preds = cls_preds.reshape(-1, num_classes)
    # cls_probs = F.softmax(cls_preds, dim=1)
    # cls_labels = cls_labels.reshape(-1)
    # cls_preserved = cls_probs[range(len(cls_probs)), cls_labels]
    # cls = focal_loss(2, cls_preserved).reshape(batch_size, -1).mean(dim=1)

    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 用平均绝对误差评价边界框的预测结果
    return float(torch.abs((bbox_preds - bbox_labels) * bbox_masks).sum())


def predict(X, net, sizes, ratios):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X, sizes, ratios)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()


Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape)
print(Y2.shape)
print(concat_preds([Y1, Y2]).shape)
print(forward(torch.zeros(2, 3, 20, 20), down_sample_blk(3, 10)).shape)
print(forward(torch.zeros(2, 3, 256, 256), base_net()).shape)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

net = TinySSD(num_classes=1, num_anchors=num_anchors)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X, sizes, ratios)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

batch_size, num_epochs, timer = 32, 20, d2l.Timer()
train_iter, _ = d2l.load_data_bananas(batch_size)
device = d2l.try_gpu()
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X, sizes, ratios)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
d2l.plt.show()

X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
output = predict(X, net.cpu(), sizes, ratios)
display(img, output.cpu(), threshold=0.9)

"""作业题解答"""
# 1.不使用Focal Loss与smooth-L1 Loss，则class err 3.44e-03, bbox mae 3.36e-03
# 只使用smooth-L1 Loss，则class err 3.28e-03, bbox mae 3.25e-03，回归误差不便于比较，但分类误差显著降低
# 同时使用Focal Loss与smooth-L1 Loss，则class err 4.29e-03, bbox mae 3.13e-03，分类不便于比较，但回归误差显著降低
# 因此可以得到结论，即Focal Loss与smooth-L1 Loss都有助于进一步改善目标检测的性能

# 2. 略
