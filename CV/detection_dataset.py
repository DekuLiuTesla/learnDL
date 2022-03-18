import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l


class BananaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print("read " + str(len(self.features)) + (" training examples" if is_train else " validation examples"))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    # 获得记录标签的csv文件的路径
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    # 读取csv文件的数据，输出大小为(num_data, 6)
    csv_data = pd.read_csv(csv_fname)
    # 将索引设置为img_name对应的一列
    csv_data = csv_data.set_index('img_name')
    # 图像和标签分开存储
    images, targets = [], []
    # 依次读取csv文件，获取图像名称及标签
    # iterrows返回(index, row)的元组
    for img_name, target in csv_data.iterrows():
        # 根据图像名称读取图像并存储
        images.append(torchvision.io.read_image(os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val',
                                                             'images', f'{img_name}')))
        # # 存储标签信息，这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    # 输出需要图像及相对位置信息
    return images, torch.tensor(targets).unsqueeze(1) / 256


def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False), batch_size)
    return train_iter, val_iter


d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
# iter(x)叭x变成迭代器Iterator，从而可以被next()函数调用并不服按返回下一个数据
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

imgs = batch[0][:10].permute(0, 2, 3, 1) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:] * edge_size], colors='w')
d2l.plt.show()


"""作业题解答"""
# 1. 主要是在尺度、旋转角度、宽高比方面有所差异，再者就是出现位置不同（一些不太可能出现的位置）

# 2. 部分操作会不同，比如随机裁剪，对于分类而言裁剪的图像只包含物体的一小部分往往不会影响分类任务，但对检测而言，
# 裁剪的图像只包含物体的一小部分时，即便得到精确的预测结果，其与真实边界框的交并比往往很小，反而会被判定为负样本，
# 进而在训练过程中错误地引导优化方向，反而带来性能降低，与数据增强的初衷相悖
