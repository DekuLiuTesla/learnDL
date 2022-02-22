import torch
from d2l import torch as d2l


def display_anchors(img, fmap_w, fmap_h, s):
    d2l.set_figsize()
    h, w = img.shape[:2]
    fmap = torch.zeros(1, 10, fmap_h, fmap_w)
    ratios = [0.5, 1, 2]
    # multibox_prior返回anchor的相对位置，因此无需调整可以直接应用在原图上
    anchors = d2l.multibox_prior(fmap, s, ratios)
    bbox_scale = torch.tensor([w, h, w, h])
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)
    d2l.plt.show()


d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(f"Height of Image: {h}")
print(f"Width of Image: {w}")
display_anchors(img, fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(img, fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(img, fmap_w=1, fmap_h=1, s=[0.8])

"""作业题解答"""
# 1. 是的，实际上尺度较小的往往对应较高的抽象层次，尺度较大的则对应较小的抽象层次，因为越深层、特征图尺度越小
# 特征图上的每个位置感受野越大、非线性性越强，得到的特征映射也更加复杂抽象

# 2. 如以下代码
'''
display_anchors(img, fmap_w=4, fmap_h=4, s=[0.25])
'''

# 3. 通过一个1*1卷积，将通道数变为(C*a+4*a)，其中C*a代表为每个位置a个锚框，每个锚框有C个可能的类别(含背景)；4*a代表每个位置a个锚框，
# 每个锚框对应四个偏移量，这样截取前C*a个通道的输出，并沿通道维度每C个做一次softmax就得到每个位置每个锚框对应各类别概率；
# 其余4*a个通道就作为偏移量，最后输出的形状为1×(C*a+4*a)×h×w
