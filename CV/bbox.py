import torch
from d2l import torch as d2l


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    # 图像中坐标的原点是图像的左上角，向右的方向为 x 轴的正方向，向下的方向为 y 轴的正方向
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # 创建新的维度并沿着该维度级联
    return torch.stack([cx, cy, h, w], dim=-1)


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    # 图像中坐标的原点是图像的左上角，向右的方向为 x 轴的正方向，向下的方向为 y 轴的正方向
    cx, cy, h, w = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    # 创建新的维度并沿着该维度级联
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, color=color, linewidth=2)


d2l.set_figsize()
img = d2l.Image.open('../img/catdog.jpg')
fig = d2l.plt.imshow(img)

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
bboxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(bboxes)) == bboxes)

fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()

"""作业题解答"""
# 1. 对边界框的标记显然更加耗时，因为需要对左上点和右下点坐标共计4个连续值进行确定，而类别是离散值，只要从有限个标签挑选一个即可，
# 难度差异是巨大的，这也导致了检测中两个任务并不平衡

# 2. 因为每个函数都需要分离最内层的四个维度方便后续坐标转化的计算
