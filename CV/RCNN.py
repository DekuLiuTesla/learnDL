import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
print(X)

# 注意水平为x方向，竖直为y方向
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
result = torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
print(result)


"""作业题解答"""
# 1. 可以视为在图片上每个位置进行以该点为中心的锚框坐标及相应概率的回归

# 2. 单、双阶段检测算法的主要区别在于单阶段少了region proposal的生成，直接在一个阶段完成对锚框位置和概率的预测
