import torch
from torch import nn
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # zip沿着第一个维度将对应元素打包成一个个元组
    # sum则沿着第一个维度对各个tensor进行相加
    return sum([d2l.corr2d(x, k) for (x, k) in zip(X, K)])


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)


def corr2d_multi_in_out_1x1(X, K):
    c_out = K.shape[0]
    c_in, h, w = X.shape
    # 先展平，再做全连接层运算
    X = X.reshape(c_in, h*w)
    K = K.reshape(c_out, c_in)
    # 全连接层中的矩阵乘法
    result = torch.matmul(K, X)

    return result.reshape(c_out, h, w)


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

K = torch.stack([K, K+1, K+2], dim=0)
print(corr2d_multi_in_out(X, K))

X = torch.normal(0, 1, (3, 3, 3))  # c_in * n_h * n_w
K = torch.normal(0, 1, (2, 3, 1, 1))  # c_out * c_in * k_h * k_w

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print((Y1 - Y2).sum().abs())
assert (Y1 - Y2).sum().abs() < 1e-6

"""作业题解答"""
# 1.1 用卷积的结合律可以容易地证明
# 1.2 相当于用两个原卷积核卷积后的结果作为新的卷积核，其大小为 k1 + k2 - 1
# 1.3 是的，交换运算次序不改变结果

# 2.1 h*w的特征图与k_h*k_w的核按照规定的步幅及padding卷积，每个输出元素对应乘法和加法次数均为k_h*k_w次(加法少一次，但此处忽略)，
#   输出特征图尺寸由6.3.2式计算，记为h_out * w_out，则总计的乘法和加法均为k_h * k_w * h_out * w_out。每个输入通道上都进行卷积，
#   随后对每个输入通道的卷积结果对应位置相加，需要再进行h_out * w_out * (c_in - 1)次加法。因此输出的每个通道对应乘法与加法次数均为
#   k_h * k_w * h_out * w_out * c_in(输入通道间相加的额外运算可略)。最后考虑各个输出通道，共计耗费的乘法和加法运算均为
#   k_h * k_w * h_out * w_out * c_in * c_out 次
# 2.2 核函数占用内存float32 * c_out * c_in *  k_h * k_w，特征图占用内存 float32 * c_in * h * w
# 2.3-2.4 暂时没有头绪

# 3. 由2.1可知计算量会变成原来的四倍；padding数量翻番也会增加运算量，但由于是padding数量加性地影响h_out和w_out，因此最终结果无法直接断点

# 4. 由2.1可知计算量会变成 h_out * w_out * c_in * c_out

# 5. 也不会完全相同，因为计算顺序的差异，可能会存在些微的舍入误差。但在四位小数精度下显示不出这样的微小区别

# 6. 可以的。多输入输出通道的本质是多个单输入输出通道结果的相加和通道维级联，单输入输出通道结果就可以采用矩阵方式运算

