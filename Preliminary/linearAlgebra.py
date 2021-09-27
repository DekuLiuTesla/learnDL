import torch
import numpy as np

X = torch.arange(24, dtype=float).reshape(-1, 2, 4)  # C*H*W
Y = X.clone() + 1
print("X: \n", X, '\n')
print("Y: \n", Y, '\n')
print("X*Y: \n", X*Y, '\n')
print("Mean of X: \n", X.mean(), '\n')  # X.mean()

X_sum_axis0 = X.sum(axis=0)
print("Sum along the first axis: \n", X_sum_axis0, '\n')

X_sum_axis1 = X.sum(axis=1)
print("Sum along the second axis: \n", X_sum_axis1, '\n')

X_sum_axis01 = X.sum(axis=[0, 1])
print("Sum along the first and second axis: \n", X_sum_axis01, '\n')

X_mean01 = X.mean(axis=[0, 1])  # X.sum(axis=[0, 1]) / (X.shape[0]*X.shape[1])
print("Mean along the first and second axis: \n", X_mean01, '\n')