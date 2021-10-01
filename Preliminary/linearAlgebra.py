import torch
import numpy as np

X = torch.arange(24, dtype=float).reshape(-1, 2, 4)  # C*H*W
Y = X.clone() + 1
print("X: \n", X, '\n')
print("Y: \n", Y, '\n')
print("X*Y: \n", X * Y, '\n')
print("Mean of X: \n", X.mean(), '\n')  # X.mean()

X_sum_axis0 = X.sum(axis=0)
print("Sum along the first axis: \n", X_sum_axis0, '\n')

X_sum_axis1 = X.sum(axis=1)
print("Sum along the second axis: \n", X_sum_axis1, '\n')

X_sum_axis01 = X.sum(axis=[0, 1])
print("Sum along the first and second axis: \n", X_sum_axis01, '\n')

X_mean01 = X.mean(axis=[0, 1])  # X.sum(axis=[0, 1]) / (X.shape[0]*X.shape[1])
print("Mean along the first and second axis: \n", X_mean01, '\n')

sum_X = X.sum(axis=1, keepdims=True)
consum_X = X.cumsum(axis=0)
print("Sum without collapsing dimension: \n", sum_X, '\n')
print(consum_X)  # X.cumsum(axis=1)
# print(X/sum_X)

a = torch.arange(4, dtype=torch.float)
b = torch.ones(4)
print("Dot Product: \n", torch.dot(a, b), '\n')  # torch.sum(a*b)

W = torch.arange(12, dtype=torch.float).reshape(3, 4)
print("Matrix-Vector Product: \n", torch.mv(W, b), '\n')

Y = torch.ones(4, 3)
print("Matrix-Matrix Product: \n", torch.mm(W, Y), '\n')

print("Norm of Matrix X: \n", torch.linalg.norm(X), '\n')
print("Norm of Vector a: \n", torch.linalg.norm(a, 1), '\n')

############# Practice #############
A = torch.rand(3, 4)  # Any size is OK
print(A)
print(A.T.T == A, '\n')

B = torch.rand(3, 4)  # Any size is OK
print(A.T + B.T == (A + B).T, '\n')

C = torch.rand(4, 4)
print((C + C.T) == (C + C.T).T)

print(len(X))  # return the first dimension
# print(X/X.sum(axis=1))  # Error, because of the mismatch of sizes
print(X.sum(axis=[0, 1, 2]))  # sum all the elements
print(torch.linalg.norm(X))  # get the Frobenius norm
