import numpy as np

A = [2, 5, 7, 10, 5, 2]
B = [3, 8, 4, 11, 3, 4]
n = len(A)

dp_array = np.zeros((n, sum(A) + 1, 2))
result = np.Inf

for x in range(A[0]):
    dp_array[0][x][0] = B[0]

x_max = A[0]
for k in range(1, n):
    x_max += A[k]
    for x in range(x_max + 1):
        if x - A[k] < 0:
            # A is assigned with no task
            dp_array[k][x][0] = dp_array[k - 1][x][0] + B[k]
        else:
            s = dp_array[k][x]
            s1 = dp_array[k - 1][x - A[k]]
            s2 = dp_array[k - 1][x]
            s[0] = min(s1[0], s2[0] + B[k])
        if k == n - 1:
            val = max(x, dp_array[k][x][0])
            if val < result:
                result = val
print("Minimum Time: ", result)
