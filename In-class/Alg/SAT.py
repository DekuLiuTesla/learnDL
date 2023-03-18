import time
import pycosat
from func_timeout import func_set_timeout, FunctionTimedOut
import numpy as np


def encode_latin_square(n):
    # 为每个位置(i, j)和每个数字k(1 <= k <= n)构造一个布尔变量
    variables = np.array(range(1, n * n * n * 2 + 1), dtype=int).reshape(n, n, n, 2).tolist()
    clauses = []

    # 通过使得第一行倒序排列来打破对称
    i = 1
    for j in range(1, n + 1):
        for k in range(1, n + 1):
            if k == j:
                clauses.append([variables[i - 1][j - 1][k - 1][0]])
                clauses.append([variables[i - 1][j - 1][k - 1][1]])
            else:
                clauses.append([-variables[i - 1][j - 1][k - 1][0]])
                clauses.append([-variables[i - 1][j - 1][k - 1][1]])

    # 对于每个位置(i, j)，其值在每一行or每一列都是唯一的
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                for r in range(i + 1, n + 1):
                    clauses.append([-variables[i - 1][j - 1][k - 1][0], -variables[r - 1][j - 1][k - 1][0]])
                    clauses.append([-variables[i - 1][j - 1][k - 1][1], -variables[r - 1][j - 1][k - 1][1]])
                for c in range(j + 1, n + 1):
                    clauses.append([-variables[i - 1][j - 1][k - 1][0], -variables[i - 1][c - 1][k - 1][0]])
                    clauses.append([-variables[i - 1][j - 1][k - 1][1], -variables[i - 1][c - 1][k - 1][1]])

    # 每个位置至少有一个取值
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            clause = []
            for k in range(1, n + 1):
                clause.append(variables[i - 1][j - 1][k - 1][0])
            clauses.append(clause)
            clause = []
            for k in range(1, n + 1):
                clause.append(variables[i - 1][j - 1][k - 1][1])
            clauses.append(clause)

    # 每个位置至多有一个取值
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                for t in range(k + 1, n + 1):
                    clauses.append([-variables[i - 1][j - 1][k - 1][0], -variables[i - 1][j - 1][t - 1][0]])
                    clauses.append([-variables[i - 1][j - 1][k - 1][1], -variables[i - 1][j - 1][t - 1][1]])

    # 两个正交矩阵在同一个位置取值必须互异
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                for t in range(1, n + 1):
                    for r in range(1, n + 1):
                        for c in range(1, n + 1):
                            if i == r and c == j:
                                continue
                            clauses.append([-variables[i - 1][j - 1][k - 1][0], -variables[i - 1][j - 1][t - 1][1],
                                           -variables[r - 1][c - 1][k - 1][0], -variables[r - 1][c - 1][t - 1][1]])

    return clauses


def decode_latin_square(solution, n):
    # 将SAT求解器的解转换为一个布尔值列表
    values = np.array(solution).reshape(n, n, n, 2)

    # 对于每个位置(i, j)，枚举所有数字k(1 <= k <= n)，如果对应的布尔变量为True，则表示第i行第j列是数字k
    latin_square = [[[0 for j in range(n)] for i in range(n)],
                    [[0 for j in range(n)] for i in range(n)]]
    for l in range(2):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if values[i][j][k][l] > 0:
                        latin_square[l][i][j] = (k + 1)
                        break
    # 将每个位置的数字放入n阶正交拉丁方阵中，并输出该方阵
    latin_square = np.array(latin_square[0]) * 10 + np.array(latin_square[1])
    print(latin_square)


@func_set_timeout(20)
def solve_latin_square(n):
    # 编码正交拉丁方阵问题
    clauses = encode_latin_square(n)

    # 使用SAT求解器求解问题
    time_start = time.time()
    solution = pycosat.solve(clauses)
    time_end = time.time()

    # 如果SAT求解器找到了一个可行解，则解码解并输出正交拉丁方阵
    if solution != "UNSAT":
        decode_latin_square(solution, n)
        print(f'Time of num {n}: {time_end - time_start} s')
    else:
        print("No solution found.")


for n in range(1, 9):
    try:
        solve_latin_square(n)
    except FunctionTimedOut as e:
        print('solve_latin_square:::', e)


