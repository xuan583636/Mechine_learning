# coding: utf-8
# 核技巧
# 《机器学习》（周志华）第六章6.9
"""
试用核技巧推广对率回归，产生"核对率回归"
"""

import readxls
import numpy as np
import matplotlib.pyplot as plt

def run(x, y):
    old_l = 0           # 记录上次计算的L
    n = 0               # 迭代次数
    b = np.zeros((18,1))
    b[17] = 1

    # 高斯核矩阵
    k = np.ones((18,18))
    for i in range(17):
        for j in range(17):
            k[i, j] = np.exp(-0.5 * np.dot((x[:, i] - x[:, j]).T, (x[:, i] - x[:, j])) )

    while(1):
        cur_l = 0
        bx = np.zeros((17, 1))
        bx = np.dot(b.T, k)
        cur_l = sum((-y * bx[0][:17]) + np.log(1 + np.exp(bx[0][:17])))

        if cur_l - old_l < 0.001:
            break

        n += 1
        old_l = cur_l
        p1 = np.zeros((17,1))
        dl = 0
        d2l = 0

        for i in range(17):
            p1[i] = 1 - 1/(1 + np.exp(bx[0][i]))
            dl -= k[:, i] * (y[i] - p1[i])
            d2l += np.dot(k[:, i], k[:, i].T) * p1[i] * (1 - p1[i])

        b = b - d2l / dl


if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.0.xlsx", 0, "Sheet1")
    x = np.array(data[0:2])
    y = np.array(data[3])

    run(x, y)