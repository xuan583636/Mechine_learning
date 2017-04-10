# -*- coding: UTF-8 -*-
# 机器学习（周志华）第三章习题3.5
"""
编程实现线性判别分析，并给出西瓜数据集3.0å上的结果
"""

import matplotlib.pyplot as plt  # 画散点图
from numpy import *

from tool import readxls


def main():
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.0.xlsx", 0, "Sheet1")
    y = data[3]
    x = mat(data[0:2])
    u = mat(zeros((2, 2)))

    # 计算均值
    for i in range(17):
        index = int(y[i])
        u[:, index] = u[:, index] + x[:, i]
    u[:, 0] /= 8
    u[:, 1] /= 9

    # 计算两类协方差矩阵和
    sw = zeros((2, 2))
    for i in range(17):
        index = int(y[i])
        temp = (x[:, i] - u[:, index])
        sw = sw + temp * temp.T

    # 求逆
    # U, S, V = linalg.svd(sw) # 奇异值分解
    # V / S * U.T 为逆
    B = linalg.inv(sw)
    w = B * (u[:, 0] - u[:, 1])

    # 绘图
    plt.title("LDA")
    plt.xlabel("Denisty")
    plt.ylabel("Sguar content")

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    index = 0
    for i in data[3]:
        if i == 1.0:
            x1.append(data[0][index])
            y1.append(data[1][index])
        else:
            x2.append(data[0][index])
            y2.append(data[1][index])
        index += 1

    plt.plot(x1, y1, 'ro', label="Good")
    plt.plot(x2, y2, 'og', label="Bad")

    W = w.T.A[0]
    pl = -(0.2 * W[0] - 0.01) / W[1]
    pr = -(0.8 * W[0] - 0.01) / W[1]

    plt.plot([0.2, 0.8], [pl, pr])

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()