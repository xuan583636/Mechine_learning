# -*- coding:UTF-8 -*-
# 机器学习（周志华）第三章习题3.3

import readxls
import matplotlib.pyplot as plt # 画散点图
from numpy import *

def main():
    # 数据准备
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.0.xlsx", 0, "Sheet1")
    old = mat(zeros((17, 2)))
    n = 0
    b = mat(zeros((17, 2)))
    b[:, -1] = 1 # 利用索引得到最后一列为1

    # 对率回归计算
    while(1):
        cur = mat(zeros((17, 2)))
        x = mat(data[0:2]).T
        y = mat(data[3]).T
        bx = multiply(x, b)
        cur = cur - multiply(y, bx) + log(1 + exp(bx))
        # 判断是否达到稳定点
        tag = cur - old
        judge = (tag < 0.0001)
        if judge.all():
            break
#         更新参数
        n = n + 1
        old = cur
        dl = mat(zeros((17, 1)))  # 一阶导
        d2l = mat(zeros((17, 1)))  # 二阶导

        p1 = 1 - 1/(1 + exp(bx))
        p1 = mat(p1)
        dl = dl - multiply(x, (y - p1))
        d2l = d2l + multiply(multiply(x, x), multiply(p1, (1-p1)))

        b = b - dl / d2l

# 绘图
    plt.title("Logarithmic regression")
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

    pl = (0.2*b[:, 0]) / b[:, 1]
    pr = (0.8*b[:, 0]) / b[:, 1]
    pl = sum(pl.T.A) / 17.0
    pr = sum(pr.T.A) / 17.0

    plt.plot([0.2, 0.8],[pl, pr])

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()