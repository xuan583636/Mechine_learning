# -*- coding:UTF-8 -*-
# 机器学习（周志华）第三章习题3.3

import readxls
import matplotlib.pyplot as plt # 画散点图
from numpy import *

def logarithmic_regression(x, y, col):
    old = 0
    n = 0
    b = mat(zeros((col, 1)))

    # 对率回归计算
    while (1):
        cur = 0
        bx = x * b
        cur = sum(- multiply(y, bx) + log(1 + exp(bx)))
        # 判断是否达到稳定点
        tag = cur - old
        judge = (abs(tag) < 0.0001)
        if judge.all():
            break
            #         更新参数
        n = n + 1
        old = cur
        dl = mat(zeros((1, col)))  # 一阶导
        d2l = mat(zeros((1, col)))  # 二阶导

        p1 = 1 - 1 / (1 + exp(bx))
        p1 = mat(p1)
        temp = multiply(x, (y - p1))
        for i in range(len(temp)):
            dl -= temp[i]
        temp = multiply(multiply(x,x), multiply(p1, (1 - p1)))
        for i in range(len(temp)):
            d2l += temp[i]

        b = b - dl.T / d2l.T

    print "迭代%d次" % n
    return b

# 绘图
def draw(data, b, y, *info):
    plt.title(info[0])
    plt.xlabel(info[1])
    plt.ylabel(info[2])

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    index = 0
    for i in y:
        if i == 1.0:
            x1.append(data[0][index])
            y1.append(data[1][index])
        else:
            x2.append(data[0][index])
            y2.append(data[1][index])
        index += 1

    plt.plot(x1, y1, 'ro', label="Good")
    plt.plot(x2, y2, 'og', label="Bad")

    b = b.A
    pl = -(0.1 * b[0]) / b[1]
    pr = -(0.9 * b[0]) / b[1]
    plt.plot([0.1, 0.9], [pl, pr])

    plt.legend()
    plt.show()

def main():
    # 数据准备
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.0.xlsx", 0, "Sheet1")
    x = mat(data[0:2]).T
    y = mat(data[3]).T
    b = logarithmic_regression(x, y, 2)

    draw(data, b, data[3], "Logarithmic regression", "Denisty", "Sguar content")

if __name__ == '__main__':
    main()