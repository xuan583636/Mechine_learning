# -*- coding:UTF-8 -*-
# 机器学习（周志华）第三章3.4

import readxls
from numpy import *
import matplotlib.pylab as plt
from logarithmic_regression import logarithmic_regression

def main():
    data = readxls.excel_table_byrow_and_col("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/UCI-iris数据集.xlsx", "Sheet1", (6,0),(12,100))
    d = mat(data)

    # 10折交差验证
    # err0 = 0
    # for tn in range(0, 9):
    #     reset = d[:, (tn*5):((tn+1)*5)]
    #     set = mat(zeros((6,0)))
    #     if tn == 0:
    #         set = d[:, ((tn+1)*5):50]
    #     else:
    #         set = d[:, 0:(tn*5)]
    #         set = hstack((set, d[:, ((tn+1)*5):50])) # 行合并
    #
    #     b = logarithmic_regression(set, 45, 4)


    # 留出法
    err1 = 0
    set = d[:, 0:40] # 数组切片
    reset = d[:, 40:50]
    set = concatenate((set, d[:, 50:90]), axis=1)
    reset = concatenate((reset, d[:, 90:100]), axis=1)

    xdata = mat(set[0:4]).T
    ydata = mat(set[5]).T
    ydata = ydata - 1
    b = logarithmic_regression(xdata, ydata, 80, 4)
    print b
    b = b.T
    list = []
    for i in range(len(b)):
        t = sum(abs(b.A[i])) / len(b.A[i])
        list.append(t)
    list = mat(list)
    x = reset[0:4]
    y = mat(reset[5]).T
    y = y - 1
    tmp = 1 / (1 + exp(-list * x))
    tmp = (tmp >= 0.5)
    tmp = tmp.A[0]
    for i in range(20):
        if tmp[i] != y[i]:
            err1 += 1
    print err1

if __name__ == '__main__':
    main()