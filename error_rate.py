# -*- coding:UTF-8 -*-
# 机器学习（周志华）第三章3.4

import readxls
from numpy import *
import matplotlib.pylab as plt
from logarithmic_regression import logarithmic_regression

def main():
    data = readxls.excel_table_byrow_and_col("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/UCI-iris数据集.xlsx", "Sheet1", (6,0),(12,100))
    d = mat(data)
    k = 100  # 样本总个数

    # 10折交差验证
    print "10折交差验证法"
    err0 = 0
    # 样本分割
    for tn in range(0, 9):
        reset = d[:, (tn*5):((tn+1)*5)]
        reset = concatenate((reset, d[:, (50+tn*5):(50+(tn+1)*5)]), axis=1)
        set = 0
        if tn == 0:
            set = d[:, ((tn+1)*5):50]
            set = concatenate((set, d[:, (50+(tn+1)*5):100]), axis=1)
        else:
            set = d[:, 0:(tn*5)]
            set = concatenate((set, d[:, ((tn + 1) * 5):50]), axis=1)
            set = concatenate((set, d[:, 0:(50+tn*5)]), axis=1)
            set = concatenate((set, d[:, (50+(tn + 1) * 5):100]), axis=1)

        xdata = set[0:5].T
        ydata = set[5].T
        ydata -= 1
        b = logarithmic_regression(xdata, ydata, 4)
        print "B的值为：", b
        b = b.T
        x = reset[0:5]
        y = mat(reset[5]).T
        y = y - 1
        tmp = 1 / (1 + exp(-b * x))
        tmp = (tmp >= 0.5)
        tmp = tmp.A[0]
        for i in range(len(y)):
            if tmp[i] != y[i]:
                err0 += 1

    print "预测样本一共%d个，预测错误%d个" % (10*10, err0)


    # 留出法
    print "\n留出法"
    err1 = 0
    set = d[:, 0:40] # 数组切片
    reset = d[:, 40:50]
    set = concatenate((set, d[:, 50:90]), axis=1)
    reset = concatenate((reset, d[:, 90:100]), axis=1)

    xdata = mat(set[0:5]).T
    ydata = mat(set[5]).T
    ydata = ydata - 1
    b = logarithmic_regression(xdata, ydata, 4)
    print "B的值为：", b
    b = b.T
    x = reset[0:5]
    y = mat(reset[5]).T
    y = y - 1
    tmp = 1 / (1 + exp(-b * x))
    tmp = (tmp >= 0.5)
    tmp = tmp.A[0]
    for i in range(20):
        if tmp[i] != y[i]:
            err1 += 1
    print "预测样本一共%d个，预测错误%d个" % (20,err1)

if __name__ == '__main__':
    main()