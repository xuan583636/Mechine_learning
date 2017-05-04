# -*- coding: utf-8 -*-
# @Time    : 2017/5/3 15:19
# @Author  : UNE
# @Project : Mechine_learning
# @File    : TSVM.py
# @Software: PyCharm

# 《机器学习》（周志华）第十三章13.4
"""
实现TSVM算法，选择两个UCI数据集，将其中30%作为测试样本，10%作为训练样本，60%作为无标记样本，分别训练出利用无标记样本的TSVM和仅利用有标记样本的SVM，并比较其性能
"""

import numpy as np
import pandas as pd
from sklearn import svm

from tool import readxls

def plc_svm(xl,yl,cl,xc,yc,cu):
    nl, ml = xl.values.shape
    nc, mc = xc.values.shape

    H = np.zeros([ml+nl+mc, ml+nl+mc])
    H[0:nl, 0:nl] = np.diag(np.ones(1, nl))     # 提取对角线
    f = np.array([np.zeros(1,nl), cl * np.ones(1, ml), cu * np.ones(1, mc)])

    A = np.zeros(2*(ml+mc), ml+nl+mc)
    for i in range(0, ml):
        A[i, 0:nl] = -yl[i] * xl[:, i].T
        A[i, nl+i] = -1
        A[ml+mc+i, nl+i] = -1

    for i in range(0, mc):
        A[i, 0:nl] = -yc[i] * xc[:, i].T
        A[ml+i, ml+nl+i] = -1
        A[2*ml+mc + i, ml+nl+i] = -1

    b = np.array([-1*np.ones(ml+mc,1), np.zeros(ml+mc, 1)])
    # 使用二次规划求解
    # 由于需要用到松弛变了，返回松弛变量和超平面参数
    


if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/UCI-iris数据集.xlsx", 0, "Sheet1")
    data = pd.DataFrame(data[12:18])
    for i in range(100,150):
        del data[i]
    data.loc[5] = 3 - 2 * data.loc[5]       # 将标记变为1，-1
    # 训练样本
    xl = data.loc[0:3, np.append(np.arange(5), np.arange(50,55))]
    xl = xl.T
    yl = data.loc[5, np.append(np.arange(5), np.arange(50,55))]
    # 测试样本
    xc = data.loc[0:3, np.append(np.arange(5,20), np.arange(55, 70))]
    xc = xc.T
    tyc = data.loc[5, np.append(np.arange(5,20), np.arange(55, 70))]
    # 无标记样本
    xt = data.loc[0:3, np.append(np.arange(20,50), np.arange(70, 100))]
    xt = xt.T
    tyt = data.loc[5, np.append(np.arange(20,50), np.arange(70,100))]

    nl, ml = xl.values.shape                # 训练样本属性与分类数
    cl = 1                                  # 训练样本的折中参数

    clf = svm.SVC()
    clf.fit(xl.values, yl.values, C=cl)
    predict = clf.predict(xt.values)
    err_svm = sum(abs(predict - tyt)) / 2   # 计算未使用无标记样本时SVM的错误个数
    print err_svm

    yt = clf.predict(xt.values)             # 对无标记进行伪标记
    cu = 0.01                               # 初始无标记样本的折中参数

    txt = xt                                # 为了计算方便需要改变xc的排序，所以使用临时变量
    while cu < cl:
        # 使用训练样本和伪标记样本训练SVM
        


