# -*- coding: utf-8 -*-
# @Time    : 2017/4/13 16:30
# @Author  : UNE
# @Site    : 
# @File    : AdaBoosw.py
# @Software: PyCharm
# 《机器学习》（周志华）第八章8.3
"""
编程实现AdaBoosw，以不剪枝决策树为基学习器，在西瓜数据集3.0å上训练一个AdaBoosw集成，并于图8.4作比较
"""

from tool import readxls
import numpy as np
import pandas as pd
from dTree import dTree

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    x = pd.DataFrame(data[6:8])
    y = pd.DataFrame(data[8])
    y = y.T
    y_index = y - 1
    y = -2 * y + 3                  # 将y映射到1，-1

    try:                            # 一维数组的情况
        m, n = y.shape
    except:
        m = 1
        n = len(y)

    set = np.arange(0, n)
    sy = np.zeros((1,17))           # 记录累积分类器的分类
    sw = np.ones((1, 17)) / 17      # 样本的权值，初始相同
    fenlei = ['√', '×']
    shuxing = ['密度', '含糖率']

    # 记录每次累积分类器的分类
    Res = pd.DataFrame(np.zeros((12,19)),dtype=object,
                       index=[1,2,3,4,5,6,7,8,9,10,11,12],
                       columns=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,'分类属性','阈值'],
                                [fenlei[int(y_index[i])] for i in y_index] + ['无','无']])

    for i in range(12):             # 产生12个分类器
        Tree = np.zeros((1,100))
        Ptr = 0
        Py = np.zeros((1,17))
        # 产生决策树基学习器
        dtree = dTree(x, y, Py, Ptr, Tree)
        # 生成决策树，返回根节点的最优属性和阈值
        minn, threshold = dtree.TreeGenerate(0, set, sw, 1)
        Py = dtree.Py
        print minn, threshold

        er = sum(np.dot((Py != y), sw.T))
        if er > 0.5 :
            break
        a = 0.5 * np.log((1 - er) / er)
        sw = sw * (np.exp(a * ((Py != y) * 2) - 1).values)
        sw = sw / sum(sw[0])
        sy = sy + a * Py

        for j in range(17):
            Res.iloc[i, j] = fenlei[int((1 - np.sign(sy[0][j]))/2)]

        Res.iloc[i, 17] = shuxing[minn]
        Res.iloc[i, 18] = threshold

    print Res