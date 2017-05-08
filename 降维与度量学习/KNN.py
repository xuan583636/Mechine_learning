# -*- coding: utf-8 -*-
# @Time    : 2017/4/30 10:03
# @Author  : UNE
# @Project : Mechine_learning
# @File    : KNN.py
# @Software: PyCharm

# 《机器学习》（周志华）第十章 10.1
"""
编程实现k邻近分类器，在西瓜数据集3.0α上比较其与决策树分类边界的异同

单变量决策树只有水平和垂直边界不同，k邻近分类器可以有曲线边界
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False       # 解决保存图像是负号'-'显示为方块的问题

from tool import readxls

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    x = np.array(data[6: 8])
    y = np.array(data[8])
    y = -2 * y + 3                          # 1 好瓜，-1坏瓜

    k = 3                                   # 近邻数
    edist = pd.Series(np.zeros(17))         # 计算k近邻边界
    for i in np.arange(0.22, 0.78, 0.01):
        for j in np.arange(0.02, 0.48, 0.01):
            # 计算各样本距离
            for l in range(17):
                # || ||²范数
                edist[l] = (np.linalg.norm([i,j] - x[:,l])) ** 2
            # 对距离排序
            edist = edist.order()
            # 选取前K个，找出其下标
            Sum = 0
            for index in edist.index[0:k]:
                Sum += y[index]
            pe = np.sign(Sum)
            if pe == 1:
                plt.plot(i, j, '.y', alpha=.5)
            else:
                plt.plot(i, j, '.g', alpha=.5)

    # 画点
    # x表示好瓜  o表示坏瓜
    # 蓝色表示样本分类 红色表示错误的分类
    for i in range(17):
        if y[i] == 1:
            plt.plot(x[0,i], x[1,i], 'ob')
        elif y[i] == -1:
            plt.plot(x[0, i], x[1, i], 'xb')
    plt.title(u"%d近邻"%k)
    plt.xlabel(u"密度")
    plt.ylabel(u"含糖率")
    plt.show()

