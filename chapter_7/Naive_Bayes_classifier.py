# -*- coding: utf-8 -*-
# @Time    : 2017/4/11 09:28
# @Author  : UNE
# @Site    : 
# @File    : Naive_Bayes_classifier.py
# @Software: PyCharm
# 《机器学习》（周志华）第七章7.3
"""
编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集3.0为训练集，对p.151"测1"样本进行评估
"""

from tool import readxls
import numpy as np
import pandas as pd
from math import pi

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    x = np.array(data[0:8])
    y = np.array(data[8])
    y = y - 1

    test = x[:, 0]              # 测试用例
    pn = [3, 3, 3, 3, 3, 2]     # 各参数项取值的个数
    pc = 0                      # 为正的概率
    nc = 0                      # 为负的概率

    for i in range(6):          # 对6种离散参数遍历
        c = np.zeros((2, 1))
        for j in range(17):     # 累积次数，计算p(xi|c)
            if x[i, j] == test[i]:
                c[y[j]] += 1
        # 拉普拉斯修正
        pc += np.log((c[0] + 1) / (8 + pn[i]))
        nc += np.log((c[1] + 1) / (9 + pn[i]))

    # 对两种连续变量参数的遍历
    pdd = pd.DataFrame(data)
    fp = lambda a: (a.ix[8] == 1)   # 判定结果为0的样本
    fn = lambda a: (a.ix[8] == 2)   # 判定结果为1的样本

    p_denisty = pdd.T[pdd.apply(fp)][6]
    p_sugar = pdd.T[pdd.apply(fp)][7]
    n_denisty = pdd.T[pdd.apply(fn)][6]
    n_sugar = pdd.T[pdd.apply(fn)][7]

    p = np.zeros((2, 2))
    p[0, 0] = np.exp(-(test[6] - p_denisty.mean()) ** 2 / (2 * p_denisty.std() ** 2)) / (
        (2 * pi) ** 0.5 * p_denisty.std())
    p[0, 1] = np.exp(-(test[6] - n_denisty.mean()) ** 2 / (2 * n_denisty.std() ** 2)) / (
        (2 * pi) ** 0.5 * n_denisty.std())
    p[1, 0] = np.exp(-(test[7] - p_sugar.mean()) ** 2 / (2 * p_sugar.std() ** 2)) / (
        (2 * pi) ** 0.5 * p_sugar.std())
    p[1, 1] = np.exp(-(test[7] - n_sugar.mean()) ** 2 / (2 * n_sugar.std() ** 2)) / (
        (2 * pi) ** 0.5 * n_sugar.std())

    for i in range(2):
        pc += np.log(p[i, 0])
        nc += np.log(p[i, 1])

    print pc, nc
