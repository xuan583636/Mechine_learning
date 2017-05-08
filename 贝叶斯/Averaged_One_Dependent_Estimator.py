# -*- coding: utf-8 -*-
# @Time    : 2017/4/11 09:30
# @Author  : UNE
# @Site    : 
# @File    : Averaged_One_Dependent_Estimator.py
# @Software: PyCharm
# 《机器学习》（周志华）第七章7.6
"""
编程实现AODE分类器，并以西瓜数据集3.0为训练集，对p.151"测1"样本进行评估
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
        tpc = 1
        tnc = 1
        for j in range(6):      # 累积记数，计算p(xi|c,xi)
            ct = np.zeros((2, 1))
            for k in range(17):
                if x[j, k] == test[i] and x[j, k] == test[j]:
                    ct[y[j]] += 1
            tpc *= (ct[0] + 1) / (c[0] + pn[j])
            tnc *= (ct[1] + 1) / (c[1] + pn[j])

        pc += ((c[0] + 1) / (8 + pn[i])) * tpc
        nc += ((c[1] + 1) / (9 + pn[i])) * tnc

    print pc, nc
