# -*- coding: utf-8 -*-
# @Time    : 2017/5/1 09:49
# @Author  : UNE
# @Project : Mechine_learning
# @File    : Relief.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from tool import readxls

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    x = np.array(data[0:8])
    y = np.array(data[8])

    n, m = x.shape
    # 标记属性是否是离散数据，1是0不是
    pf = [1,1,1,1,1,1,0,0]

    # 把连续属性用min-max归一化
    x[6, :] = (x[6, :] - min(x[6, :])) / (max(x[6, :]) - min(x[6, :]))
    x[7, :] = (x[7, :] - min(x[7, :])) / (max(x[7, :]) - min(x[7, :]))

    # 计算所有样本距离
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i < j:
                for k in range(n):
                    if pf[k] == 1:
                        dist[i, j] += int(x[k, i] != x[k, j])
                    else:
                        dist[i, j] += (x[k, i] - x[k, j])**2
            else:
                dist[i, j] = dist[j, i]

    # 初始化各属性权重
    p = np.zeros(n)
    for i in range(m):
        d = pd.Series(dist[i])
        d = d.order()
        index = d.index
        p2 = np.zeros(2)
        p2 -= 1
        ptr = 1
        # 选择最近的本类与异类
        while p2[0] < 0 or p2[1] < 0:
            py = int(y[index[ptr]] != y[i])
            if p2[py] == -1:
                p2[py] = index[ptr]
            ptr += 1

        # 计算各属性权值
        for j in range(n):
            if pf[j] == 1:
                p[j] = p[j] - int(x[j, i] != x[j, p2[0]]) + int(x[j, i] != x[j, p2[1]])
            else:
                p[j] = p[j] - (x[j, i] - x[j, p2[0]])**2 + (x[j, i] - x[j, p2[1]])**2
    print p