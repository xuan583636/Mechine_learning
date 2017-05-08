# -*- coding: utf-8 -*-
# @Time    : 2017/4/16 16:12
# @Author  : UNE
# @Project : Mechine_learning
# @File    : K_means.py
# @Software: PyCharm
# 《机器学习》（周志华）第九章9.4

"""
编程实现k均值算法，设置三组不同的k值，三组不同的初始中心点，在西瓜数据集4.0上进行实验，并讨论什么样的初始中心有利于取得好结果。
西瓜数据集4.0中30未分类样本，属性纬度2，都是连续属性。
 
K-means算法是一种局部最优的最小化类间最小化均方误差的算法，初始随机的中心不同会导致算法的迭代次数与最终结果有很大的不同。一般来说，初始的中心越集中且越靠近边缘，则会使得迭代次数更多。初始中心越分散，结果越好。
"""

from tool import readxls
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜4.xlsx", 0, "Sheet1")
    data = np.array(data)
    (m, n) = data.shape

    k = 4                                               # 设置k值
    u = data[np.random.randint(30, size=k), :]          # 产生随机均值

    while 1:
        c = np.zeros((k, 30), dtype="int64")                           # 将各类集合清空
        nums = np.zeros((k, 1), dtype="int64")
        # 对所有样本遍历，选择最近集合
        for i in range(m):
            mind = 100000
            minl = 0
            for j in range(k):
                d = np.linalg.norm(data[i, :] - u[j, :])
                if d < mind:
                    mind = d
                    minl = j
            c[minl, nums[minl]] = i
            nums[minl] += 1
        # 计算两次均值差异，并更新均值
        ut = np.zeros((k, 2))
        for i in range(k):
            for j in range(nums[i]):
                ut[i, :] += data[c[i, j], :]
            ut[i, :] /=  nums[i]
        # 迭代结束条件
        du = np.linalg.norm(ut - u)
        if du < 0.001:
            break
        else:
            u = ut

    # 使用不同的符号绘制
    ch = "o*+.>"

    for i in range(k):
        # 绘制类中的点
        plt.plot(data[c[i, 0:nums[i]], 0], data[c[i, 0:nums[i]], 1], ch[i])
        tc = data[c[i, 0:nums[i]], :]
        # 计算类凸包，并画线
        from scipy.spatial import ConvexHull
        chl = ConvexHull(tc)
        plt.plot(tc[chl.vertices, 0], tc[chl.vertices, 1], "r--", lw=2)
        # 最后一段回路的闭合
        tmp1 = [tc[chl.vertices[0], 0], tc[chl.vertices[len(chl.vertices)-1], 0]]
        tmp2 = [tc[chl.vertices[0], 1], tc[chl.vertices[len(chl.vertices)-1], 1]]
        plt.plot(tmp1, tmp2, "r--", lw=2)

    plt.xlabel("Denisty")
    plt.ylabel("Sugar")
    plt.title("K_Means")

    plt.legend()
    plt.show()
