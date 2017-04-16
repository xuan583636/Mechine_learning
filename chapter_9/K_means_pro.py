# -*- coding: utf-8 -*-
# @Time    : 2017/4/16 22:05
# @Author  : UNE
# @Project : Mechine_learning
# @File    : K_means_pro.py
# @Software: PyCharm
# 《机器学习》（周志华）第九章9.4

"""
实现一种能自动确定聚类数的改进k均值算法，编程实现并在西瓜数据集上运行。
"""

from tool import readxls
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜4.xlsx", 0, "Sheet1")
    data = np.array(data)
    (m, n) = data.shape

    old_ts = 100                                        # 当前最低的平方误差，初始设置为一个很大的值
    old_c = 0
    old_nums = 0

    for k in range(2, 10):
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

        # 计算当前误差
        ts = 0
        for i in range(k):
            for j in range(nums[i]):
                ts += (np.linalg.norm(data[c[i, j], :] - u[i, :]))**2
            # 惩罚项
            num = float(nums[i])
            ts -= (num/m) * np.log(num/m)*0.5

        if ts < old_ts:
            old_ts = ts
            old_c = c
            old_nums = nums
        else:
            break

    # 使用不同的符号绘制
    ch = "o*+.>"
    # 取前一轮k值最佳
    nums = old_nums
    c = old_c
    k = k - 1

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
