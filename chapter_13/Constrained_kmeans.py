# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 09:34
# @Author  : UNE
# @Project : Mechine_learning
# @File    : Constrained_kmeans.py
# @Software: PyCharm

# 《机器学习》（周志华）第十章10.10
"""
试为图13.7算法第10行写出违约检测算法。

首先可以通过并查集或者其他方法， 计算出所有的必连集合。 
由于必连的样本必然出现在一个类中，使用一个新样本来代替这些样本，新样本的属性参数取为必连集合中样本的均值。 
如果新样本所属的必连集合中任何一个样本与其他样本存在勿连关系，则新样本与该样本设为勿连关系。 
然后在约束K均值算法中仅需要考虑勿连约束。
"""

from tool import readxls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜4.xlsx", 0, "Sheet1")
    data = np.array(data)
    (m, n) = data.shape

    k = 3
    u = data[[6, 12, 17], :]
    c1 = {}

    # 必连标记，如果样本没有约束，则为0，如果有，则为必连集合编号
    flag_c = np.zeros([m, 1])
    # 必联集合
    c1[3] = [14, 17]
    c1[2] = [12, 20]
    c1[1] = [4, 25]
    # 设置必联标记
    for i in range(1,4):
        flag_c[c1[i]] = i
    # 勿连标记，记录样本是否有勿连标记，如果没有，为0，如果有，则为勿连序列对对应的起始序号
    flag_m = np.zeros([m, 1])
    # 勿连序列对，按第一个样本序列升序排列
    m1 = [[2, 13, 19, 21, 23, 23],[21, 23, 23, 2, 13, 19]]
    # 设置勿连标记
    for i in range(6):
        if flag_m[m1[0][i]] == 0:
            flag_m[m1[0][i]] = i

    while 1:
        flag_p = np.zeros([m, 1])
        # 对所有样本遍历，选择最近的集合
        for i in range(m):
            # 如果已经被标记就跳过
            if flag_p[i] > 0:
                continue

            # 如果属于必连集合则使用集合均值
            if flag_c[i] > 0:
                tx = np.mean(data[c1[flag_c[i][0]], :])
            else:
                tx = data[i, :]
            # 计算当前样本与所有类中心的距离
            dist = np.zeros([k, 1])
            for j in range(k):
                dist[j] = np.linalg.norm(tx - u[j, :])

            # 按照距离排序
            dist = pd.DataFrame(dist)
            dist = dist[0].order()
            # 进行最优分类选择

            # 是否满足约束标记
            mf = 0
            j = 0
            for j in range(k):
                tj = dist.iloc[j]
                ptr = int(flag_m[i])
                mf = 0
                # 由于这个实例中勿连约束只有6个，所以设置为7
                # 选择一个距离小于所有勿连样本的分类
                while ptr > 0 and ptr < 7 and m1[0][ptr]==i:
                    if flag_p[m1[1][ptr]] == tj:
                        mf = 1
                        break
                    ti = m1[1][ptr]
                    tdist = np.linalg.norm(data[ti, :] - u[tj, :])
                    if tdist < dist[j]:
                        mf = 1
                        break
                    ptr += 1
                # 如果不满足，跳出当前分类
                if mf == 1:
                    continue
                break

            # 如果没有合适分类，进行常规的约束判断，不做距离判断
            tj = 0
            if mf == 1 and j == k-1:
                for j in range(k):
                    tj = dist.iloc[j]
                    ptr = flag_m[i]
                    mf = 0
                    while ptr > 0 and ptr < 7 and m1[0][ptr]==i:
                        if flag_p[m1[1][ptr]] == tj:
                            mf = 1
                            break
                        ptr += 1
                    if mf == 1:
                        continue
                    break

            if flag_c[i] > 0:
                flag_p[c1[flag_c[i][0]]] = tj
            else:
                flag_p[i] = tj

        # 将各类集合清空
        c = np.zeros([k, 30])
        nums = np.zeros([k, 1])
        for i in range(m):
            nums[flag_p[i]] += 1
            c[flag_p[i], nums[flag_p[i]]] = i
        # 计算两次均值差异，并更新均值
        ut = np.zeros([k, 2])
        for i in range(k):
            for j in range(nums[i]):
                ut[i, :] += data[c[i, j], :]
            ut[i, :] /= nums[i]

        # 迭代结束条件
        du = np.linalg.norm(ut - u)
        if du < 0.001:
            break
        else:
            u = ut

    for i in range(k):
        # 绘制类中点
        plt.plot(data[c[i, 0:nums[i]], 1], data[c[i, 0:nums[i]], 2], 'o')
        tc = data[c[i, 0:nums[i]], :]
        # 计算类凸包，并画线
        from scipy.spatial import ConvexHull
        chl = ConvexHull(tc)
        plt.plot(tc[chl.vertices, 0], tc[chl.vertices, 1], "r--", lw=2)
        # 最后一段回路的闭合
        tmp1 = [tc[chl.vertices[0], 0], tc[chl.vertices[len(chl.vertices) - 1], 0]]
        tmp2 = [tc[chl.vertices[0], 1], tc[chl.vertices[len(chl.vertices) - 1], 1]]
        plt.plot(tmp1, tmp2, "r--", lw=2)

    # 绘制勿联约束
    for i in range(3):
        plt.plot(data[[m1[0][i],m1[1][i]], 0], data[[m1[0][i],m1[1][i]], 1], 'r--')
    # 绘制类中心
    plt.plot(u[:,0], u[:,1], '>')
    # 绘制必联约束
    for i in range(1, 4):
        plt.plot(data[c1[i], 0], data[c1[i], 1], 'r')

    plt.xlabel(u"密度")
    plt.ylabel(u"含糖率")
    plt.title(u"约束K-means")
    plt.show()
