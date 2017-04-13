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

# 全局变量
global x, y, Py, Ptr, Tree

"""
信息增益选择
    curset:当前样本集合
    sw:样本权值
输出
    n：最优属性
    threshold:连续属性返回阀值
"""
def entropy_paraselect(curset,sw):
    global x, y
#     通过样本编号与属性获取当前样本
    curx = x[:, curset[0]:curset[1]]
    cury = y[curset[0]:curset[1]]
    csw = sw[curset[0]:curset[1]]
    all_num = cury.__len__()                # 当前样本总数
    max_ent = -100                          # 为ent设初值，要最大值，所以设为一个很小的数
    minn=0                                  # 记录最优的属性编号
    threshold = 0

    for i in range(2):
        con_max_ent = -100
        con_threshold = 0
        xlisw = np.sort(curx[i], 2)
        print xlisw
        break


"""
    parent:父节点编号
    curset:当前的样本编号集合
    sw:样本权值
    height:最高高度
"""
def TreeGenerate(parent, curset, sw, height):
    global Ptr, x, y, Py, Tree
    # 新增一个节点，并记录它的父节点
    Ptr += 1
    Tree[0][Ptr] = parent
    cur = Ptr

    # 递归返回情况1：当前所有样本属于同一类
    n = len(curset)
    # 计算当前y的取值分类及各有多少个  如果只有一类表示属于同一类
    cury = y[curset]           # 通过样本编号选取所需要的样本
    y_data = np.unique(cury)                # 样本集合去重
    y_nums = []
    for i in range(len(y_data)):
        count = 0
        for j in range(len(cury)):
            if cury[j] == y_data[i]:
                count += 1
        y_nums.append(count)

    if y_nums[0] == n:
        for i in range(n):
            Py[0][curset[i]] = y_data
        return

    # 递归返回情况2:属性已经全部划分还有不同分类的样本，或者所有样本属性一致但是分类不同(这时就是有错误的样本)
    if height == 2:
        csw = sw[curset]
        tans = (sum(np.dot(csw.T, cury) > 0) * 2 - 1)
        for i in range(n):
            Py[0][curset[i]] = tans
        return

    """
    主递归
    实现了4个最优属性选择方法,使用了相同的参数与输出:信息增益，增益率，基尼指数，对率回归
    具体参数介绍间函数具体实现
    因为是相同的参数与输出，直接该函数名就能换方法
    """
    k, threshold = entropy_paraselect(curset, sw)
    # 连续属性只会分为两类 大于阀值一类 小于阀值一类 分类后继续递归
    num = [1, 1]
    tmp_set = np.zeros((2, 100))
    for i in range(n):
        if x[k, curset[i]] >= threshold:
            tmp_set[0, num[0]] = curset[i]
            num[0] += 1
        else:
            tmp_set[1, num[1]] = curset[i]
            num[1] += 1
    for i in range(2):
        # 由于用数组存编号，会有0存在，将样本分开后与当前的样本求交集  把0去掉
        ttmp_set = np.intersect1d(tmp_set[i], curset)
        TreeGenerate(cur, ttmp_set, sw, height+1)


if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    x = np.array(data[6:8])
    y = np.array(data[8])
    y_index = y - 1
    y = -2 * y + 3                  # 将y映射到1，-1

    try:                            # 一维数组的情况
        m, n = y.shape
    except:
        m = 1
        n = y.__len__()

    set = np.arange(0, n)
    sy = np.zeros((1,17))           # 记录累积分类器的分类
    sw = np.ones((1, 17)) / 17      # 样本的权值，初始相同
    fenlei = ['√', '×']
    shuxing = ['密度', '含糖率']

    # 记录每次累积分类器的分类
    Res = pd.DataFrame(np.zeros((12,19)),dtype=object,
                       index=[1,2,3,4,5,6,7,8,9,10,11,12],
                       columns=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,'分类属性','阈值'],
                                [fenlei[int(i)] for i in y_index] + ['无','无']])

    for i in range(12):             # 产生12个分类器
        Tree = np.zeros((1,100))
        Ptr = 0
        Py = np.zeros((1,17))
        # 生成决策树，返回根节点的最优属性和阈值
        (minn, threshold) = TreeGenerate(0, set, sw, 1)
        break

        er = sum(np.dot((Py != y).T, sw))
        if er > 0.5 :
            break
        a = 0.5 * np.log((1 - er) / er)
        sw = np.dot(sw.T, np.exp(a * ((Py != y) * 2) - 1))
        sw = sw / sum(sw)
        sy = sy + a * Py
        for j in range(17):
            Res[i, j] = fenlei[(3 - np.sign(sy[j])) / 2]

        Res[i, 18] = shuxing[minn]
        Res[i, 19] = threshold
