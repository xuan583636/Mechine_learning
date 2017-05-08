# -*- coding: utf-8 -*-
# @Time    : 2017/4/15 15:27
# @Author  : UNE
# @Project : Mechine_learning
# @File    : dTree.py
# @Software: PyCharm

# 《机器学习》（周志华）第八章 决策树基学习器

import numpy as np
import pandas as pd

class dTree:
    def __init__(self, x, y, Py, Ptr, Tree):
        self.x = x
        self.y = y
        self.Py = Py
        self.Ptr = Ptr
        self.Tree = Tree

    """
    parent:父节点编号
    curset:当前的样本编号集合
    sw:样本权值
    height:最高高度
    """
    def TreeGenerate(self, parent, curset, sw, height):
        # 新增一个节点，并记录它的父节点
        self.Ptr += 1
        self.Tree[0][self.Ptr] = parent
        cur = self.Ptr

        # 递归返回情况1：当前所有样本属于同一类
        n = len(curset)
        # 计算当前y的取值分类及各有多少个  如果只有一类表示属于同一类
        cury = self.y[curset]  # 通过样本编号选取所需要的样本
        y_data = np.unique(cury.values)  # 样本集合去重
        y_nums = []
        for i in range(len(y_data)):
            count = 0
            for j in cury.columns:
                if cury[j].values == y_data[i]:
                    count += 1
            y_nums.append(count)

        if y_nums[0] == n:
            for i in range(n):
                self.Py[0][curset[i]] = y_data
            return

        # 递归返回情况2:属性已经全部划分还有不同分类的样本，或者所有样本属性一致但是分类不同(这时就是有错误的样本)
        if height == 2:
            tempsw = pd.DataFrame(sw)
            csw = tempsw[curset].values

            tans = (sum(np.dot(csw, cury.T) > 0) * 2 - 1)
            for i in range(n):
                self.Py[0][curset[i]] = tans
            return

        """
        主递归
        实现了4个最优属性选择方法,使用了相同的参数与输出:信息增益，增益率，基尼指数，对率回归
        具体参数介绍间函数具体实现
        因为是相同的参数与输出，直接该函数名就能换方法
        """
        k, threshold = self.entropy_paraselect(curset, sw)
        # 连续属性只会分为两类 大于阀值一类 小于阀值一类 分类后继续递归
        num = [0, 0]
        tmp_set = -np.ones((2, 100))
        for i in range(n):
            if self.x.values[k, curset[i]] >= threshold:
                tmp_set[0, num[0]] = curset[i]
                num[0] += 1
            else:
                tmp_set[1, num[1]] = curset[i]
                num[1] += 1
        for i in range(2):
            # 由于用数组存编号，会有0存在，将样本分开后与当前的样本求交集  把-1去掉
            ttmp_set = np.intersect1d(tmp_set[i], curset)
            self.TreeGenerate(cur, ttmp_set, sw, height + 1)

        return k, threshold

    """
    信息增益选择
        curset:当前样本集合
        sw:样本权值
    输出
        n：最优属性
        threshold:连续属性返回阀值
    """

    def entropy_paraselect(self, curset, sw):
        # 通过样本编号与属性获取当前样本
        curx = self.x[curset].values
        cury = self.y[curset].values
        csw = sw[0][curset]
        all_num = len(cury[0])  # 当前样本总数
        max_ent = -100  # 为ent设初值，要最大值，所以设为一个很小的数
        minn = 0  # 记录最优的属性编号
        threshold = 0

        for i in range(2):
            con_max_ent = -100
            con_threshold = 0
            xlisw = np.sort(curx[i])
            # 计算n-1个阈值
            for j in range(all_num - 1, 0, -1):
                xlisw[j] = (xlisw[j] + xlisw[j - 1]) / 2
            # 从n-1个阀值中选最优    (循环过程中ent先递减后递增 其实只要后面的ent比前面的大就可以停止循环)
            for j in range(1, all_num):
                cur_ent = 0
                # 计算各类正负例数
                nums = np.zeros((2, 2))
                for k in range(all_num):
                    nums[int(curx[i, k] >= xlisw[j]), int((1 - cury[0][k]) / 2)] += csw[k]
                # 计算ent 连续属性只分两种情况
                for k in range(2):
                    if nums[k, 0] > nums[k, 1]:
                        p = nums[k, 0] / (nums[k, 0] + nums[k, 1])
                        cur_ent += (p * np.log2(p + 0.00001) + (1 - p) * np.log2(1 - p + 0.00001)) * (
                        nums[k, 0] + nums[k, 1]) / all_num
                # 记录该分类最优取值
                if cur_ent > con_max_ent:
                    con_max_ent = cur_ent
                    con_threshold = xlisw[j]
            if con_max_ent > max_ent:
                max_ent = con_max_ent
                minn = i
                threshold = con_threshold

        n = minn
        return n, threshold