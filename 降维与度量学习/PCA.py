# -*- coding: utf-8 -*-
# @Time    : 2017/4/30 10:54
# @Author  : UNE
# @Project : Mechine_learning
# @File    : PCA.py
# @Software: PyCharm

# 《机器学习》（周志华）第十章10.6
"""
使用matlab的PCA函数对人脸数据进行降维，并观察前20个特征向量对应的图像
"""

import numpy as np
from sklearn import decomposition
import imread

if __name__ == '__main__':
    filename_m = '/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/yalefaces/subjct%03d.gif'
    # 记录所有数据的矩阵
    info = np.zeros((243*320,166))
    # 输入，将图片保存为一列
    k = 20
    for i in range(1,167):
        filename = filename_m % i
        img = imread.imread(filename)
        info[:, i] = img

    # 进行pca分析
    pca = decomposition.PCA()
    coeff = pca.fit(info)
    # 只保留前k个特征
    coeff[:, k:166]=0
    info = info * np.dot(coeff, coeff.T)
