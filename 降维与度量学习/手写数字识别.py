# -*- coding: utf-8 -*-
# @Time    : 2017/5/8 13:59
# @Author  : UNE
# @Project : Mechine_learning
# @File    : 手写数字识别.py
# @Software: PyCharm

# 《机器学习实战》第二章

import numpy as np
from os import listdir

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def classify(inx, dataSet, labels, k):
    # 距离计算
    size = dataSet.shape[0]
    diffMat = np.tile(inx, (size, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = np.sum(sqDiffMat, axis=1)
    distance = sqDistance ** 0.5
    # 选择距离最小的K个点
    sortedDisIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDisIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1   # 赋值新值
    # 排序
    sortedclassCount = sorted(classCount.iteritems(), key=lambda asd:asd[1], reverse=True)
    return sortedclassCount[0][0]


if __name__ == '__main__':
    filename = "/Users/JJjie/Desktop/Projects/Mechine_Learning/dataset/knn-digits/"
    trainingFileList = listdir(filename+"/trainingDigits")
    m = len(trainingFileList)
    hwLabels = []
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('.')[0].split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i] = img2vector(filename+"trainingDigits/%s" % fileNameStr)

    testingFileList = listdir(filename+"/testDigits")
    mTest = len(testingFileList)
    error = 0.0
    for i in range(mTest):
        fileNameStr = testingFileList[i]
        classNumStr = int(fileNameStr.split('.')[0].split('_')[0])
        testMat = img2vector(filename + "testDigits/%s" % fileNameStr)
        classifyResult = classify(testMat, trainingMat, hwLabels, 3)
        if classifyResult != classNumStr:
            error += 1.0
    print "一共错误%d个" % error
    print "错误率：%f" % (error / mTest)
