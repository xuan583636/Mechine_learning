# -*- coding: utf-8 -*-
# @Time    : 2017/5/15 09:10
# @Author  : UNE
# @Project : Mechine_learning
# @File    : Decision_trees.py
# @Software: PyCharm

# 《机器学习实战》第三章

import math
import operator
import pandas as pd
import numpy as np
import treePlotter as plt

# 计算信息熵
def calsShannonEnt(dataset):
    numEntries = len(dataset)
    labelCount = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

# 划分数据集，留下剩下的数据集
def splitDataset(dataset, axis, value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec = np.append(reducedFeatVec, featVec[axis+1:])
            retDataset.append(reducedFeatVec)
    retDataset = np.array(retDataset)
    return retDataset

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataset):
    numFeature = dataset.shape[1]
    baseEntropy = calsShannonEnt(dataset)
    baseInfoGain = 0.0
    bastFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 得到列表中唯一数据值最快的方法
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calsShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > baseInfoGain:
            baseInfoGain = infoGain
            bastFeature = i
    return bastFeature

# 主要信息熵
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回出现频率最多的分类名称
    return sortedClassCount[0][0]

# 创建树
def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    # 到了最后一层，只有label参数，递归停止条件
    # 所有的值都一样的情况
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]   # 不改变原始表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    filename = "/Users/JJjie/Desktop/Projects/dataset/zzh_watermelon/西瓜3.xlsx"
    data = pd.read_excel(filename, header=None)
    tdata = data.loc[:5]
    tdata = tdata.append(data.loc[8], ignore_index=True)
    tdata = np.array(tdata.T.values, dtype='int64')
    labels = ['Num','Color','Root','Sound','Strike','Navel','Touch','Density','Sugar','Good&Bad']
    mytree = createTree(tdata, labels)
    print mytree
    plt.createPlot(mytree)

