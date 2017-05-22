# -*- coding: utf-8 -*-
# @Time    : 2017/5/22 08:49
# @Author  : UNE
# @Project : Mechine_learning
# @File    : CART.py
# @Software: PyCharm

# 《机器学习实战》第九章

import numpy as np

def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curlLine = line.strip().split('\t')
		fltLine = map(float, curlLine)      # 每行映射为浮点数
		dataMat.append(fltLine)
	fr.close()
	return dataMat

# 二分划分
def binSplitDataset(dataset, feature, value):
	# nonzero 给出符合的条件的index
	mat0 = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
	mat1 = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
	return mat0, mat1

def regLeaf(dataset):
	return np.mean(dataset[:, -1])
def regErr(dataset):
	return np.var(dataset[:, -1] * np.shape(dataset)[0])

# 回归树核心
def chooseBestSplit(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):
	tolS = ops[0]       # 容许的误差下降值
	tolN = ops[1]       # 最少样本数目
	if len(set(dataset[:, -1].T.tolist()[0])) == 1:
		# 没有不同的instance，所有值相等
		return None, leafType(dataset)
	m, n = np.shape(dataset)
	S = errType(dataset)
	bestS = np.inf
	bestIndex = 0
	bestValue = 0
	for featIndex in range(n-1):
		for splitVal in set(dataset[:, featIndex].T.tolist()[0]):
			mat0, mat1 = binSplitDataset(dataset, featIndex, splitVal)
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
		if S - bestS < tolS:        # 误差减少不大就退出
			return None, leafType(dataset)
		mat0, mat1 = binSplitDataset(dataset, bestIndex, bestValue)
		if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):    # 切分出的特征很小则退出
			return None, leafType(dataset)
		return bestIndex, bestValue

# leafType: 创建叶节点函数的引用，errType总方差函数的引用，ops用户定义的参数构成元组
def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataset, leafType, errType, ops)
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataset(dataset, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def isTree(obj):
	return (type(obj).__name__ == 'dict')
def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
	if np.shape(testData)[0] == 0:  # 没有测试数据时对树进行塌陷处理(返回树平均值)
		return getMean(tree)
	if isTree(tree['left']) or isTree(tree['right']):
		lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right']. rSet)
	if not isTree(tree['left'] and not isTree(tree['right'])):
		lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])
		errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
		               sum(np.power(rSet[:, -1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
		if errorMerge < errorNoMerge:
			print "merging"
			return treeMean
		else:
			return tree
	else:
		return tree

# 模型树 叶节点生成函数
def linearSlove(dataset):
	m, n = np.shape(dataset)
	X = np.mat(np.ones((m, n)))
	Y = np.mat(np.ones((m, 1)))
	X[:, 1:n] = dataset[:, 0:n-1]
	Y = dataset[:, -1]
	xTx = X.T * X
	if np.linalg.det(xTx) == 0.0:
		raise NameError('矩阵不可逆')
	ws = xTx.I * (X.T * Y)
	return ws, X, Y

# 类似于regLeaf
def modelLeaf(dataset):
	ws, X, Y = linearSlove(dataset)
	return ws
def modelErr(dataset):
	ws, X, Y = linearSlove(dataset)
	yHat = X * ws
	return sum(np.power((Y - yHat), 2))


if __name__ == '__main__':
	# testMat = np.eye(4)
	# testMat = np.mat(testMat)
	# mat0, mat1 = binSplitDataset(testMat, 1, 0.5)
	# print mat0
	# print mat1

	filename = "/Users/JJjie/Desktop/Projects/dataset/MLiA/exp9_2.txt"
	data = loadDataSet(filename)
	dataMat = np.mat(data)
	print createTree(dataMat, modelLeaf, modelErr, ops=(1,10))