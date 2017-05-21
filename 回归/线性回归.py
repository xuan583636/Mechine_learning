# -*- coding: utf-8 -*-
# @Time    : 2017/5/21 10:31
# @Author  : UNE
# @Project : Mechine_learning
# @File    : 线性回归.py
# @Software: PyCharm

# 《机器学习实战》第八章

import numpy as np

def loadDataset(filename):
    fr = open(filename)
    data = fr.readlines()
    fr.close()
    numFeat = len(data[0].split("\t")) - 1
    dataMat = []
    labelMat = []
    for line in data:
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print "矩阵是不可逆的"
        return
    ws = xTx.I * (xMat.T * yMat)    # I逆
    return ws

# 直接线性回归会有欠拟合的情况
# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))     # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j,j] = np.exp(diffMat * diffMat.T / (-2.0 * k **2))     # 核函数
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print "矩阵是不可逆的"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# 处理特征比样本多的情况
# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print "矩阵是不可逆的"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    # 数据标准化
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

# 前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = np.regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = np.rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def drawpic(xMat, yMat, yHat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    strInd = xMat[:, 1].argsort(0)
    xSort = xMat[strInd][:, 0, :]
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    ax.plot(xSort[:, 1], yHat[strInd][:,0,:])
    plt.show()


if __name__ == '__main__':
    xArr, yArr = loadDataset("/Users/JJjie/Desktop/Projects/dataset/MLiA/ex0.txt")
    ws = standRegres(xArr, yArr)
    print ws
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    print np.corrcoef(yHat.T, yMat)       # 计算相关系数
    yHat = np.dot(xMat, ws)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat = np.mat(yHat).T
    # drawpic(xMat, yMat, yHat)

    abx, aby = loadDataset("/Users/JJjie/Desktop/Projects/dataset/MLiA/abalone.txt")
    ridgeWeights = ridgeTest(abx, aby)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print np.shape(ridgeWeights)
    ax.plot(ridgeWeights)
    plt.show()


