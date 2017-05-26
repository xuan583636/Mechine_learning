# -*- coding: utf-8 -*-
# @Time    : 2017/5/26 10:30
# @Author  : UNE
# @Project : Mechine_learning
# @File    : 推荐引擎.py
# @Software: PyCharm

import numpy as np
import svdRec as svd

# 基于物品相似度的推荐系统
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas=svd.cosSim, estMethod=standEst):
    # 寻找未评级的物品
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimateScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimateScore))
    # 寻找前N个未评级物品
    return sorted(itemScores, key=lambda p:p[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I # 构建转换后的物品,只利用90%的奇异值
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j==item:
            continue
        # 寻找两个用户都评级的物品
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


if __name__ == '__main__':
    myMat = np.mat(svd.loadExData())
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print recommend(myMat, 2, simMeas=svd.ecludSim)
    myMat = np.mat(svd.loadExData2())
    U, Sigma, VT = np.linalg.svd(myMat)
    Sig2 = Sigma ** 2
    print sum(Sig2), sum(Sig2) * 0.9
    print sum(Sig2[:3]), "3维，大于90%，可以将11维变为3维"
    print recommend(myMat, 1, estMethod=svdEst)