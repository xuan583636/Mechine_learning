# coding: utf-8
# 神经网络——误差逆传播算法（BP）的实现

import readxls
from numpy import *

def BP(x, y):
    x = matrix(x)
    y = matrix(y)
    x = x.T
    y = y.T
    y = y - 1

    (m, n) = x.shape                # 输入样本的参数
    q = n+1                         # 隐层数目
    t = 1                           # 输出神经元
    v = random.random((n, q))       # 输入层与隐层之间的权值
    w = random.random((q, t))       # 隐层与输出层之间的权值
    thy = random.random(q)          # 隐层的阈值
    thj = random.random(t)          # 输出层的阈值
    ty = zeros((m,t))               # 输出层输出
    b = zeros(q)                    # 隐层输出
    gj = zeros(t)                   # 累计误差对w,thy的求导参数
    eh = zeros(q)                   # 累计误差对v,thj的求导参数
    xk = 1                          # 学习率

    kn = 0                          # 迭代次数
    sn = 0                          # 同样的累计误差积累次数
    old_ey = 0                      # 前一次迭代的累计误差

    while (1):
        kn += 1
        ey = 0                      # 当前迭代误差
        for i in range(m):
            for j in range(q):      # 计算隐层输出
                ca = sum (multiply(v[:, j], x[i]))
                b[j] = 1 / (1 + exp(-ca + thy[j]))

            for j in range(t):      # 计算输出层输出
                cb = sum(multiply(w[:, j], b))
                ty[i][j] = 1 / (1 + exp(-cb + thj[j]))

            for j in range(t):      # 计算当前累计误差
                ey += ((y[i] - ty[i][j])**2)/2

            for j in range(t):      # 计算w,thj导数参数
                gj[j] = ty[i][j] * (1 - ty[i][j]) * (y[i] - ty[i][j])

            for j in range(q):      # 计算v,thy导数参数
                teh = sum(multiply(w[j,:], gj))
                eh[j] = b[j] * (1 - b[j]) * teh

            for j in range(q):      # 更新v, thy
                thy[j] -= xk * eh[j]
                v[:, j] += xk * eh[j] * x[i, :].A[0]

            for j in range(t):      # 更新w, thj
                thj[j] -= xk * gj[j]
                w[:, j] += xk * gj[j] * b

        if (abs(old_ey - ey) < 0.0001):
            sn += 1
            if sn == 100:
                break
        else:
            old_ey = ey
            sn = 0

        print "迭代第%d次" % kn

    print "一共迭代%d次" % kn
    print "累计误差为：", ey
    print ty




def ABP(x, y):
    x = matrix(x)
    y = matrix(y)
    x = x.T
    y = y.T
    y = y - 1

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    xlen = len(data) - 1
    x = data[0: xlen]
    y = data[xlen]
    BP(x, y)
    # ABP(x, y)