# coding: utf-8
# 神经网络——误差逆传播算法（BP）的实现
# 《机器学习》（周志华）第五章习题5.5
"""
编程实现标准BP算法和累积BP算法，在西瓜数据集上分别用这两个算法训练一个单隐层网络，并进行比较

BP算法每次迭代依次计算每一个样本，最小化该样本输出值与真实值的差距，然后将修改过参数传给下一个样本，直到达到收敛条件。
这样做参数更新频繁，也可能出现参数更改相互抵销的情况，于是便有了ABP。
ABP算法每次迭代会先算出所有样本的输出，然后最小化整个样本输出与真实值的最小平方和，修改参数后进行下一次迭代。
ABP参数更新次数比BP算法少的多，但是当累计误差降到一定程度时，进一步下降会非常缓慢。
"""

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
            if sn == 10:
                break
        else:
            old_ey = ey
            sn = 0

        print "#",

    print "\n一共迭代%d次" % kn
    print "累计误差为：", ey
    print ty

def ABP(x, y):
    x = matrix(x)
    y = matrix(y)
    x = x.T
    y = y.T
    y = y - 1

    (m, n) = x.shape                # 输入样本的参数
    q = n + 1                       # 隐层数目
    t = 1                           # 输出神经元
    v = random.random((n, q))       # 输入层与隐层之间的权值
    w = random.random((q, t))       # 隐层与输出层之间的权值
    thy = random.random(q)          # 隐层的阈值
    thj = random.random(t)          # 输出层的阈值
    ty = zeros((m, t))              # 输出层输出
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
                ca = sum(multiply(v[:, j], x[i]))
                b[j] = 1 / (1 + exp(-ca + thy[j]))

            for j in range(t):      # 计算输出层输出
                cb = sum(multiply(w[:, j], b))
                ty[i][j] = 1 / (1 + exp(-cb + thj[j]))

        # 用来存储累计误差四个变量的下降方向
        tv = zeros((n, q))
        tw = zeros((q, t))
        tthy = zeros(q)
        tthj = zeros(t)

        for i in range(m):
            for j in range(t):      # 计算当前累计误差
                ey += ((y[i] - ty[i][j]) ** 2) / 2

            for j in range(t):      # 计算w,thj导数参数
                gj[j] = ty[i][j] * (1 - ty[i][j]) * (y[i] - ty[i][j])

            for j in range(q):      # 计算v,thy导数参数
                teh = sum(multiply(w[j, :], gj))
                eh[j] = b[j] * (1 - b[j]) * teh

            for j in range(q):      # 计算v,thy
                tthy[j] -= xk * eh[j]
                tv[:, j] += xk * eh[j] * x[i, :].A[0]

            for j in range(t):      # 更新w, thj
                tthj[j] -= xk * gj[j]
                tw[:, j] += xk * gj[j] * b
        # 更新参数
        v += xk * tv
        w += xk * tw
        thy += xk * tthy
        thj += xk * tthj

        if (abs(old_ey - ey) < 0.0001):
            sn += 1
            if sn == 10:
                break
        else:
            old_ey = ey
            sn = 0

        print "#",

    print "\n一共迭代%d次" % kn
    print "累计误差为：", ey
    print ty

if __name__ == '__main__':
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.xlsx", 0, "Sheet1")
    xlen = len(data) - 1
    x = data[0: xlen]
    y = data[xlen]
    BP(x, y)
    ABP(x, y)