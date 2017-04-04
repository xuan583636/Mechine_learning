# coding: UTF-8
# 径向基函数（RBF）网络
# 《机器学习》（周志华）第五章5.7
"""
根据公式（5.18和5.19），构造一个能够解决异或问题的单层RBF神经网络
"""

from numpy import *

def RBF(x, y):
    t = 10                          # 隐层神经元的数目，大于输入层
    p = random.random((4, t))       # 径向基函数的值
    ty = random.random((4, 1))      # 输出值
    w = random.random((1, t))       # 隐层第i个神经元与输出神经元的权值
    b = random.random((1, t))       # 样本与第i个神经元的中心的距离的缩放系数
    tk = 0.5

    c = random.random((t, 2))       # 隐层第i个神经元的中心

    kn = 0                          # 迭代次数
    sn = 0                          # 同样累计误差重复的次数
    old_ey = 0                      # 前一次迭代的累计误差

    while(1):
        kn += 1
        for i in range(4):
            for j in range(t):
                p[i, j] = exp(-b[:,j] * dot((x[i]-c[j]), (x[i]-c[j]).T))
                ty[i] = dot(w, p[i,:].T)

        ey = (ty - y).T * (ty - y)  # 计算累计误差

        # 更新w,b
        dw = zeros((1, t))
        db = zeros((1, t))
        for i in range(4):
            dw += (ty[i] - y[i]) * p[i, :]
            for j in range(t):
                db[0][j] -= (ty[i] - y[i]) * w[0][j] * dot((x[i] - c[j]), (x[i] - c[j]).T) * p[i, j]

        w -= tk * dw / 4
        b -= tk * db / 4

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
    x = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 1, 1, 0]
    x = matrix(x)
    y = matrix(y).T
    RBF(x, y)