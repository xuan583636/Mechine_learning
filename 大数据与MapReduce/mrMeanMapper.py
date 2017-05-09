# -*- coding: utf-8 -*-
# @Time    : 2017/5/8 19:00
# @Author  : UNE
# @Project : Mechine_learning
# @File    : mrMeanMapper.py
# @Software: PyCharm

# 《机器学习实战》第十五章
"""
python yield

"""

import sys
import numpy as np

# data = np.array(np.arange(1, 101), dtype="int64")
# print data.mean()
# exit(0)

def read_input(file):
    for line in file:
        # http://www.cnblogs.com/tqsummer/archive/2010/12/27/1917927.html
        """
        包含yield语句的函数会被特地编译成生成器。
        当函数被调用时，他们返回一个生成器对象，这个对象支持迭代器接口。
        函数也许会有个return语句，但它的作用是用来yield产生值的。
        不像一般的函数会生成值后退出，生成器函数在生成值后会自动挂起并暂停他们的执行和状态，
        他的本地变量将保存状态信息，这些信息在函数恢复时将再度有效
        """
        yield line.rstrip()         #取消尾部的字节


input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = np.mat(input)
sqInput = np.power(input, 2)    # 阶乘

print "%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput))
print >> sys.stderr, "report: still alive"
# >> 用报错模块显示（标红）
