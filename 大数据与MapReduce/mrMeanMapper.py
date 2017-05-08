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
        yield line.rstrip()         #取消尾部的字节

input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = np.mat(input)
sqInput = np.power(input, 2)    # 阶乘

print "%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput))
print >> sys.stderr, "report: still alive"
