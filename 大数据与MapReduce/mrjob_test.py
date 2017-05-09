# -*- coding: utf-8 -*-
# @Time    : 2017/5/8 20:56
# @Author  : UNE
# @Project : Mechine_learning
# @File    : mrjob_test.py
# @Software: PyCharm

# 《机器学习实战》第十五章
"""
python 使用 mrjob 自动化 MapReduce
"""

from mrjob.job import MRJob

class MRmean(MRJob):
    def __init__(self, *args, **kwargs):        # *args, 把参数收集到一个元组中, **args, 把参数收集到一个字典中
        super(MRmean, self).__init__(*args, **kwargs)   # 继承
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):
        if False: yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal ** 2

    def map_final(self):
        mn = self.inSum / self.inCount
        mnsq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnsq])

    def reduce(self, key, packedValues):
        cumVal = 0.0
        cumSumSq = 0.0
        cumN = 0.0
        for valArr in packedValues:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)

    def steps(self):
        return ([self.mr(mapper=self.map, reducer=self.reduce, mapper_final=self.map_final)])

if __name__ == '__main__':
    MRmean.run()