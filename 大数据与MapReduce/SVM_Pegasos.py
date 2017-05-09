# -*- coding: utf-8 -*-
# @Time    : 2017/5/9 12:31
# @Author  : UNE
# @Project : Mechine_learning
# @File    : SVM_Pegasos.py
# @Software: PyCharm

# 《机器学习实战》第十五章
"""
python 使用 mrjob 实现 Pegasos 算法
"""

from mrjob.job import MRJob
import pickle
import numpy as np

class MRsvm(MRJob):
    DEFAULT_INPUT_PROTOCAL = "json_value"

    def __init__(self, *args, **kwargs):
        super(MRsvm, self).__init__(*args, **kwargs)
        """
        python的pickle模块实现了基本的数据序列和反序列化。
        通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；
        通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象
        """
        self.data = pickle.load(open('<path to your Ch15 code directory>\svmDat27'))
        self.w = 0
        self.eta = 0.69
        self.dataList = []
        self.k = self.options.batchsize
        self.numMappers = 1
        self.t = 1

    def configure_options(self):
        super(MRsvm, self).configure_options()
        self.add_passthrough_option(
            '--iterations', dest='iterations', default=2, type='int',
            help='k: number of data points in a batch'
        )

    def map(self, mapperID, inVals):
        if False: yield
        if inVals[0] == 'w':
            self.w = inVals[1]
        elif inVals[0] == 'x':
            self.dataList.append(inVals[1])
        elif inVals[0] == 't':
            self.t = inVals[1]

    def map_fin(self):
        labels = self.data[:, -1]
        X = self.data[:, 0:-1]
        if self.w == 0:
            self.w = 0.001 * np.shape(X)[1]
        for index in self.dataList:
            p = np.dot(self.w, X[index, :].T)
            if np.dot(labels[index], p) < 1.0:
                yield (1, ['u', index])
        yield (1, ['w', self.w])
        yield (1, ['t', self.t])

    def reduce(self, _, packedVals):
        for valArr in packedVals:
            if valArr[0] == 'u':
                self.dataList.append(valArr[1])
            elif valArr[0] == 'w':
                self.w = valArr[1]
            elif valArr[0] == 't':
                self.t = valArr[1]
        labels = self.data[:, -1]
        X = self.data[:, 0:-1]
        wMat = np.array(self.w)
        wDelta = np.zeros(len(self.w))
        for index in self.dataList:
            wDelta += float(np.dot(labels[index], X[index, :]))
        eta = 1.0 / (2.0 * self.t)
        wMat = (1.0 - 1.0 / self.t) * wMat + (eta / self.k) * wDelta
        for mapperNum in range(1, self.numMappers+1):
            yield (mapperNum, ["w", wMat.tolist()[0]])
            if self.t < self.options.iterations:
                yield (mapperNum, ['t', self.t+1])
                for j in range(self.k / self.numMappers):
                    yield (mapperNum, ['x', np.random.randint(np.shape(self.data)[0])])

    def steps(self):
        return ([self.mr(mapper=self.map, mapper_final=self.map_fin,
                         reducer=self.reduce)] * self.options.iterations)

if __name__ == '__main__':
    MRsvm.run()