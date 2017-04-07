# coding: utf-8
# 支持向量机
# 《机器学习》（周志华）第六章习题6.2
"""
使用LIBSVM，在西瓜数据集3.0å上分别用线性核与高斯核训练一个svm，并比较其支持向量的差异
"""

import sys
# 加入import系统路径
sys.path.append("./libsvm/python")
import svmutil as svm
import readxls

if __name__ == '__main__':
    # 导入数据
    data = readxls.excel_table_byname("/Users/JJjie/Desktop/www/Mechine_Learning/dataset/西瓜3.0.xlsx", 0, "Sheet1")
    x = []
    y = data[3]
    for i in range(len(y)):
        a = [data[0][i], data[1][i]]
        x.append(a)

    prob = svm.svm_problem(y, x, isKernel=True)
    # 参数详情参见：https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    # 线性核
    param = svm.svm_parameter('-t 0 -c 4')
    m = svm.svm_train(prob, param)
    # 高斯核
    param = svm.svm_parameter('-t 2 -c 4')
    m = svm.svm_train(prob, param)

