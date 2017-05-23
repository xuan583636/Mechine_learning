# -*- coding: utf-8 -*-
# @Time    : 2017/5/23 10:22
# @Author  : UNE
# @Project : Mechine_learning
# @File    : apriori.py
# @Software: PyCharm

# 《机器学习实战》第十一章
"""
Apriori算法用于发现频繁集
"""

def loadDataSet():
	return [[1, 3, 4],
	        [2, 3, 5],
	        [1, 2, 3, 5],
	        [2, 5]]

def creatC1(dataset):
	C1 = []
	for transaction in dataset:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1)   # 对C1中每个项构建一个不变集合

# 去掉了支持度不高的点，减少时间复杂度
def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems     # 计算所有的支持度
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

def aprioriGen(Lk, k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):     # 前k-2个项相同时，将两个集合合并
			"""
			利用{0，1}{0，2}{1，2}构建，会得到三个{0，1，2}
			若只看第一个是否相同，合并，只会得到一个{0，1，2}，这样去掉了重复
			"""
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])       # 并操作符
	return retList

def apriori(dataset, minSupport=0.5):
	C1 = creatC1(dataset)
	D = map(set, dataset)
	L1, suppdata = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while (len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		suppdata.update(supK)
		L.append(Lk)
		k += 1
	return L, suppdata

#评估规则
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet-conseq]
		if conf >= minConf:
			print freqSet-conseq, "-->", conseq, "conf:", conf
			br1.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH
# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
	m = len(H)
	if len(freqSet) > (m+1):
		Hmp1 = aprioriGen(H, m+1)       # 尝试进一步合并
		Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
		if len(Hmp1) > 1:
			rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)
# 关联规则的生成函数
def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []
	for i in range(1, len(L)):      # 只获取有两个或者更多元素的集合
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if i > 1:
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

if __name__ == '__main__':
	dataset = loadDataSet()
	L, suppdata = apriori(dataset,0.5)
	print L
	rules = generateRules(L, suppdata, 0.7)
	print rules,"\n"

	# 毒蘑菇的相似特征
	filename = "/Users/JJjie/Desktop/Projects/dataset/MLiA/mushroom_9.dat"
	fr = open(filename)
	data = []
	for item in fr.readlines():
		items = item.strip().split()
		data.append(items)
	fr.close()
	L, suppdata = apriori(data, 0.3)
	print "输出所有包涵有毒特征值2的频繁特征集"
	for item in L[3]:
		if item.intersection('2'):
			print item
	print
	# rules = generateRules(L, suppdata,0.99)
	# print rules
