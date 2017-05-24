# -*- coding: utf-8 -*-
# @Time    : 2017/5/24 08:48
# @Author  : UNE
# @Project : Mechine_learning
# @File    : fpGrowth.py
# @Software: PyCharm

# 《机器学习实战》第十二章

# FP树的类定义
class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.nodeLink = None
		self.parent = parentNode
		self.children = {}

	def inc(self, numOccur):
		self.count += numOccur

	def disp(self, ind=1):  # 将树用文本形式显示
		print "\t"*ind, self.name, " ", self.count
		for child in self.children.values():
			child.disp(ind+1)

def updateTree(items, inTree, headerTable, count):
	if items[0] in inTree.children:
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]] = treeNode(items[0], count, inTree)
		if headerTable[items[0]][1] == None:
			headerTable[items[0]][1] = inTree.children[items[0]]
		else:
			updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
	if len(items) > 1:      # 对剩下的元素迭代
		updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def createTree(dataset, minSup=1):
	headerTable = {}
	for trans in dataset:
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataset[trans]
	for k in headerTable.keys():
		if headerTable[k] < minSup:
			del headerTable[k]      # 移除不满足最小支持度的元素项
	freqItemSet = set(headerTable.keys())
	if len(freqItemSet) == 0:
		return None, None           # 如果没有元素满足，则退出
	for k in headerTable:
		headerTable[k] = [headerTable[k], None]
	retTree = treeNode('Null Set', 1, None)
	for transet, count in dataset.items():
		localD = {}
		for item in transet:        # 根据全局频率对每个事物中的元素排序
			if item in freqItemSet:
				localD[item] = headerTable[item][0]
			if len(localD) > 0:
				orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
				updateTree(orderedItems, retTree, headerTable, count)   # 使用排序后的频率集对树进行填充
	return retTree, headerTable

def updateHeader(nodeToTest, targetNode):
	while nodeToTest.nodeLink != None:
		nodeToTest = nodeToTest.nodeLink
	nodeToTest.nodeLink = targetNode

def loadSimpDat():
	simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
	return simpDat

def createInitSet(dataSet):
	retDict = {}
	for trans in dataSet:
		retDict[frozenset(trans)] = 1
	return retDict

# 发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode, prefixPath):   # 迭代上溯整棵树
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
	condPats = {}
	while treeNode != None:
		prefixPath = []
		ascendTree(treeNode, prefixPath)
		if len(prefixPath) > 1:
			condPats[frozenset(prefixPath[1:])] = treeNode.count
		treeNode = treeNode.nodeLink
	return condPats

# 递归查找频繁集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None: #3. mine cond. FP-tree
			print 'conditional tree for: ',newFreqSet
			myCondTree.disp(1)
			mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()

    simDat = loadSimpDat()
    initset = createInitSet(simDat)
    print initset
    myFPtree, myHeaderTab = createTree(initset, 3)
    myFPtree.disp()

    print findPrefixPath('r', myHeaderTab['r'][1])

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print freqItems