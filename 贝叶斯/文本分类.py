# -*- coding: utf-8 -*-
# @Time    : 2017/5/16 10:04
# @Author  : UNE
# @Project : Mechine_learning
# @File    : 文本分类.py
# @Software: PyCharm

# 《机器学习实战》第四章

import numpy as np

# 词表到向量的转换函数
def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataset):
    vocabSet = set([])      # 创建一个空集
    for document in dataset:
        vocabSet = vocabSet | set(document)     # 创建两个集合的并集
    return list(vocabSet)

# 词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec
# 词袋模型
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   # 增加对应词的值
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)     # 计算属于侮辱性文章的概率
    p0Num = np.ones(numWords)       # 避免在分类当中存在0，使得最后的结果为0
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)        # 取对数，求积变为求和
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2Classify * p1vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 垃圾邮件分类
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\w*', bigString) # 除去单词，数字外的任意字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullTest = []
    for i in range(1, 26):
        filepath = '/Users/JJjie/Desktop/Projects/dataset/email/spam/%d.txt' % i
        wordList = textParse(open(filepath).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)

        filepath = '/Users/JJjie/Desktop/Projects/dataset/email/ham/%d.txt' % i
        wordList = textParse(open(filepath).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randeIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randeIndex])
        del trainingSet[randeIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)

if __name__ == '__main__':
    listOPosts, listClasses = loadDataset()
    myVocabList = createVocabList(listOPosts)
    # 将一组单词序列变为数字序列
    print "单词序列变为数字序列"
    print myVocabList
    print setOfWords2Vec(myVocabList, listOPosts[0])
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNBO(trainMat, listClasses)

    print "恶意留言分类结果"
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, "classified as: ", classifyNB(thisDoc, p0v, p1v, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, "classified as: ", classifyNB(thisDoc, p0v, p1v, pAb)

    print "垃圾邮件分类"
    spamTest()

