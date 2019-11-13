# _*_ coding: utf-8 _*_
#使用马尔可夫模型自动生成文章 
#https://zhuanlan.zhihu.com/p/25172031
## 可以尝试用于码表生成；但要解决太过随机的问题

import nltk
import random
import jieba

file = open('test/doc.txt', 'r')
#walden = file.read()
#walden = walden.split()
codelist = file.read()
codelist = codelist.replace('\n','')
codecut = [word for word in jieba.cut(codelist, cut_all=False)]



def makePairs(arr):
    pairs = []
    for i in range(len(arr)):
        if i < len(arr) - 1:
            temp = (arr[i], arr[i + 1])
            pairs.append(temp)
    return pairs


def generate(cfd, word='the', num=500):
    for i in range(num):
        # make an array with the words shown by proper count
        arr = []
        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)
        print(word, end=' ')

        # choose the word randomly from the conditional distribution
        word = arr[int((len(arr)) * random.random())]

def generateCN(cfd, word='包装', num=500):
    for i in range(num):
        # make an array with the words shown by proper count
        arr = []
        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)
        print(word, end ='')

        # choose the word randomly from the conditional distribution
        word = arr[int((len(arr)) * random.random())]


#pairs = makePairs(walden)
pairs = makePairs(codecut)
cfd = nltk.ConditionalFreqDist(pairs)
generateCN(cfd,'颜色',2)
