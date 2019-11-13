# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:13:23 2019
--------------------- 
作者：Janny张淼 
来源：CSDN 
原文：https://blog.csdn.net/zhangmiaogood1/article/details/77448129 
版权声明：本文为博主原创文章，转载请附上博文链接！
python简单实现天猫手机评论标签提取--自然语言处理

"""

#2.结巴中文分词提取关键词
import jieba
import jieba.analyse
#import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  #设置日志
#content = open('/mnt/share/jieba_test/荣耀8天猫评论.csv','rb').read()
content = open('data/verbatim.txt','r', encoding = 'utf-8').read()
tagsA = jieba.analyse.extract_tags(content, topK=20,allowPOS='a')    #allowPOS是选择提取的词性，a是形容词
tagsN = jieba.analyse.extract_tags(content, topK=20, allowPOS='n')   #allowPOS='n'，提取名词

#3.制作语料库
import pandas as pd
import numpy as np
#import logging
import codecs

words=jieba.lcut(content,cut_all=False)   #分词，精确模式

#去停用词，先自己网上找中文停用词，制作好“停用词表.txt”
stopwords = []  
#for word in open("/mnt/share/jieba_test/stopword.txt", "r"):  
#    stopwords.append(word.strip())  
stayed_line = ""  
for word in words:  
    if word not in stopwords:  
        stayed_line += word + " "   

#保存语料
file=open('data/corpus.txt','wb')
file.write(stayed_line.encode("utf-8"))
file.close()

#4.深度学习word2vec训练评论语料
from gensim.models import word2vec

sentences = word2vec.Text8Corpus('data/corpus.txt')  # 加载刚刚制作好的语料
model = word2vec.Word2Vec(sentences, size=200)  # 默认window=5

commit_index=pd.DataFrame(columns=['commit','similarity'],index=np.arange(100))  

index=0
for i in tagsN:
    for j in tagsA:
        commit_index.loc[index,:]=[i+j,model.similarity(i,j)]
        index+=1

comit_index_final=commit_index.sort(columns='similarity',ascending=False)
comit_index_final.index=commit_index.index