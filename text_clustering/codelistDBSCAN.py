# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:45:56 2019

@author: Wei
思路：
1.verbatim清理：去停词；分裂句子成短句
2.用DBSCAN算法聚类短句；
3.对cluster提取关键词
4.将关键词生成短语 ？？？ 
5.将生成的短语贴回原始数据

问题：
- 清理过程：停词表整理（去除品牌？）
- 怎样获取关键词词性，然后组合？ （用RNN学习生成描述？）-> 生成码表
- 怎样将聚类和原始列表编码做比较？
- 剩下没有聚类的数据跟生成的码表做相似度分析
"""

from time import time
import pandas as pd
#from snownlp import SnowNLP
#import re
import jieba
import numpy as np
#from gensim import corpora, models, similarities

def cleaning(sentence):
    outstr = sentence
    ls = ['“','”','/','UNK']
    for s in ls:
        outstr = outstr.replace(s,'')    
    return outstr

notlist = ['不','没','无','非','莫','弗','勿','毋','未','否','别','無']
#def tokenization(sentence):
#    result = []
#    words = [word for word in jieba.cut(sentence, cut_all=True)]
#    for word in words:
#        if word not in stopwords and word not in result:
#            result.append(word)
#    return result
#
#def seg2(sentence):
#    outstr = []
#    words = [word for word in jieba.cut(sentence, cut_all=True)]
#    for x in words:
#        if x not in outstr:
#            outstr.append(x)
#    return outstr

t0 = time()
# read and clean raw data
rawdf = pd.read_excel('data/codelist_unique.xlsx','a')
#rawdf.columns = ['ID','Code','Mapping']
#df = rawdf[['Verbatim', 'Mapping', 'Code']].copy() ##without copy df is only a view of rawdf!!!

#print('clean verbatim')
#df['cl01'] = df['Verbatim'].apply(lambda i: delNA(i))
#print('seperate sentences')
#df['frase'] = df['cl01'].apply(lambda i: getPhrases(i))

###########################################
# 展开句子，准备聚类
#df['id'] = np.arange(1,len(df)+1,1)
#rawFrases = []
#orgid = []
#mapping = []
#codelist = []
#for idx, dfrow in df.iterrows():
#    ls = dfrow['frase'].split(';')
#    for s in ls:
#        if s != '':
#            rawFrases.append(s)
#            orgid.append(dfrow['id'])
#            mapping.append(dfrow['Mapping'])
#    ls = dfrow['Mapping'].split(';')
#    for s in ls:
#        if s != '' and s not in codelist:
#            codelist.append(s)
#
#newdf = pd.DataFrame({'frase': rawFrases, 'orgid':orgid, 'mapping': mapping})
#newdf = newdf[~newdf['frase'].isnull()]
#newdf = newdf.drop_duplicates('frase')
rawdf['clean'] = rawdf['dscrpt'].apply(lambda i: cleaning(i))

#from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline

rawdf['tscut'] = rawdf['clean'].apply(lambda i:jieba.lcut(i) )
rawdf['tscut'] = [' '.join(i) for i in rawdf['tscut']]
#print("%d verbatim" % len(newdf['tscut']))
print("使用稀疏向量（Sparse Vectorizer）从训练集中抽取特征")
t0 = time()
#vectorizer = TfidfVectorizer(max_df=0.5, max_features=40000,
#                                 min_df=5, stop_words=stopwords,ngram_range=(1, 2),
#                                 use_idf=True)
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(rawdf['tscut'])

print("完成所耗费时间： %fs" % (time() - t0))
print("样本数量: %d, 特征数量: %d" % X.shape)
print('特征抽取完成！')

###############################################333
print("用LSA进行维度规约（降维）...")
t0 = time()
    
#Vectorizer的结果被归一化，这使得KMeans表现为球形k均值（Spherical K-means）以获得更好的结果。 
#由于LSA / SVD结果并未标准化，我们必须重做标准化。   
svd = TruncatedSVD(100) #100 is recommended
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
print("完成所耗费时间： %fs" % (time() - t0))
explained_variance = svd.explained_variance_ratio_.sum()
print("SVD解释方差的step: {}%".format(int(explained_variance * 100)))
print('PCA文本特征抽取完成！')

#进行实质性的DBScan聚类
#db = DBSCAN(eps=0.2, min_samples=4).fit(X)
db = DBSCAN(eps=0.2, 
            min_samples=2).fit(X) ##调整参数 min_samples 控制每个聚类最少有多少条
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

db.core_sample_indices_

labels = db.labels_
labels

clusterTitles = db.labels_
dbscandf = rawdf
dbscandf['cluster'] = clusterTitles

##看看簇群序号为0的文章的标题有哪些，通过这个能看出聚类的实际效果如何
#dbscandf[dbscandf['cluster'] == 0]['tags'].head(20)  #簇群tag为0的title名称
#
#
##看看簇群序号为20的文章的标题有哪些，通过这个能看出聚类的实际效果如何
#dbscandf[dbscandf['cluster'] == 20]['tags'].head(20)  #簇群tag为20的title名称

# 聚类数及噪点计算
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('聚类数：',n_clusters_)
print('噪点数：',n_noise_)

grouplist = list(dbscandf[dbscandf['cluster'] == -1]['dscrpt'])
for i in range(n_clusters_):
    text = ';'.join(dbscandf[dbscandf['cluster'] == i]['dscrpt'])
    grouplist.append(text)
#    if text.find(';') > 1:
#        print(text)

df1 = pd.DataFrame({'description':grouplist})
df1.to_excel('data\codelist.xlsx',sheet_name = 'a', index = False)

###############################3
## 整理数据
#from jieba import analyse
# 引入TF-IDF关键词抽取接口
#tfidf = analyse.extract_tags
#gencode = []
#for i in range(n_clusters_):
#    text = '，'.join(dbscandf[dbscandf['cluster'] == i]['dscrpt'])
#    #keywordsN = tfidf(text,topK=2, allowPOS='n')
#    #keywordsA = tfidf(text,topK=2, allowPOS='a')
#    keywordsN = tfidf(text,topK=3)
#    print(keywordsN)
#
#ttrank = analyse.textrank
#for i in range(n_clusters_):
#    text = '，'.join(dbscandf[dbscandf['cluster'] == i]['dscrpt'])
#    #keywordsN = tfidf(text,topK=2, allowPOS='n')
#    #keywordsA = tfidf(text,topK=2, allowPOS='a')
#    keywordsN = ttrank(text,topK=3)
#    print(keywordsN)

#dbscandf = dbscandf[dbscandf['cluster']!= -1]
#dbscandf[dbscandf['cluster'] == 20]['mapping']  #簇群tag为20的title名称

## 合并结果


##################################3
#去重 min_samples = 1; 286个聚类 / 1513; eps=0.2


## 参数调整 https://blog.csdn.net/u013206066/article/details/70985282




















