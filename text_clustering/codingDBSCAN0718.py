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
from snownlp import SnowNLP
import re
import jieba
import numpy as np
#from gensim import corpora, models, similarities

# seperate sentence
def getPhrases(sentence):
    if not pd.isnull(sentence): #is_chinese(sentence[0]): #sentence != "" and
        s = SnowNLP(sentence)
        outstr = ''
        for x in s.sentences:
            outstr+= x + ';'
#        s = sentence.split('。')
#        outstr = ''
#        for x in s:
#            outstr+= x + ';'
    else:
        outstr = "无"
    return outstr

def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
        else:
                return False
#原文：https://blog.csdn.net/sinat_20174131/article/details/80403049 


def delNA(sVerbatim):
    sTmp = sVerbatim
    result = "NA"
    if not pd.isnull(sTmp):
        # delete end of verbatim 删除结尾
        regex = '(无\/已?\S+)'
        p = re.compile(regex)
        m = p.search(sTmp)
        if m:
            result =  sTmp.replace(m.group(0),'')
        else:
            result =  sTmp #.replace('\n','')
        #替换括号
        result = result.replace('（','，')
        result = result.replace('）','。')
        result = result.replace(' ','，')
    return result


stopwords = []
for line in open('stopwords.txt', encoding = 'utf-8'):
    stopwords.append(line.strip())
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

def tokenization(sentence):
    result = []
    words = [word for word in jieba.cut(sentence, cut_all=True)]
    for word in words:
        if word not in stopwords and word not in result:
            result.append(word)
    return result

def seg2(sentence):
    outstr = []
    words = [word for word in jieba.cut(sentence, cut_all=True)]
    for x in words:
        if x not in outstr:
            outstr.append(x)
    return outstr

t0 = time()
# read and clean raw data
#rawdf = pd.read_excel('data/rawdata.xlsx','a')
#rawdf.columns = ['Verbatim','Code','Mapping']
rawdf = pd.read_excel('input.xlsx')
rawdf.columns = ['id','Verbatim']
df = rawdf[['id','Verbatim']]#, 'Mapping', 'Code']].copy() ##without copy df is only a view of rawdf!!!

print('clean verbatim')
df['cl01'] = df['Verbatim'].apply(lambda i: delNA(i))
print('seperate sentences')
df['frase'] = df['cl01'].apply(lambda i: getPhrases(i))

###########################################
# 展开句子，准备聚类
#df['id'] = np.arange(1,len(df)+1,1)
rawFrases = []
orgid = []
#mapping = []
#codelist = []
for idx, dfrow in df.iterrows():
    ls = dfrow['frase'].split(';')
    for s in ls:
        if s != '':
            rawFrases.append(s)
            orgid.append(dfrow['id'])
#            mapping.append(dfrow['Mapping'])
#    ls = dfrow['Mapping'].split(';')
#    for s in ls:
#        if s != '' and s not in codelist:
#            codelist.append(s)

newdf = pd.DataFrame({'frase': rawFrases, 'respid':orgid})#, 'mapping': mapping})
newdf = newdf[~newdf['frase'].isnull()]
#newdf = newdf.drop_duplicates('frase')

#from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline

newdf['tscut'] = newdf['frase'].apply(lambda i:jieba.lcut(i) )
newdf['tscut'] = [' '.join(i) for i in newdf['tscut']]
print("%d verbatim" % len(newdf['tscut']))
print("使用稀疏向量（Sparse Vectorizer）从训练集中抽取特征")
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=40000,
                                 min_df=5, stop_words=stopwords,ngram_range=(1, 2),
                                 use_idf=True)

X = vectorizer.fit_transform(newdf['tscut'])

print("完成所耗费时间： %fs" % (time() - t0))
print("样本数量: %d, 特征数量: %d" % X.shape)
print('特征抽取完成！')

###############################################333
print("用LSA进行维度规约（降维）...")
t0 = time()
    
#Vectorizer的结果被归一化，这使得KMeans表现为球形k均值（Spherical K-means）以获得更好的结果。 
#由于LSA / SVD结果并未标准化，我们必须重做标准化。   
svd = TruncatedSVD(15)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
print("完成所耗费时间： %fs" % (time() - t0))
explained_variance = svd.explained_variance_ratio_.sum()
print("SVD解释方差的step: {}%".format(int(explained_variance * 100)))
print('PCA文本特征抽取完成！')

#进行实质性的DBScan聚类
#db = DBSCAN(eps=0.2, min_samples=4).fit(X)
db = DBSCAN(eps=0.2, min_samples=10).fit(X) ##调整参数 min_samples 控制每个聚类最少有多少条
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

db.core_sample_indices_

labels = db.labels_
labels

clusterTitles = db.labels_
dbscandf = newdf
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

###############################3
## 整理数据
from jieba import analyse
# 引入TF-IDF关键词抽取接口
tfidf = analyse.extract_tags
gencode = []
print("tfidf")
for i in range(n_clusters_):
    text = '，'.join(dbscandf[dbscandf['cluster'] == i]['frase'])
    #keywordsN = tfidf(text,topK=2, allowPOS='n')
    #keywordsA = tfidf(text,topK=2, allowPOS='a')
    keywordsN = tfidf(text,topK=3)
    print(keywordsN)

print("text trank")
ttrank = analyse.textrank
for i in range(n_clusters_):
    text = '，'.join(dbscandf[dbscandf['cluster'] == i]['frase'])
    #keywordsN = tfidf(text,topK=2, allowPOS='n')
    #keywordsA = tfidf(text,topK=2, allowPOS='a')
    keywordsN = ttrank(text,topK=3)
    print(keywordsN)

#dbscandf = dbscandf[dbscandf['cluster']!= -1]
#dbscandf[dbscandf['cluster'] == 20]['mapping']  #簇群tag为20的title名称

## 合并结果


##################################3
#去重 min_samples = 1; 286个聚类 / 1513; eps=0.2


## 参数调整 https://blog.csdn.net/u013206066/article/details/70985282

xlsx = pd.ExcelWriter('cluster_result.xlsx') #save intermediate results
newdf.to_excel(xlsx, index = False)
xlsx.save()

















