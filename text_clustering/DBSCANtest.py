# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:32:58 2019
Ref: https://www.kesci.com/home/project/5c19f99de17d84002c658466
"""

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from pprint import pprint
import logging
from time import time
import numpy as np
import os
from sklearn.cluster import DBSCAN
import sklearn.pipeline

#载入虎嗅网的文本数据
import pandas as pd
import jieba
data = pd.read_excel('data/rawdata.xlsx','all')

import re
def delNA(sVerbatim):
    sTmp = sVerbatim
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
    return result

data['clean'] = data['rawdata'].apply(lambda i:delNA(i))
null = data['clean'].isnull()
data = data[~null]

'''
subset : column label or sequence of labels, optional 
用来指定特定的列，默认所有列
keep : {‘first’, ‘last’, False}, default ‘first’ 
删除重复项并保留第一次出现的项
inplace : boolean, default False 
是直接在原来数据上修改还是保留一个副本
'''
#去掉正文重复的行
data = data.drop_duplicates('clean')

##使用停用词表过滤无意义的词汇
#stwlist=[line.strip() for line in open('/home/kesci/input/stopwords7085/停用词汇总.txt',
#'r',encoding='utf-8').readlines()]
stwlist = ["，","（","。","）"]

#jieba.enable_parallel()

data['tscut'] = data['clean'].apply(lambda i:jieba.lcut(i) )

data['tscut'] = [' '.join(i) for i in data['tscut']]

print("%d verbatim" % len(data['tscut']))

print("使用稀疏向量（Sparse Vectorizer）从训练集中抽取特征")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5, max_features=40000,
                                 min_df=5, stop_words=stwlist,ngram_range=(1, 2),
                                 use_idf=True)

X = vectorizer.fit_transform(data['tscut'])

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
lsa = sklearn.pipeline.make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("完成所耗费时间： %fs" % (time() - t0))
explained_variance = svd.explained_variance_ratio_.sum()
print("SVD解释方差的step: {}%".format(int(explained_variance * 100)))

print('PCA文本特征抽取完成！')


#进行实质性的DBScan聚类
db = DBSCAN(eps=0.2, min_samples=4).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

db.core_sample_indices_

labels = db.labels_
labels

clusterTitles = db.labels_
dbscandf = data
dbscandf['cluster'] = clusterTitles

#看看簇群序号为0的文章的标题有哪些，通过这个能看出聚类的实际效果如何
dbscandf[dbscandf['cluster'] == 0]['tags'].head(20)  #簇群tag为0的title名称


#看看簇群序号为20的文章的标题有哪些，通过这个能看出聚类的实际效果如何
dbscandf[dbscandf['cluster'] == 20]['tags'].head(20)  #簇群tag为20的title名称

# 聚类数及噪点计算
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('聚类数：',n_clusters_)

print('噪点数：',n_noise_)

# #############################################################################
# 对结果可视化
import matplotlib.pyplot as plt
%matplotlib inline

# 黑色点是噪点，不参与聚类
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
          
          
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('大致聚类数: %d' % n_clusters_)
plt.savefig(os.path.join(dirname('__file__'), 'py.png'))


