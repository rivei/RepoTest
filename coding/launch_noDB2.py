# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:37:03 2019
@author: Wei

文档说明：
运行需要的文件：launch_test.py,final.db
输入：Excel格式的原始数据，赋值给rawdata。每一道题为一张工作表，格式为
    |     A       |    B     |
    |-------------|----------|
    | iobs/respid | Verbatim |
    |-------------|----------|
    |             |          |
输出：Excel文件result.xlsx, 每一题保存为一个表

"""

import datetime
import pandas as pd
import sqlite3
import re
from snownlp import SnowNLP
import jieba
from gensim import corpora, models, similarities

# =============================================================================
# 加载数据库，返回一个list; 
# list中每一行是一个子list
# 子list中的字段分别为：['PID','dscpt','vari','freq']
# =============================================================================
def loadDict(dbName):
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('SELECT * FROM codeMap')
    recs = cur.fetchall()
    rsl = []
    for rec in recs:
        rsl.append(list(rec))
    cur.close()
    dbconn.close()
    return rsl

# =============================================================================
# 讲数据库预先分词的列进行词分割
# =============================================================================
def splitList(phrases):
    ls = phrases.split('/')
    if '' in ls:
        ls.remove('')
    return ls

# =============================================================================
# 删除Verbatim中的“无/已追问”
# 输入：单一一行Verbatim字符串
# 输出：清除"无/已追问"的字符串
# =============================================================================
def delNA(sVerbatim):
    if sVerbatim == 'NA':
        return 'NA'
    sTmp = sVerbatim
    # delete end of verbatim 删除结尾
    regex = '(无[\/,，]已?\S+)'
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

# =============================================================================
# 按标点符号断句
# 输入：清理干净的Verbatim字符串
# 输出：按。/，分裂的短句字符串，以;分隔
# =============================================================================
def getPhrases(sentence):
    if sentence == 'NA':
        return 'NA'
    
    if sentence != "":
        s = SnowNLP(sentence)
        outstr = ''
        for x in s.sentences:
            outstr+= tokenization(x) + ';'
    else:
        outstr = "无"
    return outstr

# =============================================================================
# 短句分词
# 输入：句子字符串
# 输出：分词的list
# =============================================================================
def tokenization(sentence):
    result = []
    words = [word for word in jieba.cut(sentence, cut_all=False)]
    for word in words:
        if word not in stopwords: # and word not in result:
            result.append(word)
    return ','.join(result)

#def tokenization2(sentence):
#    if sentence == 'NA':
#        return 'NA'
#    result = []
#    if sentence != "":
#        s = SnowNLP(sentence)
#        for x in s.sentences:
#            words = [word for word in jieba.cut(x, cut_all=True)]
#            frases = []
#            for word in words:
#                if word not in stopwords: # and word not in result:
#                    frases.append(word)
#            result.append(list(frases))
#    else:
#        result.append('无')
#    return result

# =============================================================================
# 通过编码的ID查找匹配到的的编码描述
# 输入：含有要查找ID的SQL语句
# 输出：匹配到的编码描述
# =============================================================================
#def listPossible(strSQL,dbName):
#    dbconn = sqlite3.connect(dbName)
#    cur = dbconn.cursor()
#    result = []
#    cur.execute(strSQL)
#    recs = cur.fetchall()
#    for rec in recs:
#        result.append(list(rec))
#    cur.close()
#    dbconn.commit()
#    dbconn.close()
#    return result

# =============================================================================
# 根据阈值选出句子编码
# 输入：1.用;分割的短句字符串
#      2.相似度阈值，浮点数
# 输出：匹配到的编码TAG 
# =============================================================================
def selcode(frasesCut, threshold,dbName):
    if frasesCut == 'NA':
        return ''
    if frasesCut == '无':
        return '无'
    print('dealing:' + frasesCut)
    verCuts = frasesCut.split(';')
    mapping = ''
    for verCut in verCuts:
        if verCut != '':
            query = verCut.split(',')
            query_bow = zidian.doc2bow(query)
            
            q_lsi = lsi[query_bow]
            index = similarities.MatrixSimilarity(lsi_vec)
            sims = index[q_lsi]
            lsSim = list(enumerate(sims))
            
            dfSim = pd.DataFrame(lsSim, columns=['sID','score'])
            a = list(dfSim[dfSim.score > threshold]['sID'])
    
            if len(a) > 0:
                if len(a) == 1:
                    mapping = dfdic['dscpt'][a[0]]+';'
                else:
                    #strTmp = ''
                    for i in range(0,2): #取top 3
                        mapping += dfdic['dscpt'][a[i]]+';'
    
    if mapping == '':
        allwords = frasesCut.replace(';',',')
        query_bow = zidian.doc2bow(allwords.split(','))
        
        q_lsi = lsi[query_bow]
        index = similarities.MatrixSimilarity(lsi_vec)
        sims = index[q_lsi]
        lsSim = list(enumerate(sims))
        
        dfSim = pd.DataFrame(lsSim, columns=['sID','score'])
        a = list(dfSim[dfSim.score == max(dfSim.score)]['sID'])
        mapping = dfdic['dscpt'][a[0]]  #rsl[0][0]
    mapping.strip(';')
    print(mapping)
    return mapping




# =============================================================================
#                                   主程序
# =============================================================================
rawdata = "./data/K9777rnd.xlsx" #<----- input
output = "result.xlsx" #<-------- output

dbNew = "final.db"
## 1.载入词库（description + variance）
print('loading dictionary and build model')
dfdic = pd.DataFrame(loadDict(dbNew))
dfdic.columns = ['PID','dscpt','vari','freq']

stopwords = []
for line in open('stopwords2.txt', encoding = 'utf-8'):
    stopwords.append(line.strip())
#stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

## 2. 用variance建立词典（词袋模型 doc_vector->LSI model）
print('building corpus')
corpus = list(dfdic['vari'].apply(lambda i: splitList(i)))
numtopics = len(corpus) 

zidian = corpora.Dictionary(corpus)
doc_vec = [zidian.doc2bow(text) for text in corpus]
tfidf = models.TfidfModel(doc_vec)
tfidf_vectors = tfidf[doc_vec]
lsi = models.LsiModel(tfidf_vectors, id2word=zidian, num_topics=numtopics)
lsi_vec = lsi[tfidf_vectors]

wbRawdata = pd.read_excel(io = rawdata, sheet_name = None) #读全部数据 
xlsx = pd.ExcelWriter(output)
starttime = datetime.datetime.now()
for sht in wbRawdata:
    Qno = sht
    rawdf = wbRawdata[Qno]

    ## 4. 读入新数据，分词tokenized
    rawdf.columns = ['respid','Verbatim']
    df = rawdf[['respid', 'Verbatim']].copy()
    df = df.fillna('NA')
    print(Qno + '...clean verbatim...')
    df['cl01'] = df.apply(lambda row: delNA(row['Verbatim']), axis =1)
    print(Qno + '...seperating sentences...')
    df['frase'] = df.apply(lambda row: getPhrases(row['cl01']), axis = 1)
    print(Qno + '...finding codes...')
    #df['cut']=df['frase'].apply(lambda i: tokenization2(i))   #调整相似度阈值
    df['rsl']=df['frase'].apply(lambda i: selcode(i, 0.6, dbNew))   #调整相似度阈值
    df.to_excel(xlsx, sheet_name=Qno,index = False)
xlsx.save()

endtime = datetime.datetime.now()
wrTime = endtime - starttime 
print('runtime: ' , wrTime) #0:29:27.444146

