# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:37:03 2019

@author: Wei
"""

import datetime
import pandas as pd
import sqlite3
import re
from snownlp import SnowNLP
import jieba
from gensim import corpora, models, similarities


dbName = "final.db"
## 1.载入词库（description + variance）
def loadDict():
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
print('load dictionary and build model')
dfdic = pd.DataFrame(loadDict())
dfdic.columns = ['PID','dscpt','vari','freq']
dfdic['dscpt'] = dfdic['dscpt'].apply(lambda i: i.replace('UNK','') )
dfdic['vari'] = dfdic['vari'].apply(lambda i: i.replace('UNK/','') )

def splitList(phrases):
    ls = phrases.split('/')
    #for s in ls:
    ls.remove('')
    return ls

## 3. Training
#traindf = pd.read_excel('Wave1.xlsx','Q5A')
#rawdf2 = pd.read_excel('L0028.xlsx','Q5A')
#traindf = traindf.append(rawdf2)
#traindf = pd.read_excel('rawdata.xlsx','all')

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


## 分词前先断句！！！
def getPhrases(sentence):
    if sentence != "":
        s = SnowNLP(sentence)
        outstr = ''
        for x in s.sentences:
            outstr+= x + ';'
    else:
        outstr = "无"
    return outstr

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

def listPossible(strSQL):
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    result = []
    cur.execute(strSQL)
    recs = cur.fetchall()
    for rec in recs:
        result.append(list(rec))
    cur.close()
    dbconn.commit()
    dbconn.close()
    return result

def seg2(sentence):
    outstr = []
    words = [word for word in jieba.cut(sentence, cut_all=True)]
    for x in words:
        if x not in outstr:
            outstr.append(x)
    return outstr


def updateDescpt(dscrpt, cut):
    cuts = cut.split('/')
    if '' in cuts: cuts.remove('')
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('SELECT variant, freq FROM codeMap WHERE dscrpt = ?', (dscrpt,))
    recs = cur.fetchall()
#    print(len(recs))
    if len(recs) == 1:
        if type(recs[0][0]) == type(None):
            var = cut+'/'
        else:
            var = recs[0][0]
            varls = var.split('/')
            varls.remove('')
            for c in cuts:
                if c not in varls:
                    var += c +'/'
        freq = recs[0][1]
        try:
            cur.execute('UPDATE codeMap SET freq = ?, variant = ? WHERE dscrpt = ?', 
                        (freq+1, var, dscrpt))
#            print(var)
#            print("update "+ cut)
        except Exception as err:
            print(err)
    dbconn.commit()
    dbconn.close()

def trainCode(frases, orgCode, threshold):
    print('training:' + frases)
    orgCodes = orgCode.split(';')
    if '' in orgCodes: orgCodes.remove('')
    verCuts = frases.split(';')
    if '' in verCuts:   verCuts.remove('')

    corp = []
    for c in orgCodes:
        corp.append(seg2(c))    
    numtop = len(corp) 
    if numtop == 1: 
        for verCut in verCuts:
            updateDescpt(orgCodes[0],'/'.join(tokenization(verCut)))
        return
    
    dicts = corpora.Dictionary(corp)
    doc_v = [dicts.doc2bow(text) for text in corp]
    tfidfM = models.TfidfModel(doc_v)
    tfidf_vecs = tfidfM[doc_v]
    lsiM = models.LsiModel(tfidf_vecs, id2word=dicts, num_topics=numtop)
    lsi_v = lsiM[tfidf_vecs]

    for verCut in verCuts:
        #print(verCut)
        query = tokenization(verCut)
        query_bow = dicts.doc2bow(query)
        
        #计算相似度
        q_lsi = lsiM[query_bow]
        index = similarities.MatrixSimilarity(lsi_v)
        sims = index[q_lsi]
        lsSim = list(enumerate(sims))
        
        dfSim = pd.DataFrame(lsSim, columns=['sID','score'])
        maxScore = max(dfSim.score)
        if maxScore > threshold:
            a = dfSim[dfSim.score == maxScore].iloc[0,0]
            updateDescpt(orgCodes[a],'/'.join(query))

#print('training: clean verbatim')
#traindf['cl01'] = traindf.apply(lambda row: delNA(row['Verbatim']), axis =1)
#print('training: seperate sentences')
#traindf['frase'] = traindf.apply(lambda row: getPhrases(row['cl01']), axis = 1)
#
##idx = 1                    
##trainCode(traindf.frase[idx],traindf.Mapping[idx], 0)
#traindf.apply(lambda row: trainCode(row['frase'],row['Mapping'], 0.7), axis = 1)


## 2. 用variance建立词典（词袋模型 doc_vector->LSI model）
print('build corpus')
corpus = list(dfdic.apply(lambda row: splitList(row['vari']), axis=1))
numtopics = len(corpus) 

zidian = corpora.Dictionary(corpus)
doc_vec = [zidian.doc2bow(text) for text in corpus]
tfidf = models.TfidfModel(doc_vec)
tfidf_vectors = tfidf[doc_vec]
lsi = models.LsiModel(tfidf_vectors, id2word=zidian, num_topics=numtopics)
lsi_vec = lsi[tfidf_vectors]



## 4. 读入新数据，分词tokenized
#rawdf = pd.read_excel('L0028.xlsx','Q5A')
#rawdf2 = pd.read_excel('Wave1.xlsx','Q5A')
#rawdf = rawdf.append(rawdf2)

rawdf = pd.read_excel('rawdata.xlsx','h')
rawdf.columns = ['Verbatim','Code','Mapping']
df = rawdf[['Verbatim', 'Mapping', 'Code']].copy() ##without copy df is only a view of rawdf!!!

print('clean verbatim')
df['cl01'] = df.apply(lambda row: delNA(row['Verbatim']), axis =1)
print('seperate sentences')
df['frase'] = df.apply(lambda row: getPhrases(row['cl01']), axis = 1)


def selcode(frases, threshold):
    print('dealing:' + frases)
    verCuts = frases.split(';')
    mapping = ''
    for verCut in verCuts:
        if verCut == '':
            break
        query = tokenization(verCut) ## !!! remove stopwords
        query_bow = zidian.doc2bow(query)
        
        ## 4. 计算相似度，并获取最相似的5条数据，显示
        q_lsi = lsi[query_bow]
        index = similarities.MatrixSimilarity(lsi_vec)
        sims = index[q_lsi]
        lsSim = list(enumerate(sims))
        
        dfSim = pd.DataFrame(lsSim, columns=['sID','score'])
        a = list(dfSim[dfSim.score > threshold]['sID'])
        #print('待编码: '+ verCut)
        if len(a) > 0:
            if len(a) == 1:
                iTmp = a[0]+1
                strSQL = 'SELECT dscrpt FROM codeMap WHERE id = %s'%iTmp
            else:
                strTmp = ''
                for i in range(0,2):
                    iTmp = a[i]+1
                    strTmp += '%s,'%iTmp
                strSQL = 'SELECT dscrpt FROM codeMap WHERE id in (' + strTmp[0:len(strTmp)-1] + ')'
            rsl = listPossible(strSQL)
            #print(rsl)
            for s in rsl:
                mapping+= s[0]+';'
                #print(s[0])
    return mapping


def getNum(org, rsl):
    bas = 0
    cod = 0
    shot = 0
    if type(org) == str and org != '':
        lsO = org.split(';')
        if '' in lsO:
            lsO.remove('')
        bas = len(lsO)
    
    if type(rsl) == str and rsl != '':
        lsR = rsl.split(';')
        if '' in lsR:
            lsR.remove('')
        lsR1 = []
        for sR in lsR:
            if sR not in lsR1:
                lsR1.append(sR) ##remove dup
                if sR in lsO:
                    shot+=1    
        cod = len(lsR1)
    return  pd.Series([bas, cod, shot], index=['base','codes','shot'])

print('find codes')
starttime = datetime.datetime.now()
df['rsl']=df.apply(lambda row: selcode(row['frase'], 0.5), axis=1)   
endtime = datetime.datetime.now()
wrTime = endtime - starttime 
print('runtime: ' , wrTime)

#df.columns = ['ver','org','code','cl01','frase','rsl']
#dff = df.apply(lambda row: getNum(row['org'], row['rsl']), axis =1)
#if sum(dff.base) > 0:
#    acc = sum(dff.shot)/sum(dff.base)
#dff['diff'] = dff.codes - dff.shot
#if sum(dff.codes) > 0:
#    wro = sum(dff['diff'])/sum(dff.codes)
#print('accuracy: ' , round(acc*100, 2), '%')
#print('extra: ' , round(wro*100, 2), '%')

## 自动学习
# auto training with 0.8: accuracy:  32.78 % extra:  68.92 % at 0.5
# without training: accuracy:  22.42 % extra:  73.68 % at 0.5

# 全新数据，独立码表，不学习: accuracy:  34.02 % extra:  59.79 % @ 0.5
# 新数据，合并码表，学习前： accuracy:  25.55 %； extra:  72.23 % @0.5
# 新数据，合并码表，学习后（0.7）： accuracy:  25.55 %； extra:  72.23 % @0.5 ？？？
#df.to_excel('validate_0.5.xlsx', sheet_name='cut')

df['rsl'] = df['rsl'].apply(lambda i: i.replace('UNK',''))

xlsx = pd.ExcelWriter('h2.xlsx')
df.to_excel(xlsx, sheet_name='k')
df.to_excel(xlsx, sheet_name='h',index = False)
xlsx.save()

## 1:01:30.033802 a 
## on k: 
#runtime:  2:00:53.720081
#accuracy:  16.43 %
#extra:  91.76 %
## on c:
#runtime:  0:04:17.656007
#accuracy:  5.77 %
#extra:  96.36 %
## on d:
#runtime:  0:06:34.077246
#accuracy:  28.5 %
#extra:  80.14 %