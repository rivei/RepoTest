# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:46:46 2019

@author: Wei
"""
import pandas as pd

print('read data')
rawdf = pd.read_excel('Wave1.xlsx','Q5A')

#clean verbatim
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

print('clean verbatim')
rawdf['cl01'] = rawdf.apply(lambda row: delNA(row['Verbatim']), axis =1)

# seperate sentense
from snownlp import SnowNLP
def getPhrases(sentence):
    if sentence != "":
        s = SnowNLP(sentence)
        outstr = ''
        for x in s.sentences:
            outstr+= x + ';'
    else:
        outstr = "无"
    return outstr

print('seperate sentences')
rawdf['frase'] = rawdf.apply(lambda row: getPhrases(row['cl01']), axis = 1)

import sqlite3
dbName = "testing.db"
def addTrainSheet():
    try:
        dbconn = sqlite3.connect(dbName)
        cur = dbconn.cursor()
        
        cur.execute('''
                    CREATE TABLE IF NOT EXISTS training
                    (ID integer PRIMARY KEY AUTOINCREMENT,
                    codeID integer NOT NULL, 
                    verbatim TEXT,
                       FOREIGN KEY (codeID) REFERENCES codeMap(ID));''')
    except Exception as er:
        print('Error', er)
    finally:
        if dbconn:
            dbconn.close()

print('Add training sheet')
addTrainSheet()

# fill training sheet
def fillTrainingSheet(verfrase, codeDscr):
    vlist = verfrase.split(';')    
    dlist = codeDscr.split(';')
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    rsl = ''
    for vs in vlist:
#        print(vs)
        if vs == '':
            break
        for ds in dlist:
#            print(ds)
            if ds == '':
                break
            cur.execute('SELECT ID FROM codeMap WHERE dscrpt = ? ', (ds,))
            try:
                codeID = cur.fetchone()[0]
#                print(codeID)
                cur.execute('''INSERT INTO training (codeID, verbatim)
                    VALUES (?,?)''', (codeID, vs))
                
            except Exception as err:
                print(err)
                rsl += err + ";"
    dbconn.commit()
    dbconn.close()
    return rsl

#fillTrainingSheet(rawdf.frase[1],rawdf.Mapping[1])
#print(rawdf.frase[1])
#print(rawdf.Mapping[1])
print('fill training sheet')
rawdf['err'] = rawdf.apply(lambda row: fillTrainingSheet(row['frase'], row['Mapping']), axis = 1)

# filter correct answers:
def getMapping():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT TR.*, CM.dscrpt 
        FROM training AS TR, codeMap AS CM
        WHERE TR.codeID = CM.ID ''')
    recs = cur.fetchall()
    dbconn.commit()
    dbconn.close()
    for rec in recs:
        rsl.append(list(rec))
    return rsl

print('export data')
mapdf = pd.DataFrame(getMapping())
mapdf.to_excel('training.xlsx', sheet_name='org')









