# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:42:18 2019

@author: Wei
"""

import sqlite3

dbName = "final.db"
def initDB():
    try:
        dbconn = sqlite3.connect(dbName)
        cur = dbconn.cursor()
        
        cur.execute('''
                    CREATE TABLE IF NOT EXISTS codeMap
                    (ID integer PRIMARY KEY AUTOINCREMENT,
                    dscrpt TEXT,
                    variant TEXT,
                    freq INTEGER);''')
    except Exception as er:
        print('Error', er)
    finally:
        if dbconn:
            dbconn.close()
print('Init DB')
initDB()

def insertDescpt(descrpt, cut):
    cuts = cut.split('/')
    cuts.remove('')
    
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('SELECT variant, freq FROM codeMap WHERE dscrpt = ? ', (descrpt,))
    recs = cur.fetchall()
    #print(len(recs))
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
                        (freq+1, var, descrpt))
            print("update "+ descrpt)
        except Exception as err:
            print(err)
    else:
        cur.execute('''INSERT INTO codeMap (dscrpt, variant, freq)
            VALUES (?,?, 1)''', (descrpt,cut))
        print('insert ' + descrpt)
    dbconn.commit()
    dbconn.close()

import jieba
def seg(sentence):
    outstr = ''
    words = [word for word in jieba.cut(sentence, cut_all=False)]
    for x in words:
        outstr+= x+'/'
    return outstr


import pandas as pd
print('read data')

rawdf = pd.read_excel('codelist_FINAL.xlsx','a') ### <- codelist name
rawdf.columns = ['code','label']
rawdf['num'] = rawdf.apply(lambda row: str.isnumeric(str(row['code'])), axis = 1)
rawdf = rawdf[rawdf.num == True]

print('cut words')
rawdf['cut'] = rawdf.apply(lambda row: seg(row['label']), axis = 1)
rawdf.apply(lambda row: insertDescpt(row['label'], row['cut']), axis = 1)

def tidyupDB():
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('SELECT ID, variant FROM codeMap')
    recs = cur.fetchall()
    for rec in recs:
        if type(rec[0]) != type(None):
            vid = rec[0]
            var = rec[1]
            var = var.replace('//', '/')
        try:
            cur.execute('UPDATE codeMap SET variant = ? WHERE ID = ?',
                    (var, vid))
        except Exception as err:
            print(err)
    cur.close()
    dbconn.commit()
    dbconn.close()

tidyupDB()
