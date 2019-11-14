# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:45:34 2019

@author: Wei
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime
import re

dbName = "productDB.db"
rawdf = pd.read_excel("batch04.xlsx", "toMap")
mapdf = pd.DataFrame({'idx':np.arange(0,len(rawdf)),
                      'UPC':rawdf['UPC'],
                      'DESCRIPTION':rawdf['产品描述']
                   })
#mapdf = rawdf[['UPC','产品描述']].copy()
#mapdf.columns = ['UPC','DESCRIPTION']

'''
    I. cleaning!!
'''
#https://www.cnblogs.com/kaituorensheng/p/3554571.html
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code) #python 3 chr() replace unichr() !!
    return rstring

def getMultiPack(prodName):
    strRest = prodName
    units = ['连包','连装']
    pack = ''
    for unit in units:
        regex = '([1-9]\d*\.?\d*)' + unit
        p = re.compile(regex, re.I)
        m = p.search(prodName)
        if m:
            pack = m.group(0)
            strRest = strRest.replace(pack,'')
            pack = pack.replace(unit,'')
    return (pack, strRest)

def cleanText(idx, prodName):
    strRest = strQ2B(prodName)
    (pack, strRest) = getMultiPack(strRest)
    return pd.Series([idx, strRest, pack], 
                     index=['idx','cleaned','pack'])

print('cleanning data...')
#mapdf['cleaned'] = mapdf.apply(lambda row: strQ2B(row['DESCRIPTION']), axis = 1)
df0 = mapdf.apply(lambda row: cleanText(row['idx'],row['DESCRIPTION']), axis = 1)
mapdf = pd.merge(mapdf, df0, how = 'left', on = 'idx')


'''
    II. exclude the comine products
'''

'''
    III. 分割产品跟容量
'''
# regex extract size & pack
# regex: ([1-9]\d*\.?\d*单位)(\*?[1-9]\d?)?(\*?[1-9]\d?)?
def getSizePack(content):
    r = '-1'
    units = ['ml','l','kg','g','毫升','升','千克','克']
    for unit in units:
        #find combine first
        regex = '([1-9]\d*\.?\d*'+ unit +'?\+[1-9]\d*\.?\d*'+unit+')'
        p = re.compile(regex, re.I)
        m = p.findall(content)
        if len(m) > 0:
            for c in m:
                if r != '-1':
                    r = r +';' + c
                else:
                    r = c
            break
            
        regex = '([1-9]\d*\.?\d*' + unit + ')(\*[1-9]\d?)?(\*[1-9]\d?)?'
        p = re.compile(regex, re.I)
        m = p.findall(content)
        if len(m) > 0:
            for c in m:
                if r != '-1':
                    r += ';' + ''.join(c)
                else:
                    r = ''.join(c)
    return r

'''
处理容量
'''
def convertSize(pSize):
    sizeS = ['ML','G','毫升','克']
    sizeL = ['KG','L','升','千克']
    rsl = '-1'
    regex = '([1-9]\d*\.?\d*)'
    p = re.compile(regex, re.I)
    m = p.search(pSize)
    if m:
        r = pSize.replace(m.group(0),'')
        if r in sizeS:
            rsl = m.group(0)
        elif r in sizeL:
            rsl = str(int(float(m.group(0))*1000))
        else:
            rsl = m.group(0)
            
    return rsl

#切出容量
def splitSizePack(idx, prodName, orgpack):
    strRest = prodName
    isCmb = 0
    pSize = 'NA'
    pack = 'NA'
    s = getSizePack(prodName)
    if s.find('+') > 0:
        strRest = strRest.replace(s,'')
        isCmb = 1
        cslist = []
        for c in s.split('+'):
            cslist.append(convertSize(c.upper()))
        if len(cslist) == 1:
            pSize = cslist[0]
            pack = 1
        elif len(cslist) > 1:
            pSize = '+'.join(cslist)
            pack = len(cslist)
    elif s.find(';') > 0:
        cslist = []
        for c in s.split(';'):
            strRest = strRest.replace(c, '')
            ls = c.split('*',1)
            cs = convertSize(ls[0].upper())
            if cs not in cslist:
                cslist.append(cs)
        if len(cslist) == 1:
            pSize = cslist[0]
            pack = 1
        elif len(cslist) > 1:
            pSize = '+'.join(cslist)
            pack = len(cslist)
            isCmb = 1
    else:
        strRest = strRest.replace(s,'')
        ls = s.split('*',1)
        s = ls[0]
        pSize = convertSize(s.upper())
        if len(ls)>1:
            pack = ls[1]
        else:
            pack = 1
    if orgpack != '':
        pack = orgpack
        
    return pd.Series([idx, pSize, pack, strRest, isCmb], 
                     index=['idx','pSize','pack','rest1', 'cmb'])

print('find size & pack...')
df1 = mapdf.apply(lambda row: splitSizePack(row['idx'],row['cleaned'], row['pack']), axis = 1)
mapdf = pd.merge(mapdf, df1, how = 'left', on = 'idx')
#mapdf['cut1'] = mapdf.apply(lambda row: getSizePack(row['cleaned']), axis = 1)
#mapdf['rest1'] = mapdf.apply(lambda row: row['DESCRIPTION'].replace(row['cut1'], ''), axis = 1)
#mapdf['sizes'] = mapdf.apply(lambda row: convertSize(row['cut1']), axis = 1)


'''
    IV. 辨认品牌
'''
# 读取品牌字典并排序
def getBrandlist():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT variation, length(variation) as len FROM brands 
                 WHERE isMul = 0 ORDER BY len DESC''')
    recs = cur.fetchall()
    for rec in recs:
        rsl.append(rec[0])
    dbconn.commit()
    dbconn.close()
    return rsl

def getComplxBrand():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT variation, length(variation) as len FROM brands 
                 WHERE isMul = 1''')
    recs = cur.fetchall()
    for rec in recs:
        rsl.append(rec[0])
    dbconn.commit()
    dbconn.close()
    return rsl

blist = getBrandlist()
clist = getComplxBrand()

# 品牌辨认
def getBrand(idx, prodName):
    strRest = prodName.replace(' ','').upper()
    bmapping = []
    tocut = []
    for brandvar in clist:
        tokeep = ''
        matall = ''
        rst = brandvar
        if brandvar.find('/') > -1:
            tokeep = brandvar[brandvar.find('/')+1:len(brandvar)]
            rst = brandvar.replace('/'+tokeep,'')
            #print(rst)
        if rst.find(';') > -1:
            matall = rst.split(';')
            regex = '.*'.join(matall)
            #print(regex)
            p = re.compile(regex, re.I)
            m = p.search(strRest)
            if m:
                #print(brandvar)
                tocut.append(brandvar) 
                strRest = strRest.replace(matall[0],'')
        else:
            if strRest.find(rst) > -1:
                tocut.append(brandvar)
                strRest = strRest.replace(brandvar, '')
            
    for brandvar in blist:
        if strRest.find(brandvar) > -1:
            tocut.append(brandvar)
            strRest = strRest.replace(brandvar, '')

    if len(tocut) > 0:
        dbconn = sqlite3.connect(dbName)
        cur = dbconn.cursor()
        for c in tocut:
            cur.execute('''SELECT brand FROM brands WHERE variation = ?''',
                        (c,))
            recs = cur.fetchall()
            if len(recs) > 0 and recs[0][0] not in bmapping:
                bmapping.append(recs[0][0])
        dbconn.commit()
        dbconn.close()
        if len(bmapping) > 1:
            bMap = '+'.join(bmapping)
        else:
            bMap = bmapping[0]
        if len(tocut) >1:
            bCut = ';'.join(tocut)
        else:
            bCut = tocut[0]
    else:
        bMap = 'NA'
        bCut = 'NA'
    return pd.Series([idx, bMap, bCut, strRest], 
                     index=['idx','bMap','bCut','rest2'])

print("Finding brands...")
times = datetime.datetime.now()
df2 = mapdf.apply(lambda row: getBrand(row['idx'],row['rest1']), axis = 1)
timee = datetime.datetime.now()
print(timee-times)
mapdf = pd.merge(mapdf,df2,how = 'left', on = 'idx')


# =============================================================================
# 找口味
# =============================================================================
def getFlavorlist():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT variation, length(variation) as len FROM flavors 
                 ORDER BY len DESC''')
    recs = cur.fetchall()
    for rec in recs:
        rsl.append(rec[0])
    dbconn.commit()
    dbconn.close()
    return rsl


def getFlavor(idx, strText):
    strRest = strText.replace(' ','').upper()
    bmapping = []
    tocut = []
    flist = getFlavorlist()
    for flavor in flist:
        if strRest.find(flavor) > -1:
            tocut.append(flavor)
            strRest = strRest.replace(flavor, '')

    if len(tocut) > 0:
        dbconn = sqlite3.connect(dbName)
        cur = dbconn.cursor()
        for c in tocut:
            cur.execute('''SELECT fName FROM flavors WHERE variation = ?''',
                        (c,))
            recs = cur.fetchall()
            if len(recs) > 0 and recs[0][0] not in bmapping:
                bmapping.append(recs[0][0])
        dbconn.commit()
        dbconn.close()
        if len(bmapping) > 1:
            fMap = '+'.join(bmapping)
        else:
            fMap = bmapping[0]
        if len(tocut) >1:
            fCut = ';'.join(tocut)
        else:
            fCut = tocut[0]
    else:
        fMap = 'NA'
        fCut = 'NA'
    return pd.Series([idx, fMap, fCut, strRest], 
                     index=['idx','fMap','fCut','rest3'])
print('finding flavor...')
df3 = mapdf.apply(lambda row: getFlavor(row['idx'],row['rest2']), axis = 1)
mapdf = pd.merge(mapdf,df3,how = 'left', on = 'idx')

def getPackConList():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT contains, length(contains) as len FROM packages
                WHERE contains IS NOT NULL ORDER BY len DESC''')
    recs = cur.fetchall()
    for rec in recs:
        rsl.append(rec[0])
    dbconn.commit()
    dbconn.close()
    return rsl

def getPackBrand():
    rsl = []
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT brand FROM packages WHERE brand IS NOT NULL''')
    recs = cur.fetchall()
    for rec in recs:
        rsl.append(rec[0])
    dbconn.commit()
    dbconn.close()
    return rsl
    

def getPackage(strText, bMap):
    strRest = strText.replace(' ','').upper()
    package = ''
    plist = getPackConList()
    for contains in plist:
        if strRest.find(contains) > -1:
            dbconn = sqlite3.connect(dbName)
            cur = dbconn.cursor()
            cur.execute('''SELECT pName FROM packages WHERE contains = ?''',
                        (contains,))
            recs = cur.fetchall()
            if len(recs) > 0:
                package = recs[0][0]
                
            dbconn.commit()
            dbconn.close()
            break
        
    brands = getPackBrand()
    for brand in brands:
        if brand.upper() == bMap.upper():
            dbconn = sqlite3.connect(dbName)
            cur = dbconn.cursor()
            cur.execute('''SELECT pName FROM packages WHERE brand = ?''',
                        (brand,))
            recs = cur.fetchall()
            if len(recs) > 0:
                package = recs[0][0]
                
            dbconn.commit()
            dbconn.close()
            break
    
    return package

mapdf['package']=mapdf.apply(lambda row: getPackage(row['rest3'],row['bMap']), axis = 1)

#dfNA = df2[df2.bMap == 'NA']

mapdf.to_excel('NA.xlsx',sheet_name = 'brand')

