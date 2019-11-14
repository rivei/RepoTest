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
rawdf = pd.read_excel("batch06.xlsx", "Sheet2")
mapdf = pd.DataFrame({'idx':np.arange(0,len(rawdf)),
                      'DESCRIPTION':rawdf['品名']
                   })

# =============================================================================
# I. cleaning
# =============================================================================
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
    units = ['连包','连装','联罐']
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
df0 = mapdf.apply(lambda row: cleanText(row['idx'],row['DESCRIPTION']), axis = 1)
mapdf = pd.merge(mapdf, df0, how = 'left', on = 'idx')


# =============================================================================
# II. exclude the comine products ???
# =============================================================================


# =============================================================================
#     III. 分割产品跟容量
# =============================================================================
# regex extract size & pack
def getSizePack(content):
    r = '-1'
    units = ['ml','l','kg','g','毫升','亳升','升','千克','克']
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

## 处理容量
def convertSize(pSize):
    sizeS = ['ML','G','毫升','克','亳升']
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

## 分离容量和数量
def splitSizePack(idx, prodName, orgpack):
    strRest = prodName
    parentheses = ['(', ')','（','）']
    for p in parentheses:
        strRest = strRest.replace(p,'')
    
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


# =============================================================================
#     IV. 辨认品牌
# =============================================================================
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

def adjustBrand(idx, bMap):
    bName = bMap
    subcat = ''
    flavor = ''
    dbconn = sqlite3.connect(dbName)
    cur = dbconn.cursor()
    cur.execute('''SELECT bName, subcat, flavor FROM complex 
                 WHERE bMap = ?''', (bMap,))
    recs = cur.fetchall()
    if len(recs) == 1:
        bName = recs[0][0]
        subcat = recs[0][1]
        flavor = recs[0][2]
    dbconn.commit()
    dbconn.close()
    return pd.Series([idx, bName, subcat, flavor], 
                     index=['idx','bName','subcat','flavor'])
df2 = mapdf.apply(lambda row: adjustBrand(row['idx'], row['bMap']), axis = 1)
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

flist = getFlavorlist()
def getFlavor(idx, strText):
    strRest = strText.replace(' ','').upper()
    bmapping = []
    tocut = []
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

# =============================================================================
# 填写包装
# =============================================================================

def getPackage(strText, bMap, subcat, psize):
    strRest = strText.replace(' ','').upper()
    package = ''
    plist = [('sleek CAN','SLEEKCAN'),
				('AL BOTTLE','铝瓶'),
                ('CAN','CAN'),
				('CAN','迷你摩登罐'),
				('GLASS','玻璃瓶'),
                ('GLASS','广口瓶'),
                ('sleek CAN','细长罐'),
                ('Tetra','苗条砖'),
                ('sleek CAN','纤体听'),
                ('BAG','利乐枕'),
                ('Tetra','利乐砖'),
                ('Tetra','苗条砖'),
                ('Tetra','利乐包'),
                ('Tetra','大利乐'),
				('Tetra','康美包'),
				('Tetra','三角包'),
				('Tetra','屋顶包'),
				('Tetra','笑脸包'),
				('Tetra','多角包'),
				('Tetra','钻包'),
				('Tetra','纸盒'),
                ('sleek CAN','摩登罐'),
                ('Tetra','利乐'),
                ('CAN','听装'),
                ('CAN','罐装'),
                ('CAN','金罐'),
                ('CAN','罐'),
                ('CAN','听'),
				('PET','塑瓶'),
				('PET','PP瓶'),
                ('PET','瓶装'),
				('PET','胶瓶'),
				('BAG','百利包'),
				('BAG','透明袋'),
				('BAG','袋装'),
                ('BAG','枕式'),
                ('PET','PET')]
                    
    for contains in plist:
        if strRest.find(contains[1]) > -1:
            package += ';'+contains[0]
            
#    if strRest.find('PET') > -1:
#        package += ';PET'
    
    
    bts = ['杰事','芬特乐','福兰农庄','卓宜','普瑞达','意文']
    if bMap in bts and str.isnumeric(psize):
        if psize > 1000:
            package += ';Tetra'
    bgs = ['高丽参','力保健']
    if subcat in bgs:
        package += ';GLASS'
    
    sps = ['纯净水','饮用水','蒸馏水','矿化水','矿泉水','天然水','含气天然水',
           '无气苏打水']
    if subcat in sps:
        package += ';PET'
    #汽水 & size > 350 ->PET
	#可乐/汽水 =330 除标出瓶装/PET -> CAN 
	#可乐，雪碧，芬达 200ml -> CAN
	#可乐，雪碧，芬达 300ml -> PET
	#可乐 250ml -> AL BOTTLE
	#冰峰水 256; 汉口二厂 275 汽水   -> GLASS  
	#纯果汁,纯牛奶 >= 1000 -> Tetra (部分品牌)
	#豆奶 >=1000 ->Tetra
	#茶类 (除凉茶；奶茶 >= 500) -> PET
	#茶类:冰红茶 250 ml Tetra
	#凉茶 310 -> CAN
	#水类 -> PET
	#奶类 250 Tetra
	# 旺仔牛奶 245ml CAN 125.ETC; 250 Tetra
	# 纯牛奶 250 Tetra
	#儿童奶,乳酸奶  100 PET
	# 咖啡 180,200,240 CAN
    
	
	#初元杏仁露240ml -> can
	#康之味（异型罐）果粒桑葚汁965ml pet
	#康师傅 >= 330 PET; =310 can; = 250 Tetra
	#统一 >= 350 PET
	#伊利， 1000 Tetra; 300 - 400 PET 100 - 200 Tetra
	#安慕希 230 PET
	#蒙牛 240 BAG 
    return package.strip(';')
#bag: 190,195, 200, 210
#砖 -> Tetra
# 棒棒奶 -> 待定
# 支 -> PET
print('finding package...')
mapdf['package']=mapdf.apply(lambda row: getPackage(row['rest3'],row['bName'],
     row['subcat'], row['pSize']), axis = 1)


print('Exporting result...')
mapdf.to_excel('run06.xlsx',sheet_name = 'brand', index = False)

