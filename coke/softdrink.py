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
rawdf = pd.read_excel("batch05.xlsx")
mapdf = pd.DataFrame({'idx':np.arange(0,len(rawdf)),
                      'DESCRIPTION':rawdf['品名'],
                      'sizepack':rawdf['规格'],
                      'package':rawdf['包装材质']
                   })

# =============================================================================
#     III. 分割产品跟容量
# =============================================================================

## 分离容量和数量
def splitSizePack(idx,sizePack):
    pSize = 'NA'
    pack = 'NA'
    
    ls = sizePack.split('X')
    pack = ls[0]
    pSize = ls[1].replace(' ML','')

    return pd.Series([idx, pSize, pack], 
                     index=['idx','size','pack'])


print('find size & pack...')
df0=mapdf.apply(lambda row: splitSizePack(row['idx'],row['sizepack']),axis = 1)
mapdf = pd.merge(mapdf, df0, how = 'left', on = 'idx')


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
#def getPackConList():
#    rsl = []
#    dbconn = sqlite3.connect(dbName)
#    cur = dbconn.cursor()
#    cur.execute('''SELECT contains, length(contains) as len FROM packages
#                WHERE contains IS NOT NULL ORDER BY len DESC''')
#    recs = cur.fetchall()
#    for rec in recs:
#        rsl.append(rec[0])
#    dbconn.commit()
#    dbconn.close()
#    return rsl
#
#def getPackBrand():
#    rsl = []
#    dbconn = sqlite3.connect(dbName)
#    cur = dbconn.cursor()
#    cur.execute('''SELECT brand FROM packages WHERE brand IS NOT NULL''')
#    recs = cur.fetchall()
#    for rec in recs:
#        rsl.append(rec[0])
#    dbconn.commit()
#    dbconn.close()
#    return rsl
#def getPackage(strText, bMap, subcat, psize):
#    strRest = strText.replace(' ','').upper()
#    package = ''
#    #plist = getPackConList()
#    
#    for contains in plist:
#        if strRest.find(contains) > -1:
#            dbconn = sqlite3.connect(dbName)
#            cur = dbconn.cursor()
#            cur.execute('''SELECT pName FROM packages WHERE contains = ?''',
#                        (contains,))
#            recs = cur.fetchall()
#            if len(recs) > 0:
#                package = recs[0][0]
#                
#            dbconn.commit()
#            dbconn.close()
#            break
#        
#    #brands = getPackBrand()
#    for brand in brands:
#        if brand.upper() == bMap.upper():
#            dbconn = sqlite3.connect(dbName)
#            cur = dbconn.cursor()
#            cur.execute('''SELECT pName FROM packages WHERE brand = ?''',
#                        (brand,))
#            recs = cur.fetchall()
#            if len(recs) > 0:
#                package = recs[0][0]
#                
#            dbconn.commit()
#            dbconn.close()
#            break
#    
#    return package

def getPackage(strText, bMap, subcat, psize):
    strRest = strText.replace(' ','').upper()
    package = ''
    plist = [('sleek CAN','SLEEKCAN'),
				('AL BOTTLE','铝瓶')
                ('CAN','CAN'),
				('CAN','迷你摩登罐'),
				('GLASS','玻璃瓶'),
                ('GLASS','广口瓶'),
                ('sleek CAN','细长罐'),
                ('Tetra','苗条砖'),
                ('sleek CAN','纤体听'),
                ('BAG','利乐枕'),
                ('Tetra','利乐砖'),
                ('Tetra','苗条砖')
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
                ('BAG','枕式')]
                    
    for contains in plist:
        if strRest.find(contains[1]) > -1:
            package = contains[0]
			break
            
    if strRest.find('PET') > -1:
        package += ';PET'
    
    
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
mapdf['package']=mapdf.apply(lambda row: getPackage(row['rest3'],row['bName'],
     row['suborg'], row['pSize']), axis = 1)


print('Exporting result...')
mapdf.to_excel('NA.xlsx',sheet_name = 'brand', index = False)

