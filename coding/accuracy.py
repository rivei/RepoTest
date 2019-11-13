# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:05:35 2019

@author: Wei
"""

import pandas as pd
rsldf = pd.read_excel('trained_0.7.xlsx','cut')
rsldf.columns = ['ID','ver','org','code','cl01','frase','rsl']
def getNum(org, rsl):
    bas = 0
    cod = 0
    shot = 0
    if type(org) == str and org != '':
        lsO = org.split(';')
        lsO.remove('')
        bas = len(lsO)
    
    if type(rsl) == str and rsl != '':
        lsR = rsl.split(';')
        lsR.remove('')
#        cod = 0 #len(lsR)
        lsR1 = []
        for sR in lsR:
            if sR not in lsR1:
                lsR1.append(sR) ##remove dup
                if sR in lsO:
                    shot+=1    
        cod = len(lsR1)
    return  pd.Series([bas, cod, shot], index=['base','codes','shot'])


dff = rsldf.apply(lambda row: getNum(row['org'], row['rsl']), axis =1)
acc = sum(dff.shot)/sum(dff.base)
dff['diff'] = dff.codes - dff.shot
wro = sum(dff['diff'])/sum(dff.codes)
print('accuracy: ' , round(acc*100, 2), '%')
print('extra: ' , round(wro*100, 2), '%')


##### result #####
##### with manual tagging ####
# validation:
#accuracy:  66.92 %; extra:  60.57 % with 0.5; 
#accuracy:  56.91 %, extra:  46.37 % with 0.6; 
#accuracy:  45.87 %, extra:  32.93 % with 0.7
# used Wave1.xlsx for training, run L0028.xlsx, 
# threshold 0.5, accuracy:  55.07 %; extra:  66.59 %;
# threshold 0.6, accuracy:  45.4 %; extra:  53.77 %
# threshold 0.7, accuracy:  35.65 %; extra:  43.58 %;