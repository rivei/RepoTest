# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:03:38 2019

@author: Wei
"""

import pandas as pd
rawdf = pd.read_excel('../data/k9777rnd - 副本.xlsx','all')
rawdf = rawdf.fillna('NA')

def worddiff(cutl,cuts):
    if cutl.replace(',','') != cutl:
        cut1 = cutl.split(',')
    else:
        cut1 = [cutl]
    if cuts.replace(',','') != cuts:
        cut2 = cuts.split(',')
    else:
        cut2 = [cuts]
    stopW = ''
    addW = ''
    for s in cut1:
        if s != '' and s not in cut2:
            stopW+= s + ';'
    for s in cut2:
        if  s != '' and s not in cut1:
            addW += s + ';'
    return [stopW,addW]

stoplist = list(rawdf.apply(lambda row: worddiff(row['frase'],row['no-stop']), axis = 1))
stoplist.to_excel('stop.xlsx')
#stoplist = pd.DataFrame(stoplist, columns=['stopW','addW'])