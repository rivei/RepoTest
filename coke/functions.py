# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:42:07 2019

@author: Wei
"""
import re

def list2str(row_data, delimiter):
    values = "";
    for i in range(len(row_data)):
        if str(row_data[i]) != 'None':
            if i == 0:
                values = str(row_data[0])
            else:
                values = values + delimiter + str(row_data[i])
    return values


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


#匹配函数
def mapping(data,dictionary):
    result=[]
    for item in data:                  #数据中每一行
        f=[]                           #重置每一行的记录内容
        count=0
        for word in item:              #每一行数据切出来的每一个词
            for dic in dictionary:     #字典中的每一个大类
                if word in dic:        #如何词匹配了大类中的任一个，返回该类名
                    f.append(dic[0])   #每一行返回一个列表f
                    count+=1
        if count==0:                   #count记录有多少个匹配，=0即没有找到任何匹配
            f.append('not found')
        result.append(f)               #将每一行匹配出来的列表记录在result中
    return result               


#去重
def dup(data):
    result=[]
    for l in data:
        l=list(set(l))
        result.append(l)
    return result


#如果只有一个brand对应了多个sub，那要对sub进行判断
def adjust(databra,datasub):
    result=datasub
    for i in range(len(databra)):
        if len(databra[i])==1:      #只有一个brand的
            if result[i]==['果汁饮料','汽水']:
                result[i]=['汽水']
            if '咖啡饮料' in result[i]:
                result[i]=['咖啡饮料']
            if '儿童奶' in result[i]:
                result[i]=['儿童奶']
            if result[i]==['茶','非纯奶']:
                result[i]=['奶茶'] 
            if result[i]==['茶','果汁饮料']:
                result[i]=['果茶'] 
            if {'茶','果汁饮料'} <= set(result[i]):
                result[i]=['果茶'] 
            if '冰茶' in result[i]:
                result[i]=['冰茶']
            if result[i]==['无糖汽水','汽水']:
                result[i]=['无糖汽水']
            if (len(result[i])>1 and ('无糖汽水' in result[i])):
                result[i].remove('无糖汽水')
            if (('椰汁' in result[i]) and (set(result[i]) <= {'椰汁','非纯奶','豆奶','果汁饮料','植物饮料'})):
                result[i]=['椰汁']
            if '运动饮料' in result[i]:
                result[i]=['运动饮料']
            if '冰红茶' in result[i]:
                result[i]=['冰红茶']
            if '果茶' in result[i]:
                result[i]=['果茶']
            if set(result[i])=={'纯果汁','果汁饮料'}:
                result[i]=['纯果汁']
            if result[i]==['纯果汁','椰子水']:
                result[i]=['椰子水']
            if (('乳酸奶' in result[i]) and (set(result[i]) <= {'乳酸奶','果汁饮料','纯果汁'})):
                result[i]=['乳酸奶']
            if result[i]==['乳酸奶','汽水']:
                result[i]=['汽奶']
            if set(result[i])=={'纯牛奶','果汁饮料'}:
                result[i]=['果乳饮料']
            if {'纯牛奶','汽水'} <= set(result[i]):
                result[i]=['汽奶']
            if {'坚果浆','纯牛奶'} <= set(result[i]):
                result[i]=['非纯奶']
            if {'果汁饮料,果乳饮料'} <= set(result[i]) or {'乳酸奶','果乳饮料'} <= set(result[i]):
                result[i]=['果乳饮料']
            if {'果汁饮料','果乳饮料'} <= set(result[i]):
                result[i]=['果乳饮料']
            if '豆奶' in result[i]:
                result[i]=['豆奶']
            if {'非纯奶','果汁饮料'}<=set(result[i]):
                result[i]=['果乳饮料']
            if (len(result[i])>1 and ('茶' in result[i])):
                result[i].remove('茶')
            if '茉莉花茶' in result[i]:
                result[i]=['茉莉花茶']
            if '茉莉清茶' in result[i]:
                result[i]=['茉莉清茶']
            if '茉莉蜜茶' in result[i]:
                result[i]=['茉莉蜜茶']
            if '待定' in result[i]:
                result[i]=['待定'] 
            if (('矿泉水' in result[i]) and ('含气天然水' not in result[i])):
                result[i]=['矿泉水']
            if ({'汽水','果汁饮料'} <=set(result[i])) or ({'汽水','纯果汁'}<=set(result[i])):
                result[i]=['汽水']
    return result

def adjfla(datafla):
    result=datafla
    for i in range(len(datafla)):
        if '原味' in result[i]:
           result[i]=['原味']
        if ((len(result[i])>1) and ('牛奶' in result[i])):
            result[i].remove('牛奶')
    return result
