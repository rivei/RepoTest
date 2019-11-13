# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:49:38 2019

@author: Wei

文档说明：
运行需要的文件：autocoding.py, final.db, stopwords.txt
inputpath：Excel格式的原始数据，。每一道题为一张工作表，格式为
    |     A       |    B     |
    |-------------|----------|
    | iobs/respid | Verbatim |
    |-------------|----------|
    |             |          |
outputpath：Excel文件result.xlsx, 每一题保存为一个表

"""

from autocoding import RunCore

RunCore(inputpath = "./demo/input.xlsx",outputpath = "./demo/result.xlsx") 