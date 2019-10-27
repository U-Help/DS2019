# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:34:17 2019

@author: dell
"""

"""
ok  3. ()/[]/{} 括号里的东西
ok  4. HTML标签

2. 数字/日期，标点符号

1. 挑选所有汉字(清除其他的符号)所有，奇怪的符号(\n)
"""

import re

import pandas as pd
import os
import random

max_title_len = 30
max_content_len = 400

def title_filter(string):
    
    print(type(string))
    print(string)
    
    brack_en_filt_out = re.compile(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>")
    brack_cn_filt_out = re.compile(u"\\（.*?）|\\【.*?】|\\《.*?》")
    char_filt_in = re.compile(u'[^\u4E00-\u9FA5]')
    date_filt_out = re.compile(u'年|月|日|十|百|千|万|亿|星期|礼拜')
    
    string = brack_en_filt_out.sub(r'', string)
    string = brack_cn_filt_out.sub(r'', string)
    string = char_filt_in.sub(r'', string)
    string = date_filt_out.sub(r'', string)
    
    if len(string) > max_title_len:
        train_df['title'] = train_df['title'][:max_title_len]
    return string

def content_filter(string):
    brack_en_filt_out = re.compile(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>")
    brack_cn_filt_out = re.compile(u"\\（.*?）|\\【.*?】|\\《.*?》")
    char_filt_in = re.compile(u'[^\u4E00-\u9FA5]')
    date_filt_out = re.compile(u'年|月|日|十|百|千|万|亿|星期|礼拜')
    
    string = brack_en_filt_out.sub(r'', string)
    string = brack_cn_filt_out.sub(r'', string)
    string = char_filt_in.sub(r'', string)
    string = date_filt_out.sub(r'', string)
    
    if len(string) > max_content_len:
        train_df['title'] = train_df['title'][:max_content_len]
    return string

#train_df=pd.read_csv("Train_DataSet_Clean.csv")
#train_label_df=pd.read_csv("Train_DataSet_Label_Clean.csv")
#test_df=pd.read_csv("Test_DataSet.csv")
#
#train_df['title'] = train_df['title'].apply(title_filter);
#train_df['content'] = train_df['content'].apply(content_filter);
#
#test_df['title'] = test_df['title'].apply(title_filter);
#test_df['content'] = test_df['content'].apply(content_filter);

df = pd.DataFrame([
    [-0.532681, '测试', 0],
    [1.490752, '英特尔新cpu微架构ocean cove曝光', 1],
    [-1.387326, 'foo', 2],
    [0.814772, 'baz', ' '],     
    [-0.222552, '   ', 4],
    [-1.176781,  'qux', '  '],         
], columns='A B C'.split(), index=pd.date_range('2000-01-01','2000-01-06'))

print(df.replace(r'\s+', np.nan, regex=True))   
print(df) 