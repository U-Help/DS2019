# coding=utf-8
import numpy as np
import pandas as pd
import re
import os
import random

max_title_len = 30
max_content_len = 400

def title_filter(string):   
    brack_en_filt_out = re.compile(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>")
    brack_cn_filt_out = re.compile(u"\\（.*?）|\\【.*?】|\\《.*?》")
    char_filt_in = re.compile(u'[^\u4E00-\u9FA5]')
    date_filt_out = re.compile(u'年|月|日|十|百|千|万|亿|星期|礼拜')
    
    string = brack_en_filt_out.sub(r'', string)
    string = brack_cn_filt_out.sub(r'', string)
    string = char_filt_in.sub(r'', string)
    string = date_filt_out.sub(r'', string)
    
    if len(string) > max_title_len:
        string = string[:max_title_len]
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
        string = string[:max_content_len]
    return string

# read file
train_df=pd.read_csv("Train_DataSet_Clean.csv")
train_label_df=pd.read_csv("Train_DataSet_Label_Clean.csv")
test_df=pd.read_csv("Test_DataSet.csv")

# merge label and data
train_df=train_df.merge(train_label_df,on='id',how='left')

# eliminate data whose label is nan
train_df['label']=train_df['label'].fillna(-1)
train_df=train_df[train_df['label']!=-1]
train_df['label']=train_df['label'].astype(int)

test_df['label']=0

# fill data with '无' whose content/title is nan
test_df.title.replace(r'\s+', np.nan, regex=True)
test_df.content.replace(r'\s+', np.nan, regex=True)
train_df.title.replace(r'\s+', np.nan, regex=True)
train_df.content.replace(r'\s+', np.nan, regex=True)

test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

# filt
train_df['title'] = train_df['title'].apply(title_filter);
train_df['content'] = train_df['content'].apply(content_filter);

test_df['title'] = test_df['title'].apply(title_filter);
test_df['content'] = test_df['content'].apply(content_filter);

# fill data with '无' whose content/title is nan again
test_df.title.replace(r'\s+', np.nan, regex=True)
test_df.content.replace(r'\s+', np.nan, regex=True)
train_df.title.replace(r'\s+', np.nan, regex=True)
train_df.content.replace(r'\s+', np.nan, regex=True)

test_df['content']=test_df['content'].fillna('无')
train_df['content']=train_df['content'].fillna('无')
test_df['title']=test_df['title'].fillna('无')
train_df['title']=train_df['title'].fillna('无')

# divide data

index=set(range(train_df.shape[0]))
K_fold=[]
for i in range(5):
    if i == 4:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/5*train_df.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)


for i in range(5):
    print("Fold",i)
    os.system("mkdir data_{}_filt".format(i))
    dev_index=list(K_fold[i])
    train_index=[]
    for j in range(5):
        if j!=i:
            train_index+=K_fold[j]
    train_df.iloc[train_index].to_csv("data_{}_filt/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("data_{}_filt/dev.csv".format(i))
    test_df.to_csv("data_{}_filt/test.csv".format(i))
