# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:21:16 2019

@author: qth
"""
import jieba
#处理训练集，将训练集的文本信息和label信息合并，清洗特殊符合，同时将文本内容进行分词
def merge_feature_label(feature_name,label_name):
    feature=pd.read_csv(feature_name,sep=",")
    label=pd.read_csv(label_name,sep=",")
    data=feature.merge(label,on='id')
    data["X"]=data[["title","content"]].apply(lambda x:"".join([str(x[0]),str(x[1])]),axis=1)
    dataDropNa=data.dropna(axis=0, how='any')
    print(dataDropNa.info())
    dataDropNa["X"]=dataDropNa["X"].apply(lambda x: str(x).replace("\\n","").replace(".","").replace("\n","").replace("　","").replace("↓","").replace("/","").replace("|","").replace(" ",""))
    dataDropNa["X_split"]=dataDropNa["X"].apply(lambda x:" ".join(jieba.cut(x)))
    return dataDropNa
 
dataDropNa=merge_feature_label("Train_DataSet.csv","Train_DataSet_Label.csv")
print(dataDropNa.loc[1])