# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import jieba
#训练支持向量机模型
from sklearn import svm

data_path = '../CCF-BDCI-Sentiment-Analysis-Baseline-master/data/data_test/'

#处理训练集，将训练集的文本信息和label信息合并，清洗特殊符合，同时将文本内容进行分词
def merge_feature_label(feature_name,label_name):
    feature=pd.read_csv(feature_name,sep=",").head[20]
    label=pd.read_csv(label_name,sep=",").head[20]
    data=feature.merge(label,on='id')
    data["X"]=data[["title","content"]].apply(lambda x:"".join([str(x[0]),str(x[1])]),axis=1)
    dataDropNa=data.dropna(axis=0, how='any')
    print(dataDropNa.info())
    dataDropNa["X"]=dataDropNa["X"].apply(lambda x: str(x).replace("\\n","").replace(".","").replace("\n","").replace("　","").replace("↓","").replace("/","").replace("|","").replace(" ",""))
    dataDropNa["X_split"]=dataDropNa["X"].apply(lambda x:" ".join(jieba.cut(x)))
    return dataDropNa
 
dataDropNa=merge_feature_label(data_path+"train.csv","Train_DataSet_Label.csv")

#处理测试数据
def process_test(test_name):
    test=pd.read_csv(test_name,sep=",").head[20]
    test["X"]=test[["title","content"]].apply(lambda x:"".join([str(x[0]),str(x[1])]),axis=1)
    print(test.info())
    test["X"]=test["X"].apply(lambda x: str(x).replace("\\n","").replace(".","").replace("\n","").replace("　","").replace("↓","").replace("/","").replace("|","").replace(" ",""))
    test["X_split"]=test["X"].apply(lambda x:" ".join(jieba.cut(x)))
    return test
 
testData=process_test("Test_DataSet.csv")

#获取文本内容的tf-idf表示
xTrain=dataDropNa["X_split"]
xTest=testData["X_split"]
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
xTrain_tfidf = vec.fit_transform(xTrain)
xTest_tfidf = vec.transform(xTest)
yTrain=dataDropNa["label"]
 
#训练逻辑回归模型
clf = LogisticRegression(C=4, dual=True)
clf.fit(xTrain_tfidf, yTrain)
 
#预测测试集，并生成结果提交
preds=clf.predict_proba(xTest_tfidf)
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["label"]
print(test_pred.shape)
test_pred["id"]=list(testData["id"])
test_pred[["id","label"]].to_csv('sub_lr_baseline.csv',index=None)

lin_clf = svm.LinearSVC()
lin_clf.fit(xTrain_tfidf, yTrain)
 
#预测测试集，并生成结果提交
preds=lin_clf.predict(xTest_tfidf)
test_pred=pd.DataFrame(preds)
test_pred.columns=["label"]
test_pred["id"]=list(testData["id"])
test_pred[["id","label"]].to_csv('sub_svm_baseline.csv',index=None)