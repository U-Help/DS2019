# -*- coding: utf-8 -*-
# TF-IDF: https://www.wikiwand.com/en/Tf%E2%80%93idf
import pandas as pd, numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
import time
import thulac

def feature_scores(seg, dictionary):
    c_score, p_score, n_score = 0, 0, 0
    c_count, p_count, n_count = 0, 0, 0
    
    word_count = list(set([(word, seg.count(word)) for word in seg.split(' ')]))
    
    for word, count in word_count:
        if word in dictionary.index:
            information = dictionary.loc[word, ['情感分类', '强度', '极性']] # multiple entries
            if isinstance(information, pd.DataFrame):
                information = information.reset_index().iloc[0][1:]
            if information[2] == 0: #中性
                c_score += information[1] * count
                c_count += count
            elif information[2] == 1: #褒义
                p_score += information[1] * count
                p_count += count
            elif information[2] == 2: #贬义
                n_score += information[1] * count
                n_count += count
    c_score = 0 if c_count == 0 else c_score/c_count
    p_score = 0 if p_count == 0 else p_score/p_count
    n_score = 0 if n_count == 0 else n_score/n_count
    
    return c_score, p_score, n_score

def cut_list(sentence, model):
    segmentation = model.cut(sentence)
    return ' '.join([word[0] for word in segmentation])

if __name__ == '__main__':
    # category in dictionary, which is somehow useless
    positive = ['PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK', 'PC']
    negative = ['NA', 'NB', 'NJ', 'NH', 'NF', 'NI', 'NC', 'NG', 'NE', 'ND', 'NN', 'NK', 'NL']
    
    data_path = '../CCF-BDCI-Sentiment-Analysis-Baseline-master/data/data_test/'
    
    # using thulac module to divide the sentence
    print("Model loading...")
    thu1 = thulac.thulac(filt=True, seg_only=True)  #默认模式
    df = pd.read_excel('dictionary.xlsx').set_index('词语')
    
    # read data
    train_df=pd.read_csv(data_path+"train.csv").head(2000)
    test_df=pd.read_csv(data_path+"train.csv")[3000:3100]
    train_df['concat']=train_df[['title', 'content']].apply(''.join, axis=1).\
        apply(cut_list, model=thu1)
    test_df['concat']=test_df[['title', 'content']].apply(''.join, axis=1).\
        apply(cut_list, model=thu1)
    # extra features - c/p/n_scores
    print("Extracting feature scores...")
    scores_train = sp.csr_matrix(list(train_df['concat'].apply(feature_scores, dictionary=df))) 
    labels_train = train_df["label"]
    scores_test =sp.csr_matrix(list(test_df['title'].apply(feature_scores, dictionary=df)))
    labels_test = test_df["label"]
    
    #获取文本内容的tf-idf表示
    #    xTrain=train_df["concat"]
    #    xTest=testData["X_split"]
    #min_df=3, max_df=0.9
    vec = TfidfVectorizer(ngram_range=(1,2),use_idf=1,smooth_idf=1, sublinear_tf=1)
    tfidf_train = vec.fit_transform(train_df["concat"])
    tfidf_test = vec.transform(test_df["concat"])
    features_train = sp.hstack([scores_train, tfidf_train])
    features_test = sp.hstack([scores_test, tfidf_test])
    
    #训练逻辑回归模型
    print("Now training...")
    # logistic
#    clf = LogisticRegression(C=4)
#    clf.fit(features_train, labels_train)
    
    # svm with kernel
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svm_model = GridSearchCV(svm.SVC(), params_grid, cv=5)
    svm_model.fit(features_train, labels_train)
    clf=svm_model.best_estimator_
#    clf = svm.SVC()
#    clf.fit(features_train, labels_train)
    
    #预测测试集，并生成结果提交
    print("Now predicting...")
#    preds=clf.predict_proba(features_test)
#    preds=np.argmax(preds,axis=1)
#    test_pred=pd.DataFrame(preds)
#    test_pred.columns=["label"]
#    print(preds)
#    print(list(labels_test))
    
#    test_pred["id"]=list(test_df["id"])
#    test_pred[["id","label"]].to_csv('sub_lr_baseline.csv',index=None)
    
#    preds = clf.decision_function(features_train)
    preds = clf.predict(features_test)
    print(preds)
    print(list(labels_test))

"""
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
"""
