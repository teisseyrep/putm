# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:57:46 2023

@author: teiss
"""
import os



from putm import PUbasic, PUtm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Set classifier
clf = LogisticRegression()
clf_ex = LogisticRegression()

# Load example data:
df = pd.read_csv('data/Breast-w.csv', sep=',')
del df['BinClass']
df = df.to_numpy()
p = df.shape[1]-1
Xall = df[:,0:p]
yall = df[:,p]
model_oracle = LogisticRegression()
model_oracle.fit(Xall,yall)
prob_true=model_oracle.predict_proba(Xall)[:,1]
    

#Perform train/test split:                
X, Xtest, y, ytest = train_test_split(Xall, yall, test_size=0.25, random_state=123)

#Create PU dataset:
s = np.zeros(X.shape[0])
for i in np.arange(0,X.shape[0],1):
    if y[i]==1:
        s[i]=np.random.binomial(1, 0.5, size=1)

#Run the methods:

#Naive method (treating unlabeled examples as positive):
model = PUbasic(clf)
model.fit(X,s)
prob_y_test = model.predict_proba(Xtest)[:,1]
acc_naive = accuracy_score(ytest, np.where(prob_y_test>0.5,1,0))


#Oracle method (based on true class variable):
model= PUbasic(clf)
model.fit(X,y)
prob_y_test = model.predict_proba(Xtest)[:,1]
acc_oracle = accuracy_score(ytest, np.where(prob_y_test>0.5,1,0))


#Proposed method (Two models):
model = PUtm(clf,clf_ex,epochs=100) 
model.fit(X,s)
prob_y_test = model.predict_proba(Xtest)[:,1]
acc_tm = accuracy_score(ytest, np.where(prob_y_test>0.5,1,0))

#Print accuracy of the models:
print("ORACLE method accuracy:" ,acc_oracle)    
print("NAIVE method accuracy:" ,acc_naive)    
print("TM method accuracy:" ,acc_tm)    
