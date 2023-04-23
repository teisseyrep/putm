# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:02:24 2023

@author: teiss
"""

import numpy as np


def sigma(x):
    res = np.exp(x)/(1+np.exp(x))
    return res

  
def prepare_weighted_pu_data(X,s,ex,sx): 
    w0 = np.where(s==0)[0]
    w1 = np.where(s==1)[0]
    n1 = w1.shape[0]
    n0 = w0.shape[0]
    X0 = X[w0,:]
    X1 = X[w1,:]
    stemp = np.concatenate((np.repeat(1,n1),np.repeat(1,n0),np.repeat(0,n0)))
    Xtemp = np.concatenate((X1,X0,X0),axis=0)
    
    weights1 = np.repeat(1, n1)
    weights2 = ( (1-ex[w0])/ex[w0] ) *( sx[w0] / (1-sx[w0]) )
    weights3 = 1 - weights2
    weights = np.concatenate((weights1,weights2,weights3))
    return Xtemp, stemp, weights    


