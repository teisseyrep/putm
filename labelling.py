import numpy as np
    
def make_pu_labels(X,y,c=0.5,prob_true=None,label_scheme='scar',k=1):
    
    n = X.shape[0]
    s = np.zeros(n)
    ex_true = np.zeros(n)    
    
    if label_scheme=='scar':
        for i in np.arange(0,n,1):
            ex_true[i] = c
    elif label_scheme=='prop1':
        for i in np.arange(0,n,1):
            if X[i,0]<1:
                ex_true[i] = 0.1
            if X[i,0]>1:
                ex_true[i]=0.95
    elif label_scheme=='prop2':
        if any(prob_true)==None:
            raise Exception('Argument prob_true should be specified')
        for i in np.arange(0,n,1):
            ex_true[i] = k*prob_true[i]
    else:
        print('Argument label_scheme is not defined')
                    
    for i in np.arange(0,n,1):
        if y[i]==1:
            s[i]=np.random.binomial(1, ex_true[i], size=1)
        
    return s, ex_true
    


