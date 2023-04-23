import numpy as np
from sklearn.base import BaseEstimator
from utils import prepare_weighted_pu_data


class PUbasic(BaseEstimator):
   
    def __init__(self, clf):
        self.clf = clf
    def fit(self, X, y):
        self.clf.fit(X,y)
        return self

    def predict(self, X):
        return self.clf.predict()
        
    def predict_proba(self, Xtest):
        return self.clf.predict_proba(Xtest)    
        
    
class PUtm(BaseEstimator):
   
    def __init__(self, clf,clf_ex,epochs=100):
        self.clf = clf
        self.clf_ex = clf_ex
        self.epochs = epochs
    
        
    def fit(self, X, s):
        
        model_naive = PUbasic(self.clf)
        model_naive.fit(X,s)
        sx = model_naive.predict_proba(X)[:,1]
        
        ex = (sx+1)/2
        
        for i in np.arange(self.epochs):
        # Model for posterior probability:
            Xtemp, stemp, weights = prepare_weighted_pu_data(X,s,ex,sx)
            self.clf.fit(Xtemp,stemp,sample_weight=weights)
            yx = self.clf.predict_proba(X)[:,1]
            
            
            hat_c = np.mean(s) 
            
            yx1 = yx[np.where(s==1)]    
            val_thrs = np.quantile(yx1,q=hat_c)
            sel1 = np.where(yx>val_thrs)    
            sel2 = np.where(s==1)    
            sel = np.union1d(sel1,sel2)    
            
            if sel.shape[0]>0:
                Xsel = X[sel,:]
                ssel = s[sel]
            else:
                Xsel = X
                ssel = s
                
            # Model for propensity score:    
            self.clf_ex.fit(Xsel,ssel)
            ex = self.clf_ex.predict_proba(X)[:,1] 
                
            too_small = np.where(ex<sx)[0]
            if too_small.shape[0]>0:
                ex[too_small] = sx[too_small]
            
            sx = ex*yx 
            
        #Build final model:
        Xtemp, stemp, weights = prepare_weighted_pu_data(X,s,ex,sx)
        self.clf.fit(Xtemp,stemp,sample_weight=weights)
            
            
        return self

    def predict(self, X):
        return self.clf.predict()
        
    def predict_proba(self, Xtest):
        return self.clf.predict_proba(Xtest)        
    
    
    
