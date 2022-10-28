# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    # X: inpute variables
    # y: target
    
    n = X.shape[0]
    p = X.shape[1]
    
    y_true = y.copy()
    
    # y_pred 계산과정
    
    # 절편항 추가
    X = np.c_[np.ones(n),X]
    
    XTX = np.matmul(np.transpose(X),X)
    inv_XTX = np.linalg.inv(XTX)
    
    # beta 계산
    beta = np.matmul(np.matmul(inv_XTX,np.transpose(X)),y_true)

    y_pred = np.matmul(X, beta)
    
    # SSE, SSR, SST 계산
    SSE = sum((y_true - y_pred)**2)
    SSR = sum((y_pred - np.mean(y_pred))**2)
    SST = sum((y_true - np.mean(y_pred))**2)
    # MSR, MSE 계산
    MSR = SSR / p
    MSE = SSE / (n-p-1)
    
    # f-value 계산
    f = MSR/MSE
    
    # p-value 계산 (p : MSR의 자유도, n-p-1 : MSE의 자유도)
    pv = 1 - stats.f.cdf(f,p,n-p-1)
    
    print('-'*75)
    print('Factor         SS           DF           MS        F-value        Pr>F')
    print("Model    {:>10.4f}    {:>7}    {:>10.4f}    {:>10.4f}    {:>10.4f}".format(SSR, p, MSR, f, pv))
    print("Error    {:>10.4f}    {:>7}    {:>10.4f}".format(SSE, n-p-1, MSE))
    print('-'*75)
    print('Total    {:>10.4f}    {:>7}'.format(SST, n-1))
    print('-'*75)
    return 0

def ttest(X,y,varname=None):
    # X: inpute variables
    # y: target
    
    n = X.shape[0]
    p = X.shape[1]
    
    y_true = y.copy()
    
    # y_pred 계산과정
    
    # 절편항 추가
    X = np.c_[np.ones(n),X]
    
    XTX = np.matmul(np.transpose(X),X)
    inv_XTX = np.linalg.inv(XTX)
    
    # beta 계산
    beta = np.matmul(np.matmul(inv_XTX,np.transpose(X)),y_true)

    y_pred = np.matmul(X, beta)
    
    # SSE 계산
    SSE = sum((y_true - y_pred)**2)
    
    # t값 계산과정
    MSE = SSE / (n-p-1)
    
    se2_beta = MSE * np.diag(inv_XTX) # MSE * (inv_XTX의 대각값)   
    se_beta = np.sqrt(se2_beta)
    
    t = beta / se_beta
    
    pv = 2 * (1-stats.t.cdf(np.abs(t), n-p-1)) # 양측검정
    
    features = np.append('Const', varname)
    print('-'*55)
    print('Variable       coef         se        t       Pr>|t|')
    for i in range(p+1):
        print('{:^8}   {:>8.4f}   {:>8.4f}   {:>8.4f}   {:>8.4f}'.format(features[i],float(beta[i]),float(se_beta[i]),float(t[i]),float(pv[i]) ))
    print('-'*55)
    return 0

## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)
