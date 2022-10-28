# -*- coding: utf-8 -*-
# DO NOT CHANGE
import numpy as np
from itertools import product
from sklearn.svm import OneClassSVM
from scipy.sparse.csgraph import connected_components
import pandas as pd
import matplotlib.pyplot as plt

def get_adj_mat(X,svdd,num_cut):
    # svdd: trained svdd model by sci-kit learn using X
    # num_cut: number of cutting points on line segment 점과 점사이 몇번 check할거냐
    #######OUTPUT########
    # return adjacent matrix size of n*n (if two points are connected A_ij=1)
    svdd.fit(X)
    unbound_sv = svdd.support_[svdd.dual_coef_[0] < 1] # 경계 unbounded support vector
    non_sv = np.arange(len(X))[np.isin(np.arange(len(X)), svdd.support_) == False] # non support vector. 전체에서 sv인거 빼주기
    ij_pairs = list(product(unbound_sv, non_sv)) # 경계 안(non sv)인 애들이랑 경계에 있는(unbound sv)의 pair들
    
    #인접행렬 생성
    adj_mat = np.zeros((len(X),len(X))) # len(X)*len(X)인 0으로 채워진 matrix 생성!
    for pair in ij_pairs:
        x1, y1 = X.iloc[:,0][pair[0]], X.iloc[:,1][pair[0]] #pair의 첫번째 sv 점
        x2, y2 = X.iloc[:,0][pair[1]], X.iloc[:,1][pair[1]] #pair의 두번째 sv 점
        checking = np.c_[
            np.linspace(x1, x2, num_cut), # x1,x2 num_cut만큼 잘게 나눔..
            np.linspace(y1, y2, num_cut)  # y1,y2 num_cut만큼 잘게 나눔..
            ]
        if sum(svdd.predict(checking[1:-1])) == num_cut-2: # 잘게 나눈 값들이 모두 경계 내에 있을 경우! 같은군집
            adj_mat[pair[0]][pair[1]] = 1 # 그런애들은 1로 채워준
            adj_mat[pair[1]][pair[0]] = 1
    
    return adj_mat
    
    
    
    
def cluster_label(A,bsv): # 자기들끼리 연결 찾아 cluster 찾아주면 될거같다.
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors >> adj_mat에 없는! 아무것과도 연결되지않는,, 제외하고(bounded sv), 벗어난!
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components > 그래프가 주어졌을 때, connected component 어떻게되는지,,
    labels = connected_components(A)[1]
    labels[bsv] = -1 # bounded support vectors, label=-1
    
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1) 조건 반영해야함.
    
    labels_dict = {-1:-1} # 위의 조건 반영해서 라벨 이름 다시 지어줌 ex) 기존 이름 m : 새 이름 n 
    i = 0
    for label in list(set(labels)-{-1}):
        labels_dict[label] = i
        i += 1
    labels_sort = np.array([labels_dict[label] for label in labels])
    
    return labels_sort
    
    

ring=pd.read_csv('https://drive.google.com/uc?export=download&id=1_ygiOJ-xEPVSIvj3OzYrXtYc0Gw_Wa3a')
num_cut=20
svdd=OneClassSVM(gamma=1, nu=0.2)

##########Plot1###################
# Get SVG figure (draw line between two connected points with scatter plots)
# draw decision boundary
# mark differently for nsv, bsv, and free sv

# svdd.fit(ring)

A = get_adj_mat(ring,svdd, num_cut)


xmin, xmax = ring.iloc[:, 0].min() - 0.5, ring.iloc[:, 0].max() + 0.5
ymin, ymax = ring.iloc[:, 1].min() - 0.5, ring.iloc[:, 1].max() + 0.5
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
Z = np.c_[X.ravel(), Y.ravel()]
Z_pred = svdd.decision_function(Z)


plt.figure(figsize=(12,10))
plt.contour(  # boundary
    X,
    Y,
    Z_pred.reshape(X.shape),
    levels=[0],
    lw=2,
    colors='k'
    )
plt.scatter(  # unbounded
    ring.iloc[:, 0][svdd.support_[svdd.dual_coef_[0] < 1]],
    ring.iloc[:, 1][svdd.support_[svdd.dual_coef_[0] < 1]],
    facecolors='none',
    edgecolors='r',
    
    )
plt.scatter(  # bounded
    ring.iloc[:, 0][svdd.support_[svdd.dual_coef_[0] == 1]],
    ring.iloc[:, 1][svdd.support_[svdd.dual_coef_[0] == 1]],
    marker='x',
    c='b',
    
    )
plt.scatter(  # Non Support Vector
    ring.iloc[:, 0][np.isin(np.arange(len(ring)), svdd.support_) == False],
    ring.iloc[:, 1][np.isin(np.arange(len(ring)), svdd.support_) == False],
    marker='o',
    c='k',
    
    )

for x in range(len(A)):
    for y in range(x,len(A)):
        if A[x][y] == 1:
            plt.plot([ring.iloc[:, 0][x],ring.iloc[:, 0][y]], [ring.iloc[:, 1][x],ring.iloc[:, 1][y]] , 'k')

##########Plot2###################
# Clsuter labeling result
# different clusters should be colored using different color
# outliers (bounded support vectors) are marked with 'x'


bound_sv = svdd.support_[svdd.dual_coef_[0] == 1] # bounded sv
labels = cluster_label(A, bound_sv)


plt.figure(figsize=(12, 10))
plt.contour(
    X,
    Y,
    Z_pred.reshape(X.shape),
    levels=[0],
    lw=2,
    colors='k'
    ) # boundary
plt.scatter(
    ring.iloc[:, 0][labels != -1], 
    ring.iloc[:, 1][labels != -1],
    c=labels[labels != -1] 
    ) 
plt.scatter(
    ring.iloc[:, 0][labels == -1], # bounded sv
    ring.iloc[:, 1][labels == -1],
    marker='x', c='b'
    )






### memo ###
'''
dual coef
1이 벗어난 애들
그 이하 : 경계(boundary있는애들)

predict(ring)
1, -1
'''