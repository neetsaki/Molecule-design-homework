# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:22:03 2019

@author: neetsaki
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import scipy.io as scio
from sklearn.metrics import mean_squared_error, r2_score
import random
def regression(n,m):
    data = scio.loadmat(str(n)+"-substituted.mat")
    reg = linear_model.LinearRegression()
    X=data['X']
    X_o=data['X']
    y=data['y']
    y_o=data['y']
    seed1=random.randint(0,len(y)-1)
    seed2=random.randint(0,len(y)-1)
    X=np.delete(X,seed1,0)
    y=np.delete(y,seed1,0)
    X=np.delete(X,seed2,0)
    y=np.delete(y,seed2,0)
    #reg.fit(X,y)
    #X=np.delete(a,m,axis=0)
    #y=np.delete(b,m,axis=0)
    reg.fit(X,y)
    y_pred0= reg.predict(X)
    y_pred = reg.predict([X_o[seed1],X_o[seed2]])
    
    #y_pred_o=np.delete(y_pred,m,axis=0)
    
    #LinearRegression(copy_X=True, fit_intercept=True,n_jobs=None,normalize=False)
    # The coefficients
    print('Coefficients: \n', reg.coef_)
    #intercept
    print('Intercept: \n', reg.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"% mean_squared_error(y,y_pred0))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y,y_pred0))
    # Plot outputs
    plt.xlim(min(y),max(y))
    plt.ylim(min(y),max(y))
    plt.scatter(y,y_pred0,  color='#64363C')
    plt.scatter([y_o[seed1],y_o[seed2]],y_pred,  color='#91AD70')
    plt.plot(y,y, color='#EECC88', linewidth=3)
    plt.xlabel(r'$\log(\frac{1}{C_{50}})$')
    plt.ylabel(r'$\log(\frac{1}{C_{50}})_{predict}$')
    return seed1+1,seed2+1,[y_o[seed1],y_o[seed2]],y_pred
n=input("input regression molecule number(2~4): ")
#regression(n,5)
print(regression(n,5))
#print(np.delete(regression(n,5),1,0))