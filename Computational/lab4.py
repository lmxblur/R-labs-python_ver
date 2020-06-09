import numpy as np
from scipy.stats import lognorm,chi2
import matplotlib.pyplot as plt
import pandas as pd
def first_pdf(x):
    if x<0:
        output = 0
    else:
        output = (x**5)*np.exp(-x)
    return output 

def MHA(n,x0,props):
    x = np.repeat(x0,n)
    for i in np.arange(1,n):
        current = x[i-1]
        Y = lognorm([props],scale=np.exp(current)).rvs(size=1)
        criteria = (first_pdf(Y)*lognorm.pdf(current,props,scale=np.exp(Y)))/(first_pdf(current)*lognorm.pdf(Y,props,scale=np.exp(current)))
        alpha = np.min((1,criteria))
        if np.random.random(1) <= alpha:
            x[i] = Y
        else:
            x[i] = current
    
    return x

#log normal pdf conversion to the r
#https://stats.stackexchange.com/questions/33036/fitting-log-normal-distribution-in-r-vs-scipy


data = MHA(10000,1.5,1)
plt.figure()
plt.hist(data,bins=50,density=True)
plt.show()
plt.close()

plt.figure()
plt.plot(data)
plt.show()
plt.show()

def function_mh_chi(nstep,x0):
    x_ser = np.repeat(x0,nstep)
    for i in np.arange(1,nstep):
        X = x_ser[i-1]
        Y = chi2(np.floor(X)).rvs(size=1)
        uni = np.random.random(1)
        criteria = (first_pdf(Y)*chi2.pdf(X,int(np.floor(Y+1))))/(first_pdf(X)*chi2.pdf(Y,int(np.floor(X+1))))
        alpha = np.min((1,criteria))
        if uni <= alpha:
            x_ser[i] = Y
        else:
            x_ser[i] = X
    return x_ser

data_2 = function_mh_chi(1000,2.95)
plt.figure()
plt.plot(data_2)
plt.show()
plt.close()

plt.figure()
plt.hist(data_2,bins=50)
plt.show()
plt.close()

chemical = pd.read_csv('chemical.csv')
def samples_gibbs(Y,mu0,sigma2,N=1000):
    k=50
    res = np.zeros((N+1,k))
    res[0,:] = mu0
    for i in np.arange(1,N+1):
        res[i,0] = np.random.normal((Y[0]+res[i-1,1])/2,np.sqrt(sigma2/2),1)
        for j in np.arange(1,k-2):
            res[i,j] = np.random.normal((Y[j]+res[i-1,j+1])/2,np.sqrt(sigma2/2),1)
        res[i,k-1] = np.random.normal((Y[j]+res[i,j-1]+res[i-1,j+1])/3, np.sqrt(sigma2/3),1)
    return res

y_res = samples_gibbs(chemical.Y, np.zeros(50), sigma2=0.2)
            
        
