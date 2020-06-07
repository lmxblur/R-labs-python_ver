import numpy as np
import pandas as pd
from scipy.optimize import minimize
mortality = pd.read_csv('mortality_rate.csv')
n = mortality.shape[0]
id_index = np.random.choice(np.arange(0,n),size=np.floor(n*0.5).astype(int))
train = mortality.iloc[id_index,:]
test = mortality.iloc[np.setdiff1d(np.arange(0,n),id_index),:]
dat = np.loadtxt('data.txt')
solution_mu = np.mean(dat)
def sigma_solu(x):
    output = np.sqrt(sum((x-np.mean(x))**2)/len(x))
    return output
solution_sigma = sigma_solu(dat)

def log_liki(param,x):
    output = -1*(np.log(1/(param[1]*np.sqrt(2*np.pi))**100)-(1/(2*param[1]**2))*sum((x-param[0])**2))
    return output 


cg_method_gradient = minimize(log_liki,args=(dat), x0=[0,1],method='CG')
bfgs_method = minimize(log_liki,args=(dat), x0=[0,1],method='BFGS')

def log_liki_gradient(param,x):
    n = len(x)
    mu_gradient = -1*(sum(x-param[0])/param[1]**2)[0]
    sigma_gradient = -1*( -1*n/param[1] + sum((x-param[0])**2)/param[1]**3)[0]
    return [mu_gradient,sigma_gradient]

