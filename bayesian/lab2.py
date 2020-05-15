import numpy as np
import pandas as pd
# Question 1
## a.
temp_lin = np.loadtxt('TempLinkoping.txt',skiprows=1)  
mu_0 = np.array([-10,100,-100])
omega_0 = 1*np.identity(3)
sigma_0 = 1
v_0 = 4

from scipy.stats import chi2, multivariate_normal
def generate_sigma_pri(n):
    chisq_sample = chi2.rvs(v_0,size=n)
    inverse_chisq_sample = (v_0*sigma_0)/chisq_sample
    return inverse_chisq_sample
def estimate_beta_pri(sigma):
    output = multivariate_normal.rvs(mean=mu_0,cov=sigma*np.linalg.inv(omega_0),size=1)
    return output 

import matplotlib.pyplot as plt
def plot_estimate_pri(n):
    sigma_seq = generate_sigma_pri(n)
    plt.figure()
    plt.plot(temp_lin[:,0],temp_lin[:,1])
    for i in range(n):
        beta = estimate_beta_pri(sigma_seq[i])
        y = X@np.transpose(beta)
        plt.plot(temp_lin[:,0],y)
    plt.show()
    plt.close()

# cbind python equivalent np.c_[...]
#https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array/8505658#8505658
X = np.c_[np.ones((len(temp_lin),1)),temp_lin[:,0],temp_lin[:,0]**2]
beta_hat = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@ temp_lin[:,1]
mu_n = np.linalg.inv(np.transpose(X)@X + omega_0) @ (np.transpose(X)@X@beta_hat + omega_0@mu_0) 
omega_n = np.transpose(X) @ X + omega_0
v_n = v_0 + len(temp_lin)
sigma_n = (v_0*sigma_0 + (np.transpose(temp_lin[:,1])@temp_lin[:,1] + np.transpose(mu_0)@omega_0@mu_0 - np.transpose(mu_n)@omega_n@mu_n))/v_n
plot_estimate_pri(10)

def generate_sigma(n):
    chisq_sample = chi2.rvs(v_n,size=n)
    inverse_chisq_sample = (v_0*sigma_n)/chisq_sample
    return inverse_chisq_sample

def estimate_beta(sigma):
    output = multivariate_normal.rvs(mean=mu_n,cov=sigma*np.linalg.inv(omega_n),size=1)
    return output 

def plot_estimate(n):
    sigma_seq = generate_sigma(n)
    plt.figure()
    plt.plot(temp_lin[:,0],temp_lin[:,1])
    for i in range(n):
        beta = estimate_beta(sigma_seq[i])
        y = X@np.transpose(beta)
        plt.plot(temp_lin[:,0],y)
    plt.show()
    plt.close()

plot_estimate(10)

# Question 2
woman_work = pd.read_csv('woman_work.csv').iloc[:,1:]
X_w = np.array(woman_work.iloc[:,1:])
Y_w = np.array(woman_work.iloc[:,0])



def log_function(beta,x,y,mu,sigma):
    n = x.shape[1]
    pred = x@beta
    log_li = sum(y*pred - np.log(1+np.exp(pred)))
    log_prior = multivariate_normal.logpdf(x=beta,mean=np.zeros(n),cov= sigma)
    output = log_prior + log_li
    return -output 

mu = np.zeros(X_w.shape[1])
tau = 10
sigma = tau**2 * np.identity(X_w.shape[1])

initial_value = np.zeros(X_w.shape[1])

from scipy.optimize import minimize, HessianUpdateStrategy, fmin_bfgs
opti_result = minimize(log_function, initial_value,args=(X_w,Y_w,mu,sigma,),method='BFGS')
beta_hat = opti_result.x
# did not converge
post_cov = opti_result.hess_inv



