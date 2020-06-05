import pandas as pd
from scipy.stats import chi2,gamma,multinomial,norm,multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
rain = pd.read_csv('rainfall.csv')
def w_func(sigma):
    output = (n/sigma)/((n/sigma) + (1/(tau_0)))
    return output

def mun_func(sigma):
    w = w_func(sigma)
    output = w*mu_x + (1-w)*mu_0
    return output

def tau_func(sigma):
    output = n/sigma + 1/tau_0
    return 1/output 

def generate_sigma(mu):
    chisq_sample = chi2.rvs(v_n,size=1)
    tau_compute = (v_0*sigma_0 + sum((X-mu)**2))/ (n+v_0)
    inverse_chisq_sample = v_n*tau_compute/chisq_sample
    return inverse_chisq_sample

def generate_samples_gibbs(iteration):
    sigma_ = 1
    gibbsDraws = np.zeros(iteration)
    mu_result = np.array([])
    sigma_result = np.array([])
    for i in range(iteration):
        mu_n = mun_func(sigma_)
        tau_n = tau_func(sigma_)
        mu = np.random.normal(mu_n,np.sqrt(tau_n),1)
        mu_result = np.append(mu_result,mu)
        sigma_ = generate_sigma(mu)
        sigma_result = np.append(sigma_result,sigma_)
        gibbsDraws[i] = np.random.normal(mu,np.sqrt(sigma_),1)
    
    output = pd.DataFrame({'mu':mu_result,'sig':sigma_result})
    return output

v_0 = 1
tau_0 = 10
mu_0 = 20
sigma_0 =10
v_n = v_0 + rain.shape[0]
X = rain.X136
n= rain.shape[0]
mu_x = np.mean(X)
samples = generate_samples_gibbs(1000)
n = np.arange(1,1000+1)
y_deq = np.cumsum(samples.mu)/ np.arange(1,len(samples['mu'])+1)
y_deq_si = np.cumsum(samples.sig)/ np.arange(1,len(samples['sig'])+1)
plt.figure()
plt.plot(n,samples['mu'],label='Traceplot')
plt.plot(n,y_deq,label='Cumsum mean')
plt.legend(loc=0)
plt.show()
plt.close()

plt.figure()
plt.plot(n,samples['sig'],label='Traceplot')
plt.plot(n,y_deq_si,label='Cumsum mean')
plt.legend(loc=0)
plt.show()
plt.close()


""" x = X
# Model options
nComp = 2    # Number of mixture components

# Prior options
alpha = 10*np.repeat(1,nComp) # Dirichlet(alpha)
muPrior = np.repeat(0,nComp) # Prior mean of mu
tau2Prior = np.repeat(10,nComp) # Prior std of mu
sigma2_0 = np.repeat(np.var(x),nComp) # s20 (best guess of sigma2)
nu0 = np.repeat(4,nComp) # degrees of freedom for prior on sigma2

# MCMC options
nIter = 500 # Number of Gibbs sampling draws

# Plotting options
plotFit = True
lineColors = ["blue", "green", "magenta", 'yellow']
sleepTime = 0.1 # Adding sleep time between iterations for plotting
################   END USER INPUT ###############

###### Defining a function that simulates from the 

def rScaledInvChi2(n,df,scale):
    output = (df*scale)/chi2.rvs(df,n)
    return output 
####### Defining a function that simulates from a Dirichlet distribution


def rDirichlet(param):
    nCat = len(param)
    piDraws = np.ndarray((nCat,1))
    for j in range(nCat):
        piDraws[j] = gamma.rvs(a=1,scale=1/param[j],size=1)
    piDraws = piDraws/sum(piDraws)
    return piDraws
def S2alloc(S):
    n = S.shape[0]
    alloc = np.zeros(n)
    for i in range(n):
        alloc[i] = np.where(S[i,:]==1)[1]
    return alloc
# Simple function that converts between two different representations of the mixture allocation

# Initial value for the MCMC
nObs = len(x)
S = multinomial.rvs(size = 1 , p = np.repeat(1/nComp,nComp),n=nObs)
# nObs-by-nComp matrix with component allocations.
mu = np.quantile(x, q = np.linspace(start=0,stop=1,num = nComp))
sigma2 = np.repeat(np.var(x),nComp)
probObsInComp = np.repeat(0, nComp)

# Setting up the plot
xGrid = np.linspace(min(x)-1*np.std(x),max(x)+1*np.std(x),num = 100)
xGridMin = min(xGrid)
xGridMax = max(xGrid)
mixDensMean = np.repeat(0,len(xGrid))
effIterCount = 0

pi_seq = np.ndarray((nIter,nComp))
mu_seq = np.ndarray((nIter,nComp))
sigma_seq = np.ndarray((nIter,nComp))
for k in range(nIter):
  #message(paste('Iteration number:',k))
  alloc = S2alloc(S) # Just a function that converts between different representations of the group allocations
  nAlloc = np.sum(S,axis=0)
  #print(nAlloc)
  # Update components probabilities
  pi = rDirichlet(alpha + nAlloc)
  
  # Update mu's
  for j in range(nComp):
    precPrior = 1/tau2Prior[j]
    precData = nAlloc[j]/sigma2[j]
    precPost = precPrior + precData
    wPrior = precPrior/precPost
    muPost = wPrior*muPrior + (1-wPrior)*np.mean(x[alloc == j])
    tau2Post = 1/precPost
    mu[j] = np.random.normal(muPost,np.sqrt(tau2Post),1)
  
  
  # Update sigma2's
  for j in range(nComp):
    sigma2[j] = rScaledInvChi2(1, df = nu0[j] + nAlloc[j], scale = (nu0[j]*sigma2_0[j] + sum((x[alloc == j] - mu[j])**2))/(nu0[j] + nAlloc[j]))
  
  
  # Update allocation
  for i in range(nObs):
    for j in range(nComp):
      probObsInComp[j] = pi[j]*norm.pdf(x[i], mu[j],np.sqrt(sigma2[j]))
    
    S[i,] = multinomial.rvs(n=1, size = 1 , p = probObsInComp/sum(probObsInComp))
  
  # adding the current mu sigma and pi to the sequence
  pi_seq[k,] = pi
  mu_seq[k,] = mu
  sigma_seq[k,] = sigma2
   """
  
  # change k%%10 ==0 in order to lower the number of plot


ebay = pd.read_csv('ebay.csv')
def log_func(beta,x,y,mu,sigma):
    n = x.shape[1]
    part = beta.T @ x.T
    part = part.reshape(len(part),-1)
    log_li = np.sum(y*part - np.exp(part))
    log_prior = multivariate_normal.logpdf(beta,mean=np.zeros(n),cov=sigma)
    output = log_li + log_prior
    limit = 1e-6
    if np.min(output) < limit:
        output += np.sum(np.minimum(0.0,  output - limit)) / limit
    
    return(-1*output)

ebayx = np.array(ebay.iloc[:,1:])
ebayy = np.array(ebay.iloc[:,0]).astype(np.int).reshape((1000,-1))
mu = np.zeros(ebayx.shape[1])
sigma = 100*np.linalg.inv(ebayx.T@ebayx)
initial_value = np.zeros(ebayx.shape[1]).reshape((ebayx.shape[1],-1))

opti_result = minimize(log_func, initial_value,args=(ebayx,ebayy,mu,sigma,),method='BFGS')
beta_hat = opti_result.x
post_cov = opti_result.hess_inv

def generate_beta (n):
    output = multivariate_normal.rvs(mean=beta_hat,cov=post_cov,size=n)
    return output 

def calculatep(func,theta,**kwargs):
    x = theta
    output = func(theta,**kwargs)
    return output 

iteration = 5000
c = 1
beta_matrix = np.ndarray((iteration+1,ebayx.shape[1]))
# give initial value theta_0
beta_matrix[0,]= np.repeat(1,ebayx.shape[1])
alp_seq = np.zeros(iteration)
for i in np.arange(1,iteration+1):
    beta_minus1 = beta_matrix[i-1,:]
    beta_p =  multivariate_normal.rvs(mean = beta_minus1,cov = c*post_cov,size=1)
    # times -1 because the output is -output soexp( -... and + ...)
    alpha = np.exp(-calculatep(log_func,beta_p,x=ebayx,y=ebayy,mu=mu,sigma=sigma)+calculatep(
        log_func,beta_minus1,x=ebayx,y=ebayy,mu=mu,sigma=sigma))
    alpha = min(1,alpha)
    alp_seq[i-1] = alpha
    rand01 = np.random.random()
    if(rand01 < alpha):
        beta_matrix[i,:] = beta_p
    else:
        beta_matrix[i,:] = beta_minus1
def cumsum_mean(x):
    output = np.cumsum(x)/np.arange(1,len(x)+1)
    return output 
for i in range(beta_matrix.shape[1]):
    plt.figure()
    plt.plot(beta_matrix[2:,i])
    plt.plot(cumsum_mean(beta_matrix[2:,i]))
    plt.show()
    plt.close()
    