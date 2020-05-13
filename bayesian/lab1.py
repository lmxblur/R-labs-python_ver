import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
sns.set()
def draw_samples_beta(n,a,b):
    output = np.random.beta(a, b,size=n)
    return output
n = np.arange(10,10000,10)
mean = np.array([])
std = np.array([])
true_std = np.sqrt((17*7)/((24**2)*25))
for i in n:
    samples = draw_samples_beta(i,7,17)
    mean = np.append(mean,np.mean(samples))
    std = np.append(std,np.std(samples))
plt.figure()
plt.plot(n,mean,alpha = 0.7,label='simulated')
plt.hlines(7/(7+17), min(n), max(n),label = 'true')
plt.legend(loc=0)
plt.title('Simuated mean and true mean')
plt.show()
plt.close()

plt.figure()
plt.plot(n,std,alpha = 0.7,label='simulated')
plt.hlines(true_std, min(n), max(n),label = 'true')
plt.legend(loc=0)
plt.show()
plt.close()

## b.
prob_samples = draw_samples_beta(10000,7,17)
estimation = sum(prob_samples>0.3)/len(prob_samples)
true_prob = 1- beta.cdf(0.3,7,17)
print(f'The estimation errrosr is {abs(estimation-true_prob)/true_prob}')

## c.
log_odds_sample = draw_samples_beta(10000, 7, 17)
transform_sample = np.log(log_odds_sample/(1-log_odds_sample))
plt.figure()
sns.distplot(transform_sample,bins=50)
#sns.kdeplot(transform_sample)
plt.show()
plt.close()

## Question 2
## a.
mu = 3.7
y = np.array([44,25,45,52,30,63,19,50,34,67])
tau_2 = sum((np.log(y)-mu)**2)/len(y)

from scipy.stats import chi2
chisq_sample = chi2.rvs(len(y),size = 10000)
inverse_chisq_sample = (len(y)*tau_2)/chisq_sample
from scipy.special import gamma
def theoretical_inverse_chi (n,tau,x):
    output = (((tau*(n/2))**(n/2))/(gamma(n/2)))*((np.exp((-n*tau)/(2*x)))/(x**(1+n/2)))
    return output

x_seq = np.arange(0.01,2,0.02)
plt.figure()
sns.kdeplot(inverse_chisq_sample,label = 'Estimated')
plt.plot(x_seq,theoretical_inverse_chi(len(y),tau_2,x_seq),label='True')
plt.legend(loc=0)
plt.show()
from scipy.stats import norm
## b.
def G_func(sigma):
    output = 2*norm.cdf(np.sqrt(sigma)/np.sqrt(2))-1
    return output

plt.figure()
sns.distplot(G_func(inverse_chisq_sample))
plt.show()
plt.close()

## c.

## Question 3
## a.
#from scipy.special import yn
from mpmath import besseli
wind_radian = np.array([-2.44,2.14,2.54,1.83,2.02,2.33,-2.79,2.23,2.07,2.02])
mu = 2.39
def posterior_func(k,y,u):
    output = (1/(2*np.pi*besseli(0,k)))**len(y) * np.exp(k*(sum(np.cos(y-mu))-1))
    return output
k_seq = np.arange(0,7,0.01)
value = np.array([])
for k_si in k_seq:
    value = np.append(value,posterior_func(k_si,wind_radian,mu))
value = value.astype(float)
plt.figure()
plt.plot(k_seq,value)
plt.plot(k_seq[np.where(value == max(value))],max(value),marker='o',markersize=3,color='red')
plt.show()
plt.close()
