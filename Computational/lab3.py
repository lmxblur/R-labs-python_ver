import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace,norm
def De_invCDF(y, mu, alpha):
    output = mu - 1/alpha*np.sign(y-0.5)*np.log(1 +np.sign(y-0.5) - np.sign(y-0.5)*2*y)
    return output 
x = np.linspace(0,1,num=1000)
y = np.array([])
for i in x:
    y = np.append(y,De_invCDF(i, 0, 1))

plt.figure()
plt.plot(x,y)
plt.show()
plt.close()

xfit = np.linspace(-8,8,num=10**3)
yfit = laplace.pdf(xfit,0, 1)
xDE = De_invCDF(np.random.uniform(size=10000), 0, 1)

plt.figure()
plt.hist(xDE,bins=50,density=True,range=(-8,8))
plt.plot(xfit,yfit)

majorizing_c = 1.5

def rnom_accpt(n,mean=0,sd=1):
    laplace_mean = 0
    laplace_scale=1
    xN = np.array([])
    accepted = 0
    total = 0
    while accepted < n:
        YDE = De_invCDF(np.random.uniform(size=1), laplace_mean, laplace_scale)
        u_uni = np.random.uniform(size=1)
        u_cmp = norm.pdf(YDE, mean, sd)/(majorizing_c*laplace.pdf(x=YDE, loc=laplace_mean, scale=laplace_scale))
        if u_uni <= u_cmp:
            xN = np.append(xN,YDE)
            accepted +=1
        total+=1
    rejection_rate = (total-accepted)/total
    return [xN,rejection_rate]

res = rnom_accpt(2000)
x= np.arange(-8,8,step=0.01)
y_laplace = majorizing_c * laplace.pdf(x)
y_norm = norm.pdf(x)
plt.figure()
plt.hist(res[0],density=True,bins=50)
plt.plot(x,y_laplace)
plt.plot(x,y_norm)
plt.show()
plt.close()