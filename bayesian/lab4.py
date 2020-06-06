import pystan
import numpy as np
import pandas as pd
def arfunc(mu,phi,sigma2,T):
    output = np.array([mu])
    for i in np.arange(1,T):
        generated = mu - phi * (output[i-1]-mu) + np.random.normal(0,np.sqrt(sigma2),size=1)
        output = np.append(output,generated)
    return output
phi_seq = np.arange(-1,1,0.1)
mu=10
sigma2 = 2
T =200
realization = np.ndarray((T,len(phi_seq)))
for i in range(len(phi_seq)):
    realization[:,1] = arfunc(mu,phi_seq[i],sigma2,T)
x_t = arfunc(mu,0.3,sigma2,T)
y_t = arfunc(mu,0.95,sigma2,T)

armodel = """
data {
  int<lower=0> T;
  vector[T] x;
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma2;
}
model {
mu ~ normal(0,10);
phi ~ normal(0,100);
sigma2 ~ scaled_inv_chi_square(1,2);
  for (n in 2:T){
    x[n] ~ normal(mu + phi*(x[n-1]-mu),sqrt(sigma2));
  }
}
"""

sm = pystan.StanModel(model_code=armodel)
data = {'x':x_t,'T':T}
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=1000)
paraneters = fit.extract()
print(np.quantile(paraneters['mu'],q=[0.025,0.975]))
print(np.quantile(paraneters['phi'],q=[0.025,0.975]))
print(np.quantile(paraneters['sigma2'],q=[0.025,0.975]))
fit.plot()

campy = pd.read_csv('campy.csv')

armodel2 = """
data {
  int<lower=0> T;
  int<lower=0> c[T];
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma2;
  vector[T] x;
}
model {
mu ~ normal(0,10);
phi ~ normal(0,100);
sigma2 ~ scaled_inv_chi_square(1,2);
  for (n in 2:T){
    x[n] ~ normal(mu + phi*(x[n-1]-mu),sqrt(sigma2));
  }
  for (i in 1:T){
  c[i] ~ poisson(exp(x[i]));
  }
}
"""
sm = pystan.StanModel(model_code=armodel2)
data = {'T':T,'c':campy.c}
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=1000)
fit.plot()
paraneters = fit.extract()