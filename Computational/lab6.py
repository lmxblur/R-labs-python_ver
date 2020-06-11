import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def objective(x):
    output = (x**2/np.exp(x))-2*np.exp((-9*np.sin(x))/(x**2 + x+1))
    return output
def crossover(x,y):
    output = (x+y)/2
    return output
def mutate(x):
    output = x**2 % 30
    return output 
def rungenetic(maxiter,mutprob,skip=1):
    xplot = np.arange(0,30,0.02)
    yplot = objective(xplot)
    plt.figure()
    plt.plot(xplot,yplot)
    plt.plot(xplot[np.where(yplot==max(yplot))][0],max(yplot),marker='o',markersize=3,color='red')
    plt.show()
    plt.close()
    
    X = np.arange(0,31,5)
    values = objective(X)
    best_value = -1e10
    for i in range(maxiter):
        parent_indices = np.random.choice([1,2,3,4,5,6],size=2,replace=False)
        parent_x = X[parent_indices]
        parent_value = values[parent_indices]
        # argsort equivalent of order in R
        victim_index = np.argsort(values)[0]
        victim_x = X[victim_index]
        victim_value = values[victim_index]
        kid_index = victim_index
        kid_x = crossover(parent_x[0],parent_x[1])
        if np.random.random(1) < mutprob:
            kid_x = mutate(kid_x)
            
        kid_value = objective(kid_x)
        plt.figure()
        plt.plot(xplot,yplot)
        plt.plot(X,values,marker='o')
        plt.plot(parent_x,parent_value,marker='o')
        plt.plot(victim_x,victim_value,marker='o')
        plt.plot(kid_x,kid_value,marker='o')
        plt.show()
        plt.close()
        
        X[victim_index] = kid_x
        values[victim_index] = kid_value
        
        if best_value <max(values):
            best_index = np.argmax(values)
            best_x = X[best_index]
            best_value = values[best_index]
    
    
    plt.figure()
    plt.plot(xplot,yplot)
    plt.plot(best_x,best_value,marker='o',markersize=4)
    plt.show()
    plt.close()
    
    
rungenetic(100,0.85)


physical = pd.read_csv('physical1.csv')

plt.figure()
plt.plot(physical['X'],physical['Z'],color='red')
plt.plot(physical['X'],physical['Y'],color='blue')
plt.show()
plt.close()

def lambdaest(n,lambda_k,X,Z,Y,L,not_missing):
    output = 1/(2*n)*(sum(X*Y)+1/2*sum(X[not_missing]*Z[not_missing])+L*lambda_k)
    return output


def EM_exp(threshold,data,initial,kstep):
    lambda0 = initial
    logli_old = 0
    stop_indicator = 0
    n = data.shape[1]
    n_NA = sum(np.isnan(data['Z']))
    full_index = np.where(np.invert(np.isnan(data['Z'])))[0]
    k =1
    while(stop_indicator != 1):
        lambda1 = lambdaest(n=n,lambda_k=lambda0,X=data['X'], Y=data['Y'],Z=data['Z'],L=n_NA, not_missing = full_index)
        logli_new = 2*np.log(np.prod(data['X']))-(n*np.log(2)+2*n*np.log(lambda0)) - (1/lambda0
            )*sum(data['X']*data['Y'])- (1/(2*lambda0))* sum(
                data['X'][full_index]*data['Z'][full_index])-n_NA*lambda1/lambda0
        lambda0 = lambda1
        if abs(logli_old-logli_new) < threshold or k > kstep:
            break
        k += 1
        logli_old = logli_new
        return [lambda0,logli_new,k]
EM_run = EM_exp(0.0001,physical,100,100)
lambda_opti = EM_run[0]

plt.figure()
plt.plot(physical['X'],physical['Y'],label='Y')
plt.plot(physical['X'],physical['Z'],label='Z')
plt.plot(physical['X'],lambda_opti/physical['X'],label='Y_')
plt.plot(physical['X'],(2*lambda_opti)/physical['X'],label='Z_')
plt.legend(loc=0)
plt.show()
plt.close()

plt.figure()
plt.plot(physical['X'],physical['Z']-(2*lambda_opti)/physical['X'])
plt.plot(physical['X'],physical['Y']-lambda_opti/physical['X'])
plt.show()
plt.close()
