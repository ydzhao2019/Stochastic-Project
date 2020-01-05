import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter

def simulate_Brownian_Motion(paths, steps, T):
    deltaT = T/steps
    t = np.linspace(0, T, steps+1)
    X = np.c_[np.zeros((paths, 1)),
              np.random.randn(paths, steps)]
    return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)
def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
def StockPrice_Simulation(steps):
    t, Wt = simulate_Brownian_Motion(50000, steps, 1/12)
    t=np.linspace(0,T,steps+1)
    stockprice=S0*np.exp((r-(sigma**2)/2)*t+sigma*Wt)
    return stockprice

S0=100
K=100
T=1/12
sigma=0.2
r=0.05
N=84
deltaT=T/N

steps=N
Stockprice=StockPrice_Simulation(steps)

error=0
t=np.linspace(0, T, steps+1)
for i in range(N):
    ST=Stockprice[:,i]
    ST_1=Stockprice[:,i+1]
    deltaT=T-t[i]
    changeT=t[i+1]-t[i]
    Phi=norm.cdf((np.log(ST/K)+(r+sigma**2/2)*deltaT) / (sigma*np.sqrt(deltaT)))
    Bt_Psi= - K*np.exp(-r*deltaT)*norm.cdf((np.log(ST/K)+(r-sigma**2/2)*deltaT) / (sigma*np.sqrt(deltaT)))
    error+=(ST_1*Phi+Bt_Psi*np.exp(r*changeT))-(ST*Phi+Bt_Psi)
error=error+BlackScholesCall(S0, K, r, sigma, T)
error=error-np.maximum(Stockprice[:,N]-K,0)
plt.hist(error,weights=np.ones(len(error))/len(error),bins=np.linspace(-2,2,40))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('error')
plt.ylabel('probability')
plt.title('hedging error')
plt.show()

C0=BlackScholesCall(S0, K, r, sigma, T)
print('C0',C0)
mean=Stockprice[:,N].mean()
Price_Check=S0*np.exp(r*T)
print('mean',mean)
print('check',Price_Check)







