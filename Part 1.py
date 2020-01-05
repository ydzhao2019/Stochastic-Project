import numpy as np 
from scipy.stats import norm
import pandas as  pd
#Vanilla call/put
#1. Black_Scholes model
def BlackScholesCall(S, K, r, sigma, T):
	d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
	d2 = d1 - sigma*np.sqrt(T)
	return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
	d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
	d2 = d1 - sigma*np.sqrt(T)
	return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

#2. Bachelier model
def BachelierCall(S,K,sigma,T):
	d1=(S-K)/(S*sigma*np.sqrt(T))
	return (S-K)*norm.cdf(d1) + S*sigma*np.sqrt(T)*norm.pdf(d1)

def BachelierPut(S,K,sigma,T):
	d1=(S-K)/(S*sigma*np.sqrt(T))
	return (K-S)*norm.cdf(-d1) +S*sigma*np.sqrt(T)*norm.pdf(-d1)

#3. Black76 model
def Black76Call(F,K,r,sigma,T):
	d1= (np.log(F/K)+(1/2)*sigma**2*T) / (sigma*np.sqrt(T))
	d2=d1 - sigma*np.sqrt(T)
	return np.exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Put(F,K,r,sigma,T):
	d1= (np.log(F/K)+(1/2)*sigma**2*T) / (sigma*np.sqrt(T))
	d2=d1 - sigma*np.sqrt(T)
	return np.exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

#4. Displaced-diffusion model
'''def Displaced_diffusionCall(S,K,r,sigma,T,beta):
	Sd=S/beta
	sigmad=sigma*beta
	Kd=K+((1-beta)/beta)*S
	d1 = (np.log(Sd/Kd)+(r+sigmad**2/2)*T) / (sigmad*np.sqrt(T))
	d2 = d1 - sigmad*np.sqrt(T)
	return Sd*norm.cdf(d1) - Kd*np.exp(-r*T)*norm.cdf(d2)

def Displaced_diffusionPut(S,K,r,sigma,T,beta):
	Sd=S/beta
	sigmad=sigma*beta
	Kd=K+((1-beta)/beta)*S
	d1 = (np.log(Sd/Kd)+(r+sigmad**2/2)*T) / (sigmad*np.sqrt(T))
	d2 = d1 - sigmad*np.sqrt(T)
	return Kd*np.exp(-r*T)*norm.cdf(-d2) - Sd*norm.cdf(-d1)'''

#4. Displaced_diffusion model using Black76
def Displaced_diffusionCall(F,K,r,sigma,T,beta):
	Fd=F/beta
	sigmad=sigma*beta
	Kd=K+((1-beta)/beta)*F
	d1 = (np.log(Fd/Kd)+(sigmad**2/2)*T) / (sigmad*np.sqrt(T))
	d2 = d1 - sigmad*np.sqrt(T)
	return np.exp(-r*T)*(Fd*norm.cdf(d1) - Kd*norm.cdf(d2))

def Displaced_diffusionPut(F,K,r,sigma,T,beta):
	Fd=F/beta
	sigmad=sigma*beta
	Kd=K+((1-beta)/beta)*F
	d1 = (np.log(Fd/Kd)+(sigmad**2/2)*T) / (sigmad*np.sqrt(T))
	d2 = d1 - sigmad*np.sqrt(T)
	return np.exp(-r*T)*(Kd*norm.cdf(-d2) - Fd*norm.cdf(-d1))


Vanilla=pd.DataFrame(columns=['Vanilla call', 'Vanilla put'], index=['Black-Scholes Model',
                    'Bachelier model', 'Black76 Model', 'Displaced-diffusion Model'])
Vanilla.iloc[0,0]= BlackScholesCall(100, 100, 0.05, 0.25,1)
Vanilla.iloc[1,0]=BachelierCall(100, 100, 0.25, 1)
Vanilla.iloc[2,0]=Black76Call(100, 100, 0.05, 0.25,1)
Vanilla.iloc[3,0]=Displaced_diffusionCall(100, 100, 0.05, 0.25,1, 0.1)

Vanilla.iloc[0,1]=BlackScholesPut(100, 100, 0.05, 0.25,1)
Vanilla.iloc[1,1]=BachelierPut(100, 100, 0.25, 1)
Vanilla.iloc[2,1]=Black76Put(100, 100, 0.05, 0.25,1)
Vanilla.iloc[3,1]=Displaced_diffusionPut(100, 100, 0.05, 0.25,1, 0.1)

print(Vanilla)




