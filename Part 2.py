import pandas
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt
from scipy import interpolate
from scipy.optimize import least_squares

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
	d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
	d2 = d1 - sigma*np.sqrt(T)
	return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def impliedCallVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -BlackScholesCall(S, K, r, x, T), 1e-6, 1)
    return impliedVol

def impliedPutVolatility(S,K,r,price,T):
	impliedVol= brentq(lambda x: price -BlackScholesPut(S, K, r, x, T), 1e-6, 1)
	return impliedVol

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

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.8, x[1], x[2]))**2

    return err

#calculate T
dfCall=pandas.read_csv('goog_call.csv',parse_dates=['expiry','date'])
Days=(dfCall['expiry']-dfCall['date']).dt.days[1]
T=Days/365
print('T',T)

#calculate r
discount=pandas.read_csv('discount.csv')
f=interpolate.interp1d(discount['Day'].values,discount['Rate (%)'].values)
rf=f(Days)
r=rf/100
print('r',r)

#calculate F
S=846.9
F = S*np.exp(r*T)
print('F',F)

#calculate K and corresponding V for out the money call 
dfCall.drop(dfCall.index[dfCall['strike']<=F],inplace=True)
dfCall['price']=(dfCall['best_bid'].values+dfCall['best_offer'])/2
V_OTMC=dfCall['price']
K_OTMC=dfCall['strike']

#calculate the implied volatility for out the money call
impliedvolCall=[]
for i in range(len(K_OTMC)):
	impliedvol=impliedCallVolatility(S, K_OTMC.iloc[i], r, V_OTMC.iloc[i], T)
	impliedvolCall.append(impliedvol)
dfCall['impliedvol']=impliedvolCall

#calculate K and corresponding V for out the money put
dfPut=pandas.read_csv('goog_put.csv',parse_dates=['expiry','date'])
dfPut.drop(dfCall.index[dfCall['strike']>=F],inplace=True)
dfPut['price']=(dfPut['best_bid'].values+dfPut['best_offer'])/2
V_OTMP=dfPut['price']
K_OTMP=dfPut['strike']

#calculate the implied volatility for out the money put
impliedvolPut=[]
for i in range(len(K_OTMP)):
	impliedvol=impliedPutVolatility(S, K_OTMP.iloc[i], r, V_OTMP.iloc[i], T)
	impliedvolPut.append(impliedvol)
dfPut['impliedvol']=impliedvolPut

#concate out the money put and out the money call and plot the market implied vol
df=pandas.concat([dfPut,dfCall], axis=0)
plt.plot(df['strike'],df['impliedvol'],'go')
plt.title('Market Implied Vol Smile')
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.savefig('market.jpg',dpi=300)
plt.show()

#calibrate SABR model
initialGuess = [0.02, 0.2, 0.1]
res = least_squares(lambda x: sabrcalibration(x, df['strike'].values, df['impliedvol'].values, F, T), initialGuess)
alpha = res.x[0]
beta = 0.8
rho = res.x[1]
nu = res.x[2]
print('alpha',alpha)
print('rho',rho)
print('nu',nu)

#calculate sabrvol
sabrvol=[]
for K in df['strike']:
    sabrvoli=SABR(F, K, T, alpha, beta, rho, nu)
    sabrvol.append(sabrvoli)

#calibrate Displaced diffusion model
'''find the K which is closest to F, use that K to find corresponding at the money option price, 
find the sigma that make displaced model match that price'''
K_closest=min(df['strike'],key=lambda x: abs(x-F))
price_ATM=float(df['price'][df['strike']==K_closest])
print('price_ATM',price_ATM)
'''find the implied vol for at the money option'''
sigma_match= impliedPutVolatility(S,K_closest,r,price_ATM,T)
print('sigma_match',sigma_match)

'''for each beta in np.arange(0.2,1,0.2), use different K to get different price under displaced diffusion model, 
then use black sholes to calculate the implied vol'''
impliedvol_diff=pandas.DataFrame()
beta_diff=np.arange(0.2,1.2,0.2)
for betai in beta_diff:
    price_diff_Call=Displaced_diffusionCall(F,K_OTMC,r,sigma_match,T,betai)
    impliedvolCall_diff=[]
    for i in range(len(K_OTMC)):
        impliedvol=impliedCallVolatility(S, K_OTMC.values[i], r, price_diff_Call.iloc[i], T)
        impliedvolCall_diff.append(impliedvol)

    price_diff_Put=Displaced_diffusionPut(F,K_OTMP,r,sigma_match,T,betai)
    impliedvolPut_diff=[]
    for i in range(len(K_OTMP)):
        impliedvol=impliedPutVolatility(S, K_OTMP.values[i], r, price_diff_Put.iloc[i], T)
        impliedvolPut_diff.append(impliedvol)
    impliedvol_diffi=impliedvolPut_diff+impliedvolCall_diff
    impliedvol_diff['\u03B2 = '+str(round(betai,1))]=impliedvol_diffi

'''put impliedvol_diff, market impliedvol and sabrvol into one dataframe, use strike price as index
plot the fitted implied volatility smile against the market data'''
impliedvol_all=impliedvol_diff
impliedvol_all['SABR Model']=sabrvol
impliedvol_all['Market']=df['impliedvol'].values
impliedvol_all.index=df['strike'].values
print(impliedvol_all)
#plot DD model vol VS market implied vol
impliedvol_all[['Market','\u03B2 = 0.2','\u03B2 = 0.4','\u03B2 = 0.6','\u03B2 = 0.8','\u03B2 = 1.0']].plot(style=['go','k--','c--','b--','y--','m--'])
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('DD VS Market Implied Vol Smile')
plt.savefig('DisplaceDiffusion.jpg',dpi=300)
plt.show()
#plot sabr vol VS DD vol VS market vol
impliedvol_all[['Market','SABR Model','\u03B2 = 0.2','\u03B2 = 0.4','\u03B2 = 0.6','\u03B2 = 0.8','\u03B2 = 1.0']].plot(style=['go','r-','k--','c--','b--','y--','m--'])
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('DD VS Market VS SABR Implied Vol Smile')
plt.savefig('SABR.jpg',dpi=300)
plt.show()
#from the figure we can see that for displace model to fit market implied volatility, beta should be around 0.4

#Calculate the exact number for beta
def ddmcalibration(x, strikes, vols, F, T,S,matchedsigma):
    price_diff_Call=Displaced_diffusionCall(F,K_OTMC,r,matchedsigma,T,x)
    impliedvolCall_diff=[]
    for i in range(len(K_OTMC)):
        impliedvol=impliedCallVolatility(S, K_OTMC.values[i], r, price_diff_Call.iloc[i], T)
        impliedvolCall_diff.append(impliedvol)

    price_diff_Put=Displaced_diffusionPut(F,K_OTMP,r,matchedsigma,T,x)
    impliedvolPut_diff=[]
    for i in range(len(K_OTMP)):
        impliedvol=impliedPutVolatility(S, K_OTMP.values[i], r, price_diff_Put.iloc[i], T)
        impliedvolPut_diff.append(impliedvol)
    impliedvol_diffi=impliedvolPut_diff+impliedvolCall_diff

    err=0.0
    for i, vol in enumerate(vols):
        err += (vol - impliedvol_diffi[i])**2

    return err

res2 = least_squares(lambda x: ddmcalibration(x,df['strike'].values,df['impliedvol'].values,F,T,S,sigma_match),0.1)

beta_exact=res2.x[0]
print('beta_exact',beta_exact)

#plot sabr implied vol with different rho
sabrvol2=pandas.DataFrame()
for rhoi in [-0.5,0,0.5]:
    sabrvol=[]
    for K in df['strike']:
        sabrvoli=SABR(F, K, T, alpha, beta, rhoi, nu)
        sabrvol.append(sabrvoli)
    sabrvol2['\u03C1 = '+str(round(rhoi,1))]=sabrvol
sabrvol2.index=df['strike'].values
sabrvol2[['\u03C1 = -0.5','\u03C1 = 0','\u03C1 = 0.5']].plot(style=['b--','y--','m--'])
plt.title('SABR Vol with Different \u03C1')
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.savefig('rho.jpg',dpi=300)
plt.show()

#plot sabr implied vol with different nu
sabrvol3=pandas.DataFrame()
for nui in [0.1,0.3,0.5]:
    sabrvol=[]
    for K in df['strike']:
        sabrvoli=SABR(F, K, T, alpha, beta, rho, nui)
        sabrvol.append(sabrvoli)
    sabrvol3['\u03BD = '+str(round(nui,1))]=sabrvol
sabrvol3.index=df['strike'].values
sabrvol3[['\u03BD = 0.1','\u03BD = 0.3','\u03BD = 0.5']].plot(style=['b--','y--','m--'])
plt.title('SABR Vol with Different \u03BD')
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.savefig('nu.jpg',dpi=300)
plt.show()





















