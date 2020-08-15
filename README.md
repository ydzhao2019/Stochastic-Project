# Stochastic-Modelling-Project

Part 1: Derived Black-Scholes model, Bachelier model, Black 76 model and Displaced-diffusion model to value options;

Part 2: Calibrated Displaced-Diffusion model and SABR model with least square method to fit market implied volatility smile;

Part 3: Part 2: Combined Garch model with Empirical Martingale Simulation to value options and to explain market implied vol smile

Firstly estimate Garch model parameters using Maximum Likelihood Estimation and add S&P500 trading volume as a macroeconomic factor into Garch Variance model;

Then Generate Empirical Martingale Simulation with three types of variance model: constant variance, Garch variance, Garch+macroeconomic factors variance;

Lastly use the three types of Empirical Martingale Simulations to calculate option prices and implied vol, then compare them with market option prices and implied vol to see which variance model can fit the market implied vol better. 

Part 4: Simulated dynamic hedging strategy under Black-Sholes model using Monte Carlo and reported option hedging error.
