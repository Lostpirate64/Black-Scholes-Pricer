import numpy as np
from scipy.stats import norm

def price_call(S, K, r, q, sigma, T):
    """Black-Scholes European call option price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def price_put(S, K, r, q, sigma, T):
    """Black-Scholes European put option price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def delta_call(S, K, r, q, sigma, T):
    """
    Delta of a European call option under Black-Scholes.

    Parameters:
        S (float): Spot price of the underlying
        K (float): Strike price
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to maturity (years)

    Returns:
        float: Call option Delta
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # standard d1
    return np.exp(-q * T) * norm.cdf(d1)

def delta_put(S, K, r, q, sigma, T):
    """Delta of a European put option."""
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (norm.cdf(d1) - 1)
