import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from blackscholes.pricing import price_call, price_put  # from pricing.py

def _vega(S, K, r, q, sigma, T):
    """dPrice/dSigma (same for call/put under BS)"""
    if sigma <= 0 or T <= 0:
        return 0.0
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrtT
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def _price_bounds(S, K, r, q, T, kind):
    """no-arbitrage BS bounds used for sanity checks"""
    disc_K = K * np.exp(-r * T)
    disc_S = S * np.exp(-q * T)
    if kind == "call":
        lower = max(disc_S - disc_K, 0.0)   # intrinsic lower bound
        upper = disc_S                      # sigma -> ∞
    else:
        lower = max(disc_K - disc_S, 0.0)
        upper = disc_K
    return lower, upper

def _model_price(S, K, r, q, sigma, T, kind):
    return price_call(S, K, r, q, sigma, T) if kind == "call" else price_put(S, K, r, q, sigma, T)

def implied_vol(
    price, S, K, r, q, T, kind="call",
    tol=1e-8, max_iter=100, sigma_lo=1e-8, sigma_hi=5.0, expand_factor=2.0, max_sigma=10.0
):
    """
    black–scholes implied volatility from a market price
    returns float sigma or np.nan if not solvable
    """
    # basic validation
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    # no-arbitrage bounds
    lb, ub = _price_bounds(S, K, r, q, T, kind)
    if price < lb - 1e-12 or price > ub + 1e-12:
        return np.nan
    if abs(price - lb) < 1e-12:
        return 0.0  # effectively zero vol

    # root function: model_price(sigma) - market_price
    def f(sig):
        return _model_price(S, K, r, q, sig, T, kind) - price

    a, b = sigma_lo, sigma_hi
    fa, fb = f(a), f(b)

    # ensure sign change; expand upper bound if needed
    tries = 0
    while (np.isnan(fa) or np.isnan(fb) or fa * fb > 0) and b < max_sigma and tries < 20:
        b *= expand_factor
        fb = f(b)
        tries += 1

    # if still no bracket, try a few newton steps from a heuristic guess
    if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
        guess = 0.2 if T > 0.1 else 0.3
        sigma = max(1e-8, min(guess, max_sigma))
        for _ in range(8):
            v = _vega(S, K, r, q, sigma, T)
            if v < 1e-12:
                break
            diff = f(sigma)
            sigma -= diff / v
            if not (1e-12 < sigma <= max_sigma):
                break
            if abs(diff) < tol:
                return float(sigma)
        return np.nan

    # brent’s method (fast)
    try:
        root = brentq(f, a, b, xtol=tol, maxiter=max_iter)
        return float(root)
    except Exception:
        return np.nan

def implied_vol_array(price, S, K, r, q, T, kind="call", **kwargs):
    """vectorized wrapper: accepts arrays/lists; returns numpy array"""
    price = np.asarray(price); S = np.asarray(S); K = np.asarray(K)
    r = np.asarray(r); q = np.asarray(q); T = np.asarray(T)
    out = np.empty(np.broadcast(price, S, K, r, q, T).shape, dtype=float)
    it = np.nditer([price, S, K, r, q, T, out], op_flags=[['readonly']]*6 + [['writeonly']])
    for p_i, S_i, K_i, r_i, q_i, T_i, o_i in it:
        o_i[...] = implied_vol(float(p_i), float(S_i), float(K_i), float(r_i), float(q_i), float(T_i), kind=kind, **kwargs)
    return out
