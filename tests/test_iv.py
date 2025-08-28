import numpy as np
from blackscholes.iv import implied_vol
from blackscholes.pricing import price_call, price_put

def test_round_trip_call():
    S, K, r, q, sigma_true, T = 100, 100, 0.03, 0.01, 0.25, 0.5
    price = price_call(S, K, r, q, sigma_true, T)
    sigma_iv = implied_vol(price, S, K, r, q, T, kind="call")
    assert abs(sigma_iv - sigma_true) < 1e-6

def test_round_trip_put():
    S, K, r, q, sigma_true, T = 100, 90, 0.02, 0.00, 0.35, 1.0
    price = price_put(S, K, r, q, sigma_true, T)
    sigma_iv = implied_vol(price, S, K, r, q, T, kind="put")
    assert abs(sigma_iv - sigma_true) < 1e-6

def test_impossible_price_returns_nan():
    S, K, r, q, T = 100, 100, 0.02, 0.00, 0.5
    impossible_call = S * np.exp(-q*T) + 1.0  # above theoretical upper bound
    assert np.isnan(implied_vol(impossible_call, S, K, r, q, T, kind="call"))
