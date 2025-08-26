from blackscholes.pricing import price_call, price_put

def test_put_call_parity():
    S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0
    call = price_call(S, K, r, q, sigma, T)
    put = price_put(S, K, r, q, sigma, T)
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8
