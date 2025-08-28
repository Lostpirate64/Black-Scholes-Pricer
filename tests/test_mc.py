from blackscholes.mc import mc_price
from blackscholes.pricing import price_call, price_put

def test_mc_matches_closed_form_call():
    S, K, r, q, sigma, T = 100, 100, 0.05, 0.02, 0.2, 1.0
    mc, err = mc_price(S, K, r, q, sigma, T, kind="call", n_paths=200000, seed=42)
    exact = price_call(S, K, r, q, sigma, T)
    assert abs(mc - exact) < 3 * err  # within 3 standard errors

def test_mc_matches_closed_form_put():
    S, K, r, q, sigma, T = 100, 110, 0.03, 0.01, 0.25, 0.5
    mc, err = mc_price(S, K, r, q, sigma, T, kind="put", n_paths=200000, seed=123)
    exact = price_put(S, K, r, q, sigma, T)
    assert abs(mc - exact) < 3 * err

if __name__ == "__main__":
    price, err = mc_price(100, 100, 0.05, 0.0, 0.2, 1.0, kind="call", n_paths=100000, seed=42)
    print("MC price:", price, "±", err)
