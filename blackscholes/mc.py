import numpy as np

def mc_price(S, K, r, q, sigma, T, kind="call", n_paths=100000, antithetic=True, seed=None):
    """
    monte Carlo pricing for a european call or put under black-ccholes
    """
    if seed is not None:
        np.random.seed(seed)

    # generate random draws
    n = n_paths
    Z = np.random.normal(size=n)
    if antithetic:  # use antithetic variates to reduce variance
        Z = np.concatenate([Z, -Z])

    # simulate terminal stock price under risk-neutral measure
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S * np.exp(drift + diffusion)

    # payoff
    if kind == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # discounted expectation
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(len(payoff))  # standard error
    return price, stderr
