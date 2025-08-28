import numpy as np
import matplotlib.pyplot as plt
from blackscholes.mc import mc_price
from blackscholes.pricing import price_call

S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0
Ns = np.logspace(2, 6, 10, dtype=int)  # number of paths from 100 to 1e6
errors = []

for n in Ns:
    mc, err = mc_price(S, K, r, q, sigma, T, n_paths=n, seed=42)
    exact = price_call(S, K, r, q, sigma, T)
    errors.append(abs(mc - exact))

plt.plot(Ns, errors, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of paths (log scale)")
plt.ylabel("Error vs closed form")
plt.title("Monte Carlo convergence to Black–Scholes price")
plt.show()
