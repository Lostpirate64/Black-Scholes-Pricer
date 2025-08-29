import streamlit as st
import numpy as np
from blackscholes.pricing import price_call, price_put
from blackscholes.iv import implied_vol
from blackscholes.mc import mc_price

st.title("Black–Scholes Option Pricer")

st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
r = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
q = st.sidebar.number_input("Dividend yield (q)", value=0.0, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
T = st.sidebar.number_input("Time to maturity (T, years)", value=1.0, step=0.25)
kind = st.sidebar.radio("Option type", ["call", "put"])

st.subheader("Closed-form Black–Scholes Price")
if kind == "call":
    price = price_call(S, K, r, q, sigma, T)
else:
    price = price_put(S, K, r, q, sigma, T)
st.write(f"**Price:** {price:.4f}")

st.subheader("Implied Volatility Solver")
market_price = st.number_input("Enter observed market price", value=price)
iv = implied_vol(market_price, S, K, r, q, T, kind=kind)
st.write(f"**Implied volatility:** {iv:.4f}" if not np.isnan(iv) else "No valid IV found")

st.subheader("Monte Carlo Validation")
mc, err = mc_price(S, K, r, q, sigma, T, kind=kind, n_paths=100_000, seed=42)
st.write(f"MC Price: {mc:.4f} ± {err:.4f}")
