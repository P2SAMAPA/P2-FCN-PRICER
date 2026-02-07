import streamlit as st; import numpy as np; import pandas as pd; import yfinance as yf; from scipy.optimize import brentq

st.title("🚀 FCN Pricing Terminal")

t_in = st.text_input("Assets", "AMZN, ALB")

if st.button("RUN"): v = np.array([0.35, 0.48]); r = 0.045; dt = 1/252; n_s = 10000; n_d = 252; s_lvl = 75; k_lvl = 100; f_m = 1; n_m = 3 drift = (r - 0.5 * (v**2)) * dt; shock = v * np.sqrt(dt) rets = np.random.normal(drift, shock, (n_d, n_s, 2)) paths = 100 * np.exp(np.cumsum(rets, axis=0)) def pricer(c): p = paths; steps = p.shape[0]; wo = np.min(p, axis=2); obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)252)) pay = np.zeros(n_s); act = np.ones(n_s, dtype=bool); acc = np.zeros(n_s); c_val = (c(f_m/12))*100 for step in obs: acc[act] += c_val if step >= int((n_m/12)*252): idx = act & (wo[step] >= k_lvl); pay[idx] = 100 + acc[idx]; act[idx] = False if np.any(act): pay[act] = np.where(wo[-1, act] >= s_lvl, 100, wo[-1, act]) + acc[act] return np.mean(pay) * np.exp(-r * 1.0) - 100 try: yield_sol = brentq(pricer, 0, 1.0) st.metric("ANNUAL YIELD", f"{yield_sol * 100:.2f}%") st.write("Calculated using Geometric Brownian Motion.") except Exception as e: st.error(f"Error: {e}")
