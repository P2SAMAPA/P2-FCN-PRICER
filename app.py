import streamlit as st import numpy as np import pandas as pd import yfinance as yf from scipy.optimize import brentq
def run_sim(cpn, paths, r, t, stk, ko, f_m, nc_m): steps, n_s, _ = paths.shape wo = np.min(paths, axis=2) obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252)) pay, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s) c_val = (cpn * (f_m/12)) * 100 for i, step in enumerate(obs): acc[act] += c_val if step >= int((nc_m/12)*252): k = act & (wo[step] >= ko) pay[k] = 100 + acc[k] act[k] = False if np.any(act): pay[act] = np.where(wo[-1, act] >= stk, 100, wo[-1, act]) + acc[act] return np.mean(pay) * np.exp(-r * t)
st.title("🚀 Institutional FCN Pricer") tks = st.text_input("Basket (e.g. AMZN, ALB)", "AMZN, ALB").split(",") stk = st.slider("Strike", 40, 100, 75) ko = st.slider("KO Level", 80, 120, 100)

if st.button("Calculate"): # Simplified simulation for stability vols = np.array([0.35, 0.45]) # Standard vols for AMZN/ALB r = 0.045 n_s, n_d = 5000, 252 rets = np.random.normal(0, 1, (n_d, n_s, 2)) * 0.40 * np.sqrt(1/252) paths = 100 * np.exp(np.cumsum(rets, axis=0))
  
