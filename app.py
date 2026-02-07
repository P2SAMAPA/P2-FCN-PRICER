import streamlit as st; import numpy as np; import pandas as pd; import yfinance as yf; from scipy.optimize import brentq;

st.set_page_config(page_title="FCN Pricer", layout="wide");

def run_sim(cpn, paths, r, t, stk, ko, f_m, nc_m):

steps, n_s, n_a = paths.shape; wo = np.min(paths, axis=2); obs = np.arange(int((f_m/12)252), steps, int((f_m/12)252));

pay, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s); c_val = (cpn * (f_m/12)) * 100;

for i, step in enumerate(obs):

acc[act] += c_val;

if step >= int((nc_m/12)252):

k = act & (wo[step] >= ko); pay[k] = 100 + acc[k]; act[k] = False;

if np.any(act): pay[act] = np.where(wo[-1, act] >= stk, 100, wo[-1, act]) + acc[act];

return np.mean(pay) * np.exp(-r * t);

st.title("🚀 Institutional FCN Terminal");

tks_in = st.text_input("Underlying Basket", "AMZN, ALB");

if st.button("PRICING RUN"):

tks = [x.strip().upper() for x in tks_in.split(",")];

vols = np.array([0.32, 0.48]); r = 0.045; n_s, n_d = 10000, 252;

rets = np.random.normal((r - 0.5vols**2)(1/252), volsnp.sqrt(1/252), (n_d, n_s, 2));

paths = 100 * np.exp(np.cumsum(rets, axis=0));

target = lambda x: run_sim(x, paths, r, 1.0, 75, 100, 1, 3) - 100;

sol = brentq(target, 0, 1.0);

st.metric("SOLVED ANNUAL YIELD", f"{sol*100:.2f}%");

st.write("Calculated using Geometric Brownian Motion for high-volatility assets.");
