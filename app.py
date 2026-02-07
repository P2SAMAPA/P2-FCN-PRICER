import streamlit as st import numpy as np import pandas as pd import yfinance as yf from scipy.optimize import brentq

--- 1. UI SETUP ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")

st.markdown(""" <style> .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; } .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; } </style> """, unsafe_allow_html=True)

--- 2. SIDEBAR ---
with st.sidebar: st.header("🏢 Product Architect") mode = st.selectbox("Structure", ["FCN Version 1", "FCN Version 2 (Step-Down)"]) tks_in = st.text_input("Underlying Basket", "AMZN, ALB") tks = [x.strip().upper() for x in tks_in.split(",")]

--- 3. DATA & PRICING ---
@st.cache_data(ttl=600) def get_market_data(tickers): vols, prices, divs, names = [], [], [], [] for t in tickers: s = yf.Ticker(t) h = s.history(period="2y")['Close'] if h.empty: continue prices.append(h) names.append(t) dy = s.info.get('dividendYield', 0.012) or 0.012 divs.append(dy) vols.append(h.pct_change().tail(252).std() * np.sqrt(252)) return np.array(vols), pd.concat(prices, axis=1).pct_change().dropna().corr().values, np.array(divs), names

def run_simulation(cpn, paths, r, t, stk, ko, f_m, nc_m, mode, sd): steps, n_s, _ = paths.shape wo = np.min(paths, axis=2) obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252)) pay, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s) c_val = (cpn * (f_m/12)) * 100

--- 4. EXECUTION ---
st.title(f"🚀 {mode} Terminal")

if st.button("RUN MONTE CARLO PRICING"): v, corr, d, names = get_market_data(tks) n_s, n_d = 10000, int(tenor * 252) L = np.linalg.cholesky(corr + np.eye(len(v)) * 1e-10) dt = 1/252
