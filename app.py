import streamlit as st import numpy as np import pandas as pd import yfinance as yf from scipy.optimize import brentq

st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")

st.markdown(""" <style> .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; } .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; } </style> """, unsafe_allow_html=True)

with st.sidebar: st.header("🏢 Product Architect") mode = st.selectbox("Structure", ["FCN Version 1", "FCN Version 2 (Step-Down)"]) tks_in = st.text_input("Underlying Basket", "AMZN, ALB") tks = [x.strip().upper() for x in tks_in.split(",")] st.divider() stk_pct = st.slider("Put Strike (%)", 40, 100, 75) ko_pct = st.slider("Autocall Level (%)", 80, 120, 100) freq_opt = st.selectbox("Coupon Frequency", ["Monthly", "Quarterly"]) freq_m = 1 if freq_opt == "Monthly" else 3 tenor = st.number_input("Tenor (Years)", 0.25, 5.0, 1.0, 0.25) nc_m = st.number_input("Non-Call (Months)", 0, 24, 3) st.divider() r_val = st.number_input("Risk Free Rate (%)", 0.0, 10.0, 4.5) / 100 sd_val = st.slider("Step-Down (%)", 0.0, 2.0, 0.0) if "Version 2" in mode else 0

@st.cache_data(ttl=600) def get_market_data(tickers): vols, prices, divs, names = [], [], [], [] for t in tickers: s = yf.Ticker(t) h = s.history(period="2y")['Close'] if h.empty: continue prices.append(h) names.append(t) dy = s.info.get('dividendYield', 0.012) or 0.012 divs.append(dy) vols.append(h.pct_change().tail(252).std() * np.sqrt(252)) return np.array(vols), pd.concat(prices, axis=1).pct_change().dropna().corr().values, np.array(divs), names

def run_simulation(cpn, paths, r, t, stk, ko, f_m, nc_m, mode, sd): steps, n_s, _ = paths.shape wo = np.min(paths, axis=2) obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252)) pay, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s) c_val = (cpn * (f_m/12)) * 100 for i, step in enumerate(obs): cur_ko = ko - (i * sd) if "Version 2" in mode else ko acc[act] += c_val if step >= int((nc_m/12)*252): k = act & (wo[step] >= cur_ko) pay[k] = 100 + acc[k] act[k] = False if np.any(act): pay[act] = np.where(wo[-1, act] >= stk, 100, wo[-1, act]) + acc[act] return np.mean(pay) * np.exp(-r * t), (np.sum(wo[-1] < stk) / n_s)

st.title(f"🚀 {mode} Terminal")

if st.button("RUN MONTE CARLO PRICING"): v, corr, d, names = get_market_data(tks) n_s, n_d = 10000, int(tenor * 252) L = np.linalg.cholesky(corr + np.eye(len(v)) * 1e-10) dt = 1/252 drift = (r_val - d - 0.5 * v**2) * dt noise = np.einsum('ij,tkj->tki', L, np.random.standard_normal((n_d, n_s, len(v)))) rets = drift + (v * np.sqrt(dt)) * noise paths = np.vstack([np.ones((1, n_s, len(v)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

End of Code.
