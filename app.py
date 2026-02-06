import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- GLOBAL CONFIG & UI ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("## 🛡️ Institutional Derivatives Lab")

# --- SHARED DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data_lab(tickers, source):
    vols, prices, last_px = [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="12mo")['Close']
            prices.append(h.rename(t))
            last_px.append(h.iloc[-1])
            hist_v = h.pct_change().std() * np.sqrt(252)
            if source == "Market Implied (IV)":
                opts = s.options
                if opts:
                    chain = s.option_chain(opts[min(len(opts)-1, 1)])
                    iv = chain.calls['impliedVolatility'].median()
                    vols.append(iv if iv > 0.1 else hist_v)
                else: vols.append(hist_v)
            else: vols.append(hist_v)
        except: 
            vols.append(0.35); last_px.append(100.0)
    df = pd.concat(prices, axis=1).dropna() if prices else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(vols), corr, np.array(last_px)

# --- FCN LOGIC ---
def get_fcn_pv(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    cpn_val = (coupon_pa * (freq_m/12)) * 100
    accrued = np.zeros(n_sims)
    for d in obs_dates:
        accrued[active] += cpn_val
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + accrued[active]
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- BCN LOGIC ---
def get_bcn_pv(coupon_pa, paths, r, tenor, strike, ko, cpn_bar, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    cpn_val = (coupon_pa * (freq_m/12)) * 100
    accrued = np.zeros(n_sims)
    for d in obs_dates:
        eligible = active & (worst_of[d] >= cpn_bar)
        accrued[eligible] += cpn_val
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + accrued[active]
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- SHARED SIDEBAR ---
with st.sidebar:
    st.header("1. Market Configuration")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Vol Skew Factor", 0.0, 1.0, 0.2)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    st.header("2. Shared Parameters")
    tenor = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_label = st.selectbox("Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
    freq_m = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[freq_label]
    nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
    stk = st.slider("Put Strike %", 40, 100, 60)
    ko = st.slider("KO Barrier %", 80, 150, 100)

# --- TABBED INTERFACE ---
tab1, tab2 = st.tabs(["FCN (Fixed)", "BCN (Bonus)"])

with tab1:
    if st.button("Solve FCN Yield"):
        vols, corr, l_px = get_market_data_lab(tickers, vol_src)
        adj_v = vols * (1 + skew)
        n_sims, steps, dt = 15000, int(tenor * 252), 1/252
        L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
        z = np.random.standard_normal((steps, n_sims, len(vols)))
        eps = np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum((rf - 0.5*adj_v**2)*dt + adj_v*np.sqrt(dt)*eps, axis=0))]) * 100
        
        y_solve = brentq(lambda c: get_fcn_pv(c, paths, rf, tenor, stk, ko, freq_m, nc_m) - 100, 0, 4)
        st.metric("Solved FCN Yield", f"{y_solve*100:.2f}% p.a.")
        st.table(pd.DataFrame([{"Ticker": t, "IV": f"{vols[i]:.1%}"} for i, t in enumerate(tickers)]))

with tab2:
    cpn_bar = st.slider("Coupon Barrier %", 40, 100, 70)
    if st.button("Solve BCN Yield"):
        vols, corr, l_px = get_market_data_lab(tickers, vol_src)
        adj_v = vols * (1 + skew)
        n_sims, steps, dt = 15000, int(tenor * 252), 1/252
        L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
        z = np.random.standard_normal((steps, n_sims, len(vols)))
        eps = np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum((rf - 0.5*adj_v**2)*dt + adj_v*np.sqrt(dt)*eps, axis=0))]) * 100

        y_bcn = brentq(lambda c: get_bcn_pv(c, paths, rf, tenor, stk, ko, cpn_bar, freq_m, nc_m) - 100, 0, 5)
        st.metric("Solved BCN Yield", f"{y_bcn*100:.2f}% p.a.")
        st.info("BCN yield is higher as coupons are contingent on the barrier.")
