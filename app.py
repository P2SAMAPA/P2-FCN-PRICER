import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# 1. Setup
st.set_page_config(page_title="Institutional FCN Solver", layout="wide")
st.markdown("## 🛡️ Institutional FCN Solver")

# 2. Market Data Helpers
@st.cache_data(ttl=3600)
def get_vols_and_corr(tickers, source):
    vols = []
    prices = []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="12mo")['Close']
            prices.append(h.rename(t))
            hist_v = h.pct_change().std() * np.sqrt(252)
            
            if source == "Market Implied (IV)":
                opts = s.options
                if opts:
                    chain = s.option_chain(opts[min(len(opts)-1, 1)])
                    iv = chain.calls['impliedVolatility'].median()
                    vols.append(iv if iv > 0.05 else hist_v)
                else: vols.append(hist_v)
            else: vols.append(hist_v)
        except: vols.append(0.30) # Default to 30% vol if ticker fails
    
    df = pd.concat(prices, axis=1).dropna() if prices else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(vols), corr

# 3. Simulation Functions (Must be defined before usage)
def get_fcn_value(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
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

# 4. Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Volatility Skew", 0.0, 1.0, 0.2)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    st.divider()
    tenor = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    stk = st.slider("Put Strike %", 40, 100, 60)
    ko = st.slider("KO Barrier %", 80, 150, 100)

# 5. Main Execution
if st.button("Solve FCN Structure"):
    vols, corr = get_vols_and_corr(tickers, vol_src)
    adj_vols = vols * (1 + skew)
    
    # Monte Carlo Paths
    n_sims, steps, dt = 10000, int(tenor * 252), 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    z = np.random.standard_normal((steps, n_sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diff = adj_vols * np.sqrt(dt) * eps
    paths = np.exp(np.cumsum(drift + diff, axis=0))
    paths = np.vstack([np.ones((1, n_sims, len(vols))), paths]) * 100

    # Solve for Coupon
    try:
        y_solve = brentq(lambda c: get_fcn_value(c, paths, rf, tenor, stk, ko, 1, 3) - 100, 0.0, 2.0)
    except: y_solve = 0.1 # Fallback to 10% if math fails

    st.markdown(f"### Solved Annualized Yield: **{y_solve*100:.2f}% p.a.**")
    
    # Sensitivities
    st.divider()
    stks_range = [stk-10, stk, stk+10]
    results = []
    for s in stks_range:
        try:
            v = brentq(lambda c: get_fcn_value(c, paths, rf, tenor, s, ko, 1, 3) - 100, 0.0, 3.0)
            results.append(v * 100)
        except: results.append(0.0)
    
    df_res = pd.DataFrame([results], columns=[f"Strike {s}%" for s in stks_range], index=["Solved Yield %"])
    st.table(df_res.style.background_gradient(cmap='RdYlGn', axis=1).format("{:.2f}"))
