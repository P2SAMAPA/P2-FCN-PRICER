import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- CORE DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_market_context(tickers):
    data = []
    for t in tickers:
        s = yf.Ticker(t)
        h = s.history(period="12mo")['Close']
        data.append(h.rename(t))
    df = pd.concat(data, axis=1).dropna()
    return df, df.pct_change().corr().values

def fetch_vols(tickers, source):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        hist_v = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        if source == "Market Implied (IV)":
            try:
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 2)])
                px = s.history(period="1d")['Close'].iloc[-1]
                vols.append(chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0])
            except: vols.append(hist_v)
        else: vols.append(hist_v)
    return np.array(vols)

# --- QUANT ENGINE ---
def simulate_paths(sims, tenor, rf, vols, corr, skew):
    # CRITICAL: Applying skew to boost the diffusion term for tail-risk sensitivity
    adj_vols = np.maximum(vols * (1 + skew), 0.15) # Floor at 15% to prevent 0-yield flatlining
    steps, dt = int(tenor * 252), 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    z = np.random.standard_normal((steps, sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * eps
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

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

# --- UI LAYOUT ---
st.markdown("## 🛡️ Institutional FCN Solver (Discrete Observation)")

with st.sidebar:
    st.header("1. Market Inputs")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Vol Source", ["Market Implied (IV)", "Historical (HV)"])
    skew_f = st.slider("Vol Skew Factor", 0.0, 1.0, 0.20)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.header("2. Note Specs")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_m = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (Months)", 0, 12, 3)
    strike_p = st.slider("Put Strike %", 40, 100, 60)
    ko_p = st.slider("KO Barrier %", 80, 150, 100)

if st.button("Solve Everything"):
    df, corr = fetch_market_context(tickers)
    vols = fetch_vols(tickers, vol_src)
    paths = simulate_paths(15000, tenor_y, rf_rate, vols, corr, skew_f)
    
    # Target 100 Par Solve
    try:
        y_solve = brentq(lambda c: get_fcn_pv(c, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m) - 100, 0.0, 2.0)
    except: y_solve = 0.0

    st.markdown(f"### Solved Annualized Yield: **{y_solve*100:.2f}% p.a.**")
    
    col1, col2 = st.columns(2)
    p_loss = (np.sum(np.min(paths[-1], axis=1) < strike_p) / 15000) * 100
    col1.metric("Prob. of Capital Loss", f"{p_loss:.1f}%")
    
    # Sensitivity Tables
    st.divider()
    st.markdown("### Yield Sensitivity Matrix (% p.a.)")
    
    stks = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    bars = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    grid = []
    for b in bars:
        row = []
        for s in stks:
            try:
                # Local solve for each cell
                val = brentq(lambda c: get_fcn_pv(c, paths, rf_rate, tenor_y, s, b, freq_m, nc_m) - 100, 0.0, 3.0) * 100
                row.append(val)
            except: row.append(0.0)
        grid.append(row)
        
    df_sens = pd.DataFrame(grid, index=[f"KO {x}%" for x in bars], columns=[f"Stk {x}%" for x in stks])
    st.table(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))

    st.markdown("### Capital Loss Probability (%)")
    loss_grid = [[(np.sum(np.min(paths[-1], axis=1) < s) / 15000) * 100 for s in stks] for b in bars]
    df_loss = pd.DataFrame(loss_grid, index=[f"KO {x}%" for x in bars], columns=[f"Stk {x}%" for x in stks])
    st.table(df_loss.style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
