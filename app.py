import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")
st.markdown("## 🛡️ Institutional FCN Solver")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data_refined(tickers, source):
    vols, prices, last_px = [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="12mo")['Close']
            px = h.iloc[-1]
            prices.append(h.rename(t))
            last_px.append(px)
            
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
            vols.append(0.35)
            last_px.append(100.0)
    
    df = pd.concat(prices, axis=1).dropna() if prices else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(vols), corr, np.array(last_px)

# --- CORE LOGIC ---
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

# --- SIDEBAR (INPUTS PRESERVED) ---
with st.sidebar:
    st.header("1. Market Inputs")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Vol Skew Factor", 0.0, 1.0, 0.2)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.header("2. Note Structure")
    tenor = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_label = st.selectbox("Coupon Frequency", ["Monthly", "Quarterly", "Semi-Annual", "Annual"])
    freq_map = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6, "Annual": 12}
    freq_m = freq_map[freq_label]
    nc_m = st.number_input("Non-Call Period (Months)", 0, 24, 3)
    stk = st.slider("Put Strike %", 40, 100, 60)
    ko = st.slider("KO Barrier %", 80, 150, 100)

# --- EXECUTION ---
if st.button("Generate FCN Pricing"):
    vols, corr, last_prices = get_market_data_refined(tickers, vol_src)
    adj_vols = vols * (1 + skew)
    
    # Path Generation
    n_sims, steps, dt = 15000, int(tenor * 252), 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    z = np.random.standard_normal((steps, n_sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diff = adj_vols * np.sqrt(dt) * eps
    paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum(drift + diff, axis=0))]) * 100

    # Solve
    try:
        y_solve = brentq(lambda c: get_fcn_pv(c, paths, rf, tenor, stk, ko, freq_m, nc_m) - 100, 0.0, 4.0)
    except: y_solve = 0.0

    st.markdown(f"### Solved Annualized Yield: **{y_solve*100:.2f}% p.a.**")
    
    col1, col2 = st.columns(2)
    p_loss = (np.sum(np.min(paths[-1], axis=1) < stk) / n_sims) * 100
    col1.metric("Prob. Capital Loss", f"{p_loss:.1f}%")

    # --- NEW: COMPONENT ANALYSIS ---
    st.divider()
    st.write("### 🔍 Underlying Component Analysis")
    comp_data = []
    for i, t in enumerate(tickers):
        comp_data.append({
            "Ticker": t,
            "Spot Price": f"${last_prices[i]:.2f}",
            "Base Volatility": f"{vols[i]:.2%}",
            "Skew-Adj Vol": f"{adj_vols[i]:.2%}",
            "Distance to Strike": f"{(1 - (stk/100)):.1%}"
        })
    st.table(pd.DataFrame(comp_data))

    # SENSITIVITY MATRIX
    st.write("### Yield Sensitivity Matrix (% p.a.)")
    stks_range = [stk-10, stk, stk+10]
    bars_range = [ko+10, ko, ko-10]
    
    grid = []
    for b in bars_range:
        row = []
        for s in stks_range:
            try:
                val = brentq(lambda c: get_fcn_pv(c, paths, rf, tenor, s, b, freq_m, nc_m) - 100, 0.0, 5.0)
                row.append(val * 100)
            except: row.append(0.0)
        grid.append(row)
    
    df_res = pd.DataFrame(grid, columns=[f"Stk {s}%" for s in stks_range], index=[f"KO {b}%" for b in bars_range])
    st.table(df_res.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
