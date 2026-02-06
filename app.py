import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Professional FCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            # Get current price to find ATM options
            px = s.history(period="1d")['Close'].iloc[-1]
            opts = s.options
            # Focus on mid-term options (3-6 months) for more stable IV
            target_idx = min(len(opts)-1, 5) 
            chain = s.option_chain(opts[target_idx])
            
            # Find the option closest to current price (ATM)
            atm_iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            
            # SANITY CHECK: yfinance sometimes returns IV > 100% for stable stocks erroneously
            # We cap it at a realistic 100% and floor it at 10%
            vols.append(max(0.10, min(1.0, atm_iv)))
        except: 
            vols.append(0.20) # Conservative default for GLD/Stable assets
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now(), progress=False)['Close']
    
    # Handle single ticker case for correlation
    if len(tickers) > 1:
        corr = h.pct_change().dropna().corr().values
    else:
        corr = np.array([[1.0]])
        
    # Matrix robustness
    if np.linalg.cond(corr) > 1/np.finfo(corr.dtype).eps:
        corr = corr + np.eye(len(tickers)) * 1e-6
    return np.array(vols), corr

# --- ENGINE ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs, active, cpn_counts = np.zeros(n_sims), np.ones(n_sims, dtype=bool), np.zeros(n_sims)
    obs_dates = np.arange(nc_step, steps, obs_freq)
    
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num * cpn_per_period)
            cpn_counts[ko_mask] = num
            active[ko_mask] = False
            
    if np.any(active):
        principal = np.where(paths[-1, active] >= strike, 100, paths[-1, active])
        num = (steps - 1) // obs_freq
        payoffs[active] = principal + (num * cpn_per_period)
        cpn_counts[active] = num
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": 1 - (np.sum(active) / n_sims),
        "prob_ki": np.sum(active & (paths[-1] < strike)) / n_sims,
        "avg_cpn": np.mean(cpn_counts)
    }

def get_wo_paths(sims, tks, t_yr, rf, vols, corr, strike_val, skew_bps):
    L = np.linalg.cholesky(corr)
    dt, steps = 1/252, int(t_yr * 252)
    # Apply skew relative to Strike OTM-ness
    adj_vols = vols * (1 + (skew_bps/10000) * (100 - strike_val))
    raw = np.zeros((steps + 1, sims, len(tks)))
    raw[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((sims, len(tks))) @ L.T
        raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
    return np.min(raw, axis=2)

def solve_yield(paths, r, tenor, strike, ko, freq_m, nc_m):
    # Search range for coupon to find Par (100)
    p1 = run_valuation(0.01, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    p2 = run_valuation(0.30, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    return 0.01 + (100 - p1) * (0.30 - 0.01) / (p2 - p1)

# --- UI ---
st.title("🛡️ Institutional FCN Solver")

with st.sidebar:
    st.header("Asset Settings")
    tickers = [x.strip().upper() for x in st.text_input("Tickers (e.g. GLD)", "GLD").split(",")]
    sims = st.select_slider("Simulations", [5000, 10000], 5000)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    skew = st.slider("Volatility Skew (bps)", 0, 300, 50) # Reduced default skew for GLD
    st.divider()
    st.header("Product Parameters")
    tenor = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    freq = st.selectbox("Coupon Frequency (M)", [1, 3, 6])
    nc = st.number_input("Non-Call Period (M)", 0, 12, 3)
    strike_p = st.number_input("Strike Price %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)

if st.button("Calculate Yields"):
    with st.spinner("Fetching market data and running Monte Carlo..."):
        vols, corr = get_market_data(tickers, 12)
        
        # Display the IV used so the user knows if it's realistic
        st.info(f"Using Market Implied Vol: {vols[0]*100:.1f}%")
        
        # 1. SOLVE MAIN CASE
        wo_main = get_wo_paths(sims, tickers, tenor, rf, vols, corr, strike_p, skew)
        main_y = solve_yield(wo_main, rf, tenor, strike_p, ko_p, freq, nc)
        main_res = run_valuation(main_y, wo_main, rf, tenor, strike_p, ko_p, freq, nc)
        
        st.header(f"Solved Annualized Yield: {main_y*100:.2f}% p.a.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. of KO", f"{main_res['prob_ko']*100:.1f}%")
        c2.metric("Prob. of Capital Loss", f"{main_res['prob_ki']*100:.1f}%")
        c3.metric("Avg. Coupons Paid", f"{main_res['avg_cpn']:.2f}")

        # 2. SENSITIVITY GRID
        st.divider()
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        y_grid, ko_grid, ki_grid = [], [], []

        for kv in kk:
            y_r, ko_r, ki_r = [], [], []
            for sv in ss:
                wo = get_wo_paths(sims, tickers, tenor, rf, vols, corr, sv, skew)
                y = solve_yield(wo, rf, tenor, sv, kv, freq, nc)
                res = run_valuation(y, wo, rf, tenor, sv, kv, freq, nc)
                y_r.append(y*100); ko_r.append(res['prob_ko']*100); ki_r.append(res['prob_ki']*100)
            y_grid.append(y_r); ko_grid.append(ko_r); ki_grid.append(ki_r)

        st.subheader("Yield Sensitivity (% p.a.)")
        st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        st.subheader("Risk Metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Prob. of KO (%)**")
            st.dataframe(pd.DataFrame(ko_grid, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None).format("{:.1f}"))
        with col_b:
            st.write("**Prob. of Capital Loss (%)**")
            st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
