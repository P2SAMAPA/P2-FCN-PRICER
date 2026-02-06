import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    ivs, hvs = [], []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            # Use 6-month options for institutional pricing baseline
            chain = s.option_chain(s.options[min(len(s.options)-1, 6)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(iv)
        except: ivs.append(0.35)
        h = s.history(period=f"{lookback}mo")['Close'].pct_change().std() * np.sqrt(252)
        hvs.append(h)
    
    data = yf.download(tickers, period=f"{lookback}mo", progress=False)['Close']
    corr = data.pct_change().dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(ivs), np.array(hvs), corr

# --- ENGINE ---
def get_simulated_paths(sims, tenor, rf, vols, corr, strike_pct, skew_factor):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-7)
    dt, steps = 1/252, int(tenor * 252)
    # Applying Skew: As strike decreases, we simulate higher volatility
    adj_vols = vols * (1 + (skew_factor * (1 - strike_pct/100) * 2.5))
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * epsilon
    path_returns = np.exp(np.cumsum(drift + diffusion, axis=0))
    paths = np.vstack([np.ones((1, sims, len(vols))), path_returns]) * 100
    return np.min(paths, axis=2)

def run_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_obs = (coupon_pa * (freq_m/12)) * 100
    
    payoffs, active, cpn_earned = np.zeros(n_sims), np.ones(n_sims, dtype=bool), np.zeros(n_sims)
    obs_dates = np.arange(nc_step, steps, obs_freq)
    
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            periods = (d // obs_freq)
            payoffs[ko_mask] = 100 + (periods * cpn_per_obs)
            cpn_earned[ko_mask] = periods
            active[ko_mask] = False
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        periods = (steps - 1) // obs_freq
        payoffs[active] = principal + (periods * cpn_per_obs)
        cpn_earned[active] = periods
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": np.mean(cpn_earned)
    }

def solve_yield(p, r, t, s, k, f, nc):
    y_lo, y_hi = 0.01, 0.60
    p_lo = run_valuation(y_lo, p, r, t, s, k, f, nc)['price']
    p_hi = run_valuation(y_hi, p, r, t, s, k, f, nc)['price']
    return y_lo + (100 - p_lo) * (y_hi - y_lo) / (p_hi - p_lo)

# --- APP UI ---
st.sidebar.title("Configuration")
tk_in = st.sidebar.text_input("Tickers (CSV)", "TSLA, MSFT")
tickers = [x.strip().upper() for x in tk_in.split(",")]
vol_mode = st.sidebar.radio("Volatility", ["Market Implied", "Historical"])
skew = st.sidebar.slider("Volatility Skew Factor", 0.0, 1.0, 0.8)
sims = st.sidebar.select_slider("Simulations", [5000, 10000], 10000)
rf = st.sidebar.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

st.sidebar.subheader("Note Parameters")
tenor = st.sidebar.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
freq = st.sidebar.selectbox("Coupon Freq (M)", [1, 3, 6])
nc = st.sidebar.number_input("Non-Call (M)", 0, 12, 3)
strike_p = st.sidebar.number_input("Put Strike %", 40, 100, 60)
ko_p = st.sidebar.number_input("KO Barrier %", 70, 150, 100)

if st.sidebar.button("Solve FCN"):
    ivs, hvs, corr = get_market_data(tickers, 12)
    base_vols = ivs if vol_mode == "Market Implied" else hvs
    
    # 1. PRIMARY OUTPUTS (TOP)
    paths_main = get_simulated_paths(sims, tenor, rf, base_vols, corr, strike_p, skew)
    solved_y = solve_yield(paths_main, rf, tenor, strike_p, ko_p, freq, nc)
    res = run_valuation(solved_y, paths_main, rf, tenor, strike_p, ko_p, freq, nc)
    
    st.header(f"Solved Annualized Yield: {solved_y*100:.2f}% p.a.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    col2.metric("Prob. Capital Loss", f"{res['prob_loss']:.1f}%")
    col3.metric("Avg Coupons Paid", f"{res['avg_cpn']:.2f}")
    col4.metric("Note Price Check", f"{res['price']:.2f}")

    # 2. SENSITIVITY MATRICES (BELOW)
    st.divider()
    with st.spinner("Generating Sensitivity Tables..."):
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        y_data, ko_data, ki_data = [], [], []

        for kv in kk:
            y_r, ko_r, ki_r = [], [], []
            for sv in ss:
                p_cell = get_simulated_paths(sims, tenor, rf, base_vols, corr, sv, skew)
                yc = solve_yield(p_cell, rf, tenor, sv, kv, freq, nc)
                rc = run_valuation(yc, p_cell, rf, tenor, sv, kv, freq, nc)
                y_r.append(yc * 100); ko_r.append(rc['prob_ko']); ki_r.append(rc['prob_loss'])
            y_data.append(y_r); ko_data.append(ko_r); ki_data.append(ki_r)

        st.subheader("Yield Sensitivity (% p.a.)")
        st.dataframe(pd.DataFrame(y_data, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Probability of KO (%)**")
            st.dataframe(pd.DataFrame(ko_data, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None).format("{:.1f}"))
        with c2:
            st.write("**Probability of Capital Loss (%)**")
            st.dataframe(pd.DataFrame(ki_grid if 'ki_grid' in locals() else ki_data, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
