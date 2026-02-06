import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Institutional FCN & BCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            opts = s.options
            # Grab institutional IV (approx 6 months out)
            idx = min(len(opts)-1, 4)
            chain = s.option_chain(opts[idx])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(iv)
        except: vols.append(0.35)
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now(), progress=False)['Close']
    corr = h.pct_change().dropna().corr().values
    
    # --- ROBUSTNESS: Ensure Correlation Matrix is Positive Definite ---
    # If Cholesky fails, we add a tiny 'jitter' to the diagonal
    try:
        np.linalg.cholesky(corr)
    except np.linalg.linalg.LinAlgError:
        corr = corr + np.eye(len(tickers)) * 1e-6
        
    return np.array(vols), corr

# --- CALCULATION ENGINE ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    cpn_counts = np.zeros(n_sims)
    
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

def solve_yield(paths, r, tenor, strike, ko, freq_m, nc_m):
    p1 = run_valuation(0.01, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    p2 = run_valuation(0.40, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    if abs(p2-p1) < 1e-5: return r
    return 0.01 + (100 - p1) * (0.40 - 0.01) / (p2 - p1)

# --- APP UI ---
st.title("🛡️ Institutional FCN Solver")

with st.sidebar:
    tickers = [x.strip().upper() for x in st.text_input("Tickers", "AAPL, MSFT").split(",")]
    sims = st.select_slider("Simulations", [5000, 10000], 5000)
    rf = st.number_input("Risk Free %", 0.0, 10.0, 4.5) / 100
    skew = st.slider("Dynamic Skew (bps)", 0, 300, 120)
    
    st.divider()
    tenor = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    freq = st.selectbox("Coupon Freq (M)", [1, 3, 6])
    nc = st.number_input("Non-Call (M)", 0, 12, 3)
    strike_p = st.number_input("Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 105)

if st.button("Calculate & Generate Report"):
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        # 1. Fetch Data
        vols, corr = get_market_data(tickers, 12)
        
        # 2. Setup Progress
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        # 3. Define Ranges
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        total_cells = len(ss) * len(kk)
        
        y_grid, ko_grid, ki_grid = [], [], []
        
        # 4. Main Calculation Loop
        count = 0
        for kv in kk:
            y_r, ko_r, ki_r = [], [], []
            for sv in ss:
                count += 1
                prog_bar.progress(count / total_cells)
                status_text.text(f"Solving Matrix: Strike {sv}% | KO {kv}%")
                
                # Path Gen with Dynamic Skew
                L = np.linalg.cholesky(corr)
                adj_vols = vols * (1 + (skew/10000) * (100 - sv))
                dt, steps = 1/252, int(tenor * 252)
                raw = np.zeros((steps + 1, sims, len(tickers)))
                raw[0] = 100.0
                for t in range(1, steps + 1):
                    z = np.random.standard_normal((sims, len(tickers))) @ L.T
                    raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
                wo = np.min(raw, axis=2)
                
                # Solve
                y = solve_yield(wo, rf, tenor, sv, kv, freq, nc)
                res = run_valuation(y, wo, rf, tenor, sv, kv, freq, nc)
                
                y_r.append(y*100)
                ko_r.append(res['prob_ko']*100)
                ki_r.append(res['prob_ki']*100)
                
            y_grid.append(y_r); ko_grid.append(ko_r); ki_grid.append(ki_r)
        
        # 5. Render Tables
        status_text.success("Calculation Complete!")
        
        st.subheader("Yield Sensitivity Table (% p.a.)")
        st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Prob. of KO (%)**")
            st.dataframe(pd.DataFrame(ko_grid, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None).format("{:.1f}"))
        with c2:
            st.write("**Prob. of Capital Loss (%)**")
            st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
