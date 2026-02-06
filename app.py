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
            # Use 6-month out options for a more realistic institutional IV
            chain = s.option_chain(opts[min(3, len(opts)-1)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(iv)
        except: vols.append(0.30)
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now())['Close']
    corr = h.pct_change().corr().values
    return np.array(vols), corr

# --- IMPROVED PRICING ENGINE ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    ko_count = 0
    cpn_counts = np.zeros(n_sims)
    
    obs_dates = np.arange(nc_step, steps, obs_freq)
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            cpn_counts[ko_mask] = num_cpns
            active[ko_mask] = False
            ko_count += np.sum(ko_mask)
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps - 1) // obs_freq
        payoffs[active] = principal + (num_cpns * cpn_per_period)
        cpn_counts[active] = num_cpns
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": (ko_count / n_sims),
        "prob_ki": np.sum(active & (paths[-1] < strike)) / n_sims,
        "avg_cpn_count": np.mean(cpn_counts)
    }

# --- DYNAMIC PATH GENERATOR (Crucial for Strike Sensitivity) ---
def get_paths(sims, tks, t_yr, rf, vols, corr, strike_val, skew_bps):
    L = np.linalg.cholesky(corr)
    dt, steps = 1/252, int(t_yr * 252)
    # Apply Skew relative to the Strike: Lower strike = Higher Vol
    adj_vols = vols * (1 + (skew_bps/10000) * (100 - strike_val))
    
    raw = np.zeros((steps + 1, sims, len(tks)))
    raw[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((sims, len(tks))) @ L.T
        raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
    return np.min(raw, axis=2)

# --- INTERPOLATION SOLVER ---
def solve_par_yield(wo_paths, rf, t_yr, strike, ko, freq_m, nc_m):
    p1 = run_valuation(0.02, wo_paths, rf, t_yr, strike, ko, freq_m, nc_m)['price']
    p2 = run_valuation(0.30, wo_paths, rf, t_yr, strike, ko, freq_m, nc_m)['price']
    if abs(p2 - p1) < 1e-4: return rf
    return 0.02 + (100 - p1) * (0.30 - 0.02) / (p2 - p1)

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    tk_in = st.text_input("Tickers", "AAPL, MSFT")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    skew_bps = st.slider("Dynamic Skew (bps per 1% Strike OTM)", 0, 200, 80)
    sims = st.select_slider("Simulations", [5000, 10000, 20000], 10000)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    t_yr = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    f_m = st.selectbox("Coupon Frequency (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)
    k_in = st.number_input("Put Strike %", 40, 100, 60)
    ko_in = st.number_input("KO Barrier %", 70, 150, 100)

if st.button("Calculate Everything"):
    with st.spinner("Solving..."):
        vols, corr = get_market_data(tks, 12)
        
        # SENSITIVITY RANGES
        ss = [k_in-10, k_in-5, k_in, k_in+5, k_in+10]
        kk = [ko_in+10, ko_in+5, ko_in, ko_in-5, ko_in-10]
        
        yield_results, ko_results, ki_results, cpn_results = [], [], [], []

        for kv in kk:
            y_r, ko_r, ki_r, c_r = [], [], [], []
            for sv in ss:
                # IMPORTANT: Regenerate paths for each strike to apply Dynamic Skew
                wo = get_paths(sims, tks, t_yr, rf, vols, corr, sv, skew_bps)
                y = solve_par_yield(wo, rf, t_yr, sv, kv, f_m, nc_m)
                res = run_valuation(y, wo, rf, t_yr, sv, kv, f_m, nc_m)
                
                y_r.append(y*100); ko_r.append(res['prob_ko']*100)
                ki_r.append(res['prob_ki']*100); c_r.append(res['avg_cpn_count'])
            
            yield_results.append(y_r); ko_results.append(ko_r)
            ki_results.append(ki_r); cpn_results.append(c_r)

        # DISPLAY TABLES
        st.subheader("Yield Sensitivity (%)")
        st.dataframe(pd.DataFrame(yield_results, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Prob. of KO (%)**")
            st.dataframe(pd.DataFrame(ko_results, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None).format("{:.1f}"))
        with col2:
            st.write("**Prob. of Capital Loss (%)**")
            st.dataframe(pd.DataFrame(ki_results, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
            
        st.write("**Avg. Number of Coupons Paid**")
        st.dataframe(pd.DataFrame(cpn_results, index=kk, columns=ss).style.background_gradient(cmap='YlGn', axis=None).format("{:.2f}"))

    # Sample Paths visualization for the base case
    wo_base = get_paths(sims, tks, t_yr, rf, vols, corr, k_in, skew_bps)
    fig = go.Figure([go.Scatter(y=wo_base[:, i], mode='lines', opacity=0.2) for i in range(15)])
    fig.add_hline(y=ko_in, line_color="green", line_dash="dash", annotation_text="KO")
    fig.add_hline(y=k_in, line_color="red", line_dash="dash", annotation_text="Strike")
    st.plotly_chart(fig, use_container_width=True)
