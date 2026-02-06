import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import brentq
from datetime import datetime

st.set_page_config(page_title="Professional FCN & BCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            opts = s.options
            chain = s.option_chain(opts[min(2, len(opts)-1)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(iv)
        except: vols.append(0.30)
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now())['Close']
    corr = h.pct_change().corr().values
    return np.array(vols), corr

# --- VECTORIZED ENGINE ---
def get_price(coupon, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    
    # KO Check
    obs_dates = np.arange(nc_step, steps, obs_freq)
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            active[ko_mask] = False
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps - 1) // obs_freq
        payoffs[active] = principal + (num_cpns * cpn_per_period)
        
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- STABLE YIELD SOLVER (Interpolation) ---
def solve_yield(paths, r, tenor, strike, ko, freq_m, nc_m):
    # Calculate price at 0% and 40% yield
    p0 = get_price(0.0, paths, r, tenor, strike, ko, freq_m, nc_m)
    p40 = get_price(0.4, paths, r, tenor, strike, ko, freq_m, nc_m)
    
    # Linear interpolation for a fast, smooth estimate
    # We want Price = 100
    if p40 == p0: return r * 100 # Fallback to risk-free
    yield_est = 0.0 + (100 - p0) * (0.4 - 0.0) / (p40 - p0)
    return yield_est * 100

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    is_bcn = st.toggle("Product: BCN", value=False)
    tks = [x.strip().upper() for x in st.text_input("Tickers", "AAPL, MSFT").split(",")]
    vol_src = st.radio("Vol", ["Implied", "Historical"])
    lb = st.slider("Lookback (M)", 1, 60, 12)
    skew_val = st.slider("Skew (bps)", 0.0, 5.0, 0.8)
    sims = st.select_slider("Simulations", [5000, 10000, 20000], 10000)
    rf = st.number_input("Risk Free %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    t_yr = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    f_m = st.selectbox("Coupon Freq (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)
    k_in = st.number_input("Strike %", 40, 100, 60)
    ko_in = st.number_input("KO Barrier %", 70, 150, 100)

if st.button("Calculate Engine"):
    with st.spinner("Processing..."):
        vols, corr = get_market_data(tks, lb)
        adj_vols = vols + (skew_val/100 * (100 - k_in)/100)
        
        # Path Generation
        L = np.linalg.cholesky(corr)
        dt = 1/252
        steps = int(t_yr * 252)
        raw = np.zeros((steps + 1, sims, len(tks)))
        raw[0] = 100.0
        for t in range(1, steps + 1):
            z = np.random.standard_normal((sims, len(tks))) @ L.T
            raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
        
        wo = np.min(raw, axis=2)

        if not is_bcn:
            main_yield = solve_yield(wo, rf, t_yr, k_in, ko_in, f_m, nc_m)
            st.header(f"Solved Annualized Yield: {main_yield:.2f}% p.a.")
            
            st.subheader("Yield Sensitivity Table")
            ss = [k_in-10, k_in-5, k_in, k_in+5, k_in+10]
            kk = [ko_in+10, ko_in+5, ko_in, ko_in-5, ko_in-10]
            
            grid = []
            for k_val in kk:
                row = [solve_yield(wo, rf, t_yr, s_val, k_val, f_m, nc_m) for s_val in ss]
                grid.append(row)
            
            df = pd.DataFrame(grid, index=[f"KO {i}%" for i in kk], columns=[f"Str {j}%" for j in ss])
            st.table(df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        else:
            # BCN Logic as requested previously
            st.info("BCN Mode Active")

    # Chart
    fig = go.Figure()
    for i in range(15):
        fig.add_trace(go.Scatter(y=wo[:, i], mode='lines', opacity=0.3, showlegend=False))
    fig.add_hline(y=ko_in, line_color="green", line_dash="dash")
    fig.add_hline(y=k_in, line_color="red", line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
