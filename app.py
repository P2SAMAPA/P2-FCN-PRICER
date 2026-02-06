import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
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

# --- CORE ENGINE WITH PROBABILITY STATS ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m, is_bcn=False, bonus=0):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    ko_count = 0
    coupons_earned = np.zeros(n_sims)
    
    # KO Check loop
    obs_dates = np.arange(nc_step, steps, obs_freq)
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            coupons_earned[ko_mask] = num_cpns
            active[ko_mask] = False
            ko_count += np.sum(ko_mask)
            
    # Maturity Check for survivors
    if np.any(active):
        final_px = paths[-1, active]
        # Principal recovery
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps - 1) // obs_freq
        b_val = (bonus * 100) if is_bcn else 0
        payoffs[active] = principal + (num_cpns * cpn_per_period) + b_val
        coupons_earned[active] = num_cpns
        
    price = np.mean(payoffs) * np.exp(-r * tenor)
    
    # Probabilities
    prob_ko = (ko_count / n_sims)
    prob_ki = np.sum(active & (paths[-1] < strike)) / n_sims
    avg_coupons = np.mean(coupons_earned)
    
    return {
        "price": price, 
        "prob_ko": prob_ko, 
        "prob_ki": prob_ki, 
        "avg_cpn": avg_coupons
    }

# --- STABLE SOLVER ---
def solve_fcn_yield(paths, r, tenor, strike, ko, freq_m, nc_m):
    # Calculate price at 2 points to interpolate (avoiding solver "snapping")
    res1 = run_valuation(0.05, paths, r, tenor, strike, ko, freq_m, nc_m)
    res2 = run_valuation(0.25, paths, r, tenor, strike, ko, freq_m, nc_m)
    
    p1, p2 = res1['price'], res2['price']
    # Linear interpolation: y = y1 + (100 - p1) * (y2 - y1) / (p2 - p1)
    if abs(p2 - p1) < 1e-6: return 0.0
    solved_y = 0.05 + (100 - p1) * (0.25 - 0.05) / (p2 - p1)
    return max(0, solved_y)

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    is_bcn = st.toggle("Product Mode: BCN", value=False)
    tk_in = st.text_input("Tickers (CSV)", "AAPL, MSFT")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Implied", "Historical"])
    lb = st.slider("Lookback (Months)", 1, 60, 12)
    skew = st.slider("Skew (bps)", 0.0, 5.0, 0.8)
    sims = st.select_slider("Simulations", [5000, 10000, 20000], 10000)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    t_yr = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    f_m = st.selectbox("Coupon Freq (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)
    k_in = st.number_input("Put Strike %", 40, 100, 60)
    ko_in = st.number_input("KO Barrier %", 70, 150, 100)

if st.button("Calculate & Solve"):
    with st.spinner("Generating Correlated Paths..."):
        vols, corr = get_market_data(tks, lb)
        adj_vols = vols + (skew/100 * (100 - k_in)/100)
        
        # Path Generation (Worst-of)
        L = np.linalg.cholesky(corr)
        dt, steps = 1/252, int(t_yr * 252)
        raw = np.zeros((steps + 1, sims, len(tks)))
        raw[0] = 100.0
        for t in range(1, steps + 1):
            z = np.random.standard_normal((sims, len(tks))) @ L.T
            raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
        wo = np.min(raw, axis=2)

    if not is_bcn:
        # FCN Solution
        final_yield = solve_fcn_yield(wo, rf, t_yr, k_in, ko_in, f_m, nc_m)
        stats = run_valuation(final_yield, wo, rf, t_yr, k_in, ko_in, f_m, nc_m)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Yield", f"{final_yield*100:.2f}%")
        c2.metric("Prob. of KO", f"{stats['prob_ko']*100:.1f}%")
        c3.metric("Prob. of Capital Loss", f"{stats['prob_ki']*100:.1f}%")
        
        # SENSITIVITY TABLE
        st.subheader("Yield Sensitivity & Risk Probabilities")
        ss = [k_in-10, k_in-5, k_in, k_in+5, k_in+10]
        kk = [ko_in+10, ko_in+5, ko_in, ko_in-5, ko_in-10]
        
        yield_grid, ko_grid, ki_grid = [], [], []
        for k_val in kk:
            y_row, ko_row, ki_row = [], [], []
            for s_val in ss:
                y = solve_fcn_yield(wo, rf, t_yr, s_val, k_val, f_m, nc_m)
                stt = run_valuation(y, wo, rf, t_yr, s_val, k_val, f_m, nc_m)
                y_row.append(y*100)
                ko_row.append(stt['prob_ko']*100)
                ki_row.append(stt['prob_ki']*100)
            yield_grid.append(y_row); ko_grid.append(ko_row); ki_grid.append(ki_row)
        
        st.write("**Annualized Yield (%)**")
        st.table(pd.DataFrame(yield_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        st.write("**Probability of KO (%)**")
        st.table(pd.DataFrame(ko_grid, index=kk, columns=ss).format("{:.1f}"))
        
        st.write("**Probability of Capital Loss (%)**")
        st.table(pd.DataFrame(ki_grid, index=kk, columns=ss).format("{:.1f}"))

    # Path Visualization
    fig = go.Figure()
    for i in range(15):
        fig.add_trace(go.Scatter(y=wo[:, i], mode='lines', opacity=0.3, showlegend=False))
    fig.add_hline(y=ko_in, line_color="green", line_dash="dash", annotation_text="KO")
    fig.add_hline(y=k_in, line_color="red", line_dash="dash", annotation_text="Strike")
    st.plotly_chart(fig, use_container_width=True)
