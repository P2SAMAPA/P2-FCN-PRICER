import streamlit as st
import numpy as np
import pandas as pd  # Fixed: changed from 'import pd'
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Institutional FCN & BCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    ivs, hvs = [], []
    for t in tickers:
        s = yf.Ticker(t)
        # Implied Vol (ATM)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            # Use mid-term options for stable IV
            target_idx = min(len(s.options)-1, 5)
            chain = s.option_chain(s.options[target_idx])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(max(0.05, min(1.2, iv))) # Sanity cap
        except: ivs.append(0.30)
        
        # Historical Vol
        h = s.history(period=f"{lookback}mo")['Close'].pct_change().std() * np.sqrt(252)
        hvs.append(max(0.05, h))
    
    # Correlation
    data = yf.download(tickers, period=f"{lookback}mo", progress=False)['Close']
    if len(tickers) > 1:
        corr = data.pct_change().dropna().corr().values
    else:
        corr = np.array([[1.0]])
    
    # Ensure Matrix is Positive Definite
    corr = corr + np.eye(len(tickers)) * 1e-6
    return np.array(ivs), np.array(hvs), corr

# --- ENGINE ---
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
        final_px = paths[-1, active]
        # Real-world settle: If below strike, you get the asset value (Capital Loss)
        principal = np.where(final_px >= strike, 100, final_px)
        num = (steps - 1) // obs_freq
        payoffs[active] = principal + (num * cpn_per_period)
        cpn_counts[active] = num
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": 1 - (np.sum(active) / n_sims),
        "prob_ki": np.sum(active & (paths[-1] < strike)) / n_sims,
        "avg_cpn": np.mean(cpn_counts)
    }

def get_wo_paths(sims, t_yr, rf, vols, corr, strike_val, skew_factor):
    L = np.linalg.cholesky(corr)
    dt, steps = 1/252, int(t_yr * 252)
    
    # SKEW LOGIC: Higher volatility for lower strikes to reflect downside fear
    # If strike is 60, (1 - 60/100) = 0.4. Skew 0.8 * 0.4 = 32% vol increase.
    adj_vols = vols * (1 + (skew_factor * (1 - strike_val/100)))
    
    raw = np.zeros((steps + 1, sims, len(vols)))
    raw[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((sims, len(vols))) @ L.T
        raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
    return np.min(raw, axis=2)

def solve_yield(wo_paths, r, tenor, strike, ko, freq_m, nc_m):
    # Newton-style linear search for Par coupon
    p1 = run_valuation(0.01, wo_paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    p2 = run_valuation(0.40, wo_paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    if abs(p2-p1) < 1e-4: return r
    return 0.01 + (100 - p1) * (0.40 - 0.01) / (p2 - p1)

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    st.header("1. Asset Inputs")
    tk_in = st.text_input("Tickers (CSV)", "TSLA, MSFT")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_mode = st.radio("Vol Mode", ["Market Implied (IV)", "Historical (HV)"])
    skew_val = st.slider("Vol Skew Factor", 0.0, 1.0, 0.8)
    rf = st.number_input("Risk Free %", 0.0, 10.0, 4.5) / 100
    
    st.header("2. Product Specs")
    tenor = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    freq = st.selectbox("Coupon Freq (M)", [1, 3, 6])
    nc = st.number_input("Non-Call (M)", 0, 12, 3)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    sims = st.select_slider("Simulations", [5000, 10000, 20000], 10000)

if st.button("Calculate & Solve"):
    ivs, hvs, corr = get_market_data(tickers, 12)
    base_vols = ivs if vol_mode == "Market Implied (IV)" else hvs
    
    # 1. Main Case Result
    with st.status("Solving Par Yield...") as status:
        wo_main = get_wo_paths(sims, tenor, rf, base_vols, corr, strike_p, skew_val)
        main_y = solve_yield(wo_main, rf, tenor, strike_p, ko_p, freq, nc)
        res = run_valuation(main_y, wo_main, rf, tenor, strike_p, ko_p, freq, nc)
        
        st.header(f"Solved Annualized Yield: {main_y*100:.2f}% p.a.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. of KO", f"{res['prob_ko']*100:.1f}%")
        c2.metric("Prob. of Capital Loss", f"{res['prob_ki']*100:.1f}%")
        c3.metric("Avg. Coupons Paid", f"{res['avg_cpn']:.2f}")

        # 2. Sensitivity Matrices
        status.update(label="Building Sensitivity Tables...")
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        y_grid, ki_grid = [], []

        for kv in kk:
            y_r, ki_r = [], []
            for sv in ss:
                wo = get_wo_paths(sims, tenor, rf, base_vols, corr, sv, skew_val)
                y = solve_yield(wo, rf, tenor, sv, kv, freq, nc)
                r_val = run_valuation(y, wo, rf, tenor, sv, kv, freq, nc)
                y_r.append(y*100)
                ki_r.append(r_val['prob_ki']*100)
            y_grid.append(y_r); ki_grid.append(ki_r)
        status.update(label="Complete!", state="complete")

    st.subheader("Yield Sensitivity Table (% p.a.)")
    st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
    
    st.subheader("Capital Loss Probability (%)")
    st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
