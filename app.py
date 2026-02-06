import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback_m):
    ivs, hvs = [], []
    for t in tickers:
        s = yf.Ticker(t)
        # 1. Market Implied Vol (ATM ~6 months out)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            chain = s.option_chain(s.options[min(len(s.options)-1, 6)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(iv)
        except: ivs.append(0.35)
        # 2. Historical Vol
        h = s.history(period=f"{lookback_m}mo")['Close'].pct_change().std() * np.sqrt(252)
        hvs.append(h)
    
    # 3. Correlation Matrix
    data = yf.download(tickers, period=f"{lookback_m}mo", progress=False)['Close']
    corr = data.pct_change().dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(ivs), np.array(hvs), corr

# --- PRICING ENGINE ---
def get_simulation(sims, tenor, rf, vols, corr, strike_p, skew_val):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    # Skew Logic: Increases vol as strike gets lower to ensure "Capital Loss" isn't 0%
    adj_vols = vols * (1 + (skew_val * (1 - strike_p/100) * 3.0))
    
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
            num = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num * cpn_per_obs)
            cpn_earned[ko_mask] = num
            active[ko_mask] = False
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num = (steps - 1) // obs_freq
        payoffs[active] = principal + (num * cpn_per_obs)
        cpn_earned[active] = num
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": np.mean(cpn_earned)
    }

# --- UI ---
st.title("🛡️ Institutional Fixed Coupon Note (FCN) Pricer")

with st.sidebar:
    st.header("Asset Selection")
    tk_in = st.text_input("Tickers (CSV)", "TSLA, MSFT")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_choice = st.radio("Volatility Input", ["Market Implied (IV)", "Historical (HV)"])
    skew_f = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.8)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    st.header("Product Parameters")
    tenor_y = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    sims = st.select_slider("Simulations", [5000, 10000, 20000], 10000)

if st.button("Solve FCN Structure"):
    with st.spinner("Fetching Market Data & Simulating Paths..."):
        ivs, hvs, corr = get_market_data(tickers, 12)
        base_vols = ivs if vol_choice == "Market Implied (IV)" else hvs
        
        # 1. Main Calculation
        paths_main = get_simulation(sims, tenor_y, rf_rate, base_vols, corr, strike_p, skew_f)
        
        # Binary search for Par yield
        p_lo = run_valuation(0.01, paths_main, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)['price']
        p_hi = run_valuation(0.50, paths_main, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)['price']
        solved_y = 0.01 + (100 - p_lo) * (0.50 - 0.01) / (p_hi - p_lo)
        
        res = run_valuation(solved_y, paths_main, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

        # --- TOP LEVEL OUTPUTS ---
        st.header(f"Solved Annualized Yield: {solved_y*100:.2f}% p.a.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prob. of KO (Early Redemption)", f"{res['prob_ko']:.1f}%")
        col2.metric("Prob. of Capital Loss (At Exp)", f"{res['prob_loss']:.1f}%")
        col3.metric("Avg. Number of Coupons Paid", f"{res['avg_cpn']:.2f}")

        # --- SENSITIVITY ---
        st.divider()
        st.subheader("Structure Sensitivity Analysis")
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        y_grid, ki_grid = [], []

        for kv in kk:
            y_r, ki_r = [], []
            for sv in ss:
                p_cell = get_simulation(5000, tenor_y, rf_rate, base_vols, corr, sv, skew_f)
                c_lo = run_valuation(0.05, p_cell, rf_rate, tenor_y, sv, kv, freq_m, nc_m)['price']
                c_hi = run_valuation(0.45, p_cell, rf_rate, tenor_y, sv, kv, freq_m, nc_m)['price']
                yc = 0.05 + (100 - c_lo) * (0.45 - 0.05) / (c_hi - c_lo)
                rc = run_valuation(yc, p_cell, rf_rate, tenor_y, sv, kv, freq_m, nc_m)
                y_r.append(yc * 100); ki_r.append(rc['prob_loss'])
            y_grid.append(y_r); ki_grid.append(ki_r)

        st.write("**Yield Sensitivity Matrix (% p.a.)**")
        st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        st.write("**Capital Loss Probability Matrix (%)**")
        st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
