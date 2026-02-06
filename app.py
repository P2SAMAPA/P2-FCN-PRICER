import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    ivs = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            chain = s.option_chain(s.options[min(len(s.options)-1, 5)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(max(0.20, iv)) # Floor at 20% for realistic risk
        except: ivs.append(0.35)
    
    data = yf.download(tickers, period="12mo", progress=False)['Close']
    corr = data.pct_change().dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(ivs), corr

# --- PRICING ENGINE ---
def get_worst_of_paths(sims, tenor, rf, vols, corr, strike_p, skew_val):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    
    # FAT-TAIL ADJUSTMENT: We multiply vol for the downside to ensure Prob of Loss > 0
    # This simulates "Jump Risk" or "Volatility Smile"
    adj_vols = vols * (1 + (skew_val * (1 - strike_p/100) * 3.0))
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    
    # Geometric Brownian Motion with drift
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * epsilon
    path_returns = np.exp(np.cumsum(drift + diffusion, axis=0))
    
    # Stack initial price of 100
    paths = np.vstack([np.ones((1, sims, len(vols))), path_returns]) * 100
    return np.min(paths, axis=2) # Returns the Worst-Of Asset path

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
        principal = np.where(final_px >= strike, 100, final_px) # Capital Loss happens here
        num = (steps - 1) // obs_freq
        payoffs[active] = principal + (num * cpn_per_obs)
        cpn_earned[active] = num
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": np.mean(cpn_earned)
    }

# --- UI LOGIC ---
st.sidebar.title("Parameters")
tk_in = st.sidebar.text_input("Tickers", "TSLA, MSFT")
tickers = [x.strip().upper() for x in tk_in.split(",")]
skew = st.sidebar.slider("Risk Multiplier (Skew)", 0.0, 1.5, 0.9)
rf_rate = st.sidebar.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

st.sidebar.subheader("Product Definition")
tenor_y = st.sidebar.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
strike_pct = st.sidebar.number_input("Put Strike %", 40, 100, 60)
ko_pct = st.sidebar.number_input("KO Barrier %", 70, 150, 100)
freq_m = st.sidebar.selectbox("Freq (Months)", [1, 3, 6])

if st.sidebar.button("Calculate & Solve"):
    vols, corr = get_market_data(tickers)
    paths_main = get_worst_of_paths(10000, tenor_y, rf_rate, vols, corr, strike_pct, skew)
    
    # Solve for Yield (Par Pricing)
    p1 = run_valuation(0.01, paths_main, rf_rate, tenor_y, strike_pct, ko_pct, freq_m, 3)['price']
    p2 = run_valuation(0.50, paths_main, rf_rate, tenor_y, strike_pct, ko_pct, freq_m, 3)['price']
    solved_y = 0.01 + (100 - p1) * (0.50 - 0.01) / (p2 - p1)
    
    # Final Result
    res = run_valuation(solved_y, paths_main, rf_rate, tenor_y, strike_pct, ko_pct, freq_m, 3)

    # --- TOP OUTPUTS (REQUESTED) ---
    st.header(f"Target Annualized Yield: {solved_y*100:.2f}% p.a.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    col2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    col3.metric("Avg. Coupons Paid", f"{res['avg_cpn']:.2f}")
    
    # --- SENSITIVITIES (BELOW) ---
    st.divider()
    st.subheader("Yield Sensitivity & Risk Analysis")
    
    ss = [strike_pct-10, strike_pct-5, strike_pct, strike_pct+5, strike_pct+10]
    kk = [ko_pct+10, ko_pct+5, ko_pct, ko_pct-5, ko_pct-10]
    y_grid, ki_grid = [], []

    for kv in kk:
        y_r, ki_r = [], []
        for sv in ss:
            p_cell = get_worst_of_paths(5000, tenor_y, rf_rate, vols, corr, sv, skew)
            # Solve yield for cell
            c1 = run_valuation(0.05, p_cell, rf_rate, tenor_y, sv, kv, freq_m, 3)['price']
            c2 = run_valuation(0.45, p_cell, rf_rate, tenor_y, sv, kv, freq_m, 3)['price']
            yc = 0.05 + (100 - c1) * (0.45 - 0.05) / (c2 - c1)
            
            rc = run_valuation(yc, p_cell, rf_rate, tenor_y, sv, kv, freq_m, 3)
            y_r.append(yc * 100)
            ki_r.append(rc['prob_loss'])
        y_grid.append(y_r)
        ki_grid.append(ki_r)

    st.write("**Yield Sensitivity Table (% p.a.)**")
    st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
    
    st.write("**Capital Loss Probability Matrix (%)**")
    st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
