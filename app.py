import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_context(tickers):
    data = []
    for t in tickers:
        s = yf.Ticker(t)
        h = s.history(period="12mo")['Close']
        data.append(h.rename(t))
    df = pd.concat(data, axis=1).dropna()
    return df, df.pct_change().corr().values

def get_vol_vector(tickers, source):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        hist_vol = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        if source == "Market Implied (IV)":
            try:
                # Target ATM options ~30-60 days out
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 2)])
                px = s.history(period="1d")['Close'].iloc[-1]
                iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
                vols.append(iv)
            except: vols.append(hist_vol)
        else: vols.append(hist_vol)
    return np.array(vols)

# --- PRICING ENGINE ---
def simulate_fcn(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, n_assets = paths.shape
    worst_of = np.min(paths, axis=2)
    
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    obs_dates = np.arange(obs_interval, steps, obs_interval)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    cpns_count = np.zeros(n_sims)
    cpn_rate = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        cpns_count[active] += 1
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + (cpns_count[ko_mask] * cpn_rate)
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of[-1, active]
        # Principal protection logic
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + (cpns_count[active] * cpn_rate)
        
    # Return PV of payoff
    return np.mean(payoffs) * np.exp(-r * tenor)

def get_paths(sims, tenor, rf, vols, corr, skew):
    # CRITICAL: Volatility must be non-zero and skew-adjusted
    adj_vols = np.maximum(vols * (1 + skew), 0.01) 
    dt = 1/252
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    
    # GBM formula: S_t = S_0 * exp((r - 0.5*sigma^2)t + sigma*W_t)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * eps
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

# --- UI ---
st.title("🛡️ Institutional Fixed Coupon Note (FCN) Pricer")

with st.sidebar:
    st.header("1. Assets & Volatility")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    v_src = st.radio("Volatility Source", ["Market Implied (IV)", "Historical (HV)"])
    skew = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.2)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.header("2. Note Parameters")
    tenor = st.number_input("Tenor (Years)", 0.5, 5.0, 1.0)
    freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    stk = st.number_input("Put Strike %", 40, 100, 60)
    ko = st.number_input("KO Barrier %", 80, 150, 105)

if st.button("Solve FCN Structure"):
    with st.spinner("Simulating Market Paths..."):
        df, corr = get_market_context(tickers)
        vols = get_vol_vector(tickers, v_src)
        paths = get_paths(15000, tenor, rf, vols, corr, skew)
        
        # Solver: Find Coupon where PV(Payoffs) = 100
        # We target the range [0% , 100%] annual coupon
        try:
            target_fn = lambda c: simulate_fcn(c, paths, rf, tenor, stk, ko, freq, nc) - 100
            y_solve = brentq(target_fn, 0.0, 1.0)
        except:
            # Fallback if no solution in range
            y_solve = 0.0
            
        # Run final stats
        res_pv = simulate_fcn(y_solve, paths, rf, tenor, stk, ko, freq, nc)
        worst_final = np.min(paths[-1], axis=1)
        p_loss = (np.sum(worst_final < stk) / 15000) * 100

    st.subheader(f"Solved Annual Yield: {y_solve*100:.2f}% p.a.")
    
    col1, col2 = st.columns(2)
    col1.metric("Prob. of Capital Loss", f"{p_loss:.1f}%")
    col2.metric("Note Price Check", f"${res_pv:.2f}")

    # Sensitivities
    st.divider()
    st.write("### Yield Sensitivity Matrix (% p.a.)")
    
    s_lvls = [stk-10, stk, stk+10]
    k_lvls = [ko+10, ko, ko-10]
    
    grid = []
    for k in k_lvls:
        row = []
        for s in s_lvls:
            fn = lambda c: simulate_fcn(c, paths, rf, tenor, s, k, freq, nc) - 100
            try: val = brentq(fn, 0.0, 1.5) * 100
            except: val = 0.0
            row.append(val)
        grid.append(row)
        
    df_sens = pd.DataFrame(grid, index=[f"KO {k}%" for k in k_lvls], columns=[f"Strike {s}%" for s in s_lvls])
    st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))
