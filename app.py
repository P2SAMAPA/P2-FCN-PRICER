import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA ENGINE (FETCH ONCE) ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data = []
    for t in tickers:
        s = yf.Ticker(t)
        h = s.history(period="12mo")['Close']
        data.append(h.rename(t))
    df = pd.concat(data, axis=1).dropna()
    return df, df.pct_change().corr().values

def get_vols(tickers, source):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        # Always get HV as a floor
        hist_v = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        if source == "Market Implied (IV)":
            try:
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 2)])
                px = s.history(period="1d")['Close'].iloc[-1]
                vols.append(chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0])
            except: vols.append(hist_v)
        else: vols.append(hist_v)
    return np.array(vols)

# --- PRICING LOGIC ---
def run_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    obs_dates = np.arange(obs_interval, steps, obs_interval)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    cpns_to_date = np.zeros(n_sims)
    cpn_payment = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        cpns_to_date[active] += 1
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + (cpns_to_date[ko_mask] * cpn_payment)
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + (cpns_to_date[active] * cpn_payment)
        
    return np.mean(payoffs) * np.exp(-r * tenor)

def generate_monte_carlo(sims, tenor, rf, vols, corr, skew):
    # CRITICAL: We boost the volatility by skew to ensure the model 'feels' the risk
    adj_vols = vols * (1 + skew)
    dt = 1/252
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * eps
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

# --- DASHBOARD UI ---
st.title("🛡️ Institutional FCN Solver")

with st.sidebar:
    st.header("1. Market Inputs")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Market Implied (IV)", "Historical (HV)"])
    skew_f = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.header("2. Product Specs")
    tenor_y = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0)
    freq = st.selectbox("Coupon Frequency (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)
    put_stk = st.slider("Put Strike %", 40, 100, 60)
    ko_bar = st.slider("KO Barrier %", 80, 150, 100)

if st.button("Solve & Generate Sensitivities"):
    df, corr = get_market_data(tickers)
    vols = get_vols(tickers, vol_src)
    paths = generate_monte_carlo(15000, tenor_y, rf_rate, vols, corr, skew_f)
    
    # Solve for 100 Par using Brent's Method
    target = lambda c: run_valuation(c, paths, rf_rate, tenor_y, put_stk, ko_bar, freq, nc_m) - 100
    try:
        y_solve = brentq(target, 0.0, 2.0)
    except:
        y_solve = 0.0 # Fallback

    # Metrics
    st.subheader(f"Solved Yield: {y_solve*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    
    worst_final = np.min(paths[-1], axis=1)
    p_loss = (np.sum(worst_final < put_stk) / 15000) * 100
    
    c1.metric("Prob. Capital Loss", f"{p_loss:.1f}%")
    c2.metric("Avg Volatility", f"{np.mean(vols*(1+skew_f)):.1%}")
    c3.metric("Note Price Check", f"${run_valuation(y_solve, paths, rf_rate, tenor_y, put_stk, ko_bar, freq, nc_m):.2f}")

    # Sensitivity Matrix
    st.divider()
    st.write("### Yield Sensitivity Matrix (% p.a.)")
    
    stks = [put_stk-10, put_stk-5, put_stk, put_stk+5, put_stk+10]
    bars = [ko_bar+10, ko_bar+5, ko_bar, ko_bar-5, ko_bar-10]
    
    res_grid = []
    for b in bars:
        row = []
        for s in stks:
            t_fn = lambda c: run_valuation(c, paths, rf_rate, tenor_y, s, b, freq, nc_m) - 100
            try: row.append(brentq(t_fn, 0.0, 3.0) * 100)
            except: row.append(0.0)
        res_grid.append(row)
        
    df_sens = pd.DataFrame(res_grid, index=[f"KO {x}%" for x in bars], columns=[f"Stk {x}%" for x in stks])
    st.table(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
