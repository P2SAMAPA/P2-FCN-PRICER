import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- STEP 1: CACHE ONLY THE RAW DATA ---
@st.cache_data(ttl=3600)
def fetch_market_prices(tickers):
    """Only fetches prices and returns. No pricing logic here."""
    data_list = []
    for t in tickers:
        s = yf.Ticker(t)
        hist = s.history(period="12mo")['Close']
        data_list.append(hist.rename(t))
    df = pd.concat(data_list, axis=1).dropna()
    return df, df.pct_change().corr().values

# --- STEP 2: DYNAMIC VOLATILITY CALCULATOR (NO CACHE) ---
def get_current_vols(tickers, source="IV"):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        if source == "Market Implied (IV)":
            try:
                px_curr = s.history(period="1d")['Close'].iloc[-1]
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 4)])
                calls = chain.calls
                vols.append(calls.iloc[(calls['strike'] - px_curr).abs().argsort()[:1]]['impliedVolatility'].values[0])
            except:
                vols.append(s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252))
        else:
            vols.append(s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252))
    return np.array(vols)

# --- STEP 3: PATH GENERATION (FORCE RERUN) ---
def generate_paths(sims, tenor, rf, base_vols, corr, skew):
    # CRITICAL: Apply Skew here so it affects every path
    adj_vols = base_vols * (1 + skew)
    dt = 1/252
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    
    # New random seed every time to ensure movement
    z = np.random.standard_normal((steps, sims, len(base_vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * epsilon
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    return np.vstack([np.ones((1, sims, len(base_vols))), paths]) * 100

def price_fcn_logic(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    
    payoffs, active, coupons = np.zeros(n_sims), np.ones(n_sims, dtype=bool), np.zeros(n_sims)
    cpn_val = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        coupons[active] += 1
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            if np.any(ko_mask):
                payoffs[ko_mask] = 100 + (coupons[ko_mask] * cpn_val)
                active[ko_mask] = False
    
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + (coupons[active] * cpn_val)
        
    return np.mean(payoffs) * np.exp(-r * tenor), np.mean(coupons), (np.sum(active & (worst_of[-1] < strike))/n_sims)*100

# --- UI (UNTOUCHED PER REQUEST) ---
st.title("🛡️ Institutional Fixed Coupon Note (FCN) Pricer")

with st.sidebar:
    st.header("1. Market Configuration")
    tk_in = st.text_input("Underlying Tickers", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_source = st.radio("Volatility Input", ["Market Implied (IV)", "Historical (HV)"])
    skew_val = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.2)
    
    st.header("2. Note Structure")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_m = st.selectbox("Coupon Frequency (Months)", [1, 3, 6], index=0)
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

if st.button("Generate FCN Valuation"):
    # Clear cache only for the solve to ensure Skew is picked up
    prices_df, corr = fetch_market_prices(tickers)
    base_vols = get_current_vols(tickers, source="Market Implied (IV)" if "IV" in vol_source else "HV")
    
    # Generate paths with the Skew Applied
    paths = generate_paths(15000, tenor_y, rf_rate, base_vols, corr, skew_val)
    
    # Dense Solver
    cpns = np.linspace(0.0, 1.0, 60)
    sim_prices = [price_fcn_logic(c, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)[0] for c in cpns]
    y_solve = np.interp(100.0, sim_prices, cpns)
    
    final_px, avg_c, p_loss = price_fcn_logic(y_solve, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

    st.header(f"Solved Annual Yield: {y_solve*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Coupons", f"{avg_c:.2f}")
    c2.metric("Prob. Capital Loss", f"{p_loss:.1f}%")
    c3.metric("Selected Vol (Avg)", f"{np.mean(base_vols)*(1+skew_val):.2%}")
