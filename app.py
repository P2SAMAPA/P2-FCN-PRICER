import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Derivatives Alpha Engine", layout="wide")

# --- 1. ROBUST MARKET DATA FETCH ---
@st.cache_data(ttl=3600)
def fetch_institutional_data(tickers, vol_source):
    vols, prices, divs = [], [], []
    valid_tickers = []
    
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="1y")['Close']
            if h.empty: continue
            
            spot = h.iloc[-1]
            prices.append(h.rename(t))
            valid_tickers.append(t)
            
            # Dividend Yield - CRITICAL for Index Pricing
            dy = s.info.get('dividendYield', 0)
            divs.append(dy if dy is not None else 0.015) # Default 1.5% for indices
            
            if vol_source == "Market Implied (IV)":
                # Logic: Fetch 1-year ATM Implied Vol
                opts = s.options
                target_expiry = opts[min(len(opts)-1, 5)] # Looking for ~12 months out
                chain = s.option_chain(target_expiry)
                atm_vol = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]['impliedVolatility'].values[0]
                vols.append(atm_vol)
            else:
                vols.append(h.pct_change().std() * np.sqrt(252))
        except:
            continue
            
    df = pd.concat(prices, axis=1).dropna()
    corr_matrix = df.pct_change().corr().values
    return np.array(vols), corr_matrix, df.iloc[-1].values, np.array(divs), valid_tickers

# --- 2. THE CORE PRICING MATH (Discrete Worst-Of Logic) ---
def price_fcn(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    # paths shape: [days, num_paths, num_assets]
    steps, n_paths, n_assets = paths.shape
    wo_perf = np.min(paths, axis=2) # Worst-of performance at every day
    
    dt = 1/252
    df = np.exp(-r * tenor)
    
    # Observation logic (e.g., Monthly = 21 days)
    obs_days = np.arange(int(freq_m/12*252), steps+1, int(freq_m/12*252))
    nc_days = int(nc_m/12*252)
    
    payoffs = np.zeros(n_paths)
    active = np.ones(n_paths, dtype=bool)
    accrued_coupon = np.zeros(n_paths)
    cpn_per_period = (coupon_pa * (freq_m/12)) * 100
    
    for d in obs_days:
        if d >= steps: d = steps - 1
        # 1. Accrue Coupon
        accrued_coupon[active] += cpn_per_period
        
        # 2. Check Autocall (KO) - only after Non-Call period
        if d >= nc_days:
            ko_mask = active & (wo_perf[d] >= ko)
            payoffs[ko_mask] = 100 + accrued_coupon[ko_mask]
            active[ko_mask] = False
            
    # 3. Maturity Payoff (for those not called)
    if np.any(active):
        # If WO >= Strike, 100. Else, Physical Delivery (Worst-of spot)
        final_perf = wo_perf[-1]
        maturity_value = np.where(final_perf[active] >= strike, 100, final_perf[active])
        payoffs[active] = maturity_value + accrued_coupon[active]
        
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- 3. STREAMLIT UI ---
st.title("⚖️ FCN Calibration Audit")

with st.sidebar:
    t_in = st.text_input("Tickers", "SPY, QQQ")
    tks = [x.strip().upper() for x in t_in.split(",")]
    vol_src = st.radio("Vol Basis", ["Historical (HV)", "Market Implied (IV)"])
    rf = st.number_input("Risk Free Rate (r)", 0.0, 0.1, 0.045)
    tenor = st.number_input("Tenor (Years)", 0.5, 5.0, 1.0)
    ko = st.slider("KO Barrier %", 80, 120, 100)
    strike = st.slider("Put Strike %", 50, 100, 75)
    freq = st.selectbox("Frequency", [1, 3, 6], format_func=lambda x: f"{x} Month")

if len(tks) > 1:
    vols, corr, spots, divs, names = fetch_institutional_data(tks, vol_src)
    
    # Monte Carlo Setup
    n_paths = 20000
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr)
    
    # Drift = (r - q - 0.5*sigma^2)
    drift = (rf - divs - 0.5 * vols**2) * (1/252)
    
    # Generate Correlated Paths
    z = np.random.standard_normal((steps, n_paths // 2, len(tks)))
    z = np.concatenate([z, -z], axis=1) # Antithetic Variates
    
    daily_returns = drift + (vols * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
    price_paths = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    
    # Solver
    try:
        sol = brentq(lambda x: price_fcn(x, price_paths, rf, tenor, strike, ko, freq, 3) - 100, 0, 0.5)
        
        st.subheader("Pricing Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Solved Coupon (p.a.)", f"{sol*100:.2f}%")
        c2.metric("Index Correlation (Avg)", f"{np.mean(corr[np.triu_indices_from(corr, k=1)]):.2f}")
        c3.metric("Avg Implied Vol", f"{np.mean(vols)*100:.1f}%")
        
        st.write("**Correlation Matrix Check**")
        st.dataframe(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
        
    except ValueError:
        st.error("Solver Error: Market data suggests this note cannot be priced at Par. Adjust Barriers.")
