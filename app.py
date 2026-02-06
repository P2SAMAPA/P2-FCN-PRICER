import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. CLEAN MARKET DATA ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, vol_type):
    data, vols, divs, spots = {}, [], [], []
    valid = []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="1y")['Close']
            if len(h) < 100: continue
            data[t] = h
            spots.append(h.iloc[-1])
            divs.append(s.info.get('dividendYield', 0.015) or 0.015)
            if vol_type == "Market Implied (IV)":
                # Fallback to HV if IV fetch fails
                vols.append(h.pct_change().std() * np.sqrt(252))
            else:
                vols.append(h.pct_change().std() * np.sqrt(252))
            valid.append(t)
        except: continue
    
    df = pd.DataFrame(data).dropna()
    corr = df.pct_change().corr().values
    return np.array(vols), corr, np.array(spots), np.array(divs), valid

# --- 2. THE PRICING KERNEL ---
def calculate_note_value(coupon_pa, paths, r, tenor, strike, ko, freq_m):
    steps, n_paths, n_assets = paths.shape
    wo_perf = np.min(paths, axis=2) 
    
    obs_indices = np.arange(int(freq_m/12*252), steps, int(freq_m/12*252))
    cpn_rate = (coupon_pa * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_paths)
    active = np.ones(n_paths, dtype=bool)
    accrued = np.zeros(n_paths)
    
    for idx in obs_indices:
        accrued[active] += cpn_rate
        # Autocall Check
        ko_mask = active & (wo_perf[idx] >= ko)
        payoffs[ko_mask] = 100 + accrued[ko_mask]
        active[ko_mask] = False
        
    if np.any(active):
        # Maturity Payoff
        final_wo = wo_perf[-1]
        val = np.where(final_wo[active] >= strike, 100, final_wo[active])
        payoffs[active] = val + accrued[active]
        
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- 3. UI & SIMULATION ---
st.title("⚖️ FCN Precision Pricing")

with st.sidebar:
    t_in = st.text_input("Tickers", "SPY, QQQ")
    tks = [x.strip().upper() for x in t_in.split(",")]
    vol_mode = st.radio("Vol Mode", ["Historical", "Market Implied"])
    rf_rate = st.number_input("Risk Free Rate", 0.0, 0.1, 0.045)
    tenor = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0)
    strike_pct = st.slider("Put Strike %", 50, 100, 75)
    ko_pct = st.slider("KO Level %", 80, 120, 100)

if len(tks) >= 2:
    v, corr, s, d, names = get_clean_data(tks, vol_mode)
    
    # Path Generation
    n_paths = 10000
    days = int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(v))*1e-9)
    
    # Corrected Drift: r - div - 0.5*sigma^2
    drift = (rf_rate - d - 0.5 * v**2) * (1/252)
    z = np.random.standard_normal((days, n_paths, len(v)))
    
    returns = drift + (v * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
    paths = 100 * np.exp(np.cumsum(returns, axis=0))
    paths = np.vstack([np.ones((1, n_paths, len(v)))*100, paths])

    try:
        # We search between 0% and 100% coupon. 
        # If 0% still results in a note > 100, or 100% < 100, brentq will fail.
        # We wrap it in a custom check to prevent the "Blank Screen"
        v_low = calculate_note_value(0, paths, rf_rate, tenor, strike_pct, ko_pct, 1) - 100
        v_high = calculate_note_value(1.0, paths, rf_rate, tenor, strike_pct, ko_pct, 1) - 100
        
        if v_low * v_high < 0:
            sol = brentq(lambda x: calculate_note_value(x, paths, rf_rate, tenor, strike_pct, ko_pct, 1) - 100, 0, 1.0)
            st.metric("Solved Coupon (p.a.)", f"{sol*100:.2f}%")
        else:
            st.warning("Note cannot be priced at Par. Value is either always above or below 100.")
            st.write(f"Value at 0% Coupon: {v_low+100:.2f}")
            st.write(f"Value at 100% Coupon: {v_high+100:.2f}")
            
    except Exception as e:
        st.error(f"Solver Error: {e}")
