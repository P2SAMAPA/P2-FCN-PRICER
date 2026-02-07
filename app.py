import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from datetime import datetime, timedelta

# --- 1. GLOBAL STYLING (Restores Professional image_e2515c look) ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #dfe3e8; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #111827; color: white; font-weight: 700; transition: 0.3s; }
    .stButton>button:hover { background-color: #374151; border: none; }
    .sidebar .sidebar-content { background-image: linear-gradient(#ffffff, #f4f7f9); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATA ENGINE (Fixes the Math logic) ---
@st.cache_data(ttl=3600)
def fetch_market_context(tickers):
    vols, prices, divs, names = [], [], [], []
    for t in tickers:
        try:
            asset = yf.Ticker(t)
            hist = asset.history(period="2y")['Close']
            if hist.empty: continue
            
            prices.append(hist.rename(t))
            names.append(t)
            # Critical: Extracting div yield forces the drift to be realistic
            div_yield = asset.info.get('dividendYield', 0.015) or 0.015
            divs.append(div_yield)
            # 252-day Volatility
            vols.append(hist.pct_change().tail(252).std() * np.sqrt(252))
        except Exception: continue
    
    if len(prices) < 1: return None
    
    df_corr = pd.concat(prices, axis=1).pct_change().dropna()
    correlation_matrix = df_corr.corr().values
    return np.array(vols), correlation_matrix, np.array(divs), names

# --- 3. THE CORE PRICING SIMULATOR (Vectorized GBM) ---
def simulate_payoffs(cpn_pa, paths, r, tenor, strike, ko, freq_m, nc_m, mode, sd_step=0, b_rate=0, b_strike=0):
    steps, n_sims, n_assets = paths.shape
    worst_of_path = np.min(paths, axis=2) # Vectorized WO Logic
    
    # Identify observation dates
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    payoff = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    accrued_coupons = np.zeros(n_sims)
    cpn_per_period = (cpn_pa * (freq_m/12)) * 100
    
    for i, d in enumerate(obs_dates):
        current_ko = ko - (i * sd_step) if "Step-Down" in mode else ko
        accrued_coupons[active] += cpn_per_period
        
        # Autocall logic
        if d >= int((nc_m/12)*252):
            called = active & (worst_of_path[d] >= current_ko)
            payoff[called] = 100 + accrued_coupons[called]
            active[called] = False
            
    # Final Maturity Settlement
    if np.any(active):
        final_wo = worst_of_path[-1, active]
        if "BCN" in mode:
            bonus = np.where(final_wo >= b_strike, (b_rate * tenor) * 100, 0)
            payoff[active] = np.where(final_wo >= strike, 100 + bonus, final_wo + bonus) + accrued_coupons[active]
        else:
            payoff[active] = np.where(final_wo >= strike, 100, final_wo) + accrued_coupons[active]
            
    return np.mean(payoff) * np.exp(-r * tenor), (np.sum(worst_of_path[-1] < strike) / n_sims)

# --- 4. UX: INPUT SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2643/2643445.png", width=50)
    st.title("Product Architect")
    mode = st.selectbox("Structure Type", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks_input = st.text_input("Underlying Basket", "SPY, QQQ")
    tks = [x.strip().upper() for x in tks_input.split(",")]
    
    st.divider()
    st.subheader("🏦 Market Environment")
    rf_rate = st.number_input("Risk Free Rate (%)", 0.0, 10.0, 3.59) / 100
    tenor = st.number_input("Tenor (Years)", 0.25, 5.0, 1.0, 0.25)
    nc_m = st.number_input("Non-Call Period (Months)", 0, 24, 3)
    
    if "Step-Down" in mode:
        sd_val = st.slider("Monthly Step-Down (%)", 0.0, 1.0, 0.2, 0.1)
    else: sd_val = 0

# --- 5. MAIN INTERFACE ---
st.title(f"🚀 {mode} Pricing Terminal")
st.caption(f"Status: Ready for Valuation | Assets: {', '.join(tks)}")

c1, c2, c3 = st.columns(3)
strike_pct = c1.slider("Put Strike (%)", 40, 100, 75)
ko_pct = c2.slider("Autocall Level (%)", 80, 120, 100)
freq_opt = c3.selectbox("Coupon Frequency", ["Monthly", "Quarterly"])
freq_m = 1 if freq_opt == "Monthly" else 3

b_rate, b_strike = 0, 0
if "BCN" in mode:
    bc1, bc2 = st.columns(2)
    b_rate = bc1.number_input("Bonus Rate (p.a. %)", 0.0, 50.0, 5.0) / 100
    b_strike = bc2.number_input("Bonus Barrier (%)", 50, 150, 100)

if st.button("RUN FULL MONTE CARLO VALUATION"):
    with st.spinner("Executing 10,000 correlated paths..."):
        mkt = fetch_market_context(tks)
        if mkt:
            vols, corr, divs, names = mkt
            n_sims, n_days = 10000, int(tenor * 252)
            
            # THE MATH FIX: Cholesky + GBM
            L = np.linalg.cholesky(corr + np.eye(len(vols)) * 1e-10)
            dt = 1/252
            drift = (rf_rate - divs - 0.5 * vols**2) * dt
            
            # Vectorized correlated noise
            z = np.random.standard_normal((n_days, n_sims, len(vols)))
            correlated_z = np.einsum('ij,tkj->tki', L, z)
            
            # Path Generation
            returns = drift + (vols * np.sqrt(dt)) * correlated_z
            price_paths = np.vstack([np.ones((1, n_sims, len(vols))) * 100, 
                                   100 * np.exp(np.cumsum(returns, axis=0))])
            
            # Binary Search for Par-Coupon
            try:
                target_fn = lambda x: simulate_payoffs(x, price_paths, rf_rate, tenor, strike_pct, ko_pct, freq_m, nc_m, mode, sd_val, b_rate, b_strike)[0] - 100
                fair_cpn = brentq(target_fn, 0.0, 1.5)
                _, risk_of_loss = simulate_payoffs(fair_cpn, price_paths, rf_rate, tenor, strike_pct, ko_pct, freq_m, nc_m, mode, sd_val, b_rate, b_strike)
                
                # UX: OUTPUT CARDS (Matches image_e2677e style)
                st.divider()
                out1, out2, out3 = st.columns(3)
                out1.metric("SOLVED ANNUAL YIELD", f"{fair_cpn*100:.2f}%")
                out2.metric("PROB. CAPITAL LOSS", f"{risk_of_loss*100:.1f}%")
                out3.metric("EXP. COUPON EVENTS", f"{int(tenor * (12/freq_m))}")
                
                st.subheader("📊 Asset Correlation Matrix")
                st.dataframe(pd.DataFrame(corr, index=names, columns=names).style.background_gradient(cmap='Blues').format("{:.2f}"))
                
            except Exception as e:
                st.error(f"Solver Error: {e}. Please try adjusting inputs.")
        else:
            st.error("Data Fetch Failed. Verify Tickers on Yahoo Finance.")
