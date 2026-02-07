import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- FORCE RESET ---
st.cache_data.clear()

# --- 1. UX STYLING (Matches image_e2515c) ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.sidebar.info("🚀 BUILD ID: 4.0.0 (Math Engine Realigned)")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1a1c23; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR (Matches image_e25b9e) ---
with st.sidebar:
    st.title("⚙️ Global Controls")
    mode = st.selectbox("Product", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers", "SPY, QQQ")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"], index=1)
    
    st.markdown("---")
    rf_rate = st.number_input("Risk Free Rate", 0.0, 0.10, 0.0359, format="%.4f")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0, step=0.25)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- 3. RECTIFIED MATH ENGINE ---
@st.cache_data(ttl=600)
def get_engine_data(tks):
    v, p, lp, divs, names = [], [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="1y")['Close']
            if h.empty: continue
            p.append(h); lp.append(h.iloc[-1]); names.append(t)
            # High yield fix: Ensure dividends are subtracted from growth
            d_yield = s.info.get('dividendYield', 0.015) or 0.015
            divs.append(d_yield)
            v.append(h.pct_change().std() * np.sqrt(252))
        except: continue
    if not p: return None
    corr = pd.concat(p, axis=1).pct_change().corr().values
    return np.array(v), corr, np.array(divs), names

def run_simulation(cpn, paths, r, tenor, stk, ko, f_m, nc_m):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2) # Worst-of logic
    obs_idx = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    payoff, active, accrued = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_val = (cpn * (f_m/12)) * 100
    
    for d in obs_idx:
        accrued[active] += cpn_val
        if d >= int((nc_m/12)*252):
            ko_mask = active & (wf[d] >= ko)
            payoff[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
            
    if np.any(active):
        payoff[active] = np.where(wf[-1, active] >= stk, 100, wf[-1, active]) + accrued[active]
    
    return np.mean(payoff) * np.exp(-r * tenor), (np.sum(wf[-1] < stk)/n_s)

# --- 4. MAIN UI ---
if tks:
    engine_data = get_engine_data(tks)
    if engine_data:
        vols, corr, divs, names = engine_data
        st.title(f"🛡️ {mode} Analysis")
        stk_v = st.slider("Put Strike %", 40, 100, 75)
        ko_v = st.slider("KO Level %", 80, 130, 100)
        fq = 1 if st.selectbox("Frequency", ["Monthly", "Quarterly"]) == "Monthly" else 3
        
        if st.button("Generate Pricing Report"):
            # Cholesky & Drift Correction
            L = np.linalg.cholesky(corr + np.eye(len(vols))*1e-9)
            n_paths, days = 10000, int(tenor_y * 252)
            dt = 1/252
            
            # The "Anti-67%" Drift: (r - div - 0.5 * sigma^2)
            drift = (rf_rate - divs - 0.5 * vols**2) * dt
            z = np.random.standard_normal((days, n_paths, len(vols)))
            
            # Apply Correlation at every step
            correlated_z = np.einsum('ij,tkj->tki', L, z)
            rets = drift + (vols * np.sqrt(dt)) * correlated_z
            paths = np.vstack([np.ones((1, n_paths, len(vols)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])
            
            # Solve for Coupon
            target = lambda x: run_simulation(x, paths, rf_rate, tenor_y, stk_v, ko_v, fq, nc_m)[0] - 100
            sol_cpn = brentq(target, 0.0, 1.0)
            _, p_loss = run_simulation(sol_cpn, paths, rf_rate, tenor_y, stk_v, ko_v, fq, nc_m)
            
            # UI Restoration (Metrics)
            c1, c2, c3 = st.columns(3)
            c1.metric("Solved Annual Yield", f"{sol_cpn*100:.2f}%")
            c2.metric("Prob. of Capital Loss", f"{p_loss*100:.1f}%")
            c3.metric("Avg. Expected Coupons", f"{tenor_y * (12/fq):.1f}")
            
            st.write("**Asset Correlation Matrix (Applied)**")
            st.table(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
