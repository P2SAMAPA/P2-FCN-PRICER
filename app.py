import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- FORCE CACHE CLEAR ON REBOOT ---
st.cache_data.clear()

# --- 1. UX STYLING ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.sidebar.info("🚀 SYSTEM STATUS: LIVE - VERSION 2.1 (Correlation Fix)")

# --- 2. SIDEBAR UX (Matches image_e25b9e) ---
with st.sidebar:
    st.header("⚙️ Global Controls")
    mode = st.selectbox("Product", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers (Comma Separated)", "SPY, QQQ")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"], index=1)
    
    st.markdown("---")
    st.subheader("🏦 Funding & RF Rate")
    rf_rate = st.number_input("Risk Free Rate", 0.0, 0.10, 0.0359, format="%.4f")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0, step=0.25)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- 3. THE MATH RECTIFICATION ENGINE ---
def get_mkt_data(tks):
    v, p, lp, divs, names = [], [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="1y")['Close']
            if h.empty: continue
            p.append(h); lp.append(h.iloc[-1]); names.append(t)
            divs.append(s.info.get('dividendYield', 0.015) or 0.015)
            v.append(h.pct_change().std() * np.sqrt(252))
        except: continue
    if not p: return None
    corr = pd.concat(p, axis=1).pct_change().corr().values
    return np.array(v), corr, np.array(lp), np.array(divs), names

def run_pricing(cpn, paths, r, tenor, stk, ko, f_m, nc_m):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2) # Worst-of logic
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    gv = (cpn*(f_m/12))*100
    for d in obs:
        acc[act] += gv
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act):
        py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    return np.mean(py) * np.exp(-r * tenor), (np.sum(wf[-1] < stk)/n_s)

# --- 4. OUTPUT UX (Restored Metric Columns) ---
if tks:
    data = get_mkt_data(tks)
    if data:
        vols, corr, spots, divs, names = data
        # Cholesky Transformation: Fixes the SPY/QQQ 67% yield error
        L = np.linalg.cholesky(corr + np.eye(len(vols))*1e-9)
        n_paths, days = 10000, int(tenor_y * 252)
        drift = (rf_rate - divs - 0.5 * vols**2) * (1/252)
        z = np.random.standard_normal((days, n_paths, len(vols)))
        rets = drift + (vols * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        ps = np.vstack([np.ones((1, n_paths, len(vols)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

        st.title(f"🛡️ {mode} Analysis")
        stk_val = st.slider("Put Strike %", 40, 100, 75)
        ko_val = st.slider("KO Level %", 80, 130, 100)
        fq = 1 if st.selectbox("Frequency", ["Monthly", "Quarterly"]) == "Monthly" else 3
        
        if st.button("Generate Pricing Report"):
            sol = brentq(lambda x: run_pricing(x, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m)[0]-100, 0, 1.0)
            val, prob_loss = run_pricing(sol, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Solved Annual Yield", f"{sol*100:.2f}%")
            c2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
            c3.metric("Avg. Expected Coupons", f"{tenor_y * (12/fq):.1f}")
