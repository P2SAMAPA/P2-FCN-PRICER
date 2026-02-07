import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- UI STYLE RESTORATION ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; }
    .stButton>button { width: 100%; background-color: #1a1c23; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR UX (Matches image_e25b9e) ---
with st.sidebar:
    st.header("⚙️ Global Controls")
    mode = st.selectbox("Product", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks = [x.strip().upper() for x in st.text_input("Tickers", "SPY, QQQ").split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"], index=0)
    
    st.markdown("---")
    st.subheader("🏦 Funding & RF Rate")
    rf_rate = st.number_input("Risk Free Rate (r)", 0.0, 0.10, 0.04, step=0.01)
    tenor_y = st.number_input("Tenor (Years)", 0.5, 5.0, 1.0, step=0.5)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- PRICING ENGINE ---
@st.cache_data(ttl=600)
def get_mkt_data(tks, src):
    v, p, lp, divs, valid_names = [], [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="1y")['Close']
            if h.empty: continue
            p.append(h.rename(t)); lp.append(h.iloc[-1]); valid_names.append(t)
            divs.append(s.info.get('dividendYield', 0.015) or 0.015)
            v.append(h.pct_change().std() * np.sqrt(252))
        except: continue
    if not p: return None, None, None, None, []
    corr = pd.concat(p, axis=1).pct_change().corr().values
    return np.array(v), corr, np.array(lp), np.array(divs), valid_names

def run_pricing(cpn, paths, r, tenor, stk, ko, f_m, nc_m, mode, sd=0):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_val = (cpn*(f_m/12))*100
    for i, d in enumerate(obs):
        curr_ko = ko - (i * sd) if "Version 2" in mode else ko
        acc[act] += cpn_val
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act):
        py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    return np.mean(py) * np.exp(-r * tenor), (np.sum(wf[-1] < stk)/n_s)

# --- OUTPUT UX (Matches image_e2515c) ---
if len(tks) >= 1:
    v, corr, spots, divs, names = get_mkt_data(tks, vol_src)
    if names:
        n_paths, days = 5000, int(tenor_y * 252)
        L = np.linalg.cholesky(corr + np.eye(len(v))*1e-9)
        # Drift adjustment includes dividends to fix the yield inflation
        drift = (rf_rate - divs - 0.5 * v**2) * (1/252)
        z = np.random.standard_normal((days, n_paths, len(v)))
        rets = drift + (v * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        ps = np.vstack([np.ones((1, n_paths, len(v)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

        st.title(f"🛡️ {mode} Analysis")
        stk_val = st.slider("Put Strike %", 40, 100, 75)
        ko_val = st.slider("KO Level %", 80, 130, 100)
        fq = 1 if st.selectbox("Frequency", ["Monthly", "Quarterly"]) == "Monthly" else 3
        
        if st.button("Generate Pricing Report"):
            # Solve for Par yield
            sol = brentq(lambda x: run_pricing(x, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode)[0] - 100, 0, 0.5)
            val, prob_loss = run_pricing(sol, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Solved Annual Yield", f"{sol*100:.2f}%")
            col2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
            col3.metric("Avg. Expected Coupons", f"{tenor_y * (12/fq):.1f}")
            
            st.divider()
            st.write("**Asset Correlation (Applied to Simulation)**")
            st.table(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
