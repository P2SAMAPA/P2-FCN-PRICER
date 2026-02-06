import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- UI CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #d1d9e6; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #1a1c23; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE ---
@st.cache_data(ttl=3600)
def get_mkt_data(tks, src):
    v, p, lp = [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            p.append(h.rename(t)); lp.append(h.iloc[-1])
            hv = h.pct_change().std() * np.sqrt(252)
            if src == "Market Implied (IV)" and s.options:
                c = s.option_chain(s.options[min(len(s.options)-1, 1)])
                v.append(max(c.calls['impliedVolatility'].median(), 0.1))
            else: v.append(hv)
        except: v.append(0.35); lp.append(100.0)
    df = pd.concat(p, axis=1).dropna() if p else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tks))
    return np.array(v), corr, np.array(lp)

def run_mc_v1(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0):
    steps, n_s, _ = pths.shape; wf = np.min(pths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    for d in obs:
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= ko); py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act): py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    return np.mean(py) * np.exp(-r * tnr)

# --- FCN V2 ENGINE (Step-Down & Correlation Adjustment) ---
def run_mc_v2(c_g, pths, r, tnr, stk, ko_start, step_down, f_m, nc_m):
    steps, n_s, _ = pths.shape; wf = np.min(pths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    gv = (c_g*(f_m/12))*100
    for i, d in enumerate(obs):
        acc[act] += gv
        # Step-down logic: KO level drops over time
        current_ko = ko_start - (i * step_down)
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= current_ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act): py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    return np.mean(py) * np.exp(-r * tnr)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Global Parameters")
    mode = st.selectbox("Product Selection", ["FCN Version 1 (Stable)", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers", "NVDA, TSLA")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Volatility Skew", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("Risk-Free Rate %", 0.0, 10.0, 4.5)/100
    tenor_y = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- MODE DISPATCHER ---
if mode == "FCN Version 1 (Stable)":
    st.title("🛡️ FCN Version 1 (Stable)")
    ko_lvl = st.sidebar.slider("KO Barrier %", 80, 150, 105)
    fq = st.sidebar.selectbox("Freq", ["Monthly", "Quarterly"])
    fm, stk = {"Monthly": 1, "Quarterly": 3}[fq], st.sidebar.slider("Strike %", 40, 100, 60)
    
    if st.button("Run Stable FCN"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        y = brentq(lambda x: run_mc_v1(x, ps, rf_rate, tenor_y, stk, ko_lvl, fm, nc_m) - 100, 0, 5)
        st.metric("Solved Yield", f"{y*100:.2f}% p.a.")
        st.write("### Sensitivity Matrix (V1)"); st.table(pd.DataFrame([[brentq(lambda x: run_mc_v1(x, ps, rf_rate, tenor_y, s, b, fm, nc_m) - 100, 0, 5)*100 for s in [stk-10, stk, stk+10]] for b in [ko_lvl+10, ko_lvl, ko_lvl-10]], columns=[stk-10, stk, stk+10], index=[ko_lvl+10, ko_lvl, ko_lvl-10]).style.background_gradient(cmap='RdYlGn'))

elif mode == "FCN Version 2 (Step-Down)":
    st.title("🛡️ FCN Version 2: Step-Down & Enhanced Analysis")
    ko_start = st.sidebar.slider("Initial KO %", 80, 150, 105)
    step_val = st.sidebar.slider("Step-Down per Obs (%)", 0.0, 2.0, 0.5)
    fq = st.sidebar.selectbox("Freq", ["Monthly", "Quarterly"])
    fm, stk = {"Monthly": 1, "Quarterly": 3}[fq], st.sidebar.slider("Strike %", 40, 100, 60)
    
    if st.button("Run Version 2 Enhanced FCN"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        y = brentq(lambda x: run_mc_v2(x, ps, rf_rate, tenor_y, stk, ko_start, step_val, fm, nc_m) - 100, 0, 5)
        
        st.metric("V2 Enhanced Yield", f"{y*100:.2f}% p.a.")
        st.info(f"Note features a step-down mechanism where the KO barrier drops by {step_val}% every observation.")
        
        # New Correlation Sensitivity
        st.subheader("⛓️ Correlation Stress Test")
        corrs = [0.2, 0.5, 0.8]
        corr_y = []
        for cr in corrs:
            c_mod = np.full((len(v), len(v)), cr); np.fill_diagonal(c_mod, 1.0)
            L_mod = np.linalg.cholesky(c_mod + np.eye(len(v))*1e-8)
            ps_mod = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L_mod, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
            corr_y.append(brentq(lambda x: run_mc_v2(x, ps_mod, rf_rate, tenor_y, stk, ko_start, step_val, fm, nc_m) - 100, 0, 5)*100)
        
        st.table(pd.DataFrame([corr_y], columns=[f"Corr {c}" for c in corrs], index=["Solved Yield %"]))

else:
    st.title("🛡️ BCN Solver")
    ko_b = st.sidebar.slider("KO Barrier %", 80, 150, 105)
    fq_b = st.sidebar.selectbox("Guar Freq", ["Monthly", "Quarterly"])
    fm_b, b_ref = {"Monthly": 1, "Quarterly": 3}[fq_b], st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    g_cpn, b_rate = st.number_input("Guar Rate %", 0.0, 20.0, 4.0)/100, st.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
    
    if st.button("Solve BCN"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        try:
            ki = brentq(lambda x: run_mc_v1(g_cpn, ps, rf_rate, tenor_y, x, ko_b, fm_b, nc_m, b_rate, b_ref) - 100, 0.01, 150)
            st.metric("Required KI Barrier", f"{ki:.2f}%")
        except: st.error("Solver Error: Adjust coupons.")
