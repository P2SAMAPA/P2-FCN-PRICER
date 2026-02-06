import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- INSTITUTIONAL UI CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #d1d9e6; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1a1c23; color: white; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
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
    return np.array(v), df.pct_change().corr().values if not df.empty else np.eye(len(tks)), np.array(lp)

def run_mc(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0):
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Global Parameters")
    mode = st.selectbox("Product Selection", ["FCN Solver", "BCN Solver"])
    tk_in = st.text_input("Underlying Tickers", "NVDA, TSLA")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Volatility Skew", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("Risk-Free Rate %", 0.0, 10.0, 4.5)/100
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
    ko_lvl = st.slider("KO Barrier %", 80, 150, 105)

# --- FCN INTERFACE ---
if mode == "FCN Solver":
    st.title("🛡️ Institutional FCN Solver")
    fq_label = st.sidebar.selectbox("Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
    fm, stk_lvl = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[fq_label], st.sidebar.slider("Put Strike %", 40, 100, 60)
    
    if st.button("Generate FCN Pricing"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tenor_y*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        y = brentq(lambda x: run_mc(x, ps, rf_rate, tenor_y, stk_lvl, ko_lvl, fm, nc_m) - 100, 0, 5)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Solved Yield", f"{y*100:.2f}% p.a.")
        c2.metric("Strike / KI", f"{stk_lvl}%")
        c3.metric("Autocall / KO", f"{ko_lvl}%")
        
        st.subheader("📊 Sensitivity Analysis")
        stks, kos = [stk_lvl-10, stk_lvl, stk_lvl+10], [ko_lvl+10, ko_lvl, ko_lvl-10]
        gy, gl = [], []
        for b in kos:
            ry, rl = [], []
            for s in stks:
                try: ry.append(brentq(lambda x: run_mc(x, ps, rf_rate, tenor_y, s, b, fm, nc_m) - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Yield Matrix (%)**")
            st.table(pd.DataFrame(gy, index=kos, columns=stks).style.background_gradient(cmap='RdYlGn'))
        with col2:
            st.write("**Capital Loss Probability (%)**")
            st.table(pd.DataFrame(gl, index=kos, columns=stks).style.background_gradient(cmap='Reds'))

# --- BCN INTERFACE ---
else:
    st.title("🛡️ Institutional BCN Solver")
    fq_b = st.sidebar.selectbox("Guaranteed Freq", ["Monthly", "Quarterly"])
    fm_b, b_ref = {"Monthly": 1, "Quarterly": 3}[fq_b], st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    
    col1, col2 = st.columns(2)
    g_cpn = col1.number_input("Guaranteed Rate %", 0.0, 20.0, 4.0)/100
    b_rate = col2.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
    
    if st.button("Solve Required KI Barrier"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tenor_y*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        
        try:
            # We widen the solver range to 0.01 to 150 to catch high-value note scenarios
            ki = brentq(lambda x: run_mc(g_cpn, ps, rf_rate, tenor_y, x, ko_lvl, fm_b, nc_m, b_rate, b_ref) - 100, 0.01, 150)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Required KI Barrier", f"{ki:.2f}%")
            c2.metric("Total Potential Yield", f"{(g_cpn + b_rate)*100:.1f}%")
            c3.metric("Initial Ref Price", f"100.00%")
            
            st.subheader("📊 Barrier Sensitivity")
            gs, bs = [g_cpn-0.01, g_cpn, g_cpn+0.01], [b_rate-0.02, b_rate, b_rate+0.02]
            gb = []
            for b in bs:
                rb = []
                for g in gs:
                    try: rb.append(brentq(lambda x: run_mc(g, ps, rf_rate, tenor_y, x, ko_lvl, fm_b, nc_m, b, b_ref) - 100, 0.01, 150))
                    except: rb.append(0.0)
                gb.append(rb)
            
            st.table(pd.DataFrame(gb, index=[f"Bonus {x*100:.1f}%" for x in bs], columns=[f"Guar {x*100:.1f}%" for x in gs]).style.background_gradient(cmap='RdYlGn_r'))
            
            st.subheader("🔍 Market Reference Spot")
            st.table(pd.DataFrame({"Ticker": tks, "Spot Price": [f"${x:.2f}" for x in lp], "Implied Vol": [f"{x*100:.1f}%" for x in v]}))
            
        except ValueError:
            st.error("⚠️ **Mathematical Limit Reached**: The note is currently valued above par ($100) even with no protection barrier. Please increase the coupons or lower the interest rate/autocall level.")
