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
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #d1d9e6; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
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

# --- SIMULATION ENGINE ---
def price_note(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0):
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
st.sidebar.title("🛠️ Global Controls")
mode = st.sidebar.selectbox("Product Selection", ["FCN Solver", "BCN Solver"])
with st.sidebar:
    tk_in = st.text_input("Underlying Tickers", "NVDA, TSLA")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Volatility Skew", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("Risk-Free Rate %", 0.0, 10.0, 4.5)/100
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
    ko_lvl = st.slider("KO Barrier %", 80, 150, 105)

# --- FCN MODE ---
if mode == "FCN Solver":
    st.title("🛡️ Institutional FCN Solver")
    fq_label = st.sidebar.selectbox("Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
    fm = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[fq_label]
    stk_lvl = st.sidebar.slider("Put Strike %", 40, 100, 60)
    
    if st.button("Calculate FCN Yield"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tenor_y*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        
        y_solve = brentq(lambda x: price_note(x, ps, rf_rate, tenor_y, stk_lvl, ko_lvl, fm, nc_m) - 100, 0, 5)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Yield", f"{y_solve*100:.2f}%")
        c2.metric("Put Strike", f"{stk_lvl}%")
        c3.metric("KO Barrier", f"{ko_lvl}%")
        
        st.divider()
        st.subheader("📊 FCN Sensitivity Analysis")
        stks, kos = [stk_lvl-10, stk_lvl, stk_lvl+10], [ko_lvl+10, ko_lvl, ko_lvl-10]
        gy, gl = [], []
        for b in kos:
            ry, rl = [], []
            for s in stks:
                try: ry.append(brentq(lambda x: price_note(x, ps, rf_rate, tenor_y, s, b, fm, nc_m) - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)
            
        col_y, col_l = st.columns(2)
        with col_y:
            st.write("**Yield Matrix (% p.a.)**")
            st.table(pd.DataFrame(gy, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='RdYlGn'))
        with col_l:
            st.write("**Cap. Loss Probability (%)**")
            st.table(pd.DataFrame(gl, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='Reds'))

# --- BCN MODE ---
else:
    st.title("🛡️ Institutional BCN Solver")
    fq_b = st.sidebar.selectbox("Guaranteed Frequency", ["Monthly", "Quarterly"])
    fm_b = {"Monthly": 1, "Quarterly": 3}[fq_b]
    b_ref = st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    
    col_g, col_b = st.columns(2)
    g_cpn = col_g.number_input("Guaranteed Rate %", 0.0, 20.0, 4.0)/100
    b_rate = col_b.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
    
    if st.button("Solve Required KI Barrier"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tenor_y*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        
        try:
            # Solve for Strike (x) to get price=100
            ki_res = brentq(lambda x: price_note(g_cpn, ps, rf_rate, tenor_y, x, ko_lvl, fm_b, nc_m, b_rate, b_ref) - 100, 1, 100)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Required KI Barrier", f"{ki_res:.2f}%")
            c2.metric("Guaranteed Yield", f"{g_cpn*100:.1f}%")
            c3.metric("Bonus Yield", f"{b_rate*100:.1f}%")
            
            st.divider()
            st.subheader("📊 BCN KI Sensitivity")
            gs, bs = [g_cpn-0.01, g_cpn, g_cpn+0.01], [b_rate-0.02, b_rate, b_rate+0.02]
            gb = []
            for b in bs:
                rb = []
                for g in gs:
                    try: rb.append(brentq(lambda x: price_note(g, ps, rf_rate, tenor_y, x, ko_lvl, fm_b, nc_m, b, b_ref) - 100, 1, 120))
                    except: rb.append(0.0)
                gb.append(rb)
            
            st.table(pd.DataFrame(gb, index=[f"Bonus {x*100:.1f}%" for x in bs], columns=[f"Guar {x*100:.1f}%" for x in gs]).style.background_gradient(cmap='RdYlGn_r').format("{:.2f}"))
            
            st.subheader("🔍 Market Reference")
            st.table(pd.DataFrame({"Underlying": tks, "Spot (100%)": [f"${x:.2f}" for x in lp], "Volatility": [f"{x*100:.1f}%" for x in v]}))
            
        except ValueError:
            st.error("⚠️ **Mathematical Limit Reached**: The coupons are too high for a par issuance. Lower the Guaranteed or Bonus rates, or increase the KO Barrier.")
