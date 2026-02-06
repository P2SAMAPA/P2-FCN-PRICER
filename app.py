import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- CORE THEME & CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_mkt(tks, src):
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

# --- PRICING ENGINE ---
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

# --- SIDEBAR CONTROL ---
st.sidebar.header("🕹️ Global Controls")
md = st.sidebar.selectbox("Product Mode", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
with st.sidebar:
    tk = st.text_input("Underlying Tickers", "NVDA, TSLA").split(",")
    tks = [x.strip().upper() for x in tk]
    src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    skw = st.slider("Volatility Skew", 0.0, 1.0, 0.2)
    rf = st.number_input("Risk-Free Rate %", 0.0, 10.0, 4.5)/100
    tnr = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc = st.number_input("Non-Call Period (Months)", 0, 24, 3)
    ko = st.slider("Autocall Barrier (KO) %", 80, 150, 105)

# --- FCN INTERFACE ---
if md == "Fixed Coupon Note (FCN)":
    st.title("🛡️ Institutional FCN Solver")
    fq = st.sidebar.selectbox("Coupon Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
    fm, stk = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[fq], st.sidebar.slider("Put Strike %", 40, 100, 60)
    
    if st.button("Generate Professional FCN Pricing"):
        with st.spinner("Simulating Market Paths..."):
            v, c, lp = get_mkt(tks, src); av = v*(1+skw)
            L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
            z = np.random.standard_normal((int(tnr*252), 10000, len(v)))
            ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
            
            y = brentq(lambda x: run_mc(x, ps, rf, tnr, stk, ko, fm, nc) - 100, 0, 5)
            
            # Professional Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Solved Yield (p.a.)", f"{y*100:.2f}%")
            m2.metric("Put Strike (KI)", f"{stk}%")
            m3.metric("KO Barrier", f"{ko}%")
            
            # Sensitivity Tables
            st.subheader("📊 Yield & Risk Sensitivity Matrix")
            sr, br = [stk-10, stk, stk+10], [ko+10, ko, ko-10]
            gy, gl = [], []
            for b in br:
                ry, rl = [], []
                for s in sr:
                    try: ry.append(brentq(lambda x: run_mc(x, ps, rf, tnr, s, b, fm, nc) - 100, 0, 5)*100)
                    except: ry.append(0.0)
                    rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
                gy.append(ry); gl.append(rl)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Annualized Yield (%)**")
                st.table(pd.DataFrame(gy, columns=[f"Stk {s}%" for s in sr], index=[f"KO {b}%" for b in br]).style.background_gradient(cmap='RdYlGn'))
            with col2:
                st.write("**Capital Loss Probability (%)**")
                st.table(pd.DataFrame(gl, columns=[f"Stk {s}%" for s in sr], index=[f"KO {b}%" for b in br]).style.background_gradient(cmap='Reds'))

# --- BCN INTERFACE ---
else:
    st.title("🛡️ Institutional BCN Solver")
    fqb = st.sidebar.selectbox("Guaranteed Frequency", ["Monthly", "Quarterly"])
    fmb, b_f = {"Monthly": 1, "Quarterly": 3}[fqb], st.sidebar.slider("Bonus Reference Strike %", 100, 130, 100)
    
    col_in1, col_in2 = st.columns(2)
    cg = col_in1.number_input("Guaranteed Coupon % (Fixed)", 0.0, 20.0, 4.0)/100
    brt = col_in2.number_input("Bonus Rate % (Contingent)", 0.0, 40.0, 8.0)/100
    
    if st.button("Generate Professional BCN Solve"):
        with st.spinner("Simulating Contingent Payoffs..."):
            v, c, lp = get_mkt(tks, src); av = v*(1+skw)
            L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
            z = np.random.standard_normal((int(tnr*252), 10000, len(v)))
            ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
            
            try:
                ki = brentq(lambda x: run_mc(cg, ps, rf, tnr, x, ko, fmb, nc, brt, b_f) - 100, 1, 100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Required KI Barrier", f"{ki:.2f}%")
                m2.metric("Guaranteed Coupon", f"{cg*100:.1f}%")
                m3.metric("Bonus Trigger", f"{b_f}%")
                
                st.subheader("📊 KI Barrier Sensitivity")
                gr, br = [cg-0.01, cg, cg+0.01], [brt-0.02, brt, brt+0.02]
                gb = []
                for b in br:
                    rb = []
                    for g in gr:
                        try: rb.append(brentq(lambda x: run_mc(g, ps, rf, tnr, x, ko, fmb, nc, b, b_f) - 100, 1, 100))
                        except: rb.append(0.0)
                    gb.append(rb)
                
                st.table(pd.DataFrame(gb, columns=[f"Guar {x*100:.1f}%" for x in gr], index=[f"Bonus {x*100:.1f}%" for x in br]).style.background_gradient(cmap='RdYlGn_r').format("{:.2f}"))
                
                st.write("### 🔍 Market Reference Data")
                st.table(pd.DataFrame({"Ticker": tks, "Spot Price (100%)": [f"${x:.2f}" for x in lp], "Implied Vol": [f"{x*100:.1f}%" for x in v]}))
                
            except ValueError:
                st.error("⚠️ **Mathematical Limit Reached**: The current coupons are too high for a par issuance. Please lower the Guaranteed or Bonus rates.")
