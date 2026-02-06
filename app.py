import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- PRO UI CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #d1d9e6; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stTable { background-color: #ffffff; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1a1c23; color: white; font-weight: bold; }
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

def run_mc_core(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0, step_down=0):
    steps, n_s, _ = pths.shape; wf = np.min(pths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_count = np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    
    for i, d in enumerate(obs):
        # Apply Step-down if V2
        curr_ko = ko - (i * step_down)
        # Accrue Coupons
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        cpn_count[act] += 1
        
        # Check Autocall
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]
            act[ko_m] = False
            
    if np.any(act):
        py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    
    return np.mean(py) * np.exp(-r * tnr), np.mean(cpn_count), (np.sum(wf[-1] < stk)/n_s)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🕹️ Global Controls")
    mode = st.selectbox("Select Product", ["FCN Version 1 (Stable)", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers", "NVDA, TSLA")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Vol Skew", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("RF Rate %", 0.0, 10.0, 4.5)/100
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- MAIN INTERFACE ---
if "FCN" in mode:
    st.title(f"🛡️ Institutional {mode}")
    ko_val = st.sidebar.slider("Autocall Level (KO) %", 80, 150, 105)
    stk_val = st.sidebar.slider("Protection Barrier (Put Strike) %", 40, 100, 60)
    fq = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly"])
    fm = {"Monthly": 1, "Quarterly": 3}[fq]
    step_d = st.sidebar.slider("Step-Down % (V2 Only)", 0.0, 2.0, 0.5) if "Version 2" in mode else 0.0

    if st.button("Generate Professional Pricing Report"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        
        # Solve
        y_solve = brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)
        _, avg_cpn, prob_loss = run_mc_core(y_solve, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Solved Yield (p.a.)", f"{y_solve*100:.2f}%")
        m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
        m3.metric("Expected Coupons", f"{avg_cpn:.1f}")
        m4.metric("Protection Barrier", f"{stk_val}%")

        st.divider()
        
        # Sensitivity Tables
        st.subheader("📊 Multi-Factor Sensitivity Matrix")
        stks, kos = [stk_val-10, stk_val, stk_val+10], [ko_val+10, ko_val, ko_val-10]
        gy, gl = [], []
        for b in kos:
            ry, rl = [], []
            for s in stks:
                try: 
                    res_y = brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, s, b, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)*100
                    ry.append(res_y)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)

        c_left, c_right = st.columns(2)
        with c_left:
            st.write("**Annualized Yield (%)** - *KO Barrier (Rows) vs Strike (Cols)*")
            st.table(pd.DataFrame(gy, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='RdYlGn'))
        with c_right:
            st.write("**Capital Loss Probability (%)**")
            st.table(pd.DataFrame(gl, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='Reds'))

else:
    # --- BCN Solver (Restored UX) ---
    st.title("🛡️ Institutional BCN Solver")
    ko_b = st.sidebar.slider("KO Level %", 80, 150, 105)
    fq_b = st.sidebar.selectbox("Guar Freq", ["Monthly", "Quarterly"])
    fm_b, b_ref = {"Monthly": 1, "Quarterly": 3}[fq_b], st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    col_a, col_b = st.columns(2)
    g_cpn = col_a.number_input("Guaranteed Rate %", 0.0, 20.0, 4.0)/100
    b_rate = col_b.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
    
    if st.button("Solve Required KI Barrier"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        try:
            ki = brentq(lambda x: run_mc_core(g_cpn, ps, rf_rate, tenor_y, x, ko_b, fm_b, nc_m, b_rate, b_ref)[0] - 100, 0.01, 150)
            st.metric("Required KI Barrier", f"{ki:.2f}%")
            
            st.subheader("📊 KI Sensitivity Matrix")
            gs, bs = [g_cpn-0.01, g_cpn, g_cpn+0.01], [b_rate-0.02, b_rate, b_rate+0.02]
            gb = []
            for b in bs:
                rb = []
                for g in gs:
                    try: rb.append(brentq(lambda x: run_mc_core(g, ps, rf_rate, tenor_y, x, ko_b, fm_b, nc_m, b, b_ref)[0] - 100, 0.01, 150))
                    except: rb.append(0.0)
                gb.append(rb)
            st.table(pd.DataFrame(gb, index=[f"Bonus {x*100:.1f}%" for x in bs], columns=[f"Guar {x*100:.1f}%" for x in gs]).style.background_gradient(cmap='RdYlGn_r'))
        except: st.error("Solver Error: The note value is too high for par issuance.")
