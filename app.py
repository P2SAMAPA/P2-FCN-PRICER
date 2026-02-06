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

# --- MARKET DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_rf_rates():
    try:
        # ^IRX = 13-week T-Bill, ^FVX = 5yr (used as proxy if others fail), ^TNX = 10yr
        # We fetch the 13-week and the 1-year Treasury Bill
        tb3m = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        # 1-Year proxy usually requires checking the 12-month bill or interpolation
        # For reliability in Yahoo, we'll use ^IRX as the base 3m rate
        return tb3m
    except:
        return 0.045 # Fallback to 4.5% if API is down

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

# --- PRICING LOGIC ---
def run_mc_core(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0, step_down=0):
    steps, n_s, _ = pths.shape; wf = np.min(pths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_count = np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    for i, d in enumerate(obs):
        curr_ko = ko - (i * step_down)
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        cpn_count[act] += 1
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
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
    
    st.subheader("🏦 Funding & RF Rate")
    rf_base_val = get_rf_rates()
    rf_choice = st.selectbox("Base Rate", ["3m T-Bill", "1 Year Treasury", "3m T-Bill + Spread"])
    
    # Logic for Spread
    spread_bps = 0
    if rf_choice == "3m T-Bill + Spread":
        # Increments of 10bps from 0 to 5% (500 bps)
        spread_bps = st.slider("Spread (bps)", 0, 500, 100, step=10)
    
    # Map choice to value
    if "3m T-Bill" in rf_choice:
        rf_rate = rf_base_val + (spread_bps / 10000)
    else:
        # Using a slight premium for 1Y vs 3M as proxy if yfinance 1Y ticker is flaky
        rf_rate = rf_base_val + 0.002 
    
    st.caption(f"Effective Rate: {rf_rate*100:.2f}%")
    
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- DISPATCHER ---
# (The rest of the logic remains identical to the previous version to maintain stability)
if mode == "FCN Version 1 (Stable)":
    st.title("🛡️ Institutional FCN (Stable V1)")
    ko_val = st.sidebar.slider("Autocall Level (KO) %", 80, 150, 105)
    stk_val = st.sidebar.slider("Protection Barrier (Put Strike) %", 40, 100, 60)
    fq = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly"])
    fm = {"Monthly": 1, "Quarterly": 3}[fq]

    if st.button("Generate Pricing Report (V1)"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        y_solve = brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m)[0] - 100, 0, 5)
        val, avg_cpn, prob_loss = run_mc_core(y_solve, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Solved Yield (p.a.)", f"{y_solve*100:.2f}%")
        m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
        m3.metric("Expected Coupons", f"{avg_cpn:.1f}")
        m4.metric("Protection Barrier", f"{stk_val}%")
        
        st.divider()
        stks, kos = [stk_val-10, stk_val, stk_val+10], [ko_val+10, ko_val, ko_val-10]
        gy, gl = [], []
        for b in kos:
            ry, rl = [], []
            for s in stks:
                try: ry.append(brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, s, b, fm, nc_m)[0] - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)
        cl, cr = st.columns(2)
        with cl:
            st.write("**Annualized Yield (%)** - *KO (Rows) vs Strike (Cols)*")
            st.table(pd.DataFrame(gy, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='RdYlGn'))
        with cr:
            st.write("**Capital Loss Probability (%)**")
            st.table(pd.DataFrame(gl, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='Reds'))

elif mode == "FCN Version 2 (Step-Down)":
    st.title("🛡️ Institutional FCN (Step-Down V2)")
    ko_val = st.sidebar.slider("Initial Autocall Level %", 80, 150, 105)
    stk_val = st.sidebar.slider("Protection Barrier %", 40, 100, 60)
    fq = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly"])
    fm = {"Monthly": 1, "Quarterly": 3}[fq]
    step_d = st.sidebar.slider("Step-Down % per period", 0.0, 2.0, value=0.5, step=0.5)

    if st.button("Generate Pricing Report (V2)"):
        v, c, lp = get_mkt_data(tks, vol_src); av = v*(1+skew)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf_rate-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, np.random.standard_normal((int(tenor_y*252), 10000, len(v)))), 0))])*100
        y_solve = brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)
        val, avg_cpn, prob_loss = run_mc_core(y_solve, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("V2 Solved Yield", f"{y_solve*100:.2f}%")
        m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
        m3.metric("Expected Coupons", f"{avg_cpn:.1f}")
        m4.metric("Step-Down Rate", f"{step_d}%")
        
        st.divider()
        stks, kos = [stk_val-10, stk_val, stk_val+10], [ko_val+10, ko_val, ko_val-10]
        gy, gl = [], []
        for b in kos:
            ry, rl = [], []
            for s in stks:
                try: ry.append(brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, s, b, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)
        cl, cr = st.columns(2)
        with cl:
            st.write("**Annualized Yield (%)** - *KO (Rows) vs Strike (Cols)*")
            st.table(pd.DataFrame(gy, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='RdYlGn'))
        with cr:
            st.write("**Capital Loss Probability (%)**")
            st.table(pd.DataFrame(gl, index=[f"KO {k}%" for k in kos], columns=[f"Stk {s}%" for s in stks]).style.background_gradient(cmap='Reds'))

else:
    st.title("🛡️ Institutional BCN Solver")
    ko_b = st.sidebar.slider("KO Level %", 80, 150, 105)
    fq_b = st.sidebar.selectbox("Guar Freq", ["Monthly", "Quarterly"])
    fm_b, b_ref = {"Monthly": 1, "Quarterly": 3}[fq_b], st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    ga, ba = st.columns(2)
    g_cpn = ga.number_input("Guar %", 0.0, 20.0, 4.0)/100
    b_rate = ba.number_input("Bonus %", 0.0, 40.0, 8.0)/100
    
    if st.button("Solve BCN"):
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
        except: st.error("Solver Error: Note value too high.")
