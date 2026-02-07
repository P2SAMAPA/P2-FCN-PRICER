import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. GLOBAL STYLING (LOCKED) ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INPUT UX (SIDEBAR - DO NOT TOUCH) ---
with st.sidebar:
    st.header("🏢 Product Architect")
    mode = st.selectbox("Structure", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks_in = st.text_input("Underlying Basket", "SPY, QQQ")
    tks = [x.strip().upper() for x in tks_in.split(",")]
    
    st.divider()
    st.subheader("⚙️ Note Parameters")
    stk_pct = st.slider("Put Strike (%)", 40, 100, 75)
    ko_pct = st.slider("Autocall Level (%)", 80, 120, 100)
    freq_opt = st.selectbox("Coupon Frequency", ["Monthly", "Quarterly"])
    freq_m = 1 if freq_opt == "Monthly" else 3
    tenor = st.number_input("Tenor (Years)", 0.25, 5.0, 1.0, 0.25)
    nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
    
    st.divider()
    st.subheader("📈 Market Environment")
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    rf_source = st.selectbox("Base Rate Source", ["3m T-Bill", "1 Year UST", "SOFR", "3m T-Bill + Spread"])
    spread = st.slider("Spread (%)", 0.0, 5.0, 1.0, 0.1) if "Spread" in rf_source else 0.0
    
    @st.cache_data(ttl=3600)
    def get_base_rate(src):
        try: return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        except: return 0.045
    
    r_final = get_base_rate(rf_source) + (spread / 100)
    st.caption(f"Effective Model Rate: {r_final*100:.2f}%")
    sd_val = st.slider("Step-Down (%)", 0.0, 2.0, 0.0) if "Version 2" in mode else 0

# --- 3. PRICING ENGINE (MATHEMATICAL REPAIR) ---
@st.cache_data(ttl=600)
def get_mkt_context(tickers):
    v, p, d, names = [], [], [], []
    for t in tickers:
        s = yf.Ticker(t); h = s.history(period="2y")['Close']
        if h.empty: continue
        p.append(h); names.append(t)
        # Fix: Factor in Dividend Yield to reduce simulated risk
        d.append(s.info.get('dividendYield', 0.015) or 0.015)
        v.append(h.pct_change().tail(252).std() * np.sqrt(252))
    return np.array(v), pd.concat(p, axis=1).pct_change().dropna().corr().values, np.array(d), names

def run_valuation(cpn, paths, r, t, stk, ko, f_m, nc_m, mode, sd):
    steps, n_s, n_a = paths.shape
    wo = np.min(paths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    pay, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    c_val = (cpn * (f_m/12)) * 100
    for i, d in enumerate(obs):
        cur_ko = ko - (i * sd) if "Version 2" in mode else ko
        acc[act] += c_val
        if d >= int((nc_m/12)*252):
            k = act & (wo[d] >= cur_ko)
            pay[k] = 100 + acc[k]; act[k] = False
    if np.any(act):
        pay[act] = np.where(wo[-1, act] >= stk, 100, wo[-1, act]) + acc[act]
    return np.mean(pay) * np.exp(-r * t), (np.sum(wo[-1] < stk)/n_s)

# --- 4. OUTPUT UX (LOCKED) ---
st.title(f"🚀 {mode} Institutional Terminal")

if st.button("GENERATE PRICING & MATRIX SENSITIVITY"):
    vols, corr, divs, names = get_mkt_context(tks)
    n_s, n_d = 10000, int(tenor * 252)
    # Fix: Cholesky decomposition of Correlation Matrix
    L = np.linalg.cholesky(corr + np.eye(len(vols)) * 1e-10)
    
    def gen_paths():
        dt = 1/252
        drift = (r_final - divs - 0.5 * vols**2) * dt
        noise = np.einsum('ij,tkj->tki', L, np.random.standard_normal((n_d, n_s, len(vols))))
        rets = drift + (vols * np.sqrt(dt)) * noise
        return np.vstack([np.ones((1, n_s, len(vols)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

    paths = gen_paths()
    sol = brentq(lambda x: run_valuation(x, paths, r_final, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)[0]-100, 0, 1.0)
    _, base_loss = run_valuation(sol, paths, r_final, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("SOLVED ANNUAL YIELD", f"{sol*100:.2f}%")
    m2.metric("PROB. CAPITAL LOSS", f"{base_loss*100:.1f}%")
    m3.metric("EXP. COUPON EVENTS", f"{int(tenor * (12/freq_m))}")

    # SENSITIVITY MATRICES (Yield & Loss)
    st.divider()
    st.subheader("📊 Yield Sensitivity Matrix (KO Level vs Put Strike)")
    ko_range = [ko_pct-5, ko_pct, ko_pct+5]
    stk_range = [stk_pct-10, stk_pct-5, stk_pct, stk_pct+5, stk_pct+10]
    
    y_data, l_data = [], []
    for k in ko_range:
        y_r, l_r = [], []
        for s in stk_range:
            try:
                sy = brentq(lambda x: run_valuation(x, paths, r_final, tenor, s, k, freq_m, nc_m, mode, sd_val)[0]-100, 0, 1.5)
                _, sl = run_valuation(sy, paths, r_final, tenor, s, k, freq_m, nc_m, mode, sd_val)
                y_r.append(f"{sy*100:.2f}%"); l_r.append(f"{sl*100:.1f}%")
            except: y_r.append("N/A"); l_r.append("N/A")
        y_data.append(y_r); l_data.append(l_r)

    st.table(pd.DataFrame(y_data, index=[f"KO {k}%" for k in ko_range], columns=[f"Strike {s}%" for s in stk_range]))
    st.subheader("📉 Capital Loss Probability Matrix")
    st.table(pd.DataFrame(l_data, index=[f"KO {k}%" for k in ko_range], columns=[f"Strike {s}%" for s in stk_range]))

    st.divider()
    st.subheader("🔍 Model Assumptions")
    d1, d2 = st.columns(2)
    d1.write("**Correlation Matrix**")
    d1.dataframe(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
    d2.write("**Asset Parameters**")
    d2.table(pd.DataFrame({"Asset": names, "Vol": vols, "Div Yield": divs}))
