import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PRODUCT ARCHITECT (Input UX) ---
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
    rf_choice = st.selectbox("Base Rate Source", ["3m T-Bill", "SOFR", "Manual"])
    rf_val = st.number_input("Risk Free Rate (%)", 0.0, 10.0, 3.59) / 100
    
    sd_val = st.slider("Step-Down (%)", 0.0, 2.0, 0.0) if "Version 2" in mode else 0

# --- 3. PRICING ENGINE ---
@st.cache_data(ttl=600)
def get_mkt_data(tickers, v_src):
    v, p, d, names = [], [], [], []
    for t in tickers:
        s = yf.Ticker(t); h = s.history(period="2y")['Close']
        if h.empty: continue
        p.append(h); names.append(t)
        d.append(s.info.get('dividendYield', 0.015) or 0.015)
        if v_src == "Market Implied (IV)":
            try:
                opt = s.option_chain(s.options[0]).calls
                v.append(opt.iloc[len(opt)//2]['impliedVolatility'])
            except: v.append(h.pct_change().std() * np.sqrt(252))
        else: v.append(h.pct_change().std() * np.sqrt(252))
    return np.array(v), pd.concat(p, axis=1).pct_change().dropna().corr().values, np.array(d), names

def engine(cpn, paths, r, t, stk, ko, f_m, nc_m, mode, sd):
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

# --- 4. OUTPUT UX ---
st.title(f"🚀 {mode} Institutional Terminal")

if st.button("EXECUTE PRICING & SENSITIVITY"):
    vols, corr, divs, names = get_mkt_data(tks, vol_src)
    n_s, n_d = 10000, int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(vols)) * 1e-10)
    
    def gen_paths(v_adj=1.0):
        v = vols * v_adj
        drift = (rf_val - divs - 0.5 * v**2) * (1/252)
        z = np.random.standard_normal((n_d, n_s, len(v)))
        rets = drift + (v * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        return np.vstack([np.ones((1, n_s, len(v)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

    paths = gen_paths()
    sol = brentq(lambda x: engine(x, paths, rf_val, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)[0]-100, 0, 1.0)
    val, p_loss = engine(sol, paths, rf_val, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)

    # OUTPUT CARDS
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("SOLVED ANNUAL YIELD", f"{sol*100:.2f}%")
    c2.metric("PROB. CAPITAL LOSS", f"{p_loss*100:.1f}%")
    c3.metric("EXP. COUPON EVENTS", f"{int(tenor * (12/freq_m))}")

    # SENSITIVITY TABLES
    st.divider()
    st.subheader("📊 Sensitivity Analysis")
    s1, s2 = st.columns(2)
    
    # Delta Sensitivity (Spot Shift)
    delta_data = []
    for shift in [0.9, 0.95, 1.0, 1.05, 1.1]:
        v_shift, _ = engine(sol, paths * shift, rf_val, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)
        delta_data.append({"Spot Shift": f"{int((shift-1)*100)}%", "Note Value": f"{v_shift:.2f}"})
    s1.write("**Price Delta (Spot Shift)**")
    s1.table(pd.DataFrame(delta_data))

    # Vega Sensitivity (Vol Shift)
    vega_data = []
    for v_shift in [0.8, 0.9, 1.0, 1.1, 1.2]:
        p_v = gen_paths(v_adj=v_shift)
        v_note, _ = engine(sol, p_v, rf_val, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)
        vega_data.append({"Vol Multiplier": f"{v_shift}x", "Note Value": f"{v_note:.2f}"})
    s2.write("**Vega Sensitivity (Vol Shift)**")
    s2.table(pd.DataFrame(vega_data))

    # DIAGNOSTICS
    st.divider()
    st.subheader("🔍 Model Diagnostics")
    d1, d2 = st.columns(2)
    d1.write("**Correlation Matrix**")
    d1.dataframe(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
    d2.write("**Model Assumptions**")
    d2.table(pd.DataFrame({"Asset": names, "Vol": vols, "Div Yield": divs}))
