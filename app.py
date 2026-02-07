import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. GLOBAL STYLING ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: INPUT UX ---
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
    
    # Unified Rf Logic: Source + Spread
    rf_source = st.selectbox("Base Rate Source", ["3m T-Bill", "1 Year UST", "SOFR", "3m T-Bill + Spread"])
    spread = st.slider("Spread (%)", 0.0, 5.0, 1.0, 0.1) if "Spread" in rf_source else 0.0
    
    @st.cache_data(ttl=3600)
    def get_base_rate(src):
        ticker_map = {"3m T-Bill": "^IRX", "1 Year UST": "^IRX", "SOFR": "^IRX"} # Proxies
        try: return yf.Ticker(ticker_map.get(src, "^IRX")).history(period="1d")['Close'].iloc[-1] / 100
        except: return 0.045 # Fallback
    
    r_final = get_base_rate(rf_source) + (spread / 100)
    st.caption(f"Effective Model Rate: {r_final*100:.2f}%")
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
        v.append(h.pct_change().tail(252).std() * np.sqrt(252))
    return np.array(v), pd.concat(p, axis=1).pct_change().dropna().corr().values, np.array(d), names

def run_mc(cpn, paths, r, t, stk, ko, f_m, nc_m, mode, sd):
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
st.title(f"🛡️ {mode} Institutional Terminal")

if st.button("GENERATE PRICING & SENSITIVITY REPORT"):
    vols, corr, divs, names = get_mkt_data(tks, vol_src)
    n_s, n_d = 10000, int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(vols)) * 1e-10)
    
    def gen_paths(base_p=100):
        drift = (r_final - divs - 0.5 * vols**2) * (1/252)
        rets = drift + (vols * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, np.random.standard_normal((n_d, n_s, len(vols))))
        return np.vstack([np.ones((1, n_s, len(vols)))*base_p, base_p * np.exp(np.cumsum(rets, axis=0))])

    paths = gen_paths()
    sol = brentq(lambda x: run_mc(x, paths, r_final, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)[0]-100, 0, 1.0)
    val, p_loss = run_mc(sol, paths, r_final, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)

    # 3-COLUMN METRICS
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("SOLVED ANNUAL YIELD", f"{sol*100:.2f}%")
    m2.metric("PROB. CAPITAL LOSS", f"{p_loss*100:.1f}%")
    m3.metric("EXP. COUPON EVENTS", f"{int(tenor * (12/freq_m))}")

    # STRIKE & KO SENSITIVITY TABLES
    st.divider()
    st.subheader("📊 Price Sensitivity Analysis")
    s1, s2 = st.columns(2)
    
    # Put Strike Sensitivity
    stk_sens = []
    for s_shift in [stk_pct-10, stk_pct-5, stk_pct, stk_pct+5, stk_pct+10]:
        v_note, _ = run_mc(sol, paths, r_final, tenor, s_shift, ko_pct, freq_m, nc_m, mode, sd_val)
        stk_sens.append({"Put Strike %": f"{s_shift}%", "Note Value": f"{v_note:.2f}"})
    s1.write("**Sensitivity to Put Strike**")
    s1.table(pd.DataFrame(stk_sens))

    # KO Level Sensitivity
    ko_sens = []
    for k_shift in [ko_pct-5, ko_pct-2, ko_pct, ko_pct+2, ko_pct+5]:
        v_note, _ = run_mc(sol, paths, r_final, tenor, stk_pct, k_shift, freq_m, nc_m, mode, sd_val)
        ko_sens.append({"Autocall Level %": f"{k_shift}%", "Note Value": f"{v_note:.2f}"})
    s2.write("**Sensitivity to KO Level**")
    s2.table(pd.DataFrame(ko_sens))

    # DIAGNOSTICS
    st.divider()
    st.subheader("🔍 Model Assumptions")
    col_a, col_b = st.columns(2)
    col_a.write("**Correlation Matrix**")
    col_a.dataframe(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
    col_b.write("**Asset Parameters**")
    col_b.table(pd.DataFrame({"Asset": names, "Vol": vols, "Div Yield": divs}))
