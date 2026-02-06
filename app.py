import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. RESTORED PROFESSIONAL UI STYLING ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #d1d9e6; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stTable { background-color: #ffffff; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1a1c23; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_rf_rates():
    try:
        val = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        return val if val > 0 else 0.045
    except: return 0.045

@st.cache_data(ttl=3600)
def get_mkt_data(tks, src):
    v, p, lp, divs = [], [], [], []
    valid_names = []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            if h.empty: continue
            p.append(h.rename(t)); lp.append(h.iloc[-1]); valid_names.append(t)
            dy = s.info.get('dividendYield', 0.015) or 0.015
            divs.append(dy)
            if src == "Market Implied (IV)" and s.options:
                try:
                    c = s.option_chain(s.options[min(len(s.options)-1, 2)])
                    v.append(c.calls.iloc[(c.calls['strike'] - lp[-1]).abs().argsort()[:1]]['impliedVolatility'].values[0])
                except: v.append(h.pct_change().std() * np.sqrt(252))
            else: v.append(h.pct_change().std() * np.sqrt(252))
        except: continue
    if not p: return None, None, None, None, []
    df = pd.concat(p, axis=1).dropna()
    corr = df.pct_change().corr().values
    return np.array(v), corr, np.array(lp), np.array(divs), valid_names

# --- 3. PRICING KERNEL ---
def run_pricing_logic(cpn_pa, paths, r, tenor, stk, ko, f_m, nc_m, mode, sd=0, b_r=0, b_f=0):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_cnt = np.zeros(n_s)
    gv, bv = (cpn_pa*(f_m/12))*100, (b_r*(f_m/12))*100
    for i, d in enumerate(obs):
        curr_ko = ko - (i * sd) if "Version 2" in mode else ko
        acc[act] += gv
        cpn_cnt[act] += 1
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act):
        f_p = wf[-1, act]
        if "BCN" in mode:
            bonus = np.where(f_p >= b_f, bv * (len(obs)), 0)
            py[act] = np.where(f_p >= stk, 100 + bonus, f_p + bonus) + acc[act]
        else:
            py[act] = np.where(f_p >= stk, 100, f_p) + acc[act]
    return np.mean(py) * np.exp(-r * tenor), np.mean(cpn_cnt), (np.sum(wf[-1] < stk)/n_s)

# --- 4. RESTORED INPUT UX (SIDEBAR) ---
with st.sidebar:
    st.header("🕹️ Global Controls")
    mode = st.selectbox("Product", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks = [x.strip().upper() for x in st.text_input("Tickers", "SPY, QQQ").split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    
    st.subheader("🏦 Funding & RF Rate")
    rf_base_val = get_rf_rates()
    rf_choice = st.selectbox("Base Rate", ["3m T-Bill", "1 Year Treasury", "3m T-Bill + Spread"])
    spread_bps = st.slider("Spread (bps)", 0, 500, 100, step=10) if "Spread" in rf_choice else 0
    rf_rate = rf_base_val + (spread_bps / 10000)
    if "1 Year" in rf_choice: rf_rate += 0.002
    st.caption(f"Effective Rate: {rf_rate*100:.2f}%")
    
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- 5. EXECUTION & OUTPUT UX ---
if len(tks) >= 1:
    v, corr, spots, divs, names = get_mkt_data(tks, vol_src)
    if names:
        n_paths, days = 10000, int(tenor_y * 252)
        L = np.linalg.cholesky(corr + np.eye(len(v))*1e-9)
        drift = (rf_rate - divs - 0.5 * v**2) * (1/252)
        z = np.random.standard_normal((days, n_paths // 2, len(v)))
        z = np.concatenate([z, -z], axis=1)
        rets = drift + (v * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        ps = np.vstack([np.ones((1, n_paths, len(v)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

        st.title(f"🛡️ {mode} Analysis")
        
        if "BCN" in mode:
            c1, c2 = st.columns(2)
            g_cpn = c1.number_input("Guar %", 0.0, 20.0, 4.0)/100
            b_rate = c2.number_input("Bonus %", 0.0, 40.0, 8.0)/100
            b_ref = st.slider("Bonus Ref Strike %", 80, 120, 100)
            fq = st.selectbox("Frequency", [1, 3], format_func=lambda x: "Monthly" if x==1 else "Quarterly")
            
            if st.button("Solve Required Protection Barrier"):
                sol, _, prob_loss = run_pricing_logic(g_cpn, ps, rf_rate, tenor_y, 0, 100, fq, nc_m, mode, b_r=b_rate, b_f=b_ref)
                st.metric("Required Protection Barrier", f"{sol:.2f}%")
        else:
            stk_val = st.slider("Put Strike %", 40, 100, 75)
            ko_val = st.slider("KO Level %", 80, 130, 100)
            fq = st.selectbox("Frequency", [1, 3], format_func=lambda x: "Monthly" if x==1 else "Quarterly")
            sd = st.sidebar.slider("Step-Down %", 0.0, 2.0, 0.5) if "Version 2" in mode else 0
            
            if st.button("Generate Pricing Report"):
                sol = brentq(lambda x: run_pricing_logic(x, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode, sd=sd)[0] - 100, 0, 1.0)
                val, avg_cpn, prob_loss = run_pricing_logic(sol, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode, sd=sd)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Solved Annual Yield", f"{sol*100:.2f}%")
                m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
                m3.metric("Avg. Expected Coupons", f"{avg_cpn:.1f}")
                
                st.divider()
                st.write("**Correlation Matrix**")
                st.table(pd.DataFrame(corr, index=names, columns=names).style.background_gradient(cmap='Greens'))
    else:
        st.error("❌ Data fetch failed. Check tickers.")
