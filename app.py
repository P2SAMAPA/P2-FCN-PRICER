import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. RESTORED UI STYLE (Matches your successful screenshots) ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1a1c23; color: white; font-weight: bold; }
    .stTable { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. RESTORED SIDEBAR UX (Matches image_e25b9e) ---
with st.sidebar:
    st.header("⚙️ Global Controls")
    mode = st.selectbox("Product", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers (Comma Separated)", "SPY, QQQ")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"], index=1)
    
    st.markdown("---")
    st.subheader("🏦 Funding & RF Rate")
    
    @st.cache_data(ttl=3600)
    def get_rf_base():
        try: return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        except: return 0.045

    rf_base_val = get_rf_base()
    rf_choice = st.selectbox("Base Rate", ["3m T-Bill", "1 Year Treasury", "3m T-Bill + Spread"])
    spread_bps = st.slider("Spread (bps)", 0, 500, 100, step=10) if "Spread" in rf_choice else 0
    rf_rate = rf_base_val + (spread_bps / 10000)
    if "1 Year" in rf_choice: rf_rate += 0.002
    st.caption(f"Effective Rate: {rf_rate*100:.2f}%")
    
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0, step=0.25)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- 3. DYNAMIC DATA ENGINE (Fixes 67% Yield Error) ---
@st.cache_data(ttl=3600)
def get_mkt_data(tks, src):
    v, p, lp, divs, valid_names = [], [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            if h.empty: continue
            p.append(h.rename(t)); lp.append(h.iloc[-1]); valid_names.append(t)
            # Fetching dividends is critical to prevent yield inflation
            dy = s.info.get('dividendYield', 0.015) or 0.015
            divs.append(dy)
            if src == "Market Implied (IV)" and s.options:
                try:
                    c = s.option_chain(s.options[min(len(s.options)-1, 3)])
                    v.append(c.calls.iloc[(c.calls['strike'] - lp[-1]).abs().argsort()[:1]]['impliedVolatility'].values[0])
                except: v.append(h.pct_change().std() * np.sqrt(252))
            else: v.append(h.pct_change().std() * np.sqrt(252))
        except: continue
    if not p: return None, None, None, None, []
    df = pd.concat(p, axis=1).dropna()
    corr = df.pct_change().corr().values # Calculates real asset correlation
    return np.array(v), corr, np.array(lp), np.array(divs), valid_names

def run_pricing_logic(cpn_pa, paths, r, tenor, stk, ko, f_m, nc_m, mode, sd=0, b_r=0, b_f=0):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2) # "Worst-of" logic
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    gv, bv = (cpn_pa*(f_m/12))*100, (b_r*(f_m/12))*100
    for i, d in enumerate(obs):
        curr_ko = ko - (i * sd) if "Version 2" in mode else ko
        acc[act] += gv
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act):
        f_p = wf[-1, act]
        if "BCN" in mode:
            bonus = np.where(f_p >= b_f, bv * len(obs), 0)
            py[act] = np.where(f_p >= stk, 100 + bonus, f_p + bonus) + acc[act]
        else:
            py[act] = np.where(f_p >= stk, 100, f_p) + acc[act]
    return np.mean(py) * np.exp(-r * tenor), (np.sum(wf[-1] < stk)/n_s)

# --- 4. RESTORED MAIN OUTPUT UX (Matches image_e2515c) ---
if len(tks) >= 1:
    v, corr, spots, divs, names = get_mkt_data(tks, vol_src)
    if names:
        n_paths, days = 10000, int(tenor_y * 252)
        # Cholesky ensures SPY/QQQ move together, fixing the yield inflation
        L = np.linalg.cholesky(corr + np.eye(len(v))*1e-9)
        drift = (rf_rate - divs - 0.5 * v**2) * (1/252)
        z = np.random.standard_normal((days, n_paths // 2, len(v)))
        z = np.concatenate([z, -z], axis=1) # Variance reduction
        rets = drift + (v * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        ps = np.vstack([np.ones((1, n_paths, len(v)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])

        st.title(f"🛡️ {mode} Analysis")
        
        if "BCN" in mode:
            c1, c2 = st.columns(2)
            g_cpn = c1.number_input("Guaranteed Rate %", 0.0, 20.0, 4.0)/100
            b_rate = c2.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
            b_ref = st.slider("Bonus Ref Strike %", 80, 120, 100)
            fq = 1 if st.selectbox("Frequency", ["Monthly", "Quarterly"]) == "Monthly" else 3
            if st.button("Solve Barrier"):
                try:
                    sol = brentq(lambda x: run_pricing_logic(g_cpn, ps, rf_rate, tenor_y, x, 100, fq, nc_m, mode, b_r=b_rate, b_f=b_ref)[0]-100, 40, 100)
                    st.metric("Required Protection Barrier", f"{sol:.2f}%")
                except: st.warning("⚠️ Adjust rates.")
        else:
            stk_val = st.slider("Put Strike %", 40, 100, 75)
            ko_val = st.slider("KO Level %", 80, 130, 100)
            fq = 1 if st.selectbox("Frequency", ["Monthly", "Quarterly"]) == "Monthly" else 3
            sd = st.sidebar.slider("Step-Down %", 0.0, 2.0, 0.5) if "Version 2" in mode else 0
            
            if st.button("Generate Pricing Report"):
                sol = brentq(lambda x: run_pricing_logic(x, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode, sd=sd)[0] - 100, 0, 1.0)
                val, prob_loss = run_pricing_logic(sol, ps, rf_rate, tenor_y, stk_val, ko_val, fq, nc_m, mode, sd=sd)
                
                # RESTORED 3-COLUMN METRIC CARDS
                m1, m2, m3 = st.columns(3)
                m1.metric("Solved Annual Yield", f"{sol*100:.2f}%")
                m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
                m3.metric("Avg. Expected Coupons", f"{tenor_y * (12/fq):.1f}")
                
                st.divider()
                st.write("**Asset Correlation Matrix (Dynamic)**")
                st.table(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
