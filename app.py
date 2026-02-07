import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. CLEAN SLATE CACHE CLEAR ---
st.cache_data.clear()

# --- 2. INSTITUTIONAL UI STYLING ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #0f172a; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1e293b; color: #38bdf8; }
    [data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. INPUT PANEL UX (All Controls in Sidebar) ---
with st.sidebar:
    st.title("🏦 Product Architect")
    mode = st.selectbox("Structure", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks_in = st.text_input("Underlying Basket", "SPY, QQQ")
    tks = [x.strip().upper() for x in tks_in.split(",")]
    
    st.divider()
    st.subheader("🛠️ Note Parameters")
    stk_pct = st.slider("Put Strike (%)", 40, 100, 75)
    ko_pct = st.slider("Autocall Level (%)", 80, 120, 100)
    freq_val = st.selectbox("Coupon Frequency", ["Monthly", "Quarterly"])
    freq_m = 1 if freq_val == "Monthly" else 3
    tenor = st.number_input("Tenor (Years)", 0.25, 5.0, 1.0, 0.25)
    nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
    
    st.divider()
    st.subheader("📈 Market Environment")
    rf_rate = st.number_input("Risk Free Rate (%)", 0.0, 10.0, 3.59) / 100
    sd_val = st.slider("Step-Down (%)", 0.0, 2.0, 0.0) if "Version 2" in mode else 0

# --- 4. THE PRICING ENGINE (Vectorized & Correlated) ---
@st.cache_data(ttl=600)
def get_market_engine(tickers):
    v, p, d, names = [], [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t); h = s.history(period="2y")['Close']
            if h.empty: continue
            p.append(h); names.append(t)
            # Critical: Dividend yield lowers the solved coupon
            d.append(s.info.get('dividendYield', 0.015) or 0.015)
            v.append(h.pct_change().tail(252).std() * np.sqrt(252))
        except: continue
    if not p: return None
    corr = pd.concat(p, axis=1).pct_change().dropna().corr().values
    return np.array(v), corr, np.array(d), names

def run_valuation(cpn, paths, r, tenor, strike, ko, f_m, nc_m, mode, sd):
    steps, n_s, n_a = paths.shape
    wo_path = np.min(paths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    payoff, active, accrued = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_event = (cpn * (f_m/12)) * 100
    
    for i, d in enumerate(obs):
        cur_ko = ko - (i * sd) if "Version 2" in mode else ko
        accrued[active] += cpn_event
        if d >= int((nc_m/12)*252):
            called = active & (wo_path[d] >= cur_ko)
            payoff[called] = 100 + accrued[called]
            active[called] = False
            
    if np.any(active):
        final_wo = wo_path[-1, active]
        payoff[active] = np.where(final_wo >= strike, 100, final_wo) + accrued[active]
    
    return np.mean(payoff) * np.exp(-r * tenor), (np.sum(wo_path[-1] < strike)/n_s)

# --- 5. OUTPUT PANEL UX ---
st.title(f"🛡️ {mode} Pricing Report")
st.info(f"Analysis for basket: **{', '.join(tks)}** | Tenor: **{tenor}Y**")

if st.button("EXECUTE INSTITUTIONAL VALUATION"):
    mkt = get_market_engine(tks)
    if mkt:
        vols, corr, divs, names = mkt
        n_sims, n_days = 10000, int(tenor * 252)
        
        # FIX: The Cholesky Correlation + Drift Logic
        L = np.linalg.cholesky(corr + np.eye(len(vols)) * 1e-10)
        dt = 1/252
        drift = (rf_rate - divs - 0.5 * vols**2) * dt
        z = np.random.standard_normal((n_days, n_sims, len(vols)))
        
        # Correlate returns at every time step
        rets = drift + (vols * np.sqrt(dt)) * np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols)))*100, 100 * np.exp(np.cumsum(rets, axis=0))])
        
        # Solve for the Par Coupon (PV=100)
        try:
            target = lambda x: run_valuation(x, paths, rf_rate, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)[0] - 100
            sol_cpn = brentq(target, 0.0, 1.0)
            _, p_loss = run_valuation(sol_cpn, paths, rf_rate, tenor, stk_pct, ko_pct, freq_m, nc_m, mode, sd_val)
            
            # THE THREE-COLUMN METRIC CARDS
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("SOLVED ANNUAL YIELD", f"{sol_cpn*100:.2f}%")
            m2.metric("PROB. CAPITAL LOSS", f"{p_loss*100:.1f}%")
            m3.metric("EXP. COUPON EVENTS", f"{int(tenor * (12/freq_m))}")
            
            st.divider()
            st.subheader("📊 Model Diagnostics")
            col_a, col_b = st.columns([1, 1])
            col_a.write("**Asset Correlation Matrix**")
            col_a.dataframe(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
            
            col_b.write("**Market Volatility (Used)**")
            vol_df = pd.DataFrame({"Asset": names, "Annual Vol": [f"{v*100:.1f}%" for v in vols]})
            col_b.table(vol_df)
            
        except Exception as e:
            st.error(f"Pricing Engine Error: {e}. Try increasing Risk Free Rate or lowering Strike.")
    else:
        st.error("Failed to fetch market data. Check tickers.")
