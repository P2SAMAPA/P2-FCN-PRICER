import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- CORE ENGINE CONFIG ---
st.set_page_config(page_title="Multi-Structure Pricing Lab", layout="wide")

@st.cache_data(ttl=3600)
def get_mkt_data(tickers, vol_src):
    v, p, lp, divs = [], [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t); h = s.history(period="1y")['Close']
            p.append(h.rename(t)); lp.append(h.iloc[-1])
            dy = s.info.get('dividendYield', 0.015) or 0.015
            divs.append(dy)
            if vol_src == "Market Implied (IV)" and s.options:
                chain = s.option_chain(s.options[min(len(s.options)-1, 2)])
                v.append(chain.calls.iloc[(chain.calls['strike'] - lp[-1]).abs().argsort()[:1]]['impliedVolatility'].values[0])
            else: v.append(h.pct_change().std() * np.sqrt(252))
        except: v.append(0.20); lp.append(100.0); divs.append(0.015)
    df = pd.concat(p, axis=1).dropna()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(v), corr, np.array(lp), np.array(divs)

# --- UNIFIED MONTE CARLO KERNEL ---
def run_pricing_logic(cpn_pa, paths, r, tenor, strike, ko, f_m, nc_m, mode, step_down=0, b_rate=0, b_ref=0):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2) # Worst-of performance
    obs_idx = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    
    payoffs, active, accrued = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_per_period = (cpn_pa * (f_m/12)) * 100
    
    for i, d in enumerate(obs_idx):
        accrued[active] += cpn_per_period
        curr_ko = ko - (i * step_down) if "Version 2" in mode else ko
        
        # Autocall Check (after Non-Call period)
        if d >= int((nc_m/12)*252):
            ko_m = active & (wf[d] >= curr_ko)
            payoffs[ko_m] = 100 + accrued[ko_m]
            active[ko_m] = False
            
    if np.any(active):
        # Maturity Payoff
        final_perf = wf[-1, active]
        if "BCN" in mode:
            # Bonus logic: if worst-of > bonus reference, pay bonus
            bonus = np.where(final_perf >= b_ref, b_rate * 100, 0)
            payoffs[active] = np.where(final_perf >= strike, 100 + bonus, final_perf + bonus) + accrued[active]
        else:
            # Standard FCN logic
            payoffs[active] = np.where(final_perf >= strike, 100, final_perf) + accrued[active]
            
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- UI CONTROLS ---
with st.sidebar:
    st.header("Global Configuration")
    mode = st.selectbox("Product Type", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tks = [x.strip().upper() for x in st.text_input("Tickers", "SPY, QQQ").split(",")]
    vol_src = st.radio("Vol Source", ["Historical", "Market Implied (IV)"])
    rf = st.number_input("RF Rate", 0.0, 0.1, 0.045)
    tenor = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0)
    strike = st.slider("Put Strike %", 50, 100, 75)
    ko = st.slider("KO Level %", 80, 120, 100)
    fq = st.selectbox("Frequency", [1, 3], format_func=lambda x: "Monthly" if x==1 else "Quarterly")

# --- EXECUTION ---
if len(tks) >= 2:
    vols, corr, spots, divs, names = get_mkt_data(tks, vol_src)
    n_paths, days = 10000, int(tenor * 252)
    L = np.linalg.cholesky(corr + np.eye(len(vols))*1e-9)
    drift = (rf - divs - 0.5 * vols**2) * (1/252)
    
    # Path Generation (Antithetic)
    z = np.random.standard_normal((days, n_paths // 2, len(vols)))
    z = np.concatenate([z, -z], axis=1)
    rets = drift + (vols * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
    paths = 100 * np.exp(np.cumsum(rets, axis=0))
    paths = np.vstack([np.ones((1, n_paths, len(vols)))*100, paths])

    st.title(f"📊 {mode} Pricing Report")
    
    if "BCN" in mode:
        b_rate = st.number_input("Bonus Rate %", 0.0, 20.0, 5.0) / 100
        b_ref = st.slider("Bonus Ref Strike %", 90, 110, 100)
        if st.button("Solve BCN Barrier"):
            # Solving for the required Put Strike to hit Par
            sol = brentq(lambda x: run_pricing_logic(0.04, paths, rf, tenor, x, ko, fq, 3, mode, b_rate=b_rate, b_ref=b_ref) - 100, 10, 100)
            st.metric("Required Protection Barrier", f"{sol:.2f}%")
    else:
        sd = st.sidebar.slider("Step-Down %", 0.0, 2.0, 0.5) if "Version 2" in mode else 0
        if st.button("Run FCN Solver"):
            sol = brentq(lambda x: run_pricing_logic(x, paths, rf, tenor, strike, ko, fq, 3, mode, step_down=sd) - 100, 0, 1.0)
            st.metric("Solved Annual Coupon", f"{sol*100:.2f}%")

    st.write("**Market Context**")
    st.table(pd.DataFrame({'Ticker': tks, 'Vol': vols*100, 'Div Yield': divs*100}))
