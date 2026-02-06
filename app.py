import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- 1. PROFESSIONAL UI CONFIG (Move to top to ensure styling loads) ---
st.set_page_config(page_title="Multi-Structure Pricing Lab", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1e1e1e; color: white; font-weight: bold; }
    .stTable { background-color: white; border-radius: 10px; }
    .stHeader { color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST MARKET DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_mkt_data(tickers, vol_src):
    v, p, lp, divs = [], [], [], []
    valid_names = []
    
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="1y")['Close']
            if h.empty: continue
            
            p.append(h.rename(t))
            spot = h.iloc[-1]
            lp.append(spot)
            valid_names.append(t)
            
            # Dividend Yield - Critical for Index (SPY/QQQ) pricing
            dy = s.info.get('dividendYield', 0.015) or 0.015
            divs.append(dy)
            
            if vol_src == "Market Implied (IV)" and s.options:
                try:
                    # Target ATM IV for 3-6 month expiry
                    expiry_idx = min(len(s.options)-1, 3)
                    chain = s.option_chain(s.options[expiry_idx])
                    calls = chain.calls
                    atm_vol = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]['impliedVolatility'].values[0]
                    v.append(atm_vol)
                except:
                    v.append(h.pct_change().std() * np.sqrt(252))
            else:
                v.append(h.pct_change().std() * np.sqrt(252))
        except Exception:
            continue
            
    if not p:
        return None, None, None, None, []
        
    df = pd.concat(p, axis=1).dropna()
    corr = df.pct_change().corr().values
    return np.array(v), corr, np.array(lp), np.array(divs), valid_names

# --- 3. UNIFIED PRICING KERNEL ---
def run_pricing_logic(cpn_pa, paths, r, tenor, strike, ko, f_m, nc_m, mode, step_down=0, b_rate=0, b_ref=0):
    steps, n_s, n_a = paths.shape
    wf = np.min(paths, axis=2) # Worst-of performance
    obs_idx = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    
    payoffs, active, accrued = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    cpn_per_period = (cpn_pa * (f_m/12)) * 100
    
    for i, d in enumerate(obs_idx):
        accrued[active] += cpn_per_period
        # Apply Step-down if V2
        curr_ko = ko - (i * step_down) if "Version 2" in mode else ko
        
        # Autocall Check after Non-Call
        if d >= int((nc_m/12)*252):
            ko_m = active & (wf[d] >= curr_ko)
            payoffs[ko_m] = 100 + accrued[ko_m]
            active[ko_m] = False
            
    if np.any(active):
        final_perf = wf[-1, active]
        if "BCN" in mode:
            bonus = np.where(final_perf >= b_ref, b_rate * 100, 0)
            # Payoff = 100 if above strike, else physical delivery + bonus + coupons
            payoffs[active] = np.where(final_perf >= strike, 100 + bonus, final_perf + bonus) + accrued[active]
        else:
            # FCN Logic
            payoffs[active] = np.where(final_perf >= strike, 100, final_perf) + accrued[active]
            
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Global Controls")
    mode = st.selectbox("Product Selection", ["FCN Version 1", "FCN Version 2 (Step-Down)", "BCN Solver"])
    t_input = st.text_input("Underlying Tickers", "SPY, QQQ")
    tks = [x.strip().upper() for x in t_input.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    rf = st.number_input("Risk-Free Rate %", 0.0, 10.0, 4.5) / 100
    tenor = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    strike_val = st.slider("Put Strike (Protection) %", 40, 100, 75)
    ko_val = st.slider("Autocall (KO) Level %", 80, 130, 100)
    fq = st.selectbox("Coupon Frequency", [1, 3], format_func=lambda x: "Monthly" if x==1 else "Quarterly")
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)

# --- 5. MAIN PAGE EXECUTION ---
if len(tks) >= 1:
    vols, corr, spots, divs, names = get_mkt_data(tks, vol_src)
    
    if names:
        # Standard Monte Carlo Setup
        n_paths, days = 10000, int(tenor * 252)
        L = np.linalg.cholesky(corr + np.eye(len(vols))*1e-9)
        # Drift: r - div - 0.5 * sigma^2
        drift = (rf - divs - 0.5 * vols**2) * (1/252)
        
        # Antithetic Variates for Variance Reduction
        z = np.random.standard_normal((days, n_paths // 2, len(vols)))
        z = np.concatenate([z, -z], axis=1)
        rets = drift + (vols * np.sqrt(1/252)) * np.einsum('ij,tkj->tki', L, z)
        paths = 100 * np.exp(np.cumsum(rets, axis=0))
        paths = np.vstack([np.ones((1, n_paths, len(vols)))*100, paths])

        st.markdown(f"## 🛡️ {mode} Analysis: {', '.join(names)}")
        
        if "BCN" in mode:
            col1, col2 = st.columns(2)
            g_rate = col1.number_input("Guaranteed Rate % (p.a.)", 0.0, 20.0, 4.0) / 100
            b_rate = col2.number_input("Bonus Rate % (p.a.)", 0.0, 40.0, 8.0) / 100
            b_ref = st.slider("Bonus Reference Strike %", 80, 120, 100)
            
            if st.button("Solve Required Protection Barrier"):
                try:
                    # Check if solution is possible
                    v100 = run_pricing_logic(g_rate, paths, rf, tenor, 100, ko_val, fq, nc_m, mode, b_rate=b_rate, b_ref=b_ref)
                    v40 = run_pricing_logic(g_rate, paths, rf, tenor, 40, ko_val, fq, nc_m, mode, b_rate=b_rate, b_ref=b_ref)
                    
                    if (v100 - 100) * (v40 - 100) < 0:
                        sol = brentq(lambda x: run_pricing_logic(g_rate, paths, rf, tenor, x, ko_val, fq, nc_m, mode, b_rate=b_rate, b_ref=b_ref) - 100, 40, 100)
                        st.metric("Required Protection Barrier", f"{sol:.2f}%")
                    else:
                        st.warning("⚠️ Market condition / Coupon mismatch. Try adjusting rates to allow par pricing.")
                except Exception as e:
                    st.error(f"Solver Error: {e}")

        else:
            # FCN Logic
            step_d = st.sidebar.slider("Step-Down % per period", 0.0, 2.0, value=0.5, step=0.5) if "Version 2" in mode else 0
            if st.button("Calculate Market Coupon"):
                try:
                    sol = brentq(lambda x: run_pricing_logic(x, paths, rf, tenor, strike_val, ko_val, fq, nc_m, mode, step_down=step_d) - 100, 0, 1.0)
                    st.metric("Solved Annual Coupon", f"{sol*100:.2f}%")
                    
                    st.divider()
                    st.write("**Correlation Matrix**")
                    st.table(pd.DataFrame(corr, index=names, columns=names).style.format("{:.2f}"))
                except Exception as e:
                    st.error(f"Solver Error: {e}")
    else:
        st.error("❌ Invalid Tickers. Please verify symbols on Yahoo Finance.")
