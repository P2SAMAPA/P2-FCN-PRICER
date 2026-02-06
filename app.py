import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")

# --- SHARED DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, source):
    vols, prices, last_px = [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            h = s.history(period="12mo")['Close']
            prices.append(h.rename(t))
            last_px.append(h.iloc[-1])
            hist_v = h.pct_change().std() * np.sqrt(252)
            if source == "Market Implied (IV)":
                opts = s.options
                if opts:
                    chain = s.option_chain(opts[min(len(opts)-1, 1)])
                    iv = chain.calls['impliedVolatility'].median()
                    vols.append(iv if iv > 0.1 else hist_v)
                else: vols.append(hist_v)
            else: vols.append(hist_v)
        except: 
            vols.append(0.35); last_px.append(100.0)
    df = pd.concat(prices, axis=1).dropna() if prices else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(vols), corr, np.array(last_px)

# --- FCN PRICER ENGINE ---
def get_fcn_pv(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    cpn_val = (coupon_pa * (freq_m/12)) * 100
    accrued = np.zeros(n_sims)
    for d in obs_dates:
        accrued[active] += cpn_val
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + accrued[active]
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- BCN PRICER ENGINE (REVERSED: SOLVES FOR STRIKE) ---
def get_bcn_pv(strike, paths, r, tenor, fixed_cpn, bonus_cpn, ko, cpn_bar, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    
    fixed_val = (fixed_cpn * (freq_m/12)) * 100
    bonus_val = (bonus_cpn * (freq_m/12)) * 100
    accrued = np.zeros(n_sims)
    
    for d in obs_dates:
        # Fixed always pays, Bonus only if above barrier
        accrued[active] += fixed_val
        eligible = active & (worst_of[d] >= cpn_bar)
        accrued[eligible] += bonus_val
        
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + accrued[active]
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- NAVIGATION ---
mode = st.sidebar.selectbox("Select Product", ["FCN Pricer", "BCN Solver"])

if mode == "FCN Pricer":
    st.markdown("## 🛡️ Institutional FCN Solver")
    with st.sidebar:
        st.header("1. Market Inputs")
        tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
        tickers = [x.strip().upper() for x in tk_in.split(",")]
        vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
        skew = st.slider("Vol Skew Factor", 0.0, 1.0, 0.2)
        rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
        st.header("2. Note Structure")
        tenor = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
        freq_label = st.selectbox("Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
        freq_m = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[freq_label]
        nc_m = st.number_input("Non-Call (Months)", 0, 24, 3)
        stk_fcn = st.slider("Put Strike %", 40, 100, 60)
        ko_fcn = st.slider("KO Barrier %", 80, 150, 100)

    if st.button("Generate FCN Pricing"):
        vols, corr, l_px = get_market_data(tickers, vol_src)
        adj_v = vols * (1 + skew)
        n_sims, steps, dt = 15000, int(tenor * 252), 1/252
        L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
        z = np.random.standard_normal((steps, n_sims, len(vols)))
        eps = np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum((rf - 0.5*adj_v**2)*dt + adj_v*np.sqrt(dt)*eps, axis=0))]) * 100
        
        y_solve = brentq(lambda c: get_fcn_pv(c, paths, rf, tenor, stk_fcn, ko_fcn, freq_m, nc_m) - 100, 0, 4)
        st.markdown(f"### Solved Annualized Yield: **{y_solve*100:.2f}% p.a.**")
        
        c1, c2 = st.columns(2)
        p_loss = (np.sum(np.min(paths[-1], axis=1) < stk_fcn) / n_sims) * 100
        c1.metric("Prob. Capital Loss", f"{p_loss:.1f}%")
        
        st.divider()
        st.write("### 🔍 Underlying Component Analysis")
        st.table(pd.DataFrame([{"Ticker": t, "Spot": f"${l_px[i]:.2f}", "IV": f"{vols[i]:.1%}"} for i, t in enumerate(tickers)]))

elif mode == "BCN Solver":
    st.markdown("## 🛡️ Institutional BCN Solver (Bonus Coupon)")
    with st.sidebar:
        st.header("1. Market Inputs")
        tk_in_b = st.text_input("Tickers (CSV)", "NVDA, TSLA")
        tickers_b = [x.strip().upper() for x in tk_in_b.split(",")]
        vol_src_b = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
        skew_b = st.slider("Vol Skew Factor", 0.0, 1.0, 0.2)
        rf_b = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
        st.header("2. BCN Structure")
        tenor_b = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
        freq_label_b = st.selectbox("Frequency", ["Monthly", "Quarterly"])
        freq_m_b = {"Monthly": 1, "Quarterly": 3}[freq_label_b]
        nc_m_b = st.number_input("Non-Call (Months)", 0, 24, 3)
        ko_b = st.slider("KO Barrier %", 80, 150, 105)
        cpn_bar_b = st.slider("Coupon Barrier %", 40, 100, 70)

    # BCN Input Table for Coupons
    st.write("### Coupon Inputs")
    col_c1, col_c2 = st.columns(2)
    f_cpn = col_c1.number_input("Fixed Coupon (% p.a.)", 0.0, 50.0, 5.0) / 100
    b_cpn = col_c2.number_input("Bonus Coupon (% p.a.)", 0.0, 50.0, 10.0) / 100

    if st.button("Solve Required Strike"):
        vols, corr, l_px = get_market_data(tickers_b, vol_src_b)
        adj_v = vols * (1 + skew_b)
        n_sims, steps, dt = 15000, int(tenor_b * 252), 1/252
        L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
        z = np.random.standard_normal((steps, n_sims, len(vols)))
        eps = np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum((rf_b - 0.5*adj_v**2)*dt + adj_v*np.sqrt(dt)*eps, axis=0))]) * 100

        try:
            # Solving for STRIKE instead of Yield
            stk_solve = brentq(lambda s: get_bcn_pv(s, paths, rf_b, tenor_b, f_cpn, b_cpn, ko_b, cpn_bar_b, freq_m_b, nc_m_b) - 100, 1.0, 100.0)
            st.metric("Required Put Strike", f"{stk_solve:.2f}%")
            
            # Sensitivity Matrix for BCN
            st.divider()
            st.write("### BCN Strike Sensitivity (Fixed vs Bonus Coupon)")
            f_range = [f_cpn-0.02, f_cpn, f_cpn+0.02]
            b_range = [b_cpn-0.05, b_cpn, b_cpn+0.05]
            
            grid = []
            for b in b_range:
                row = []
                for f in f_range:
                    try:
                        s_val = brentq(lambda s: get_bcn_pv(s, paths, rf_b, tenor_b, f, b, ko_b, cpn_bar_b, freq_m_b, nc_m_b) - 100, 1.0, 150.0)
                        row.append(s_val)
                    except: row.append(0.0)
                grid.append(row)
            
            df_sens = pd.DataFrame(grid, columns=[f"Fixed {f*100:.1f}%" for f in f_range], index=[f"Bonus {b*100:.1f}%" for b in b_range])
            st.table(df_sens.style.background_gradient(cmap='RdYlGn_r').format("{:.2f}"))
        except:
            st.error("Structure not solvable. Try reducing coupons or increasing KO.")
