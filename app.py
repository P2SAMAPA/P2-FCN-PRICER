
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

# --- FCN ENGINE ---
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

# --- BCN ENGINE ---
def get_bcn_pv(ki_barrier, paths, r, tenor, g_cpn, b_rate, ko, b_ref_stk, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    g_val, b_val = (g_cpn * (freq_m/12)) * 100, (b_rate * (freq_m/12)) * 100
    accrued = np.zeros(n_sims)
    for d in obs_dates:
        accrued[active] += g_val
        bonus_eligible = active & (worst_of[d] >= b_ref_stk)
        accrued[bonus_eligible] += b_val
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + accrued[ko_mask]
            active[ko_mask] = False
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= ki_barrier, 100, final_px) + accrued[active]
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
        
        # --- RESTORED FCN SENSITIVITY TABLES ---
        st.divider()
        st.write("### Yield Sensitivity Matrix (% p.a.)")
        stks_range = [stk_fcn-10, stk_fcn, stk_fcn+10]
        bars_range = [ko_fcn+10, ko_fcn, ko_fcn-10]
        grid_y, grid_l = [], []
        for b in bars_range:
            row_y, row_l = [], []
            for s in stks_range:
                try:
                    val = brentq(lambda c: get_fcn_pv(c, paths, rf, tenor, s, b, freq_m, nc_m) - 100, 0, 5)
                    row_y.append(val * 100)
                except: row_y.append(0.0)
                # Capital Loss Prob Calculation
                row_l.append((np.sum(np.min(paths[-1], axis=1) < s) / n_sims) * 100)
            grid_y.append(row_y)
            grid_l.append(row_l)
            
        st.table(pd.DataFrame(grid_y, columns=[f"Stk {s}%" for s in stks_range], index=[f"KO {b}%" for b in bars_range]).style.background_gradient(cmap='RdYlGn'))
        
        st.write("### Capital Loss Probability (%)")
        st.table(pd.DataFrame(grid_l, columns=[f"Stk {s}%" for s in stks_range], index=[f"KO {b}%" for b in bars_range]).style.background_gradient(cmap='Reds'))

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
        freq_label_b = st.selectbox("Guaranteed Coupon Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
        freq_m_b = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[freq_label_b]
        nc_m_b = st.number_input("Non-Call (Months)", 0, 24, 3)
        ko_b = st.slider("KO Barrier %", 80, 150, 105)
        b_ref_stk = st.slider("Bonus Reference Strike %", 100, 130, 100)

    st.write("### BCN Parameters")
    c1, c2 = st.columns(2)
    g_cpn = c1.number_input("Guaranteed Coupon Rate (% p.a.)", 0.0, 20.0, 4.0) / 100
    b_rate = c2.number_input("Bonus Rate (% p.a.)", 0.0, 40.0, 8.0) / 100

    if st.button("Solve KI Barrier"):
        vols, corr, l_px = get_market_data(tickers_b, vol_src_b)
        adj_v = vols * (1 + skew_b)
        n_sims, steps, dt = 15000, int(tenor_b * 252), 1/252
        L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
        z = np.random.standard_normal((steps, n_sims, len(vols)))
        eps = np.einsum('ij,tkj->tki', L, z)
        paths = np.vstack([np.ones((1, n_sims, len(vols))), np.exp(np.cumsum((rf_b - 0.5*adj_v**2)*dt + adj_v*np.sqrt(dt)*eps, axis=0))]) * 100

        st.write("#### Initial Reference Strikes (Spot)")
        st.table(pd.DataFrame([{"Ticker": t, "Spot (100%)": f"${l_px[i]:.2f}"} for i, t in enumerate(tickers_b)]))

        try:
            ki_solve = brentq(lambda s: get_bcn_pv(s, paths, rf_b, tenor_b, g_cpn, b_rate, ko_b, b_ref_stk, freq_m_b, nc_m_b) - 100, 10.0, 100.0)
            st.metric("Required KI Barrier (Put Strike)", f"{ki_solve:.2f}%")
            
            st.divider()
            st.write("### KI Barrier Sensitivity (Guaranteed vs Bonus Rate)")
            g_range = [g_cpn-0.01, g_cpn, g_cpn+0.01]
            b_range = [b_rate-0.02, b_rate, b_rate+0
