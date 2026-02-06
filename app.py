import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")

@st.cache_data(ttl=3600)
def get_market_data(tickers, source):
    vols, prices, last_px = [], [], []
    for t in tickers:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            prices.append(h.rename(t)); last_px.append(h.iloc[-1])
            hist_v = h.pct_change().std() * np.sqrt(252)
            if source == "Market Implied (IV)" and s.options:
                chain = s.option_chain(s.options[min(len(s.options)-1, 1)])
                vols.append(max(chain.calls['impliedVolatility'].median(), 0.1))
            else: vols.append(hist_v)
        except: vols.append(0.35); last_px.append(100.0)
    df = pd.concat(prices, axis=1).dropna() if prices else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tickers))
    return np.array(vols), corr, np.array(last_px)

def run_mc(cpn_guar, paths, r, tenor, strike, ko, freq_m, nc_m, b_rate=0, b_ref=0):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    payoffs, active, accrued = np.zeros(n_sims), np.ones(n_sims, dtype=bool), np.zeros(n_sims)
    g_val, b_val = (cpn_guar*(freq_m/12))*100, (b_rate*(freq_m/12))*100
    for d in obs:
        accrued[active] += g_val
        if b_rate > 0: accrued[active & (worst_of[d] >= b_ref)] += b_val
        if d >= int((nc_m/12)*252):
            ko_m = active & (worst_of[d] >= ko)
            payoffs[ko_m] = 100 + accrued[ko_m]
            active[ko_m] = False
    if np.any(active):
        payoffs[active] = np.where(worst_of[-1, active] >= strike, 100, worst_of[-1, active]) + accrued[active]
    return np.mean(payoffs) * np.exp(-r * tenor)

mode = st.sidebar.selectbox("Select Product", ["FCN Pricer", "BCN Solver"])
with st.sidebar:
    tk = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk.split(",")]
    vol_src = st.radio("Vol Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Vol Skew", 0.0, 1.0, 0.2)
    rf = st.number_input("RF Rate %", 0.0, 10.0, 4.5) / 100
    tenor = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)
    ko = st.slider("KO Barrier %", 80, 150, 105)

if mode == "FCN Pricer":
    st.markdown("## 🛡️ Institutional FCN Solver")
    freq = st.sidebar.selectbox("Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
    f_m = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[freq]
    stk = st.sidebar.slider("Put Strike %", 40, 100, 60)
    if st.button("Generate FCN"):
        v, c, lp = get_market_data(tickers, vol_src)
        av = v * (1 + skew); steps = int(tenor * 252)
        L = np.linalg.cholesky(c + np.eye(len(c)) * 1e-8)
        z = np.random.standard_normal((steps, 10000, len(v)))
        paths = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf - 0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), axis=0))]) * 100
        y = brentq(lambda cp: run_mc(cp, paths, rf, tenor, stk, ko, f_m, nc_m) - 100, 0, 5)
        st.metric("Solved Yield", f"{y*100:.2f}% p.a.")
        s_r, b_r = [stk-10, stk, stk+10], [ko+10, ko, ko-10]
        gy, gl = [], []
        for b in b_r:
            ry, rl = [], []
            for s in s_r:
                try: ry.append(brentq(lambda cp: run_mc(cp, paths, rf, tenor, s, b, f_m, nc_m) - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(paths[-1], axis=1) < s) / 10000) * 100)
            gy.append(ry); gl.append(rl)
        st.write("### Yield Sensitivity (%)"); st.table(pd.DataFrame(gy, columns=s_r, index=b_r).style.background_gradient(cmap='RdYlGn'))
        st.write("### Capital Loss Prob (%)"); st.table(pd.DataFrame(gl, columns=s_r, index=b_r).style.background_gradient(cmap='Reds'))

else:
    st.markdown("## 🛡️ Institutional BCN Solver")
    freq_b = st.sidebar.selectbox("Guar. Frequency", ["Monthly", "Quarterly"])
    f_m_b = {"Monthly": 1, "Quarterly": 3}[freq_b]
    b_ref_val = st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
    g_cpn = st.number_input("Guar. Coupon %", 0.0, 20.0, 4.0)/100
    b_cpn = st.number_input("Bonus Rate %", 0.0, 40.0, 8.0)/100
    if st.button("Solve KI Barrier"):
        v, c, lp = get_market_data(tickers, vol_src)
        av = v * (1 + skew); steps = int(tenor * 252)
        L = np.linalg.cholesky(c + np.eye(len(c)) * 1e-8)
        z = np.random.standard_normal((steps, 10000, len(v)))
        paths = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf - 0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), axis=0))]) * 100
        ki = brentq(lambda s: run_mc(g_cpn, paths, rf, tenor, s, ko, f_m_b, nc_m_b, b_cpn, b_ref_val) - 100, 10, 100)
        st.metric("Required KI Barrier", f"{ki:.2f}%")
        gr, br = [g_cpn-0.01, g_cpn, g_cpn+0.01], [b_cpn-0.02, b_cpn, b_cpn+0.02]
        gb = []
        for b in br:
            rb = []
            for g in gr:
                try: rb.append(brentq(lambda s: run_mc(g, paths, rf, tenor, s, ko, f_m_b, nc_m_b, b, b_ref_val) - 100, 10, 150))
                except: rb.append(0.0)
            gb.append(rb)
        st.write("### KI Barrier Sensitivity"); st.table(pd.DataFrame(gb, columns=[f"{x*100:.1f}%" for x in gr], index=[f"{x*100:.1f}%" for x in br]).style.background_gradient(cmap='RdYlGn_r'))
