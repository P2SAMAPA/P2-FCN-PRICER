import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Derivatives Lab", layout="wide")

@st.cache_data(ttl=3600)
def get_mkt(tks, src):
    v, p, lp = [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            p.append(h.rename(t)); lp.append(h.iloc[-1])
            hv = h.pct_change().std() * np.sqrt(252)
            if src == "Market Implied (IV)" and s.options:
                c = s.option_chain(s.options[min(len(s.options)-1, 1)])
                v.append(max(c.calls['impliedVolatility'].median(), 0.1))
            else: v.append(hv)
        except: v.append(0.35); lp.append(100.0)
    df = pd.concat(p, axis=1).dropna() if p else pd.DataFrame()
    return np.array(v), df.pct_change().corr().values if not df.empty else np.eye(len(tks)), np.array(lp)

def run_mc(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0):
    steps, n_s, _ = pths.shape; wf = np.min(pths, axis=2)
    obs = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    py, act, acc = np.zeros(n_s), np.ones(n_s, dtype=bool), np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    for d in obs:
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= ko); py[ko_m] = 100 + acc[ko_m]; act[ko_m] = False
    if np.any(act): py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    return np.mean(py) * np.exp(-r * tnr)

# --- UI & LOGIC ---
md = st.sidebar.selectbox("Mode", ["FCN", "BCN"])
with st.sidebar:
    tk = st.text_input("Tickers", "NVDA, TSLA").split(",")
    tks = [x.strip().upper() for x in tk]
    src = st.radio("Vol", ["Historical (HV)", "Market Implied (IV)"])
    skw, rf = st.slider("Skew", 0.0, 1.0, 0.2), st.number_input("RF %", 0.0, 10.0, 4.5)/100
    tnr, nc = st.number_input("Tenor (Y)", 0.5, 3.0, 1.0), st.number_input("NC (M)", 0, 24, 3)
    ko = st.slider("KO %", 80, 150, 105)

if md == "FCN":
    fq = st.sidebar.selectbox("Freq", ["Monthly", "Quarterly", "Semi-Annual"])
    fm = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}[fq]
    stk = st.sidebar.slider("Strike %", 40, 100, 60)
    if st.button("Run FCN"):
        v, c, lp = get_mkt(tks, src); av = v*(1+skw)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tnr*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        y = brentq(lambda x: run_mc(x, ps, rf, tnr, stk, ko, fm, nc) - 100, 0, 5)
        st.metric("Yield", f"{y*100:.2f}% p.a.")
        sr, br = [stk-10, stk, stk+10], [ko+10, ko, ko-10]
        gy, gl = [], []
        for b in br:
            ry, rl = [], []
            for s in sr:
                try: ry.append(brentq(lambda x: run_mc(x, ps, rf, tnr, s, b, fm, nc) - 100, 0, 5)*100)
                except: ry.append(0.0)
                rl.append((np.sum(np.min(ps[-1], 1) < s)/10000)*100)
            gy.append(ry); gl.append(rl)
        st.write("### Yield %"); st.table(pd.DataFrame(gy, sr, br).T.style.background_gradient(cmap='RdYlGn'))
        st.write("### Capital Loss %"); st.table(pd.DataFrame(gl, sr, br).T.style.background_gradient(cmap='Reds'))
else:
    fqb = st.sidebar.selectbox("Guar. Freq", ["Monthly", "Quarterly"])
    fmb, b_f = {"Monthly": 1, "Quarterly": 3}[fqb], st.sidebar.slider("Bonus Ref %", 100, 130, 100)
    cg, brt = st.number_input("Guar %", 0.0, 20.0, 4.0)/100, st.number_input("Bonus %", 0.0, 40.0, 8.0)/100
    if st.button("Run BCN"):
        v, c, lp = get_mkt(tks, src); av = v*(1+skw)
        L = np.linalg.cholesky(c + np.eye(len(c))*1e-8)
        z = np.random.standard_normal((int(tnr*252), 10000, len(v)))
        ps = np.vstack([np.ones((1, 10000, len(v))), np.exp(np.cumsum((rf-0.5*av**2)*(1/252) + av*np.sqrt(1/252)*np.einsum('ij,tkj->tki', L, z), 0))])*100
        ki = brentq(lambda x: run_mc(cg, ps, rf, tnr, x, ko, fmb, nc, brt, b_f) - 100, 10, 100)
        st.metric("KI Barrier", f"{ki:.2f}%")
        gr, br = [cg-0.01, cg, cg+0.01], [brt-0.02, brt, brt+0.02]
        gb = []
        for b in br:
            rb = []
            for g in gr:
                try: rb.append(brentq(lambda x: run_mc(g, ps, rf, tnr, x, ko, fmb, nc, b, b_f) - 100, 10, 120))
                except: rb.append(0.0)
            gb.append(rb)
        st.write("### KI Sensitivity"); st.table(pd.DataFrame(gb, [f"{x*100:.1f}" for x in gr], [f"{x*100:.1f}" for x in br]).T.style.background_gradient(cmap='RdYlGn_r'))
