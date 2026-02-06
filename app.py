import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- PRO UI CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")

# --- IMPROVED MARKET DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_mkt_data(tks, src):
    v, p, lp, divs = [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t); h = s.history(period="12mo")['Close']
            p.append(h.rename(t)); lp.append(h.iloc[-1])
            # Fetch Dividend Yield
            d_yield = s.info.get('dividendYield', 0)
            if d_yield is None: d_yield = 0
            divs.append(d_yield)
            
            hv = h.pct_change().std() * np.sqrt(252)
            if src == "Market Implied (IV)" and s.options:
                # Target ATM Vol (closest to spot)
                chain = s.option_chain(s.options[min(len(s.options)-1, 2)])
                calls = chain.calls
                atm_vol = calls.iloc[(calls['strike'] - lp[-1]).abs().argsort()[:1]]['impliedVolatility'].values[0]
                v.append(atm_vol)
            else: v.append(hv)
        except: v.append(0.35); lp.append(100.0); divs.append(0.01)
    
    df = pd.concat(p, axis=1).dropna() if p else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tks))
    return np.array(v), corr, np.array(lp), np.array(divs)

# --- ANTITHETIC VARIATES MC ENGINE ---
def run_mc_core(c_g, pths, r, tnr, stk, ko, f_m, nc_m, divs, b_r=0, b_f=0, step_down=0):
    # pths shape: [steps, paths, assets]
    steps, n_s, n_a = pths.shape
    wf = np.min(pths, axis=2)
    obs_idx = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    
    py = np.zeros(n_s)
    act = np.ones(n_s, dtype=bool)
    acc = np.zeros(n_s)
    cpn_count = np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    
    for i, d in enumerate(obs_idx):
        curr_ko = ko - (i * step_down)
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        cpn_count[act] += 1
        
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]
            act[ko_m] = False
            
    if np.any(act):
        # Payoff at maturity: Worst-of performance if below strike
        py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    
    return np.mean(py) * np.exp(-r * tnr), np.mean(cpn_count), (np.sum(wf[-1] < stk)/n_s)

# --- APP LOGIC (Updating paths with Dividend Drift) ---
# Inside your st.button block:
# drift = (rf_rate - divs - 0.5 * av**2) * (1/252)
# diffusion = av * np.sqrt(1/252)
# z = np.random.standard_normal((steps, n_paths // 2, n_assets))
# z_full = np.concatenate([z, -z], axis=1) # Antithetic Variates for better convergence
