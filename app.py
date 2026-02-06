import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_context(tickers):
    data = []
    for t in tickers:
        s = yf.Ticker(t)
        h = s.history(period="12mo")['Close']
        data.append(h.rename(t))
    df = pd.concat(data, axis=1).dropna()
    return df, df.pct_change().corr().values

def get_vols(tickers, source):
    v_list = []
    for t in tickers:
        s = yf.Ticker(t)
        hist_v = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        if source == "Market Implied (IV)":
            try:
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 2)])
                px = s.history(period="1d")['Close'].iloc[-1]
                v_list.append(chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0])
            except: v_list.append(hist_v)
        else: v_list.append(hist_v)
    return np.array(v_list)

# --- VALUATION ENGINE ---
def run_fcn_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_steps = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    
    payoffs, active = np.zeros(n_sims), np.ones(
