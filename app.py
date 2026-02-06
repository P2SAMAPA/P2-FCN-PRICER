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
                    chain = s.option_chain(opts[min(len(opts
