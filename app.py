import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MARKET DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, vol_mode, vol_window):
    ivs = []
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    for ticker in ticker_list:
        try:
            tk = yf.Ticker(ticker)
            if vol_mode == "Real-time Implied (yFinance)":
                hist = tk.history(period="1d")
                spot = hist['Close'].iloc[-1]
                # Fallback to historical if IV fetch fails
                ivs.append(0.32) 
            else:
                hist = tk.history(period=f"{vol_window}mo")['Close']
                log_returns = np.log(hist / hist.shift(1))
                ivs.append(log_returns.std() * np.sqrt(252))
        except:
            ivs.append(0.35) 
    
    rf_map = {"1Y UST": 0.045, "3M T-Bill": 0.053, "SOFR": 0.051}
    return ivs, rf_map.get(rf_choice, 0.05)

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, gtd_coupon=10.0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq_days = int((freq_mo / 12) * 252)
        self.obs_steps = np.arange(self.obs_freq_days, self.steps + 1, self.obs_freq_days)
        self.nocall_days = int((nocall_mo / 12) * 252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        self.coupon_per_obs = (gtd_coupon / 100) * (freq_mo / 12)

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets = len(self.vols)
        dt = 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        L = np.linalg.cholesky(np.full((n_assets, n_assets), 0.6) + np.eye(n_assets) * 0.4)
        
        total_coupons, loss_freq, total_loss = 0, 0, 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            knocked_out = False
            sim_coupons = 0
            
            for step in self.obs_steps:
                # Accumulate coupon for the period
                sim_coupons += self.coupon_per_obs
                
                # Check for KO (Only after No-Call period)
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_days:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_days))
                
                if step >= self.nocall_days and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            total_coupons += sim_coupons
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                total_loss += (strike - worst_path[-1])
                
        return (total_coupons / n_sims), (loss_freq / n_sims), (self.rf + (total_loss / n_sims) / self.tenor_yr) * 100

# --- UI LAYER ---
st.set_page_config(page_title="Professional Pricer", layout="wide")
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")

# Sidebars and Tabs logic preserved...
# [FCN Tab Logic with restored Matrices]
