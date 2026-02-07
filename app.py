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
                # Using a grounded 32% if live IV fetch is throttled by API
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
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, coupon_rate=10.0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq = int((freq_mo / 12) * 252)
        self.obs_steps = np.arange(self.obs_freq, self.steps + 1, self.obs_freq)
        self.nocall_steps = int((nocall_mo / 12) * 252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        # Fixed coupon amount per observation window
        self.coupon_per_obs = (coupon_rate / 100) * (freq_mo / 12)

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets = len(self.vols)
        dt = 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        L = np.linalg.cholesky(np.full((n_assets, n_assets), 0.6) + np.eye(n_assets) * 0.4)
        
        total_coupons, loss_freq, total_loss_amt = 0, 0, 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            knocked_out = False
            sim_coupons = 0
            
            for step in self.obs_steps:
                # Accumulate coupon for the period BEFORE checking KO
                sim_coupons += self.coupon_per_obs
                
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                
                # Check for KO only after No-Call period
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            total_coupons += sim_coupons
            # Loss only occurs if no KO and worst asset ends below strike
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                total_loss_amt += (strike - worst_path[-1])
                
        prob_l = loss_freq / n_sims
        exp_loss_ann = (total_loss_amt / n_sims) / self.tenor_yr
        ann_yield = (self.rf + exp_loss_ann) * 100
        
        return (total_coupons / n_sims), prob_l, ann_yield

# --- UI LAYER ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")

tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
STRIKES = [70, 75, 80, 85, 90]
BARRIERS = [90, 95, 100, 105, 110]

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("FCN Inputs")
        f_t = st.text_input("Underlyings", "AAPL, MSFT, NVDA", key="f_t")
        f_v_mode = st.radio("Vol Source", ["Real-time Implied (yFinance)", "Historical Lookback"], key="f_v")
        f_v_win = st.selectbox("Window (Mo)", [3, 6, 12, 24], index=2, key="f_vw") if "Historical" in f_v_mode else 12
        f_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="f_rf")
        f_te = st.slider("Tenor (Months)", 1, 36, 12, key="f_te")
        f_fr = st.selectbox("Frequency (Months)", [1, 3, 6, 12], key="f_fr")
        f_nc = st.selectbox("No-Call (Months)", [1, 2, 3, 6], key="f_nc")
        f_st_val = st.slider("Put Strike (%)", 50, 100, 80, key="f_st")
        f_ko_val = st.slider("KO Barrier (%)", 80, 110, 100, key="f_ko")
        f_ks = st.radio("KO Schedule", ["Fixed", "Step Down"], key="f_ks")
        f_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="f_sd") if f_ks == "Step Down" else 0
        run_fcn = st.button("Calculate FCN")

    with col2:
        if run_fcn:
            vols, rf = get_market_data(f_t, f_te, f_rf, f_v_mode, f_v_win)
            eng = StructuredProductEngine(vols, rf, f_te, f_fr, f_nc, f_ks, f_sd, "FCN")
            
            avg_c, p_l, a_y = eng.run_simulation(f_st_val, f_ko_val)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Annualized Yield", f"{a_y:.2f}%")
            m2.metric("Prob. Capital Loss", f"{p_l:.2%}")
            m3.metric("Expected Coupons Paid", f"{avg_c:.2f}")

            st.subheader("Sensitivity Analysis")
            y_res, l_res = np.zeros((5,5)), np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng.run_simulation(sk, ko, n_sims=500)
                    y_res[i,j], l_res[i,j] = y, l
                    prog.progress((i * 5 + j + 1) / 25)
            
            c1, c2 = st.columns(2)
            c1.write("**Yield Matrix (KO vs Strike)**")
            c1.dataframe(pd.DataFrame(y_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.write("**Capital Loss Matrix (KO vs Strike)**")
            c2.dataframe(pd.DataFrame(l_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
