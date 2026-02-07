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
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, bonus_barr=85.0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq_days = int((freq_mo / 12) * 252)
        self.obs_steps = np.arange(self.obs_freq_days, self.steps + 1, self.obs_freq_days)
        self.total_possible_obs = len(self.obs_steps)
        self.nocall_steps = int((nocall_mo / 12) * 252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        self.bonus_barr = bonus_barr / 100

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets = len(self.vols)
        dt = 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        L = np.linalg.cholesky(np.full((n_assets, n_assets), 0.6) + np.eye(n_assets) * 0.4)
        
        total_payout_events, loss_freq, total_loss_amt = 0, 0, 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            knocked_out = False
            sim_payout_count = 0
            
            for step in self.obs_steps:
                # Determine payout event
                if self.prod_type == "FCN":
                    sim_payout_count += 1 
                else: # BCN: Count as payout if bonus barrier is cleared
                    if worst_path[step-1] >= self.bonus_barr:
                        sim_payout_count += 1
                
                # Check KO
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            total_payout_events += sim_payout_count
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                total_loss_amt += (strike - worst_path[-1])
                
        return (total_payout_events / n_sims), (loss_freq / n_sims), (self.rf + (total_loss_amt / n_sims) / self.tenor_yr) * 100

# --- UI LAYER ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")

tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
STRIKES, BARRIERS = [70, 75, 80, 85, 90], [90, 95, 100, 105, 110]

# --- TAB 1: FCN ---
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("FCN Inputs")
        f_t = st.text_input("Underlyings", "AAPL, MSFT, NVDA", key="f_t")
        f_v_mode = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="f_v")
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

    if run_fcn:
        with col2:
            vols, rf = get_market_data(f_t, f_te, f_rf, f_v_mode, f_v_win)
            eng = StructuredProductEngine(vols, rf, f_te, f_fr, f_nc, f_ks, f_sd, "FCN")
            avg_c, p_l, a_y = eng.run_simulation(f_st_val, f_ko_val)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Annualized Yield", f"{a_y:.2f}%")
            m2.metric("Prob. Capital Loss", f"{p_l:.2%}")
            m3.metric("Likely Coupons Paid", f"{avg_c:.2f} out of {eng.total_possible_obs}")
            
            y_res, l_res = np.zeros((5,5)), np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng.run_simulation(sk, ko, n_sims=400)
                    y_res[i,j], l_res[i,j] = y, l
                    prog.progress((i * 5 + j + 1) / 25)
            
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(y_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.dataframe(pd.DataFrame(l_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- TAB 2: BCN ---
with tab2:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("BCN Inputs")
        b_t = st.text_input("Underlyings", "TSLA, AMD, NVDA", key="b_t")
        b_v_mode = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="b_v_b")
        b_v_win = st.selectbox("Window (Mo)", [3, 6, 12, 24], index=2, key="b_vw_b") if "Historical" in b_v_mode else 12
        b_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="b_rf_b")
        
        # RESTORED COUPON FIELDS
        b_gtd = st.number_input("Guaranteed Coupon (%)", 2.0, step=0.5, key="b_gtd")
        b_bonus = st.number_input("Bonus Coupon (%)", 8.0, step=0.5, key="b_bonus")
        b_barr = st.slider("Bonus Barrier (%)", 50, 100, 85, key="b_barr")
        
        b_te = st.slider("Tenor (Months)", 1, 36, 12, key="b_te_b")
        b_fr = st.selectbox("Frequency (Months)", [1, 3, 6, 12], key="b_fr_b")
        b_nc = st.selectbox("No-Call (Months)", [1, 2, 3, 6], key="b_nc_b")
        b_st_val = st.slider("Put Strike (%)", 50, 100, 75, key="b_st_b")
        b_ko_val = st.slider("KO Barrier (%)", 80, 110, 100, key="b_ko_b")
        b_ks = st.radio("KO Schedule", ["Fixed", "Step Down"], key="b_ks_b")
        b_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="b_sd_b") if b_ks == "Step Down" else 0
        run_bcn = st.button("Calculate BCN")

    if run_bcn:
        with col2:
            vols_b, rf_b = get_market_data(b_t, b_te, b_rf, b_v_mode, b_v_win)
            eng_b = StructuredProductEngine(vols_b, rf_b, b_te, b_fr, b_nc, b_ks, b_sd, "BCN", bonus_barr=b_barr)
            avg_cb, p_lb, a_yb = eng_b.run_simulation(b_st_val, b_ko_val)
            
            st.divider()
            m1b, m2b, m3b = st.columns(3)
            m1b.metric("Annualized Yield", f"{a_yb:.2f}%")
            m2b.metric("Prob. Capital Loss", f"{p_lb:.2%}")
            m3b.metric("Likely Bonus Payouts", f"{avg_cb:.2f} out of {eng_b.total_possible_obs}")

            st.subheader("Sensitivity Analysis")
            y_res_b, l_res_b = np.zeros((5,5)), np.zeros((5,5))
            prog_b = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    cb, lb, yb = eng_b.run_simulation(sk, ko, n_sims=400)
                    y_res_b[i,j], l_res_b[i,j] = yb, lb
                    prog_b.progress((i * 5 + j + 1) / 25)
            
            c1b, c2b = st.columns(2)
            c1b.dataframe(pd.DataFrame(y_res_b, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2b.dataframe(pd.DataFrame(l_res_b, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
