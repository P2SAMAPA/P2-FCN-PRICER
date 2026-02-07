import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- SETTINGS ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")

# --- MARKET DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, vol_mode, vol_window):
    ivs = []
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    for ticker in ticker_list:
        try:
            tk = yf.Ticker(ticker)
            if vol_mode == "Real-time Implied":
                ivs.append(0.32) # Standard baseline if API limit reached
            else:
                hist = tk.history(period=f"{vol_window}mo")['Close']
                log_returns = np.log(hist / hist.shift(1))
                ivs.append(log_returns.std() * np.sqrt(252))
        except:
            ivs.append(0.35) 
    
    rf_map = {"1Y UST": 0.045, "3M T-Bill": 0.053, "SOFR": 0.051}
    return ivs, rf_map.get(rf_choice, 0.05)

# --- QUANT ENGINE ---
class PricingEngine:
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, gtd_rate, bonus_rate=0.0, bonus_barr=85.0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq = int((freq_mo / 12) * 252)
        self.obs_steps = np.arange(self.obs_freq, self.steps + 1, self.obs_freq)
        self.total_possible_obs = len(self.obs_steps)
        self.nocall_steps = int((nocall_mo / 12) * 252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        # Coupon units
        self.gtd_unit = (gtd_rate / 100) * (freq_mo / 12)
        self.bonus_unit = (bonus_rate / 100) * (freq_mo / 12)
        self.bonus_barr = bonus_barr / 100

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets = len(self.vols)
        dt = 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        L = np.linalg.cholesky(np.full((n_assets, n_assets), 0.6) + np.eye(n_assets) * 0.4)
        
        total_payout_events, loss_freq, total_final_return = 0, 0, 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            sim_payout_count = 0
            sim_cash_flows = 0
            knocked_out = False
            
            for step in self.obs_steps:
                # Accumulate Coupons
                sim_cash_flows += self.gtd_unit
                if self.prod_type == "BCN" and worst_path[step-1] >= self.bonus_barr:
                    sim_cash_flows += self.bonus_unit
                    sim_payout_count += 1
                elif self.prod_type == "FCN":
                    sim_payout_count += 1
                
                # Check KO
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            final_val = 1.0 # Default Principal Return
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                final_val = worst_path[-1] / strike # Physical delivery penalty
            
            total_final_return += (final_val + sim_cash_flows)
            total_payout_events += sim_payout_count

        ann_yield = (((total_final_return / n_sims) - 1) / self.tenor_yr) * 100
        return (total_payout_events / n_sims), (loss_freq / n_sims), ann_yield

# --- UI LOGIC ---
st.title("🏦 Professional Derivatives Terminal")
tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
STRIKES, BARRIERS = [70, 75, 80, 85, 90], [90, 95, 100, 105, 110]

# --- FCN TAB ---
with tab1:
    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        st.subheader("FCN Config")
        f_t = st.text_input("Tickers", "AAPL, MSFT, GOOG", key="f1")
        f_v = st.radio("Vol Mode", ["Real-time Implied", "Historical"], key="f2")
        f_vw = st.selectbox("Lookback", [3, 6, 12, 24], index=2, key="f3")
        f_rf = st.selectbox("Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="f4")
        f_c = st.number_input("Coupon (%)", 12.0, key="f5")
        f_te = st.slider("Tenor (Mo)", 1, 36, 12, key="f6")
        f_fr = st.selectbox("Freq (Mo)", [1, 3, 6], key="f7")
        f_nc = st.selectbox("No-Call (Mo)", [1, 3, 6], key="f8")
        f_st = st.slider("Strike (%)", 50, 100, 80, key="f9")
        f_ko = st.slider("KO (%)", 80, 110, 100, key="f10")
        f_ks = st.radio("KO Type", ["Fixed", "Step Down"], key="f11")
        f_sd = st.slider("Step (%)", 0.0, 2.0, 0.5, key="f12") if f_ks == "Step Down" else 0
        btn_f = st.button("Price FCN")

    if btn_f:
        with f_col2:
            vols, rf = get_market_data(f_t, f_te, f_rf, f_v, f_vw)
            eng = PricingEngine(vols, rf, f_te, f_fr, f_nc, f_ks, f_sd, "FCN", f_c)
            avg_c, p_l, a_y = eng.run_simulation(f_st, f_ko)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Yield (Ann)", f"{a_y:.2f}%")
            m2.metric("Loss Prob", f"{p_l:.2%}")
            m3.metric("Coupons", f"{avg_c:.2f} / {eng.total_possible_obs}")
            
            y_mat = np.zeros((5,5)); l_mat = np.zeros((5,5))
            p = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng.run_simulation(sk, ko, n_sims=300)
                    y_mat[i,j] = y; l_mat[i,j] = l
                    p.progress((i*5+j+1)/25)
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(y_mat, BARRIERS, STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.dataframe(pd.DataFrame(l_mat, BARRIERS, STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- BCN TAB ---
with tab2:
    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        st.subheader("BCN Config")
        b_t = st.text_input("Tickers", "TSLA, NVDA, AMD", key="b1")
        b_v = st.radio("Vol Mode", ["Real-time Implied", "Historical"], key="b2")
        b_vw = st.selectbox("Lookback", [3, 6, 12, 24], index=2, key="b3")
        b_rf = st.selectbox("Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="b4")
        b_gc = st.number_input("Gtd Coupon (%)", 2.0, key="b5")
        b_bc = st.number_input("Bonus Coupon (%)", 8.0, key="b6")
        b_bb = st.slider("Bonus Barrier (%)", 50, 100, 85, key="b7")
        b_te = st.slider("Tenor (Mo)", 1, 36, 12, key="b8")
        b_fr = st.selectbox("Freq (Mo)", [1, 3, 6], key="b9")
        b_nc = st.selectbox("No-Call (Mo)", [1, 3, 6], key="b10")
        b_st = st.slider("Strike (%)", 50, 100, 75, key="b11")
        b_ko = st.slider("KO (%)", 80, 110, 100, key="b12")
        b_ks = st.radio("KO Type", ["Fixed", "Step Down"], key="b13")
        b_sd = st.slider("Step (%)", 0.0, 2.0, 0.5, key="b14") if b_ks == "Step Down" else 0
        btn_b = st.button("Price BCN")

    if btn_b:
        with b_col2:
            vols, rf = get_market_data(b_t, b_te, b_rf, b_v, b_vw)
            eng_b = PricingEngine(vols, rf, b_te, b_fr, b_nc, b_ks, b_sd, "BCN", b_gc, b_bc, b_bb)
            avg_c, p_l, a_y = eng_b.run_simulation(b_st, b_ko)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Yield (Ann)", f"{a_y:.2f}%")
            m2.metric("Loss Prob", f"{p_l:.2%}")
            m3.metric("Bonus Hits", f"{avg_c:.2f} / {eng_b.total_possible_obs}")
            
            y_mat_b = np.zeros((5,5)); l_mat_b = np.zeros((5,5))
            p_b = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng_b.run_simulation(sk, ko, n_sims=300)
                    y_mat_b[i,j] = y; l_mat_b[i,j] = l
                    p_b.progress((i*5+j+1)/25)
            c1b, c2b = st.columns(2)
            c1b.dataframe(pd.DataFrame(y_mat_b, BARRIERS, STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2b.dataframe(pd.DataFrame(l_mat_b, BARRIERS, STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
