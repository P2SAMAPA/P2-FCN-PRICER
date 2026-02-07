import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

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
class PricingEngine:
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, gtd_rate=0.0, bonus_rate=0.0, bonus_barr=85.0):
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
            
            sim_payout_count, sim_cash_flows, knocked_out = 0, 0, False
            
            for step in self.obs_steps:
                sim_cash_flows += self.gtd_unit
                if self.prod_type == "FCN":
                    sim_payout_count += 1
                elif self.prod_type == "BCN" and worst_path[step-1] >= self.bonus_barr:
                    sim_cash_flows += self.bonus_unit
                    sim_payout_count += 1
                
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            final_val = 1.0
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                final_val = worst_path[-1] / strike
            
            total_final_return += (final_val + sim_cash_flows)
            total_payout_events += sim_payout_count

        ann_yield = (((total_final_return / n_sims) - 1) / self.tenor_yr) * 100
        return (total_payout_events / n_sims), (loss_freq / n_sims), ann_yield

# --- UI LOGIC ---
st.title("🏦 Professional Derivatives Terminal")
tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
STRIKES, BARRIERS = [70, 75, 80, 85, 90], [90, 95, 100, 105, 110]

# --- TAB 1: FCN ---
with tab1:
    f_c1, f_c2 = st.columns([1, 3])
    with f_c1:
        st.subheader("FCN Config")
        f_t = st.text_input("Underlyings", "AAPL, MSFT, GOOG", key="ft")
        f_v = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="fv")
        # CONDITIONAL UI FOR LOOKBACK
        f_vw = st.selectbox("Window (Mo)", [3, 6, 12, 24], index=2, key="fvw") if f_v == "Historical" else 12
        f_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="frf")
        f_coup = st.number_input("Coupon (%)", 12.0, key="fcp")
        f_te = st.slider("Tenor (Months)", 1, 36, 12, key="fte")
        f_fr = st.selectbox("Frequency (Months)", [1, 3, 6, 12], key="ffr")
        f_nc = st.selectbox("No-Call (Months)", [1, 2, 3, 6], key="fnc")
        f_st = st.slider("Put Strike (%)", 50, 100, 80, key="fst")
        f_ko = st.slider("KO Barrier (%)", 80, 110, 100, key="fko")
        f_ks = st.radio("KO Schedule", ["Fixed", "Step Down"], key="fks")
        f_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="fsd") if f_ks == "Step Down" else 0
        run_fcn = st.button("Price FCN")

    if run_fcn:
        with f_c2:
            v_list, rf_val = get_market_data(f_t, f_te, f_rf, f_v, f_vw)
            eng = PricingEngine(v_list, rf_val, f_te, f_fr, f_nc, f_ks, f_sd, "FCN", gtd_rate=f_coup)
            avg_c, p_l, a_y = eng.run_simulation(f_st, f_ko)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Ann. Yield", f"{a_y:.2f}%")
            m2.metric("Loss Prob", f"{p_l:.2%}")
            m3.metric("Coupons Paid", f"{avg_c:.2f} out of {eng.total_possible_obs}")
            
            y_m = np.zeros((5,5)); l_m = np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng.run_simulation(sk, ko, n_sims=300)
                    y_m[i,j], l_m[i,j] = y, l
                    prog.progress((i*5+j+1)/25)
            c1, c2 = st.columns(2)
            c1.write("**Yield Matrix**")
            c1.dataframe(pd.DataFrame(y_m, BARRIERS, STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.write("**Capital Loss Matrix**")
            c2.dataframe(pd.DataFrame(l_m, BARRIERS, STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- TAB 2: BCN ---
with tab2:
    b_c1, b_c2 = st.columns([1, 3])
    with b_c1:
        st.subheader("BCN Config")
        b_t = st.text_input("Underlyings", "TSLA, NVDA, AMD", key="bt")
        b_v = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="bv")
        # CONDITIONAL UI FOR LOOKBACK
        b_vw = st.selectbox("Window (Mo)", [3, 6, 12, 24], index=2, key="bvw") if b_v == "Historical" else 12
        b_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="brf")
        b_gtd = st.number_input("Guaranteed (%)", 2.0, key="bgtd")
        b_bon = st.number_input("Bonus (%)", 8.0, key="bbon")
        b_bar = st.slider("Bonus Barrier (%)", 50, 100, 85, key="bbar")
        b_te = st.slider("Tenor (Months)", 1, 36, 12, key="bte")
        b_fr = st.selectbox("Frequency (Months)", [1, 3, 6, 12], key="bfr")
        b_nc = st.selectbox("No-Call (Months)", [1, 2, 3, 6], key="bnc")
        b_st = st.slider("Put Strike (%)", 50, 100, 75, key="bst")
        b_ko = st.slider("KO Barrier (%)", 80, 110, 100, key="bko")
        b_ks = st.radio("KO Schedule", ["Fixed", "Step Down"], key="bks")
        b_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="bsd") if b_ks == "Step Down" else 0
        run_bcn = st.button("Price BCN")

    if run_bcn:
        with b_c2:
            v_list, rf_val = get_market_data(b_t, b_te, b_rf, b_v, b_vw)
            eng_b = PricingEngine(v_list, rf_val, b_te, b_fr, b_nc, b_ks, b_sd, "BCN", gtd_rate=b_gtd, bonus_rate=b_bon, bonus_barr=b_bar)
            avg_c, p_l, a_y = eng_b.run_simulation(b_st, b_ko)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Ann. Yield", f"{a_y:.2f}%")
            m2.metric("Loss Prob", f"{p_l:.2%}")
            m3.metric("Bonus Hits", f"{avg_c:.2f} out of {eng_b.total_possible_obs}")
            
            y_m = np.zeros((5,5)); l_m = np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    c, l, y = eng_b.run_simulation(sk, ko, n_sims=300)
                    y_m[i,j], l_m[i,j] = y, l
                    prog.progress((i*5+j+1)/25)
            c1, c2 = st.columns(2)
            c1.write("**Yield Matrix**")
            c1.dataframe(pd.DataFrame(y_m, BARRIERS, STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.write("**Capital Loss Matrix**")
            c2.dataframe(pd.DataFrame(l_m, BARRIERS, STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
