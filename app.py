import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import io
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, vol_mode, vol_window):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    ivs, prices = [], pd.DataFrame()
    
    for ticker in ticker_list:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="24mo")['Close'] 
            if not hist.empty:
                prices[ticker] = hist
                if vol_mode == "Real-time Implied": 
                    ivs.append(0.32) 
                else:
                    vol_hist = hist.tail(vol_window * 21)
                    log_returns = np.log(vol_hist / vol_hist.shift(1))
                    ivs.append(log_returns.std() * np.sqrt(252))
            else:
                ivs.append(0.35)
        except: 
            ivs.append(0.35)
    
    # --- FIX: Safety check for single ticker or empty data ---
    if len(ticker_list) > 1 and not prices.empty:
        corr_matrix = prices.pct_change().corr().values
        indices = np.triu_indices(len(ticker_list), k=1)
        corr_values = corr_matrix[indices]
        avg_hist_corr = np.nanmean(corr_values) if len(corr_values) > 0 else 0.6
    else:
        avg_hist_corr = 0.0 # No correlation possible for a single asset
        
    rf_map = {"1Y UST": 0.045, "3M T-Bill": 0.053, "SOFR": 0.051}
    return ivs, rf_map.get(rf_choice, 0.05), avg_hist_corr

# --- PRICING ENGINE ---
class PricingEngine:
    def __init__(self, vols, rf, tenor_mo, prod_type, correlation=0.6, freq_mo=1, nocall_mo=1, ko_style="Fixed", step_down=0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.prod_type = prod_type
        self.correlation = correlation
        # FCN specific params
        self.freq_mo = freq_mo
        self.nocall_mo = nocall_mo
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21

    def run_fcn_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets, dt = len(self.vols), 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        
        # Stabilize Correlation
        safe_corr = max(0.0, min(0.99, self.correlation))
        corr_matrix = np.full((n_assets, n_assets), safe_corr)
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix += np.eye(n_assets) * 1e-9
        L = np.linalg.cholesky(corr_matrix)
        
        steps = int(self.tenor_yr * 252)
        obs_freq = max(1, int((self.freq_mo / 12) * 252))
        obs_steps = np.arange(obs_freq, steps + 1, obs_freq)
        nocall_steps = int((self.nocall_mo / 12) * 252)
        
        total_life_months, total_profit, loss_freq = 0, 0, 0
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * (1/252) + self.vols * np.sqrt(1/252) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            sim_life_periods, knocked_out = len(obs_steps), False
            
            for i, step in enumerate(obs_steps):
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - nocall_steps))
                if step >= nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    sim_life_periods = i + 1
                    break
            
            final_p = 1.0
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                final_p = worst_path[-1]
            
            total_profit += (1.0 - final_p)
            total_life_months += (sim_life_periods * self.freq_mo)

        avg_yield = (self.rf + (total_profit / n_sims) / self.tenor_yr) * 100
        return (total_life_months / n_sims), (loss_freq / n_sims), avg_yield

    def run_bcn_simulation(self, strike_pct, n_sims=2000):
        n_assets = len(self.vols)
        strike = strike_pct / 100
        
        # Stabilize Correlation
        safe_corr = max(0.0, min(0.99, self.correlation))
        corr_matrix = np.full((n_assets, n_assets), safe_corr)
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix += np.eye(n_assets) * 1e-9
        
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)
        
        total_upside_participation, total_downside_loss, prob_above_strike = 0, 0, 0
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, n_assets) @ L.T
            terminal_prices = np.exp((self.rf - 0.5 * self.vols**2) * self.tenor_yr + self.vols * np.sqrt(self.tenor_yr) * Z)
            worst_performance = np.min(terminal_prices)
            
            if worst_performance >= strike:
                prob_above_strike += 1
                total_upside_participation += max(0, worst_performance - 1.0)
            else:
                total_downside_loss += (1.0 - worst_performance)

        avg_participation = total_upside_participation / n_sims
        avg_downside = total_downside_loss / n_sims
        prob_payout = prob_above_strike / n_sims
        fixed_coupon = ((avg_downside - avg_participation) / prob_payout) * 100 if prob_payout > 0 else 0
        return fixed_coupon, (1 - prob_payout), (avg_participation * 100)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Risk Configuration")
    corr_mode = st.selectbox("Correlation Method", ["Manual Slider", "Historical (Live Calc)", "Implied (Live + Buffer)"])
    
    if corr_mode == "Manual Slider":
        active_corr_input = st.slider("Manual Correlation", 0.0, 1.0, 0.6, 0.1)
        buffer_val = 0.0
    elif corr_mode == "Implied (Live + Buffer)":
        buffer_val = st.slider("Correlation Buffer (+)", 0.0, 0.5, 0.2, 0.05)
        active_corr_input = 0.0
    else:
        buffer_val = 0.0
        active_corr_input = 0.0

st.title("🏦 Derivatives Desk: FCN & BCN Pricer")
tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])

# --- FCN TAB ---
with tab1:
    f_c1, f_c2 = st.columns([1, 3])
    with f_c1:
        st.header("FCN Config")
        f_t = st.text_input("Underlyings", "AAPL, MSFT, GOOG", key="ft")
        f_v = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="fv")
        f_vw = st.selectbox("Lookback (Mo)", [3, 6, 12, 24], index=2, key="fvw") if f_v == "Historical" else 12
        f_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="frf")
        f_te, f_fr, f_nc = st.slider("Tenor (Mo)", 1, 36, 12, key="fte"), st.selectbox("Frequency (Mo)", [1, 3, 6], key="ffr"), st.selectbox("No-Call (Mo)", [1, 3, 6], key="fnc")
        f_st, f_ko = st.slider("Strike (%)", 10, 100, 80, key="fst"), st.slider("KO Barrier (%)", 50, 150, 100, key="fko")
        f_ks = st.radio("KO Schedule", ["Fixed", "Step Down"], key="fks")
        f_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="fsd") if f_ks == "Step Down" else 0
        run_fcn = st.button("Calculate Yield")

    if run_fcn:
        STRIKES = [f_st - 20, f_st - 10, f_st, f_st + 10, f_st + 20]
        BARRIERS = [f_ko - 20, f_ko - 10, f_ko, f_ko + 10, f_ko + 20]
        with f_c2:
            v, rf, h_c = get_market_data(f_t, f_te, f_rf, f_v, f_vw)
            if corr_mode == "Manual Slider": final_c = active_corr_input
            elif corr_mode == "Historical (Live Calc)": final_c = h_c
            else: final_c = min(1.0, h_c + buffer_val)
            
            eng = PricingEngine(v, rf, f_te, "FCN", correlation=final_c, freq_mo=f_fr, nocall_mo=f_nc, ko_style=f_ks, step_down=f_sd)
            life, loss, yld = eng.run_fcn_simulation(f_st, f_ko)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Output Yield", f"{yld:.2f}%"); m2.metric("Loss Prob", f"{loss:.2%}"); m3.metric("Exp. Life (Months)", f"{life:.2f}")
            
            y_m, l_m = np.zeros((5,5)), np.zeros((5,5))
            p = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    l_val, ls, y = eng.run_fcn_simulation(sk, ko, n_sims=300)
                    y_m[i,j], l_m[i,j] = y, ls
                    p.progress((i*5+j+1)/25)
            df_y, df_l = pd.DataFrame(y_m, BARRIERS, STRIKES), pd.DataFrame(l_m, BARRIERS, STRIKES)
            ca, cb = st.columns(2)
            ca.write("### Yield Matrix"); ca.dataframe(df_y.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            cb.write("### Loss Matrix"); cb.dataframe(df_l.style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- BCN TAB ---
with tab2:
    bc1, bc2 = st.columns([1, 3])
    with bc1:
        st.header("New BCN Config")
        b_t = st.text_input("Underlyings", "AAPL, MSFT, GOOG", key="bt")
        b_v = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="bv")
        b_vw = st.selectbox("Lookback (Mo)", [3, 6, 12, 24], index=2, key="bvw") if b_v == "Historical" else 12
        b_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="brf")
        b_te, b_st = st.slider("Tenor (Mo)", 1, 24, 12, key="bte"), st.slider("Put Strike (%)", 50, 100, 85, key="bst")
        run_bcn = st.button("Calculate BCN")

    if run_bcn:
        STRIKES_B = [b_st - 10, b_st - 5, b_st, b_st + 5, b_st + 10]
        TENORS_B = [max(1, b_te - 6), max(1, b_te - 3), b_te, b_te + 3, b_te + 6]
        with bc2:
            v_b, rf_b, h_c_b = get_market_data(b_t, b_te, b_rf, b_v, b_vw)
            if corr_mode == "Manual Slider": final_c = active_corr_input
            elif corr_mode == "Historical (Live Calc)": final_c = h_c_b
            else: final_c = min(1.0, h_c_b + buffer_val)
            
            eng_b = PricingEngine(v_b, rf_b, b_te, "BCN", correlation=final_c)
            fixed_x, prob_loss, avg_kicker = eng_b.run_bcn_simulation(b_st)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Affordable Fixed Coupon (X)", f"{fixed_x:.2f}%"); m2.metric("Prob. of Capital Loss", f"{prob_loss:.2%}"); m3.metric("Avg. Expected Bonus", f"{avg_kicker:.2f}%")
            
            st.info(f"**Payout Logic:** If at {b_te} months the worst stock is > {b_st}%, you receive 100% + {fixed_x:.2f}% + Worst Stock Return. Otherwise, delivery of Worst Stock.")
            
            res_coupon, res_loss = np.zeros((5,5)), np.zeros((5,5))
            p_bar = st.progress(0)
            for i, te in enumerate(TENORS_B):
                for j, sk in enumerate(STRIKES_B):
                    temp_eng = PricingEngine(v_b, rf_b, te, "BCN", correlation=final_c)
                    x, p_l, _ = temp_eng.run_bcn_simulation(sk, n_sims=400)
                    res_coupon[i,j], res_loss[i,j] = x, p_l
                    p_bar.progress((i*5 + j + 1) / 25)
            
            df_coupon = pd.DataFrame(res_coupon, index=[f"{t} Mo" for t in TENORS_B], columns=[f"{s}%" for s in STRIKES_B])
            df_loss = pd.DataFrame(res_loss, index=[f"{t} Mo" for t in TENORS_B], columns=[f"{s}%" for s in STRIKES_B])
            
            col_a, col_b = st.columns(2)
            col_a.write("### Sensitivity: Fixed Coupon X%"); col_a.dataframe(df_coupon.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            col_b.write("### Sensitivity: Capital Loss Prob"); col_b.dataframe(df_loss.style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
