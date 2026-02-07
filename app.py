import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MARKET DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, spread_bps):
    ivs = []
    target_date = datetime.now() + timedelta(days=tenor_mo * 30)
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker.strip().upper())
            exp = tk.options
            closest_exp = min(exp, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
            chain = tk.option_chain(closest_exp).calls
            spot = tk.history(period="1d")['Close'].iloc[-1]
            atm_option = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
            ivs.append(atm_option['impliedVolatility'].values[0])
        except:
            ivs.append(0.35) 
    
    rf_benchmarks = {
        "3M T-Bill": 0.0535, "1Y UST": 0.0485, "SOFR": 0.0531,
        "3M T-Bill + Spread": 0.0535 + (spread_bps / 10000)
    }
    return ivs, rf_benchmarks.get(rf_choice, 0.05)

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, gtd_coupon=0, bonus_coupon=0, bonus_barrier=0):
        self.tickers = tickers
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        safe_freq = max(1, freq_mo)
        self.obs_steps = np.arange(int((safe_freq/12)*252), self.steps + 1, int((safe_freq/12)*252))
        self.nocall_steps = int((nocall_mo/12)*252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        self.gtd_coupon = gtd_coupon / 100
        self.bonus_coupon = bonus_coupon / 100
        self.bonus_barrier = bonus_barrier / 100

    def run_simulation(self, strike_pct, ko_pct, n_sims=2000):
        n_assets = len(self.tickers)
        dt = 1/252
        strike = strike_pct / 100
        ko_barrier = ko_pct / 100
        
        corr_mat = np.full((n_assets, n_assets), 0.5)
        np.fill_diagonal(corr_mat, 1.0)
        L = np.linalg.cholesky(corr_mat)
        
        total_payout_magnitude = 0
        loss_frequency = 0
        total_loss_amount = 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            drift = (self.rf - 0.5 * self.vols**2) * dt
            diffusion = self.vols * np.sqrt(dt) * Z
            paths = np.exp(np.cumsum(drift + diffusion, axis=0))
            worst_path = np.min(paths, axis=1)
            
            knocked_out = False
            sim_payout = 0
            
            for step in self.obs_steps:
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))

                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    # Pay accrued coupons for final period
                    sim_payout += self.gtd_coupon / len(self.obs_steps)
                    if self.prod_type == "BCN" and worst_path[step-1] >= self.bonus_barrier:
                        sim_payout += self.bonus_coupon / len(self.obs_steps)
                    break
                
                # Period Coupon
                sim_payout += self.gtd_coupon / len(self.obs_steps)
                if self.prod_type == "BCN" and worst_path[step-1] >= self.bonus_barrier:
                    sim_payout += self.bonus_coupon / len(self.obs_steps)
            
            total_payout_magnitude += sim_payout
            if not knocked_out and worst_path[-1] < strike:
                loss_frequency += 1
                total_loss_amount += (strike - worst_path[-1])
                
        prob_l = loss_frequency / n_sims
        expected_loss_ann = (total_loss_amount / n_sims) / self.tenor_yr
        ann_yield = (self.rf + expected_loss_ann) * 100
        
        return (total_payout_magnitude / n_sims), prob_l, ann_yield

# --- STREAMLIT UI ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")
st.title("🏦 Derivatives Pricing Terminal")

tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])

# Shared constants for matrices
STRIKES = [70, 75, 80, 85, 90]
BARRIERS = [90, 95, 100, 105, 110]

# --- TAB 1: FCN ---
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("FCN Inputs")
        f_tickers = st.text_input("Underlyings", "AAPL, MSFT, GOOG", key="f_t")
        f_rf_choice = st.selectbox("Rf Rate", ["3M T-Bill", "1Y UST", "SOFR", "3M T-Bill + Spread"], key="f_rf")
        f_spread = st.slider("Spread (bps)", 0, 500, 100, key="f_s") if "Spread" in f_rf_choice else 0
        f_tenor = st.slider("Tenor (Months)", 1, 36, 12, key="f_te")
        f_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12], key="f_fr")
        f_nocall = st.selectbox("No-Call Period (Months)", [1, 2, 3, 6], key="f_nc")
        f_strike = st.slider("Put Strike (%)", 50, 100, 80, key="f_st")
        f_ko = st.slider("KO Barrier (%)", 80, 110, 100, key="f_ko")
        f_ko_style = st.radio("KO Schedule", ["Fixed", "Step Down"], key="f_ks")
        f_step = st.slider("Monthly Step Down (%)", 0.0, 2.0, 0.5, key="f_sd") if f_ko_style == "Step Down" else 0
        run_fcn = st.button("Price FCN")

    with col2:
        if run_fcn:
            t_list = [t.strip().upper() for t in f_tickers.split(",")]
            vols, rf = get_market_data(t_list, f_tenor, f_rf_choice, f_spread)
            eng = StructuredProductEngine(t_list, vols, rf, f_tenor, f_freq, f_nocall, f_ko_style, f_step, "FCN")
            avg_p, p_l, a_y = eng.run_simulation(f_strike, f_ko)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Est. Annualized Yield", f"{a_y:.2f}%")
            m2.metric("Prob. of Capital Loss", f"{p_l:.2%}")
            m3.metric("Likely Avg Coupon Paid", f"{avg_p:.4f}")

            st.subheader("Sensitivity Analysis")
            y_res = np.zeros((5,5)); l_res = np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    _, cl, cy = eng.run_simulation(sk, ko, n_sims=400)
                    y_res[i,j] = cy; l_res[i,j] = cl
                    prog.progress((i * 5 + j + 1) / 25)
            
            st.write("**Yield Matrix (Rows: KO, Cols: Strike)**")
            st.dataframe(pd.DataFrame(y_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            st.write("**Capital Loss Matrix**")
            st.dataframe(pd.DataFrame(l_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- TAB 2: BCN ---
with tab2:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("BCN Inputs")
        b_tickers = st.text_input("Underlyings", "TSLA, NVDA, AMD", key="b_t")
        b_rf_choice = st.selectbox("Rf Rate", ["3M T-Bill", "1Y UST", "SOFR", "3M T-Bill + Spread"], key="b_rf")
        b_spread = st.slider("Spread (bps)", 0, 500, 100, key="b_s") if "Spread" in b_rf_choice else 0
        b_gtd = st.number_input("Guaranteed Coupon (Annual %)", 2.0, key="b_g")
        b_bonus = st.number_input("Bonus Coupon (Annual %)", 8.0, key="b_b")
        b_barr = st.slider("Bonus Coupon Barrier (%)", 50, 100, 85, key="b_ba")
        b_tenor = st.slider("Tenor (Months)", 1, 36, 12, key="b_te")
        b_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12], key="b_fr")
        b_nocall = st.selectbox("No-Call Period (Months)", [1, 2, 3, 6], key="b_nc")
        b_strike = st.slider("Put Strike (%)", 50, 100, 75, key="b_st")
        b_ko = st.slider("KO Barrier (%)", 80, 110, 100, key
