import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MARKET DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, spread_bps, vol_mode, vol_window):
    ivs = []
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    for ticker in ticker_list:
        try:
            tk = yf.Ticker(ticker)
            if vol_mode == "Real-time Implied (yFinance)":
                target_date = datetime.now() + timedelta(days=tenor_mo * 30)
                exp = tk.options
                closest_exp = min(exp, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
                chain = tk.option_chain(closest_exp).calls
                spot = tk.history(period="1d")['Close'].iloc[-1]
                atm_option = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
                ivs.append(atm_option['impliedVolatility'].values[0])
            else:
                hist = tk.history(period=f"{vol_window}mo")['Close']
                log_returns = np.log(hist / hist.shift(1))
                ivs.append(log_returns.std() * np.sqrt(252))
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
        
        # Simple correlation assumption
        corr_mat = np.full((n_assets, n_assets), 0.6)
        np.fill_diagonal(corr_mat, 1.0)
        L = np.linalg.cholesky(corr_mat)
        
        total_payout_magnitude = 0
        loss_frequency = 0
        total_loss_amount = 0
        
        for _ in range(n_sims):
            # Generate correlated paths
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            drift = (self.rf - 0.5 * self.vols**2) * dt
            diffusion = self.vols * np.sqrt(dt) * Z
            paths = np.exp(np.cumsum(drift + diffusion, axis=0))
            
            # TRACK WORST PERFORMER AT EACH STEP
            worst_performer_path = np.min(paths, axis=1)
            
            knocked_out = False
            sim_payout = 0
            
            # Observation Loop
            for step in self.obs_steps:
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))

                # Check for KO
                if step >= self.nocall_steps and worst_performer_path[step-1] >= curr_ko:
                    knocked_out = True
                    sim_payout += (self.gtd_coupon / len(self.obs_steps))
                    if self.prod_type == "BCN" and worst_performer_path[step-1] >= self.bonus_barrier:
                        sim_payout += (self.bonus_coupon / len(self.obs_steps))
                    break # Trade ends
                
                # Pay regular coupons if no KO
                sim_payout += (self.gtd_coupon / len(self.obs_steps))
                if self.prod_type == "BCN" and worst_performer_path[step-1] >= self.bonus_barrier:
                    sim_payout += (self.bonus_coupon / len(self.obs_steps))
            
            total_payout_magnitude += sim_payout
            
            # CAPITAL LOSS LOGIC: ONLY IF NOT KO'd
            if not knocked_out:
                final_worst = worst_performer_path[-1]
                if final_worst < strike:
                    loss_frequency += 1
                    total_loss_amount += (strike - final_worst)
                
        prob_l = loss_frequency / n_sims
        expected_loss_ann = (total_loss_amount / n_sims) / self.tenor_yr
        ann_yield = (self.rf + expected_loss_ann) * 100
        
        return (total_payout_magnitude / n_sims), prob_l, ann_yield

# --- STREAMLIT UI ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")

tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])

STRIKES = [70, 75, 80, 85, 90]
BARRIERS = [90, 95, 100, 105, 110]

# --- TAB 1: FCN ---
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
        run_fcn = st.button("Calculate FCN Value")

    with col2:
        if run_fcn:
            vols, rf = get_market_data(f_t, f_te, f_rf, 0, f_v_mode, f_v_win)
            eng = StructuredProductEngine(f_t.split(","), vols, rf, f_te, f_fr, f_nc, f_ks, f_sd, "FCN")
            
            # Initial Run
            _, p_l, a_y = eng.run_simulation(f_st_val, f_ko_val)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Est. Annualized Yield", f"{a_y:.2f}%")
            m2.metric("Prob. of Capital Loss", f"{p_l:.2%}")
            m3.metric("Expected Coupons Paid", f"{(_*100):.2f}")

            st.subheader("Sensitivity Analysis")
            y_res, l_res = np.zeros((5,5)), np.zeros((5,5))
            prog = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    _, cl, cy = eng.run_simulation(sk, ko, n_sims=1000)
                    y_res[i,j], l_res[i,j] = cy, cl
                    prog.progress((i * 5 + j + 1) / 25)
            
            c1, c2 = st.columns(2)
            c1.write("**Yield Matrix (KO vs Strike)**")
            c1.dataframe(pd.DataFrame(y_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            c2.write("**Capital Loss Matrix (KO vs Strike)**")
            c2.dataframe(pd.DataFrame(l_res, index=BARRIERS, columns=STRIKES).style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)

# --- TAB 2: BCN ---
# (Logic mirrored from Tab 1 with BCN specific coupon inputs)
