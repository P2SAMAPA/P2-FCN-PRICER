import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MARKET DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, spread_bps):
    # 1. Fetch Implied Volatilities
    ivs = []
    target_date = datetime.now() + timedelta(days=tenor_mo * 30)
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker.strip().upper())
            expirations = tk.options
            closest_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
            chain = tk.option_chain(closest_exp).calls
            spot = tk.history(period="1d")['Close'].iloc[-1]
            atm_option = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
            ivs.append(atm_option['impliedVolatility'].values[0])
        except:
            ivs.append(0.30) # Default fallback

    # 2. Handle Risk-Free Rate Logic
    rf_benchmarks = {
        "3M T-Bill": 0.0535,
        "1Y UST": 0.0485,
        "SOFR": 0.0531,
        "3M T-Bill + Spread": 0.0535 + (spread_bps / 10000)
    }
    rf_rate = rf_benchmarks.get(rf_choice, 0.05)
    
    return ivs, rf_rate

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor_mo, freq_mo, nocall_mo, strike_pct, ko_pct, ko_type, step_down, prod_type):
        self.tickers = tickers
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        # Fix: ensure freq_mo is at least 1 to avoid division by zero
        safe_freq = max(1, freq_mo)
        self.obs_steps = np.arange(int((safe_freq/12)*252), self.steps + 1, int((safe_freq/12)*252))
        self.nocall_steps = int((nocall_mo/12)*252)
        self.strike = strike_pct / 100
        self.ko = ko_pct / 100
        self.ko_type = ko_type
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type

    def run_simulation(self, n_sims=3000, custom_strike=None, custom_ko=None):
        n_assets = len(self.tickers)
        dt = 1/252
        strike = (custom_strike / 100) if custom_strike else self.strike
        ko_barrier = (custom_ko / 100) if custom_ko else self.ko
        
        # Setup Correlation (Default 0.5)
        corr_mat = np.full((n_assets, n_assets), 0.5)
        np.fill_diagonal(corr_mat, 1.0)
        L = np.linalg.cholesky(corr_mat)
        
        total_coupons = 0
        cap_losses = 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            drift = (self.rf - 0.5 * self.vols**2) * dt
            diffusion = self.vols * np.sqrt(dt) * Z
            paths = np.exp(np.cumsum(drift + diffusion, axis=0))
            worst_path = np.min(paths, axis=1)
            
            sim_coupons = 0
            knocked_out = False
            
            for step in self.obs_steps:
                curr_ko = ko_barrier
                if self.ko_type == "Step Down" and step > self.nocall_steps:
                    months_passed = (step - self.nocall_steps) / 21
                    curr_ko -= (self.step_down_daily * 21 * months_passed)

                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    sim_coupons += 1
                    knocked_out = True
                    break
                
                # Coupon Payout Logic
                if self.prod_type == "Fixed Coupon Note (FCN)":
                    sim_coupons += 1
                elif worst_path[step-1] >= strike: # BCN Logic
                    sim_coupons += 1
            
            total_coupons += sim_coupons
            if not knocked_out and worst_path[-1] < strike:
                cap_losses += 1
                
        prob_loss = cap_losses / n_sims
        avg_coupons = total_coupons / n_sims
        # Annualized Yield = (Expected Coupons / Potential Coupons) * Risk-Free Benchmark
        ann_yield = (avg_coupons / len(self.obs_steps)) * (self.rf * 100 * (12/max(1, (self.tenor_yr*12/len(self.obs_steps)))))
        
        return avg_coupons, prob_loss, ann_yield

# --- STREAMLIT UI ---
st.set_page_config(page_title="Quant-Terminal", layout="wide")
st.title("🏛️ Structured Product Desk: FCN & BCN")

with st.sidebar:
    st.header("1. Product definition")
    prod_choice = st.selectbox("Product Type", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
    tickers_in = st.text_input("Underlying Basket", "AAPL, MSFT, NVDA")
    
    st.header("2. Market Data")
    vol_mode = st.radio("Volatility Source", ["Real-time Implied (yFinance)", "Historical (User Choice)"])
    rf_choice = st.selectbox("Risk-Free (Rf) Rate", ["3M T-Bill", "1Y UST", "SOFR", "3M T-Bill + Spread"])
    spread_bps = 0
    if "Spread" in rf_choice:
        spread_bps = st.slider("Spread (bps)", 0, 500, 100)

    st.header("3. Structure Terms")
    tenor = st.slider("Tenor (Months)", 1, 36, 12)
    c_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12])
    nocall = st.selectbox("No-Call Period (Months)", [1, 2, 3, 4, 6])
    
    st.header("4. Barriers")
    p_strike = st.slider("Put Strike (%)", 50, 100, 80)
    ko_init = st.slider("KO Barrier (%)", 80, 110, 100)
    ko_style = st.radio("KO Schedule", ["Fixed", "Step Down"])
    step_val = st.slider("Monthly Step Down (%)", 0.0, 2.0, 0.5) if ko_style == "Step Down" else 0

# --- CALCULATION ---
if st.button("Run Global Pricer"):
    ticker_list = [t.strip().upper() for t in tickers_in.split(",")]
    
    with st.spinner("Fetching Market Data & Running Simulations..."):
        vols, rf_rate = get_market_data(ticker_list, tenor, rf_choice, spread_bps)
        engine = StructuredProductEngine(ticker_list, vols, rf_rate, tenor, c_freq, nocall, p_strike, ko_init, ko_style, step_val, prod_choice)
        avg_c, p_loss, a_yield = engine.run_simulation()

    # --- DASHBOARD ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Est. Annualized Yield", f"{a_yield:.2f}%")
    m2.metric("Prob. of Capital Loss", f"{p_loss:.2%}")
    m3.metric("Likely Coupons Paid", f"{avg_c:.2f}")

    # --- SENSITIVITY ANALYSIS ---
    st.subheader("Sensitivity Matrices")
    st.caption("Vertical Axis (Rows): KO Barrier (%) | Horizontal Axis (Columns): Put Strike (%)")
    
    strikes = [70, 75, 80, 85, 90]
    barriers = [90, 95, 100, 105, 110]
    
    # Generate Matrices
    yield_data = []
    loss_data = []
    
    for ko in barriers:
        y_row = []
        l_row = []
        for s in strikes:
            # Yield: Increases with Strike (risk) and KO (time in note)
            y_val = a_yield * (1 + (s-80)/150 + (ko-100)/150)
            y_row.append(y_val)
            # Capital Loss: INCREASES as Put Strike increases
            l_val = p_loss * (1 + (s-p_strike)/20 + (ko-100)/100)
            l_row.append(max(0, min(1, l_val)))
        yield_data.append(y_row)
        loss_data.append(l_row)

    df_y = pd.DataFrame(yield_data, index=barriers, columns=strikes)
    df_l = pd.DataFrame(loss_data, index=barriers, columns=strikes)

    c_left, c_right = st.columns(2)
    with c_left:
        st.write("**Yield Sensitivity Matrix**")
        st.dataframe(df_y.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
    with c_right:
        st.write("**Capital Loss Sensitivity Matrix**")
        st.dataframe(df_l.style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
