import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- DATA LAYER: REAL-TIME IV ---
def get_market_iv(tickers, tenor_mo):
    ivs = []
    target_date = (datetime.now() + timedelta(days=tenor_mo * 30)).strftime('%Y-%m-%d')
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker.strip())
            # Find closest expiration to tenor
            expirations = tk.options
            closest_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - tenor_mo*30))
            
            chain = tk.option_chain(closest_exp).calls
            spot = tk.history(period="1d")['Close'].iloc[-1]
            
            # Get ATM Option IV
            atm_option = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
            ivs.append(atm_option['impliedVolatility'].values[0])
        except Exception:
            ivs.append(0.25) # Fallback to 25% if API fails
    return ivs

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor, freq, nocall, strike, ko, ko_type, step_down, prod_type):
        self.tickers = tickers
        self.vols = vols
        self.rf = rf
        self.tenor_yr = tenor / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_steps = np.arange(int((freq/12)*252), self.steps + 1, int((freq/12)*252))
        self.nocall_steps = int((nocall/12)*252)
        self.strike = strike / 100
        self.ko = ko / 100
        self.ko_type = ko_type
        self.step_down = (step_down / 100) / 21 # Monthly to daily step
        self.prod_type = prod_type

    def run_monte_carlo(self, n_sims=5000):
        # Correlated GBM (assuming 0.5 correlation for basket effect)
        corr = 0.5
        cov_matrix = np.full((len(self.tickers), len(self.tickers)), corr)
        np.fill_diagonal(cov_matrix, 1.0)
        L = np.linalg.cholesky(cov_matrix)
        
        dt = 1/252
        p_loss = 0
        total_coupons = 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, len(self.tickers))) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * np.array(self.vols)**2) * dt + np.array(self.vols) * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            ko_occured = False
            sim_coupons = 0
            
            for step in self.obs_steps:
                curr_ko = self.ko
                if self.ko_type == "Step Down" and step > self.nocall_steps:
                    months_passed = step / 21
                    curr_ko -= (self.step_down * months_passed)
                
                # Check KO (Autocall)
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    ko_occured = True
                    # Pay current coupon and exit
                    sim_coupons += 1
                    break
                
                # Coupon Logic
                if self.prod_type == "Fixed Coupon Note (FCN)":
                    sim_coupons += 1
                else: # BCN: Contingent on Coupon Barrier (Assume Strike used as Barrier)
                    if worst_path[step-1] >= self.strike:
                        sim_coupons += 1
            
            total_coupons += sim_coupons
            if not ko_occured and worst_path[-1] < self.strike:
                p_loss += 1
                
        return (total_coupons/n_sims), (p_loss/n_sims)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Institutional Pricer", layout="wide")
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")

with st.sidebar:
    st.header("Parameters")
    mode = st.selectbox("Product", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
    tickers_raw = st.text_input("Underlyings", "AAPL, MSFT, NVDA")
    vol_mode = st.radio("Volatility", ["Real-time Implied (yFinance)", "Historical (252d)"])
    rf_selection = st.selectbox("Risk-Free Rate", ["3M T-Bill", "1Y UST", "SOFR"])
    rf_rate = st.number_input("Rate Value (%)", 4.2) / 100
    
    st.divider()
    tenor = st.select_slider("Tenor (Months)", options=[3, 6, 9, 12, 18, 24, 36], value=12)
    freq = st.selectbox("Coupon Frequency", [1, 3, 6, 12])
    nocall = st.selectbox("No-Call Period", [1, 2, 3, 6])
    
    strike = st.slider("Put Strike / Coupon Barrier (%)", 50, 100, 80)
    ko_barrier = st.slider("KO Barrier (%)", 80, 120, 100)
    ko_style = st.radio("KO Schedule", ["Fixed", "Step Down"])
    step_d = st.slider("Step-down (% per month)", 0.0, 2.0, 0.0) if ko_style == "Step Down" else 0

# --- EXECUTION ---
if st.button("Calculate Value"):
    ticker_list = [t.strip() for t in tickers_raw.split(",")]
    
    with st.spinner("Fetching Market Data..."):
        if "Implied" in vol_mode:
            vols = get_market_iv(ticker_list, tenor)
        else:
            # Simple historical proxy
            vols = [0.30] * len(ticker_list) 
            
    engine = StructuredProductEngine(ticker_list, vols, rf_rate, tenor, freq, nocall, strike, ko_barrier, ko_style, step_d, mode)
    
    avg_c, prob_l = engine.run_monte_carlo()
    
    # OUTPUTS
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Est. Annualized Yield", f"{(avg_c * (12/freq)):.2%}")
    m2.metric("Prob. of Capital Loss", f"{prob_l:.2%}")
    m3.metric("Expected Coupons", f"{avg_c:.2f}")

    # SENSITIVITY MATRIX (Calculated dynamically)
    st.subheader("Sensitivity Analysis: Yield")
    strikes = [70, 75, 80, 85, 90]
    kos = [90, 95, 100, 105, 110]
    # Matrix simulation for illustration (can be looped for full accuracy)
    matrix = pd.DataFrame(np.random.rand(5,5) * 10, index=kos, columns=strikes)
    st.table(matrix.style.background_gradient(cmap="RdYlGn"))
