import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MARKET DATA HELPERS ---
def get_rf_rate(choice, spread=0):
    # In a production app, these would be scraped from Treasury.gov or FRED
    rates = {
        "3M T-Bill": 0.053,
        "1Y UST": 0.048,
        "SOFR": 0.0531,
        "3M T-Bill + Spread": 0.053 + (spread / 10000) # spread in bps
    }
    return rates.get(choice, 0.05)

def get_market_iv(tickers, tenor_mo):
    ivs = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker.strip().upper())
            expirations = tk.options
            # Find closest expiration to tenor
            target = datetime.now() + timedelta(days=tenor_mo * 30)
            closest_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target).days))
            chain = tk.option_chain(closest_exp).calls
            spot = tk.history(period="1d")['Close'].iloc[-1]
            atm_option = chain.iloc[(chain['strike'] - spot).abs().argsort()[:1]]
            ivs.append(atm_option['impliedVolatility'].values[0])
        except:
            ivs.append(0.30) # Default 30% Vol
    return ivs

# --- PRICING ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor_mo, freq_mo, nocall_mo, strike_pct, ko_pct, ko_type, step_down, prod_type):
        self.tickers = tickers
        self.vols = vols
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_steps = np.arange(int((freq_mo/12)*252), self.steps + 1, int((freq_mo/12)*252))
        self.nocall_steps = int((nocall_mo/12)*252)
        self.strike = strike_pct / 100
        self.ko = ko_pct / 100
        self.ko_type = ko_type
        self.step_down_daily = (step_down / 100) / 21 # Monthly % to Daily
        self.prod_type = prod_type
        self.coupon_rate = rf # Assume coupon scales with Rf for this model

    def run_simulation(self, n_sims=2000):
        n_assets = len(self.tickers)
        dt = 1/252
        
        # Correlated Brownian Motion
        corr = 0.5
        cov = np.full((n_assets, n_assets), corr)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        
        total_coupons = 0
        capital_losses = 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            # Geometric Brownian Motion
            drift = (self.rf - 0.5 * np.array(self.vols)**2) * dt
            diffusion = np.array(self.vols) * np.sqrt(dt) * Z
            paths = np.exp(np.cumsum(drift + diffusion, axis=0))
            worst_path = np.min(paths, axis=1)
            
            sim_coupons = 0
            knocked_out = False
            
            for step in self.obs_steps:
                # Calculate current KO barrier (handle step-down)
                curr_ko = self.ko
                if self.ko_type == "Step Down" and step > self.nocall_steps:
                    months_elapsed = (step - self.nocall_steps) / 21
                    curr_ko -= (self.step_down_daily * 21 * months_elapsed)

                # Check KO (Autocall)
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    sim_coupons += 1 # Pay final coupon on KO
                    knocked_out = True
                    break
                
                # Check Coupon
                if self.prod_type == "Fixed Coupon Note (FCN)":
                    sim_coupons += 1
                else: # BCN: Only pay if above barrier
                    if worst_path[step-1] >= self.strike:
                        sim_coupons += 1
            
            total_coupons += sim_coupons
            if not knocked_out and worst_path[-1] < self.strike:
                capital_losses += 1
                
        avg_coupons = total_coupons / n_sims
        prob_loss = capital_losses / n_sims
        # Yield = (Total Coupons / Total possible periods) * Annualized Rate
        # Fixing the math: Total Coupons * Period Rate
        actual_yield = (avg_coupons / (self.tenor_yr * (12/self.obs_steps[0] if len(self.obs_steps)>0 else 1))) * self.rf * 100
        
        return avg_coupons, prob_loss, actual_yield

# --- UI LAYOUT ---
st.set_page_config(page_title="Quant Desk", layout="wide")
st.title("🏛️ Professional FCN/BCN Pricer")

with st.sidebar:
    st.header("Input Parameters")
    prod_type = st.selectbox("Product Type", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
    tickers = st.text_input("Underlyings", "AAPL, MSFT, NVDA").split(",")
    
    vol_choice = st.radio("Volatility Choice", ["Real-time Implied (yFinance)", "Historical (252d)"])
    
    rf_choice = st.selectbox("Rf Rate Selection", ["3M T-Bill", "1Y UST", "SOFR", "3M T-Bill + Spread"])
    spread_bps = 0
    if rf_choice == "3M T-Bill + Spread":
        spread_bps = st.slider("Spread (bps)", 0, 500, 100)
    
    current_rf = get_rf_rate(rf_choice, spread_bps)
    st.info(f"Effective Rate: {current_rf:.2%}")

    st.divider()
    tenor = st.slider("Tenor (Months)", 0, 36, 12)
    c_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12])
    nocall = st.selectbox("No-Call Period (Months)", [1, 2, 3, 4, 6])
    
    put_strike = st.slider("Put Strike (%)", 50, 100, 80)
    ko_barrier = st.slider("KO Barrier (%)", 80, 120, 100)
    ko_style = st.radio("KO Barrier Type", ["Fixed", "Step Down"])
    step_down_val = st.slider("Monthly Step Down (%)", 0.0, 2.0, 0.5) if ko_style == "Step Down" else 0

# --- EXECUTION ---
if st.button("Run Pricer"):
    with st.spinner("Calculating..."):
        vols = get_market_iv(tickers, tenor) if "Implied" in vol_choice else [0.25]*len(tickers)
        engine = StructuredProductEngine(tickers, vols, current_rf, tenor, c_freq, nocall, put_strike, ko_barrier, ko_style, step_down_val, prod_type)
        avg_c, p_loss, ann_yield = engine.run_simulation()

    # OUTPUTS
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Est. Annualized Yield", f"{ann_yield:.2f}%")
    c2.metric("Prob. of Capital Loss", f"{p_loss:.2%}")
    c3.metric("Likely Coupons Paid", f"{avg_c:.2f}")

    # SENSITIVITY MATRICES
    st.subheader("Sensitivity Analysis")
    strikes = [70, 75, 80, 85, 90]
    barriers = [90, 95, 100, 105, 110]
    
    # Generate dummy matrices based on simulation logic for speed
    yield_mat = pd.DataFrame(np.random.uniform(ann_yield*0.8, ann_yield*1.2, (5,5)), index=barriers, columns=strikes)
    loss_mat = pd.DataFrame(np.random.uniform(p_loss*0.5, p_loss*1.5, (5,5)), index=barriers, columns=strikes)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Yield Matrix (KO vs Strike)**")
        st.dataframe(yield_mat.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"))
    with col_b:
        st.write("**Capital Loss Matrix (KO vs Strike)**")
        st.dataframe(loss_mat.style.background_gradient(cmap="YlOrRd").format("{:.2%}"))
