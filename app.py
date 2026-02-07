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
            ivs.append(0.30)
    
    rf_benchmarks = {
        "3M T-Bill": 0.0535, "1Y UST": 0.0485, "SOFR": 0.0531,
        "3M T-Bill + Spread": 0.0535 + (spread_bps / 10000)
    }
    return ivs, rf_benchmarks.get(rf_choice, 0.05)

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, tickers, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type):
        self.tickers = tickers
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.freq_steps = int((freq_mo/12)*252)
        self.obs_steps = np.arange(self.freq_steps, self.steps + 1, self.freq_steps)
        self.nocall_steps = int((nocall_mo/12)*252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type

    def run_simulation(self, strike_pct, ko_pct, n_sims=2000):
        n_assets = len(self.tickers)
        dt = 1/252
        strike = strike_pct / 100
        ko_barrier = ko_pct / 100
        
        # Correlated Paths (0.5 Correlation)
        corr_mat = np.full((n_assets, n_assets), 0.5)
        np.fill_diagonal(corr_mat, 1.0)
        L = np.linalg.cholesky(corr_mat)
        
        total_coupons = 0
        losses = 0
        
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
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))

                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    sim_coupons += 1
                    knocked_out = True
                    break
                
                # FCN pays fixed; BCN only pays if above strike
                if self.prod_type == "Fixed Coupon Note (FCN)":
                    sim_coupons += 1
                elif worst_path[step-1] >= strike:
                    sim_coupons += 1
            
            total_coupons += sim_coupons
            if not knocked_out and worst_path[-1] < strike:
                losses += 1
                
        avg_c = total_coupons / n_sims
        prob_l = losses / n_sims
        # Yield is scaled by the risk profile (Strike/KO interaction)
        # Note: In real markets, yield is an input, price is output. 
        # Here we simulate theoretical yield based on probability of being in-the-money.
        ann_yield = (avg_c / (self.tenor_yr * (12/max(1, (self.obs_steps[0] if len(self.obs_steps)>0 else 1))))) * (self.rf * 100) * (1 + (strike - 0.7))
        
        return avg_c, prob_l, ann_yield

# --- UI ---
st.set_page_config(page_title="Pricer Pro", layout="wide")
st.title("🛡️ Institutional FCN & BCN Desk")

with st.sidebar:
    prod_choice = st.selectbox("Product", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
    tickers_in = st.text_input("Underlyings", "AAPL, MSFT, NVDA")
    rf_choice = st.selectbox("Rf Rate", ["3M T-Bill", "1Y UST", "SOFR", "3M T-Bill + Spread"])
    spread_bps = st.slider("Spread (bps)", 0, 500, 100) if "Spread" in rf_choice else 0
    tenor = st.slider("Tenor (Months)", 1, 36, 12)
    c_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6, 12])
    nocall = st.selectbox("No-Call Period (Months)", [1, 2, 3, 4, 6])
    p_strike = st.slider("Put Strike (%)", 50, 100, 80)
    ko_init = st.slider("KO Barrier (%)", 80, 110, 100)
    ko_style = st.radio("KO Style", ["Fixed", "Step Down"])
    step_val = st.slider("Monthly Step Down (%)", 0.0, 2.0, 0.5) if ko_style == "Step Down" else 0

if st.button("Run Full Matrix Pricer"):
    ticker_list = [t.strip().upper() for t in tickers_in.split(",")]
    vols, rf_rate = get_market_data(ticker_list, tenor, rf_choice, spread_bps)
    engine = StructuredProductEngine(ticker_list, vols, rf_rate, tenor, c_freq, nocall, ko_style, step_val, prod_choice)
    
    # 1. Base Case
    avg_c, p_loss, a_yield = engine.run_simulation(p_strike, ko_init)
    
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Est. Annualized Yield", f"{a_yield:.2f}%")
    m2.metric("Prob. of Capital Loss", f"{p_loss:.2%}")
    m3.metric("Likely Coupons Paid", f"{avg_c:.2f}")

    # 2. Sensitivity Matrices (Horizontal = Put Strike, Vertical = KO)
    st.subheader("Sensitivity Analysis")
    st.caption("Rows: KO Barrier (%) | Columns: Put Strike (%)")
    
    strikes = [70, 75, 80, 85, 90]
    barriers = [90, 95, 100, 105, 110]
    
    yield_results = np.zeros((len(barriers), len(strikes)))
    loss_results = np.zeros((len(barriers), len(strikes)))

    # Progress bar for the matrix simulation
    progress = st.progress(0)
    total_cells = len(barriers) * len(strikes)
    
    for i, ko in enumerate(barriers):
        for j, strike in enumerate(strikes):
            # Run a smaller simulation for each cell for speed
            _, cell_loss, cell_yield = engine.run_simulation(strike, ko, n_sims=1000)
            yield_results[i, j] = cell_yield
            loss_results[i, j] = cell_loss
            progress.progress((i * len(strikes) + j + 1) / total_cells)

    df_y = pd.DataFrame(yield_results, index=barriers, columns=strikes)
    df_l = pd.DataFrame(loss_results, index=barriers, columns=strikes)

    c_left, c_right = st.columns(2)
    with c_left:
        st.write("**Yield Matrix**")
        st.dataframe(df_y.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
    with c_right:
        st.write("**Capital Loss Matrix**")
        st.dataframe(df_l.style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
