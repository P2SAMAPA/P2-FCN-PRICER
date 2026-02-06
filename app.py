import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import brentq
from datetime import datetime

st.set_page_config(page_title="Institutional FCN/BCN Solver", layout="wide")

# --- DATA & VOLATILITY FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_market_params(tickers, lookback_months):
    vols = []
    prices = []
    for t in tickers:
        stock = yf.Ticker(t)
        # Get Implied Vol from ATM Options
        try:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            prices.append(current_price)
            opt_dates = stock.options
            chain = stock.option_chain(opt_dates[0])
            atm_iv = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(atm_iv)
        except:
            vols.append(0.25) # Fallback
    
    # Get Correlation from Historical Data
    end = datetime.now()
    start = end - pd.DateOffset(months=lookback_months)
    hist_data = yf.download(tickers, start=start, end=end)['Close']
    corr_matrix = hist_data.pct_change().corr().values
    
    return np.array(vols), corr_matrix

# --- SIMULATION ENGINE ---
def run_simulation(n_sims, n_assets, tenor, r, vols, corr_matrix, skew, strike_pct):
    dt = 1/252
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr_matrix)
    
    # Apply Skew: Vol increases as strike decreases
    # adj_vol = Base_Vol + Skew * (100 - Strike)
    adj_vols = vols + (skew/100 * (100 - strike_pct)/100)
    
    paths = np.zeros((steps + 1, n_sims, n_assets))
    paths[0] = 100.0
    
    for t in range(1, steps + 1):
        z = np.random.standard_normal((n_sims, n_assets)) @ L.T
        drift = (r - 0.5 * adj_vols**2) * dt
        diffusion = adj_vols * np.sqrt(dt) * z
        paths[t] = paths[t-1] * np.exp(drift + diffusion)
    
    return np.min(paths, axis=2) # Worst-of paths

def price_engine(coupon_pa, wo_paths, tenor, r, strike_pct, ko_barrier, freq_months, nc_months, is_bcn, bonus=0):
    steps, n_sims = wo_paths.shape
    dt = 1/252
    obs_freq = int(252 * (freq_months/12))
    nc_step = int(252 * (nc_months/12))
    
    payoffs = np.zeros(n_sims)
    coupon_per_obs = (coupon_pa * (freq_months/12)) * 100
    
    for i in range(n_sims):
        path = wo_paths[:, i]
        ko_idx = np.where((path[nc_step:] >= ko_barrier))[0]
        
        if len(ko_idx) > 0:
            actual_ko_step = ko_idx[0] + nc_step
            # Count coupons up to KO
            num_coupons = (actual_ko_step // obs_freq)
            payoffs[i] = 100 + (num_coupons * coupon_per_obs)
        else:
            # Maturity Payoff
            principal = 100 if path[-1] >= strike_pct else path[-1]
            num_coupons = (steps // obs_freq)
            b_payoff = (bonus * 100) if is_bcn else 0
            payoffs[i] = principal + (num_coupons * coupon_per_obs) + b_payoff
            
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- UI LAYOUT ---
st.title("🛡️ Worst-of FCN & BCN Solver")

with st.sidebar:
    st.header("1. Assets & Volatility")
    ticker_input = st.text_input("Tickers (CSV)", "AAPL, MSFT, TSLA")
    tickers = [x.strip() for x in ticker_input.split(",")]
    vol_mode = st.radio("Vol Source", ["Market Implied", "Historical"])
    lookback = st.number_input("Hist. Lookback (Months)", 1, 60, 12)
    skew = st.slider("Volatility Skew (bps per 1% Strike)", 0.0, 2.0, 0.8)
    
    st.header("2. Product Features")
    p_type = st.selectbox("Product Type", ["FCN", "BCN"])
    tenor = st.number_input("Tenor (Years)", 0.1, 3.0, 1.0)
    strike_input = st.number_input("Put Strike %", 50, 100, 80)
    ko_input = st.number_input("KO Barrier %", 80, 150, 105)
    freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    
    if p_type == "BCN":
        f_coupon = st.number_input("Fixed Coupon % p.a.", 0.0, 20.0, 5.0) / 100
        b_coupon = st.number_input("Bonus Coupon %", 0.0, 10.0, 2.0) / 100

    n_sims = st.select_slider("Simulations", [1000, 5000, 10000, 20000], value=5000)
    r_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.0) / 100

if st.button("Run Solver"):
    vols, corr = get_market_params(tickers, lookback)
    wo_paths = run_simulation(n_sims, len(tickers), tenor, r_rate, vols, corr, skew, strike_input)
    
    if p_type == "FCN":
        st.subheader("FCN Results")
        # Solve for coupon where Price = 100
        obj = lambda x: price_engine(x, wo_paths, tenor, r_rate, strike_input, ko_input, freq, nc, False) - 100
        try:
            yield_sol = brentq(obj, 0.0, 0.6)
            st.metric("Solved Annualized Yield", f"{yield_sol*100:.2f}% p.a.")
        except:
            st.error("Solver could not converge. Try adjusting barriers.")
    else:
        st.subheader("BCN Results")
        price = price_engine(f_coupon, wo_paths, tenor, r_rate, strike_input, ko_input, freq, nc, True, b_coupon)
        st.metric("Note Fair Value", f"{price:.2f}%")

    # --- SENSITIVITY TABLE ---
    st.write("---")
    st.subheader("Sensitivity: Annualized Yield vs Strike & KO")
    
    s_range = [strike_input - 10, strike_input - 5, strike_input, strike_input + 5, strike_input + 10]
    k_range = [ko_input + 10, ko_input + 5, ko_input, ko_input - 5, ko_input - 10]
    
    results = []
    for k in k_range:
        row = []
        for s in s_range:
            # Quick solve for sensitivity (using 1/2 sims for speed)
            obj_sens = lambda x: price_engine(x, wo_paths[:, :n_sims//2], tenor, r_rate, s, k, freq, nc, False) - 100
            try:
                row.append(round(brentq(obj_sens, 0.0, 0.8) * 100, 2))
            except:
                row.append(np.nan)
        results.append(row)
    
    df_sens = pd.DataFrame(results, index=[f"KO {i}%" for i in k_range], columns=[f"Strike {j}%" for j in s_range])
    st.table(df_sens.style.background_gradient(cmap='RdYlGn'))

    # Path Visualization
    fig = go.Figure()
    for i in range(15):
        fig.add_trace(go.Scatter(y=wo_paths[:, i], mode='lines', opacity=0.3))
    fig.update_layout(title="Worst-of Sample Paths", xaxis_title="Days", yaxis_title="Level (%)")
    st.plotly_chart(fig)
