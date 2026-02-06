import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import brentq
from datetime import datetime

st.set_page_config(page_title="FCN & BCN Structuring Tool", layout="wide")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_market_params(tickers, lookback_months):
    vols = []
    for t in tickers:
        stock = yf.Ticker(t)
        try:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            opt_dates = stock.options
            chain = stock.option_chain(opt_dates[0])
            atm_iv = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(atm_iv)
        except:
            vols.append(0.25)
    
    end = datetime.now()
    start = end - pd.DateOffset(months=lookback_months)
    hist_data = yf.download(tickers, start=start, end=end)['Close']
    corr_matrix = hist_data.pct_change().corr().values
    return np.array(vols), corr_matrix

# --- SIMULATION ENGINE ---
def generate_wo_paths(n_sims, n_assets, tenor, r, vols, corr_matrix, skew, strike_pct):
    dt = 1/252
    steps = int(tenor * 252)
    L = np.linalg.cholesky(corr_matrix)
    
    # Skew Adjustment: Higher vol for lower strikes
    adj_vols = vols + (skew/100 * (100 - strike_pct)/100)
    
    paths = np.zeros((steps + 1, n_sims, n_assets))
    paths[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((n_sims, n_assets)) @ L.T
        drift = (r - 0.5 * adj_vols**2) * dt
        diffusion = adj_vols * np.sqrt(dt) * z
        paths[t] = paths[t-1] * np.exp(drift + diffusion)
    
    return np.min(paths, axis=2)

def price_structure(coupon_pa, wo_paths, tenor, r, strike_pct, ko_barrier, freq_months, nc_months, is_bcn, bonus=0):
    steps, n_sims = wo_paths.shape
    obs_freq = max(1, int(252 * (freq_months/12)))
    nc_step = int(252 * (nc_months/12))
    coupon_per_obs = (coupon_pa * (freq_months/12)) * 100
    
    payoffs = np.zeros(n_sims)
    for i in range(n_sims):
        path = wo_paths[:, i]
        # Check KO
        ko_indices = np.where(path[nc_step:] >= ko_barrier)[0]
        if len(ko_indices) > 0:
            ko_step = ko_indices[0] + nc_step
            num_coupons = (ko_step // obs_freq)
            payoffs[i] = 100 + (num_coupons * coupon_per_obs)
        else:
            # Maturity Payoff
            principal = 100 if path[-1] >= strike_pct else path[-1]
            num_coupons = (steps // obs_freq)
            b_payoff = (bonus * 100) if is_bcn else 0
            payoffs[i] = principal + (num_coupons * coupon_per_obs) + b_payoff
            
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    product_choice = st.radio("Select Product", ["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
    st.divider()
    tickers = [x.strip() for x in st.text_input("Tickers (CSV)", "AAPL, MSFT").split(",")]
    vol_mode = st.radio("Volatility", ["Market Implied", "Historical"])
    lookback = st.slider("Historical Lookback (Months)", 1, 60, 12)
    skew = st.slider("Skew (bps/1% Strike)", 0.0, 2.0, 0.8)
    n_sims = st.select_slider("Sims", [1000, 5000, 10000, 20000], 5000)
    r_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.subheader("Note Parameters")
    tenor = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    freq = st.selectbox("Coupon Freq (Months)", [1, 3, 6])
    nc = st.number_input("Non-Call (Months)", 0, 12, 3)
    strike_base = st.number_input("Put Strike %", 50, 100, 80)
    ko_base = st.number_input("KO Barrier %", 80, 150, 105)

    if product_choice == "BCN (Bonus Coupon Note)":
        bcn_fixed = st.number_input("Fixed Coupon % p.a.", 0.0, 20.0, 5.0) / 100
        bcn_bonus = st.number_input("Bonus at Maturity %", 0.0, 10.0, 2.0) / 100

# --- CALCULATION ---
if st.button("Solve & Generate Table"):
    with st.spinner("Running Monte Carlo..."):
        vols, corr = get_market_params(tickers, lookback)
        
        # Base Simulation for the Charts
        wo_paths = generate_wo_paths(n_sims, len(tickers), tenor, r_rate, vols, corr, skew, strike_base)
        
        if "FCN" in product_choice:
            # Main Output
            obj = lambda x: price_structure(x, wo_paths, tenor, r_rate, strike_base, ko_base, freq, nc, False) - 100
            yield_sol = brentq(obj, 0.0, 0.8)
            st.metric("Annualized Yield", f"{yield_sol*100:.2f}% p.a.")

            # Correct Sensitivity Logic
            st.subheader("Yield Sensitivity Table")
            s_range = [strike_base-10, strike_base-5, strike_base, strike_base+5, strike_base+10]
            k_range = [ko_base+10, ko_base+5, ko_base, ko_base-5, ko_base-10]
            
            results = []
            for k in k_range:
                row = []
                for s in s_range:
                    # Regenerate paths for each strike to account for Skew correctly
                    sens_paths = generate_wo_paths(2000, len(tickers), tenor, r_rate, vols, corr, skew, s)
                    obj_s = lambda x: price_structure(x, sens_paths, tenor, r_rate, s, k, freq, nc, False) - 100
                    try: row.append(round(brentq(obj_s, 0.0, 1.0) * 100, 2))
                    except: row.append("N/A")
                results.append(row)
            
            df_sens = pd.DataFrame(results, index=[f"KO {i}%" for i in k_range], columns=[f"Strike {j}%" for j in s_range])
            st.table(df_sens.style.background_gradient(cmap='RdYlGn'))

        else: # BCN Logic
            price = price_structure(bcn_fixed, wo_paths, tenor, r_rate, strike_base, ko_base, freq, nc, True, bcn_bonus)
            st.metric("Note Price", f"{price:.2f}%")
            st.info("Sensitivity for BCN usually focuses on Price vs Vol/Spot. Would you like a Price sensitivity table?")

        # Visuals
        fig = go.Figure()
        for i in range(20):
            fig.add_trace(go.Scatter(y=wo_paths[:, i], mode='lines', opacity=0.3, showlegend=False))
        fig.add_hline(y=ko_base, line_dash="dash", line_color="green", annotation_text="KO")
        fig.add_hline(y=strike_base, line_dash="dash", line_color="red", annotation_text="Strike")
        st.plotly_chart(fig, use_container_width=True)
