import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import brentq
from datetime import datetime

st.set_page_config(page_title="Institutional FCN & BCN Solver", layout="wide")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_market_params(tickers, lookback_months):
    vols = []
    for t in tickers:
        stock = yf.Ticker(t)
        try:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            opt_dates = stock.options
            # Use the second or third expiry to get a more stable "Institutional" IV
            chain = stock.option_chain(opt_dates[min(2, len(opt_dates)-1)])
            atm_iv = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(atm_iv)
        except:
            vols.append(0.30) # Default if data fetch fails
    
    end = datetime.now()
    start = end - pd.DateOffset(months=lookback_months)
    hist_data = yf.download(tickers, start=start, end=end)['Close']
    corr_matrix = hist_data.pct_change().corr().values
    return np.array(vols), corr_matrix

# --- PRICING ENGINE ---
def price_structure(coupon_pa, wo_paths, tenor, r, strike_pct, ko_barrier, freq_months, nc_months, is_bcn, bonus=0):
    n_steps, n_sims = wo_paths.shape
    obs_freq = max(1, int(252 * (freq_months/12)))
    nc_step = int(252 * (nc_months/12))
    coupon_per_obs = (coupon_pa * (freq_months/12)) * 100
    
    payoffs = np.zeros(n_sims)
    for i in range(n_sims):
        path = wo_paths[:, i]
        # Knock-Out Logic
        ko_happened = False
        for step in range(nc_step, n_steps, obs_freq):
            if path[step] >= ko_barrier:
                # Paid 100% + coupons earned up to this point
                num_coupons = (step // obs_freq)
                payoffs[i] = 100 + (num_coupons * coupon_per_obs)
                ko_happened = True
                break
        
        if not ko_happened:
            # Maturity Payoff (Worst-of Performance)
            principal = 100 if path[-1] >= strike_pct else path[-1]
            num_coupons = (n_steps - 1) // obs_freq
            b_payoff = (bonus * 100) if is_bcn else 0
            payoffs[i] = principal + (num_coupons * coupon_per_obs) + b_payoff
            
    return np.mean(payoffs) * np.exp(-r * tenor)

# --- APP LAYOUT ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    product_choice = st.radio("Product Type", ["FCN", "BCN"])
    tickers = [x.strip().upper() for x in st.text_input("Tickers (CSV)", "AAPL, MSFT").split(",")]
    vol_mode = st.radio("Volatility Source", ["Market Implied", "Historical"])
    lookback = st.slider("Historical Lookback (Months)", 1, 60, 12)
    skew = st.slider("Skew (bps per 1% Strike)", 0.0, 3.0, 0.8)
    n_sims = st.select_slider("Simulations", [1000, 5000, 10000, 20000], 5000)
    r_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.subheader("Note Parameters")
    tenor = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    strike_val = st.number_input("Put Strike / KI Barrier %", 40, 100, 60)
    ko_val = st.number_input("KO Barrier %", 70, 150, 100)

    if product_choice == "BCN":
        bcn_f = st.number_input("Fixed Coupon % p.a.", 0.0, 30.0, 5.0) / 100
        bcn_b = st.number_input("Bonus at Maturity %", 0.0, 20.0, 2.0) / 100

# --- EXECUTION ---
if st.button("Calculate & Generate Sensitivity"):
    with st.spinner("Fetching data and running Monte Carlo..."):
        vols, corr = get_market_params(tickers, lookback)
        
        # Adjust Vol for Skew at the base Strike
        adj_vols = vols + (skew/100 * (100 - strike_val)/100)
        
        # Generate correlated paths
        dt = 1/252
        steps = int(tenor * 252)
        L = np.linalg.cholesky(corr)
        raw_paths = np.zeros((steps + 1, n_sims, len(tickers)))
        raw_paths[0] = 100.0
        for t in range(1, steps + 1):
            z = np.random.standard_normal((n_sims, len(tickers))) @ L.T
            raw_paths[t] = raw_paths[t-1] * np.exp((r_rate - 0.5 * adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
        
        wo_paths = np.min(raw_paths, axis=2)

        if product_choice == "FCN":
            # 1. Main Solve
            obj = lambda c: price_structure(c, wo_paths, tenor, r_rate, strike_val, ko_val, freq, nc, False) - 100
            try:
                # Brentq search between 0% and 100% coupon
                final_yield = brentq(obj, 0.0, 1.0)
                st.metric("Solved Annualized Yield", f"{final_yield*100:.2f}% p.a.")
            except ValueError:
                st.error("Solver failed. The barriers might be too high/low for a Par structure.")

            # 2. Sensitivity Table
            st.subheader("Yield Sensitivity Table (Strike vs KO)")
            strikes = [strike_val-10, strike_val-5, strike_val, strike_val+5, strike_val+10]
            kos = [ko_val+10, ko_val+5, ko_val, ko_val-5, ko_val-10]
            
            sens_results = []
            for k in kos:
                row = []
                for s in strikes:
                    # Inner solver for each scenario
                    # We reuse wo_paths but adjust the strike/ko barriers passed to the engine
                    obj_s = lambda x: price_structure(x, wo_paths, tenor, r_rate, s, k, freq, nc, False) - 100
                    try:
                        row.append(round(brentq(obj_s, 0.0, 1.0) * 100, 2))
                    except:
                        row.append("N/A")
                sens_results.append(row)
            
            df_sens = pd.DataFrame(sens_results, index=[f"KO {i}%" for i in kos], columns=[f"Strike {j}%" for j in strikes])
            st.table(df_sens.style.background_gradient(cmap='RdYlGn', axis=None))

        else: # BCN Calculation
            price = price_structure(bcn_f, wo_paths, tenor, r_rate, strike_val, ko_val, freq, nc, True, bcn_b)
            st.metric("BCN Fair Value", f"{price:.2f}%")

    # Path Chart
    fig = go.Figure()
    for i in range(min(15, n_sims)):
        fig.add_trace(go.Scatter(y=wo_paths[:, i], mode='lines', opacity=0.4, showlegend=False))
    fig.add_hline(y=ko_val, line_dash="dash", line_color="green", annotation_text="KO Barrier")
    fig.add_hline(y=strike_val, line_dash="dash", line_color="red", annotation_text="Put Strike")
    fig.update_layout(title="Worst-of Sample Paths", xaxis_title="Trading Days", yaxis_title="Level (%)")
    st.plotly_chart(fig, use_container_width=True)
