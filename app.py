import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="FCN & BCN Structure Pricer", layout="wide")

# --- CORE MATH FUNCTIONS ---
def get_historical_data(tickers, months):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=months)
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    returns = np.log(data / data.shift(1)).dropna()
    vols = returns.std() * np.sqrt(252)
    corr = returns.corr()
    return vols, corr

def run_monte_carlo(n_sims, n_steps, n_assets, T, r, vols, corr_matrix, skew, strike_pct):
    dt = T / n_steps
    # Adjust vol for skew: Vol_adj = Vol + Skew * (Strike - 100)
    # Note: Strike is usually < 100, so a negative skew increases vol for lower strikes
    adj_vols = vols + (skew * (strike_pct - 100) / 100)
    
    # Cholesky for correlation
    L = np.linalg.cholesky(corr_matrix)
    
    # Simulate paths
    # Result shape: [steps, sims, assets]
    paths = np.zeros((n_steps + 1, n_sims, n_assets))
    paths[0] = 100.0 # Standardized start
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal((n_sims, n_assets))
        correlated_z = z @ L.T
        drift = (r - 0.5 * adj_vols**2) * dt
        diffusion = adj_vols * np.sqrt(dt) * correlated_z
        paths[t] = paths[t-1] * np.exp(drift + diffusion)
        
    return paths

# --- APP INTERFACE ---
st.title("🛡️ Structured Product Pricer: FCN & BCN")
st.markdown("Worst-of Basket Pricing Engine with Monte Carlo Simulation")

with st.sidebar:
    st.header("1. Global Parameters")
    product_type = st.selectbox("Product Type", ["FCN (Fixed Coupon)", "BCN (Bonus Coupon)"])
    tickers = st.text_input("Tickers (comma separated)", "AAPL, MSFT, GOOGL").split(",")
    tickers = [t.strip() for t in tickers]
    
    vol_mode = st.radio("Volatility Mode", ["Implied (Manual)", "Historical"])
    if vol_mode == "Historical":
        lookback = st.slider("Lookback Period (Months)", 1, 60, 12)
        vols_data, corr_matrix = get_historical_data(tickers, lookback)
        st.write("Calculated Annualized Vols:")
        st.write(vols_data)
    else:
        manual_vol = st.number_input("Flat Implied Vol (%)", value=25.0) / 100
        vols_data = np.array([manual_vol] * len(tickers))
        corr_val = st.slider("Fixed Correlation", -1.0, 1.0, 0.5)
        corr_matrix = np.full((len(tickers), len(tickers)), corr_val)
        np.fill_diagonal(corr_matrix, 1.0)

    st.header("2. Note Features")
    tenor = st.number_input("Tenor (Years)", value=1.0)
    strike_pct = st.number_input("Put Strike / KI Barrier (%)", value=80.0)
    ko_barrier = st.number_input("KO Barrier (%)", value=105.0)
    coupon_freq = st.selectbox("Coupon Frequency (Months)", [1, 3, 6], index=0)
    non_call = st.number_input("Non-Call Period (Months)", value=3)
    
    st.header("3. Pricing Logic")
    r = st.number_input("Risk Free Rate (%)", value=4.5) / 100
    skew = st.number_input("Volatility Skew (bps per 1% Strike)", value=0.5) / 100
    n_sims = st.select_slider("Simulations", options=[1000, 5000, 10000], value=5000)

    if product_type == "FCN (Fixed Coupon)":
        coupon_pa = st.number_input("Fixed Coupon (% p.a.)", value=10.0) / 100
        bonus_coupon = 0.0
    else:
        coupon_pa = st.number_input("Fixed Coupon (% p.a.)", value=5.0) / 100
        bonus_coupon = st.number_input("At-Maturity Bonus (%)", value=3.0) / 100

# --- COMPUTATION ---
if st.button("Calculate Note Value"):
    n_steps = int(tenor * 252)
    paths = run_monte_carlo(n_sims, n_steps, len(tickers), tenor, r, vols_data.values if vol_mode == "Historical" else vols_data, corr_matrix, skew, strike_pct)
    
    # Calculate Worst-of paths
    wo_paths = np.min(paths, axis=2) # Shape: [steps, sims]
    
    # Observation logic
    obs_steps = np.arange(0, n_steps + 1, int(252 * (coupon_freq/12)))
    non_call_step = int(252 * (non_call/12))
    
    total_payoffs = np.zeros(n_sims)
    ko_count = 0
    
    for i in range(n_sims):
        path = wo_paths[:, i]
        ko_event = False
        coupons_paid = 0
        
        for step in obs_steps[1:]:
            # Check for KO
            if step >= non_call_step and path[step] >= ko_barrier:
                ko_event = True
                total_payoffs[i] = 100 + (coupons_paid + 1) * (coupon_pa * (coupon_freq/12) * 100)
                ko_count += 1
                break
            # Pay regular coupon
            coupons_paid += 1
            
        if not ko_event:
            # Maturity Payoff
            final_perf = path[-1]
            principal = 100 if final_perf >= strike_pct else final_perf
            total_coupons = coupons_paid * (coupon_pa * (coupon_freq/12) * 100)
            bonus = (bonus_coupon * 100) if product_type == "BCN (Bonus Coupon)" else 0
            total_payoffs[i] = principal + total_coupons + bonus

    fair_value = np.exp(-r * tenor) * np.mean(total_payoffs)
    ann_yield = ((np.mean(total_payoffs) / 100) - 1) / tenor

    # --- OUTPUTS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Fair Value", f"{fair_value:.2f}%")
    col2.metric("Est. Annualized Yield", f"{ann_yield*100:.2f}%")
    col3.metric("KO Probability", f"{(ko_count/n_sims)*100:.1f}%")

    # --- SENSITIVITY TABLE ---
    st.subheader("Sensitivity Analysis (Annualized Yield %)")
    
    strikes = [strike_pct * x for x in [0.9, 0.95, 1.0, 1.05, 1.1]]
    kos = [ko_barrier * x for x in [0.9, 0.95, 1.0, 1.05, 1.1]]
    
    sens_data = np.zeros((len(kos), len(strikes)))
    for r_idx, k in enumerate(kos):
        for c_idx, s in enumerate(strikes):
            # Simplified sensitivity: adjusting yield linearly for display
            sens_data[r_idx, c_idx] = ann_yield * 100 * (s/strike_pct) * (ko_barrier/k)

    sens_df = pd.DataFrame(sens_data, index=[f"KO {x:.0f}%" for x in kos], columns=[f"Strike {x:.0f}%" for x in strikes])
    st.table(sens_df.style.background_gradient(cmap='RdYlGn'))

    # Path Visualization
    st.subheader("Sample Worst-of Path Simulation")
    fig = go.Figure()
    for i in range(min(10, n_sims)):
        fig.add_trace(go.Scatter(y=wo_paths[:, i], mode='lines', opacity=0.4, showlegend=False))
    
    fig.add_hline(y=ko_barrier, line_dash="dash", line_color="green", annotation_text="KO Barrier")
    fig.add_hline(y=strike_pct, line_dash="dash", line_color="red", annotation_text="Put Strike")
    st.plotly_chart(fig, use_container_width=True)
