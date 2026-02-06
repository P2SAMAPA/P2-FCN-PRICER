import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

# --- SETTINGS ---
st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- ENGINE ---
def run_valuation_discrete(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    
    # Calculate exactly when payments happen
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    obs_dates = obs_dates[obs_dates >= nc_steps]
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    coupons_paid = np.zeros(n_sims)
    cpn_value = (coupon_pa * (freq_m / 12)) * 100
    
    # Track the state at each observation date
    for d in obs_dates:
        # 1. Accrue coupon for all active notes
        coupons_paid[active] += 1
        
        # 2. Check for Auto-Call (KO)
        ko_mask = active & (worst_of[d] >= ko)
        if np.any(ko_mask):
            payoffs[ko_mask] = 100 + (coupons_paid[ko_mask] * cpn_value)
            active[ko_mask] = False
            
    # 3. Handle Maturity for remaining notes
    if np.any(active):
        final_px = worst_of[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        payoffs[active] = principal + (coupons_paid[active] * cpn_value)
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (worst_of[-1] < strike)) / n_sims) * 100,
        "avg_cpns": np.mean(coupons_paid),
        "cpn_vector": coupons_paid
    }

# --- UI LAYOUT ---
with st.sidebar:
    st.header("1. Market Inputs")
    tk_in = st.text_input("Tickers (CSV)", "SLV")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_mode = st.radio("Volatility Source", ["Implied Vol (IV)", "Historical Vol (HV)"])
    skew_f = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.8)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.header("2. Product Specs")
    tenor_y = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Coupon/KO Frequency (Months)", [1, 3, 6], index=0)
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)

# --- MAIN EXECUTION ---
if st.button("Solve & Generate Sensitivities"):
    # (Simulation code omitted for brevity - using standard GBM logic)
    # paths = get_simulation(...) 
    
    # Solve for Yield
    coupons = np.linspace(0.0, 1.0, 40)
    prices = [run_valuation_discrete(c, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)['price'] for c in coupons]
    y_main = np.interp(100.0, prices, coupons)
    res = run_valuation_discrete(y_main, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

    # 1. METRICS UX
    st.subheader(f"Solved Yield: {y_main*100:.2f}% p.a.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of KO", f"{res['prob_ko']:.1f}%")
    m2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    # This will now show a realistic number (e.g., 8.4 out of 12)
    m3.metric("Avg. Coupons Paid", f"{res['avg_cpns']:.2f}")

    # 2. FREQUENCY CHART
    st.write("### Coupon Distribution")
    cpn_counts = pd.Series(res['cpn_vector']).value_counts(normalize=True).sort_index() * 100
    fig = px.bar(x=cpn_counts.index, y=cpn_counts.values, 
                 labels={'x': 'Number of Coupons Received', 'y': 'Probability (%)'})
    st.plotly_chart(fig, use_container_width=True)

    # 3. SENSITIVITY TABLES WITH BOLD LEGENDS
    st.divider()
    strikes = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    barriers = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    y_data, loss_data = [], []
    for b in barriers:
        y_row, loss_row = [], []
        for s in strikes:
            # Solve and calculate for each cell
            y_cell = np.interp(100.0, [run_valuation_discrete(c, paths, rf_rate, tenor_y, s, b, freq_m, nc_m)['price'] for c in coupons], coupons)
            l_cell = run_valuation_discrete(y_cell, paths, rf_rate, tenor_y, s, b, freq_m, nc_m)['prob_loss']
            y_row.append(y_cell * 100); loss_row.append(l_cell)
        y_data.append(y_row); loss_data.append(loss_row)

    # UI: Explicitly naming index and columns for the UX
    df_y = pd.DataFrame(y_data, index=[f"KO: {b}%" for b in barriers], columns=[f"Strike: {s}%" for s in strikes])
    df_l = pd.DataFrame(loss_data, index=[f"KO: {b}%" for b in barriers], columns=[f"Strike: {s}%" for s in strikes])

    st.write("### 📈 Yield Sensitivity (% p.a.)")
    st.caption("Rows: Auto-Call Barrier Level | Columns: Downside Put Strike Level")
    st.dataframe(df_y.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    st.write("### ⚠️ Capital Loss Risk (%)")
    st.caption("Rows: Auto-Call Barrier Level | Columns: Downside Put Strike Level")
    st.dataframe(df_l.style.background_gradient(cmap='Reds').format("{:.1f}"))
