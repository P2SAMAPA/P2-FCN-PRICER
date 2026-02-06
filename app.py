import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# --- ENGINE: DISCRETE OBSERVATION LOGIC ---
def run_valuation_discrete(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims = paths.shape
    # Calculate indices for discrete observations (e.g., every 21 days for monthly)
    obs_interval = int((freq_m / 12) * 252)
    nc_steps = int((nc_m / 12) * 252)
    
    # Define observation dates, ensuring we don't exceed the path length
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    obs_dates = obs_dates[obs_dates >= nc_steps]
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    cpn_earned = np.zeros(n_sims)
    cpn_per_period = (coupon_pa * (freq_m / 12)) * 100
    
    # Check KO only on specific observation dates (Discrete European style)
    for d in obs_dates:
        # KO triggers if Worst-Of is >= barrier on THIS specific day
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = d // obs_interval
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            cpn_earned[ko_mask] = num_cpns
            active[ko_mask] = False
            
    # Final Maturity Payoff
    if np.any(active):
        final_px = paths[-1, active]
        # Principal protection logic at the 60% strike
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps_total - 1) // obs_interval
        payoffs[active] = principal + (num_cpns * cpn_per_period)
        cpn_earned[active] = num_cpns
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": np.mean(cpn_earned)
    }

def solve_for_yield_discrete(paths, r, t, s, k, f, nc):
    coupons = np.linspace(0.0, 0.60, 30)
    prices = [run_valuation_discrete(c, paths, r, t, s, k, f, nc)['price'] for c in coupons]
    # Solve for the coupon where the Note Price equals Par (100)
    return np.interp(100.0, prices, coupons)

# --- UI IMPLEMENTATION ---
st.title("🛡️ Institutional FCN Solver (Discrete Observation)")

with st.sidebar:
    st.header("Parameters")
    tk_in = st.text_input("Tickers", "MSFT, GOOGL")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    freq_m = st.selectbox("Coupon & Observation Frequency", [1, 3, 6], index=1, help="KO Checked Monthly, Quarterly, etc.")
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    strike_p = st.slider("Put Strike %", 40, 100, 60)
    ko_p = st.slider("KO Barrier %", 80, 120, 100)
    skew_f = st.slider("Vol Skew", 0.0, 1.0, 0.2)

if st.button("Calculate Everything"):
    # Fetching Data and Simulating (using previous get_market_data/get_simulation logic)
    # ... [Assuming standard GBM simulation code is used here] ...
    
    # 1. PRIMARY SOLVE
    # yield_res = solve_for_yield_discrete(...)
    
    # 2. SENSITIVITY TABLE WITH LABELS
    ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    # Generate Matrices using the Discrete Solver
    # ... [Looping logic same as before, using solve_for_yield_discrete] ...
    
    st.write("### Yield Sensitivity Table (% p.a.)")
    # df_yield.index.name = "KO Barrier (%)"
    # df_yield.columns.name = "Put Strike (%)"
    # st.dataframe(df_yield)
