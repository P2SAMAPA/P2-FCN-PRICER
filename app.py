import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import cholesky
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

@st.cache_data(ttl=1800)
def fetch_stock_data(tickers, lookback_months=60, iv_maturity_days=30):
    end_date = datetime.now()
    start_date_hist = end_date - pd.Timedelta(days=lookback_months * 30 + 60)
    data = yf.download(tickers, start=start_date_hist, end=end_date, progress=False, repair=True)['Close']
    if data.empty:
        raise ValueError("No historical price data. Check tickers or connection.")
    log_returns = np.log(data / data.shift(1)).dropna()
    
    vols = log_returns.std() * np.sqrt(252)
    corr_matrix = log_returns.corr()
    
    dividends = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dividends[ticker] = info.get('trailingAnnualDividendYield', 0.0) or 0.0
        except:
            dividends[ticker] = 0.0
    
    return vols, corr_matrix, dividends

def price_note(product, tickers, T, freq, non_call, KO, bonus_barrier, fixed_coupon, bonus_coupon, strike, rf, n_sims=10000, lookback_months=60):
    vols, corr_matrix, dividends = fetch_stock_data(tickers, lookback_months)
    
    num_stocks = len(tickers)
    num_periods = int(T * freq)
    dt = T / num_periods
    
    vol_vector = vols.values
    cov_matrix = np.diag(vol_vector) @ corr_matrix.values @ np.diag(vol_vector)
    chol_matrix = cholesky(cov_matrix, lower=True)
    
    drifts = rf - np.array([dividends[t] for t in tickers]) - 0.5 * vol_vector**2
    
    times = np.arange(1, num_periods + 1) * dt
    disc_factors = np.exp(-rf * times)
    
    np.random.seed(42)
    disc_principals = np.zeros(n_sims)
    annuities = np.zeros(n_sims)
    
    for sim in range(n_sims):
        normals = np.random.normal(size=(num_periods, num_stocks))
        correlated_normals = normals @ chol_matrix.T
        
        increments = drifts * dt + vol_vector * np.sqrt(dt) * correlated_normals
        
        log_paths = np.cumsum(increments, axis=0)
        paths = np.exp(log_paths)
        
        full_paths = np.vstack([np.ones(num_stocks), paths])
        
        terminated = False
        term_period = num_periods
        redemption = 1.0
        
        coupon_annuity = 0.0
        
        for period in range(1, num_periods + 1):
            worst = np.min(full_paths[period])
            
            coupon = fixed_coupon / freq
            if product == "BCN" and worst >= bonus_barrier:
                coupon += bonus_coupon / freq
            coupon_annuity += disc_factors[period - 1] * coupon
            
            if not terminated and period > non_call and worst >= KO:
                terminated = True
                term_period = period
                redemption = 1.0
                break
        
        if not terminated:
            worst = np.min(full_paths[-1])
            redemption = 1.0 if worst >= strike else worst
        
        disc_principal = disc_factors[term_period - 1] * redemption
        
        disc_principals[sim] = disc_principal
        annuities[sim] = coupon_annuity
    
    avg_disc_principal = np.mean(disc_principals)
    avg_annuity = np.mean(annuities)
    
    if avg_annuity == 0:
        return 0.0
    
    annualized_yield = (1 - avg_disc_principal) / avg_annuity
    
    return annualized_yield

# Main app
st.title("Structured Note Pricer")
product = st.selectbox("Select Product", ["FCN (Fixed Coupon Note)", "BCN (Bonus Coupon Note)"])

with st.form("inputs"):
    tickers_str = st.text_input("Basket tickers (comma-separated, e.g. AAPL,MSFT,NVDA)", "AAPL,MSFT,NVDA")
    T = st.number_input("Tenor in years (e.g. 3)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    freq = st.number_input("Coupon frequency per year (e.g. 4 = quarterly)", min_value=1, max_value=12, value=4)
    non_call = st.number_input("Non-call periods (e.g. 4)", min_value=0, max_value=20, value=4)
    KO = st.number_input("Knock-Out barrier (e.g. 1.00 = 100%)", min_value=0.5, max_value=1.5, value=1.00, step=0.05)
    put_strike = st.number_input("Put strike (e.g. 0.70 = 70%)", min_value=0.3, max_value=1.0, value=0.70, step=0.05)
    rf = st.number_input("Risk-free rate (e.g. 0.045 = 4.5%)", min_value=0.0, max_value=0.10, value=0.045, step=0.005)
    sims = st.number_input("Monte Carlo simulations (10000 recommended)", min_value=1000, max_value=100000, value=10000, step=1000)
    lookback_months = st.number_input("Lookback months for vol/corr/dividends", min_value=1, max_value=120, value=60, step=1)
    
    if product == "BCN (Bonus Coupon Note)":
        bonus_barrier = st.number_input("Bonus barrier (e.g. 1.0 = 100%)", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
        fixed_coupon = st.number_input("Fixed coupon rate p.a. (e.g. 0.05 = 5%)", min_value=0.0, max_value=0.20, value=0.05, step=0.005)
        bonus_coupon = st.number_input("Bonus coupon rate p.a. (e.g. 0.02 = 2%)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)
    else:
        fixed_coupon = st.number_input("Fixed coupon rate p.a. (e.g. 0.05 = 5%)", min_value=0.0, max_value=0.20, value=0.05, step=0.005)
        bonus_barrier = 1.0
        bonus_coupon = 0.0

    col1, col2 = st.columns(2)
    with col1:
        show_sensitivity = st.checkbox("Show sensitivity table", value=True)
    with col2:
        sensitivity_param = st.radio("Vary parameter", ["KO barrier", "Put strike"], horizontal=True, disabled=not show_sensitivity)
    
    submitted = st.form_submit_button("Calculate")

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
    else:
        with st.spinner("Running..."):
            try:
                results = price_note(product, tickers, tenor, freq, non_call_periods, ko_barrier, bonus_barrier, fixed_coupon, bonus_coupon, put_strike, rf, sims, lookback_months)

                st.success(f"Implied Annualized Yield p.a.: **{results*100:.2f}%**")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Worst-of Autocallable Structured Notes • European barriers • GBM Monte Carlo • Yahoo Finance data • Indication only")
