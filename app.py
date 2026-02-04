import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import cholesky
from datetime import datetime

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(tickers, lookback_years=5):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=lookback_years * 365 + 60)  # buffer
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    if data.empty:
        raise ValueError("No data downloaded – check tickers or internet.")
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

def price_fcn(tickers, T, freq, non_call, KO, strike, rf, n_sims=10000, lookback_years=5):
    vols, corr_matrix, dividends = fetch_stock_data(tickers, lookback_years)
    
    num_stocks = len(tickers)
    num_periods = int(T * freq)
    dt = T / num_periods
    
    vol_vector = vols.values
    cov_matrix = np.diag(vol_vector) @ corr_matrix.values @ np.diag(vol_vector)
    try:
        chol_matrix = cholesky(cov_matrix, lower=True)
    except:
        raise ValueError("Invalid correlation/vol matrix – try more lookback years or valid tickers.")
    
    drifts = rf - np.array([dividends.get(t, 0.0) for t in tickers]) - 0.5 * vol_vector**2
    
    times = np.arange(1, num_periods + 1) * dt
    disc_factors = np.exp(-rf * times)
    
    np.random.seed(42)
    disc_principals = np.zeros(n_sims)
    annuities = np.zeros(n_sims)
    
    for sim in range(n_sims):
        normals = np.random.normal(size=(num_periods, num_stocks))
        increments = drifts * dt + vol_vector * np.sqrt(dt) * (normals @ chol_matrix.T)
        log_paths = np.cumsum(increments, axis=0)
        paths = np.exp(log_paths)
        full_paths = np.vstack([np.ones(num_stocks), paths])  # t=0 = 1.0
        
        terminated = False
        term_period = num_periods
        redemption = 1.0
        
        for period in range(1, num_periods + 1):
            worst = np.min(full_paths[period])
            if period > non_call and worst >= KO:
                terminated = True
                term_period = period
                redemption = 1.0
                break
        
        if not terminated:
            worst = np.min(full_paths[-1])
            redemption = 1.0 if worst >= strike else worst
        
        annuity = np.sum(disc_factors[:term_period]) / freq
        disc_principal = disc_factors[term_period - 1] * redemption
        
        disc_principals[sim] = disc_principal
        annuities[sim] = annuity
    
    avg_disc_principal = np.mean(disc_principals)
    avg_annuity = np.mean(annuities)
    
    if avg_annuity < 1e-10:
        return 0.0
    
    annualized_yield = (1 - avg_disc_principal) / avg_annuity
    return annualized_yield

# ────────────────────────────────────────────────
st.title("Fixed Coupon Note (Autocallable Worst-of) Pricer")
st.caption("Simplified Monte Carlo GBM model – for indication only, not financial advice.")

with st.form("inputs"):
    tickers_str = st.text_input("Basket tickers (comma-separated, e.g., AAPL,MSFT,NVDA)", "AAPL,MSFT,NVDA")
    tenor = st.number_input("Tenor in years", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    freq = st.number_input("Coupon frequency per year (e.g., 4 for quarterly)", min_value=1, max_value=12, value=4)
    non_call_periods = st.number_input("Non-call / lockout periods", min_value=0, max_value=20, value=4)
    ko_barrier = st.number_input("Knock-Out barrier (decimal, e.g., 1.00 = 100%)", min_value=0.5, max_value=1.5, value=1.00, step=0.05)
    put_strike = st.number_input("Put strike / protection (decimal, e.g., 0.70 = 70%)", min_value=0.3, max_value=1.0, value=0.70, step=0.05)
    rf = st.number_input("Continuous risk-free rate (decimal, e.g., 0.045 = 4.5%)", min_value=0.0, max_value=0.10, value=0.045, step=0.005)
    sims = st.number_input("Monte Carlo simulations (higher = more accurate, slower)", min_value=1000, max_value=100000, value=10000, step=1000)
    lookback = st.number_input("Lookback years for vol/corr/dividends", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Calculate Yield")

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
    else:
        with st.spinner("Fetching market data and running simulations... (may take 10-90 seconds)"):
            try:
                result = price_fcn(tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf, sims, lookback)
                st.success(f"Implied Annualized Yield p.a.: **{result*100:.2f}%**")
            except Exception as e:
                st.error(f"Error: {str(e)}. Check inputs, tickers, or try fewer sims.")

st.markdown("---")
st.caption("Assumptions: Worst-of basket performance; European-style barriers (checked at period ends); GBM paths with constant vol/corr/div; no jumps or stoch vol. Uses real-time Yahoo Finance data.")
