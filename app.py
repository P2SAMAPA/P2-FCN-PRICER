import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import cholesky
from datetime import datetime
import matplotlib.pyplot as plt

@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, lookback_months=60):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=lookback_months * 30 + 30)  # approx 30 days/month + buffer
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        repair=True
    )['Close']
    if data.empty:
        raise ValueError("No price data downloaded. Check tickers or connection.")
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

def price_fcn(tickers, T, freq, non_call, KO, strike, rf, n_sims=10000, lookback_months=60):
    vols, corr_matrix, dividends = fetch_stock_data(tickers, lookback_months)
    
    num_stocks = len(tickers)
    num_periods = int(T * freq)
    dt = T / num_periods
    
    vol_vector = vols.values
    cov_matrix = np.diag(vol_vector) @ corr_matrix.values @ np.diag(vol_vector)
    try:
        chol_matrix = cholesky(cov_matrix, lower=True)
    except:
        raise ValueError("Invalid cov matrix – try more lookback months or valid tickers.")
    
    drifts = rf - np.array([dividends.get(t, 0.0) for t in tickers]) - 0.5 * vol_vector**2
    
    times = np.arange(1, num_periods + 1) * dt
    disc_factors = np.exp(-rf * times)
    
    np.random.seed(42)
    disc_principals = np.zeros(n_sims)
    annuities = np.zeros(n_sims)
    term_periods = np.zeros(n_sims)
    
    viz_sims = min(20, n_sims)
    worst_paths_viz = np.zeros((viz_sims, num_periods + 1))
    worst_paths_viz[:, 0] = 1.0
    
    autocall_count = 0
    put_hit_count = 0
    
    for sim in range(n_sims):
        normals = np.random.normal(size=(num_periods, num_stocks))
        increments = drifts * dt + vol_vector * np.sqrt(dt) * (normals @ chol_matrix.T)
        log_paths = np.cumsum(increments, axis=0)
        paths = np.exp(log_paths)
        full_paths = np.vstack([np.ones(num_stocks), paths])
        
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
            if worst < strike:
                put_hit_count += 1
            redemption = 1.0 if worst >= strike else worst
        else:
            autocall_count += 1
        
        annuity = np.sum(disc_factors[:term_period]) / freq
        disc_principal = disc_factors[term_period - 1] * redemption
        
        disc_principals[sim] = disc_principal
        annuities[sim] = annuity
        term_periods[sim] = term_period
        
        if sim < viz_sims:
            worst_path = np.minimum.reduce(full_paths, axis=1)
            worst_paths_viz[sim] = worst_path
    
    avg_disc_principal = np.mean(disc_principals)
    avg_annuity = np.mean(annuities)
    
    annualized_yield = (1 - avg_disc_principal) / avg_annuity if avg_annuity >= 1e-10 else 0.0
    
    prob_autocall = autocall_count / n_sims
    prob_put_hit = put_hit_count / n_sims
    
    expected_coupons = np.mean(term_periods)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.linspace(0, T, num_periods + 1)
    for i in range(viz_sims):
        ax.plot(time_axis, worst_paths_viz[i], alpha=0.5, linewidth=1)
    ax.axhline(y=KO, color='g', linestyle='--', label='KO Barrier')
    ax.axhline(y=strike, color='r', linestyle='--', label='Put Strike')
    obs_times = np.arange(1/freq, T + 1/freq, 1/freq)
    for ot in obs_times:
        ax.axvline(x=ot, color='gray', linestyle=':', alpha=0.3)
    ax.set_title(f'Simulated Worst-of Paths (First {viz_sims} Sims)')
    ax.set_xlabel('Years')
    ax.set_ylabel('Performance (Initial = 1.0)')
    ax.legend()
    ax.grid(True)
    
    results = {
        'yield_pa': annualized_yield,
        'prob_autocall': prob_autocall,
        'prob_survival': 1 - prob_autocall,
        'prob_put_hit': prob_put_hit,
        'expected_coupons': expected_coupons,
        'fig': fig
    }
    return results

# ────────────────────────────────────────────────
st.title("Fixed Coupon Note (Autocallable Worst-of) Pricer")
st.caption("Monte Carlo GBM – indication only, not advice.")

with st.form("inputs"):
    tickers_str = st.text_input(
        "Basket tickers (comma-separated, e.g. AAPL,MSFT,NVDA)",
        value="AAPL,MSFT,NVDA"
    )
    tenor = st.number_input(
        "Tenor in years (e.g. 3)",
        min_value=0.5, max_value=10.0, value=3.0, step=0.5
    )
    freq = st.number_input(
        "Coupon frequency per year (e.g. 4 for quarterly)",
        min_value=1, max_value=12, value=4
    )
    non_call_periods = st.number_input(
        "Non-call / lockout periods (e.g. 4)",
        min_value=0, max_value=20, value=4
    )
    ko_barrier = st.number_input(
        "Knock-Out barrier as decimal (e.g. 1.00 = 100%)",
        min_value=0.5, max_value=1.5, value=1.00, step=0.05
    )
    put_strike = st.number_input(
        "Put strike as decimal (e.g. 0.70 = 70%)",
        min_value=0.3, max_value=1.0, value=0.70, step=0.05
    )
    rf = st.number_input(
        "Continuous risk-free rate (e.g. 0.045 = 4.5%)",
        min_value=0.0, max_value=0.10, value=0.045, step=0.005
    )
    sims = st.number_input(
        "Monte Carlo simulations (e.g. 10000; higher = more accurate but slower)",
        min_value=1000, max_value=100000, value=10000, step=1000
    )
    lookback_months = st.number_input(
        "Lookback months for vol/corr/div data (e.g. 60 for 5 years)",
        min_value=1, max_value=120, value=60, step=1
    )
    submitted = st.form_submit_button("Calculate")

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
    else:
        with st.spinner("Fetching market data and running simulations... (may take 10-90 seconds)"):
            try:
                results = price_fcn(tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf, sims, lookback_months)
                st.success(f"Implied Annualized Yield p.a.: **{results['yield_pa']*100:.2f}%**")
                
                st.write(f"Probability of Autocall (early redemption): **{results['prob_autocall']*100:.2f}%**")
                st.write(f"Probability of Survival to Maturity: **{results['prob_survival']*100:.2f}%**")
                st.write(f"Probability of hitting Put Strike at maturity (capital loss): **{results['prob_put_hit']*100:.2f}%**")
                st.write(f"  → This is the prob. investor suffers principal loss")
                st.write(f"Expected Coupons Paid: **{results['expected_coupons']:.2f}** (out of {int(tenor * freq)})")
                
                st.subheader("Simulated Worst-of Paths")
                st.pyplot(results['fig'])
                plt.close(results['fig'])
            except Exception as e:
                st.error(f"Error: {str(e)}. Check inputs, tickers, or try fewer sims.")

st.markdown("---")
st.caption("Assumptions: Worst-of basket performance; European-style barriers (checked at period ends); GBM paths with constant vol/corr/div; no jumps or stoch vol. Uses real-time Yahoo Finance data.")
