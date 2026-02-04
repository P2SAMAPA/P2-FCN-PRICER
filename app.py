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
    start_date = end_date - pd.Timedelta(days=lookback_months * 30 + 30)
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
        raise ValueError("Invalid correlation/vol matrix – try more lookback months or different tickers.")
    
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
    total_redemption_when_hit = 0.0  # For average loss severity
    
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
                total_redemption_when_hit += worst
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
    
    avg_loss_severity = (1.0 - total_redemption_when_hit / put_hit_count) if put_hit_count > 0 else 0.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.linspace(0, T, num_periods + 1)
    for i in range(viz_sims):
        ax.plot(time_axis, worst_paths_viz[i], alpha=0.5, linewidth=1)
    ax.axhline(y=KO, color='g', linestyle='--', label='KO Barrier')
    ax.axhline(y=strike, color='r', linestyle='--', label='Put Strike')
    obs_times = np.arange(1/freq, T + 1/freq, 1/freq)
    for ot in obs_times:
        ax.axvline(x=ot, color='gray', linestyle=':', alpha=0.3)
    ax.set_title(f'Simulated Worst-of Paths (First {viz_sims} Simulations)')
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
        'avg_loss_severity': avg_loss_severity,
        'fig': fig
    }
    return results

# ————————————————————————————————————————
st.title("Fixed Coupon Note (Autocallable Worst-of) Pricer")
st.caption("Monte Carlo GBM – indication only")

with st.form("inputs"):
    tickers_str = st.text_input("Basket tickers (comma-separated, e.g. AAPL,MSFT,NVDA)", "AAPL,MSFT,NVDA")
    tenor = st.number_input("Tenor in years (e.g. 3)", 0.5, 10.0, 3.0, 0.5)
    freq = st.number_input("Coupon frequency per year (e.g. 4 = quarterly)", 1, 12, 4)
    non_call_periods = st.number_input("Non-call / lockout periods (e.g. 4)", 0, 20, 4)
    ko_barrier = st.number_input("Knock-Out barrier (e.g. 1.00 = 100%)", 0.5, 1.5, 1.00, 0.05)
    put_strike = st.number_input("Put strike (e.g. 0.70 = 70%)", 0.3, 1.0, 0.70, 0.05)
    rf = st.number_input("Risk-free rate (e.g. 0.045 = 4.5%)", 0.0, 0.10, 0.045, 0.005)
    sims = st.number_input("Monte Carlo simulations (10000 recommended)", 1000, 100000, 10000, 1000)
    lookback_months = st.number_input("Lookback months for vol/corr/div (e.g. 60 = 5 years)", 1, 120, 60, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        show_sensitivity = st.checkbox("Show sensitivity table", value=True)
    with col2:
        sensitivity_param = st.radio("Vary parameter", ["KO barrier", "Put strike"], horizontal=True, disabled=not show_sensitivity)
    
    submitted = st.form_submit_button("Calculate", use_container_width=True)

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                main_results = price_fcn(tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf, sims, lookback_months)
                
                st.success(f"Implied Annualized Yield p.a.: **{main_results['yield_pa']*100:.2f}%**")
                st.write(f"• Probability of Autocall: **{main_results['prob_autocall']*100:.2f}%**")
                st.write(f"• Probability of Survival to Maturity: **{main_results['prob_survival']*100:.2f}%**")
                st.write(f"• Probability of Capital Loss (Put hit): **{main_results['prob_put_hit']*100:.2f}%**")
                if main_results['prob_put_hit'] > 0:
                    st.write(f"• Average Loss Severity when Put hit: **-{main_results['avg_loss_severity']*100:.2f}%** of principal")
                st.write(f"• Expected Coupons Paid: **{main_results['expected_coupons']:.2f}** out of {int(tenor * freq)}")

                st.subheader("Simulated Worst-of Paths (first 20 paths)")
                st.pyplot(main_results['fig'])
                plt.close(main_results['fig'])

                if show_sensitivity:
                    st.subheader("Sensitivity Analysis")
                    with st.spinner("Calculating sensitivity table..."):
                        if sensitivity_param == "KO barrier":
                            levels = [0.90, 0.95, 1.00, 1.05, 1.10]
                            param_name = "KO Level"
                        else:
                            levels = [0.60, 0.65, 0.70, 0.75, 0.80]
                            param_name = "Put Strike"

                        table_data = []
                        for level in levels:
                            if sensitivity_param == "KO barrier":
                                res = price_fcn(tickers, tenor, freq, non_call_periods, level, put_strike, rf, max(5000, sims//2), lookback_months)
                            else:
                                res = price_fcn(tickers, tenor, freq, non_call_periods, ko_barrier, level, rf, max(5000, sims//2), lookback_months)
                            table_data.append({
                                param_name: f"{level:.0%}",
                                "Yield p.a.": f"{res['yield_pa']*100:.2f}%",
                                "Prob Capital Loss": f"{res['prob_put_hit']*100:.2f}%"
                            })
                        df = pd.DataFrame(table_data)
                        st.table(df)

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Worst-of Autocallable Fixed Coupon Note • European barriers • Real-time Yahoo Finance data • GBM model")
