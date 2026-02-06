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
        raise ValueError("No historical price data. Check tickers/internet.")

    log_returns = np.log(data / data.shift(1)).dropna()
    hist_vols = log_returns.std() * np.sqrt(252)
    corr_matrix_hist = log_returns.corr()

    implied_vols = pd.Series(index=tickers, dtype=float)
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                implied_vols[ticker] = hist_vols[ticker]
                continue

            target_date = end_date + timedelta(days=iv_maturity_days)
            closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
            opt_chain = stock.option_chain(closest_exp)
            calls, puts = opt_chain.calls, opt_chain.puts

            current_price = stock.history(period='1d')['Close'].iloc[-1]
            atm_strike_call = calls.iloc[(calls['strike'] - current_price).abs().argmin()]['strike']
            atm_strike_put = puts.iloc[(puts['strike'] - current_price).abs().argmin()]['strike']

            call_iv = calls[calls['strike'] == atm_strike_call]['impliedVolatility'].values
            put_iv = puts[puts['strike'] == atm_strike_put]['impliedVolatility'].values

            ivs = []
            if len(call_iv) > 0 and not np.isnan(call_iv[0]):
                ivs.append(call_iv[0])
            if len(put_iv) > 0 and not np.isnan(put_iv[0]):
                ivs.append(put_iv[0])

            implied_vols[ticker] = np.mean(ivs) if ivs else hist_vols[ticker]
        except Exception:
            implied_vols[ticker] = hist_vols[ticker]

    dividends = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dividends[ticker] = info.get('trailingAnnualDividendYield', 0.0) or 0.0
        except:
            dividends[ticker] = 0.0

    return hist_vols, implied_vols, corr_matrix_hist, dividends

def price_note(product, tickers, T, freq, non_call, KO, strike, rf, n_sims=10000,
               lookback_months=60, iv_maturity_days=30, use_implied_vol=True,
               skew_factor=1.0, equicorr_override=0.0, bonus_barrier=1.0, bonus_coupon=0.0):
    
    hist_vols, implied_vols, corr_hist, dividends = fetch_stock_data(tickers, lookback_months, iv_maturity_days)

    vols = implied_vols if use_implied_vol else hist_vols
    vols = vols * skew_factor

    num_stocks = len(tickers)
    num_periods = int(T * freq)
    dt = T / num_periods

    vol_vector = vols.values

    if equicorr_override > 0.0001:
        corr_matrix = np.full((num_stocks, num_stocks), equicorr_override)
        np.fill_diagonal(corr_matrix, 1.0)
    else:
        corr_matrix = corr_hist.values

    cov_matrix = np.diag(vol_vector) @ corr_matrix @ np.diag(vol_vector)
    try:
        chol_matrix = cholesky(cov_matrix, lower=True)
    except:
        st.warning("Covariance matrix not positive semi-definite. Using fallback.")
        cov_matrix = np.diag(vol_vector**2)
        chol_matrix = cholesky(cov_matrix, lower=True)

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
        
        coupon_annuity = 0.0
        
        for period in range(1, num_periods + 1):
            worst = np.min(full_paths[period])
            
            # Coupon logic
            coupon = 1.0 / freq  # For FCN, unit coupon for annuity; for BCN, adjust below
            if product == "BCN":
                coupon = fixed_coupon / freq
                if worst >= bonus_barrier:
                    coupon += bonus_coupon / freq
            coupon_annuity += disc_factors[period - 1] * coupon if period <= term_period else 0  # Only add up to term

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
        
        annuity = coupon_annuity  # already limited to term_period
        
        disc_principal = disc_factors[term_period - 1] * redemption
        
        disc_principals[sim] = disc_principal
        annuities[sim] = annuity
        term_periods[sim] = term_period
        
        if sim < viz_sims:
            worst_path = np.minimum.reduce(full_paths, axis=1)
            worst_paths_viz[sim] = worst_path
    
    avg_disc_principal = np.mean(disc_principals)
    avg_annuity = np.mean(annuities)
    
    annualized_yield = (1 - avg_disc_principal) / avg_annuity if avg_annuity > 1e-10 else 0.0
    
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
    tickers_str = st.text_input("Basket tickers", "AAPL,MSFT,NVDA")
    tenor = st.number_input("Tenor years", 0.5, 10.0, 3.0, 0.5)
    freq = st.number_input("Coupons/year", 1, 12, 4)
    non_call_periods = st.number_input("Non-call periods", 0, 20, 4)
    ko_barrier = st.number_input("KO barrier", 0.5, 1.5, 1.00, 0.05)
    put_strike = st.number_input("Put strike", 0.3, 1.0, 0.70, 0.05)
    rf = st.number_input("Risk-free rate", 0.0, 0.10, 0.045, 0.005)
    sims = st.number_input("Simulations", 1000, 100000, 10000, 1000)
    lookback = st.number_input("Lookback years", 1, 10, 5)
    submitted = st.form_submit_button("Calculate")

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Enter tickers.")
    else:
        with st.spinner("Running... (10-90s)"):
            try:
                results = price_note(product, tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf, sims, lookback_months)
                st.success(f"Implied Yield p.a.: **{results['yield_pa']*100:.2f}%**")
                
                st.write(f"Probability of Autocall (early redemption): **{results['prob_autocall']*100:.2f}%**")
                st.write(f"Probability of Survival to Maturity: **{results['prob_survival']*100:.2f}%**")
                st.write(f"Probability of hitting Put Strike at maturity (capital loss): **{results['prob_put_hit']*100:.2f}%**")
                st.write(f"  (i.e. survives to maturity **and** worst-of final level < {put_strike*100:.0f}%)")
                st.write(f"Expected Coupons Paid: **{results['expected_coupons']:.2f}** (out of {int(tenor * freq)})")
                
                st.subheader("Simulated Worst-of Paths")
                st.pyplot(results['fig'])
                plt.close(results['fig'])
            except Exception as e:
                st.error(f"Error: {str(e)}")
st.markdown("---")
st.caption("Worst-of, European barriers at coupon dates, GBM, real Yahoo data.")
