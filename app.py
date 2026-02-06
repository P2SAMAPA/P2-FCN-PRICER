import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import cholesky, eigh
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
    
    hist_vols = log_returns.std() * np.sqrt(252)
    corr_matrix = log_returns.corr().fillna(0).clip(-0.99, 0.99)
    
    # Regularize to PSD
    eigenvalues, eigenvectors = eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    dividends = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dividends[ticker] = info.get('trailingAnnualDividendYield', 0.0) or 0.0
        except:
            dividends[ticker] = 0.0
    
    # Implied vols fallback
    implied_vols = pd.Series(hist_vols, index=tickers)
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                continue

            target_date = end_date + timedelta(days=iv_maturity_days)
            closest_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
            opt_chain = stock.option_chain(closest_exp)
            calls, puts = opt_chain.calls, opt_chain.puts

            current_price = stock.history(period='1d')['Close'].iloc[-1]
            atm_strike = calls.iloc[(calls['strike'] - current_price).abs().argmin()]['strike']

            iv_call = calls[calls['strike'] == atm_strike]['impliedVolatility'].values
            iv_put = puts[puts['strike'] == atm_strike]['impliedVolatility'].values

            ivs = []
            if len(iv_call) > 0 and not np.isnan(iv_call[0]):
                ivs.append(iv_call[0])
            if len(iv_put) > 0 and not np.isnan(iv_put[0]):
                ivs.append(iv_put[0])

            if ivs:
                implied_vols[ticker] = np.mean(ivs)
        except:
            pass

    return hist_vols, implied_vols, corr_matrix, dividends

def price_note(product, tickers, tenor, freq, non_call, KO, strike, rf, n_sims=10000,
               lookback_months=60, iv_maturity_days=30, use_implied_vol=True,
               skew_factor=1.0, equicorr_override=0.0,
               bonus_barrier=1.0, fixed_coupon=0.05, bonus_coupon=0.0):
    
    hist_vols, implied_vols, corr_matrix, dividends = fetch_stock_data(tickers, lookback_months, iv_maturity_days)

    vols = implied_vols if use_implied_vol else hist_vols
    vols = vols * skew_factor

    num_stocks = len(tickers)
    num_periods = int(tenor * freq)
    dt = tenor / num_periods

    vol_vector = vols.values

    if equicorr_override > 0.0001:
        corr_matrix = np.full((num_stocks, num_stocks), equicorr_override)
        np.fill_diagonal(corr_matrix, 1.0)

    cov_matrix = np.diag(vol_vector) @ corr_matrix @ np.diag(vol_vector)

    try:
        chol_matrix = cholesky(cov_matrix, lower=True)
    except:
        st.warning("Cov matrix not PSD – using diagonal fallback")
        cov_matrix = np.diag(vol_vector**2)
        chol_matrix = cholesky(cov_matrix, lower=True)

    drifts = rf - np.array([dividends.get(t, 0.0) for t in tickers]) - 0.5 * vol_vector**2

    times = np.arange(1, num_periods + 1) * dt
    disc_factors = np.exp(-rf * times)

    disc_principals = np.zeros(n_sims)
    annuities = np.zeros(n_sims)
    term_periods = np.zeros(n_sims)
    put_hit_count = 0
    total_redemption_when_hit = 0.0

    viz_sims = min(20, n_sims)
    worst_paths_viz = np.zeros((viz_sims, num_periods + 1))
    worst_paths_viz[:, 0] = 1.0

    autocall_count = 0

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

            coupon = fixed_coupon / freq
            if product == "BCN" and worst >= bonus_barrier:
                coupon += bonus_coupon / freq
            coupon_annuity += disc_factors[period - 1] * coupon

            if not terminated and period > non_call and worst >= KO:
                terminated = True
                term_period = period
                redemption = 1.0
                autocall_count += 1
                break  # stop coupons

        if not terminated:
            worst = np.min(full_paths[-1])
            if worst < strike:
                put_hit_count += 1
                total_redemption_when_hit += worst
            redemption = 1.0 if worst >= strike else worst

        disc_principal = disc_factors[term_period - 1] * redemption

        disc_principals[sim] = disc_principal
        annuities[sim] = coupon_annuity
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

    avg_loss_severity = (1.0 - total_redemption_when_hit / put_hit_count) if put_hit_count > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.linspace(0, tenor, num_periods + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, viz_sims))
    for i in range(viz_sims):
        ax.plot(time_axis, worst_paths_viz[i], color=colors[i], alpha=0.7, linewidth=1.5)
    ax.axhline(y=KO, color='g', linestyle='--', label='KO Barrier')
    ax.axhline(y=strike, color='r', linestyle='--', label='Put Strike')
    if product == "BCN":
        ax.axhline(y=bonus_barrier, color='b', linestyle='--', label='Bonus Barrier')
    obs_times = np.arange(1/freq, tenor + 1/freq, 1/freq)
    for ot in obs_times:
        ax.axvline(x=ot, color='gray', linestyle=':', alpha=0.3)
    ax.set_title(f'Simulated Worst-of Paths (First {viz_sims} Sims)')
    ax.set_xlabel('Years')
    ax.set_ylabel('Performance (Initial = 1.0)')
    ax.set_ylim(0, 2.0)
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

# ────────────────────────────────────────────────
st.title("Structured Note Pricer (FCN & BCN)")
product = st.selectbox("Select Product", ["FCN (Fixed Coupon Note)", "BCN (Bonus Coupon Note)"])

with st.form("inputs"):
    tickers_str = st.text_input("Basket tickers (comma-separated, e.g. SLV,AAPL,MSFT)", "SLV")
    tenor = st.number_input("Tenor in years (e.g. 1)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    freq = st.number_input("Coupon frequency per year (e.g. 4 = quarterly)", min_value=1, max_value=12, value=4)
    non_call_periods = st.number_input("Non-call / lockout periods (e.g. 0 for 1-year)", min_value=0, max_value=20, value=0)
    ko_barrier = st.number_input("Knock-Out barrier (e.g. 1.00 = 100%)", min_value=0.5, max_value=1.5, value=1.00, step=0.05)
    put_strike = st.number_input("Put strike (e.g. 0.60 = 60%)", min_value=0.3, max_value=1.0, value=0.60, step=0.05)
    rf = st.number_input("Risk-free rate (e.g. 0.045 = 4.5%)", min_value=0.0, max_value=0.10, value=0.045, step=0.005)
    sims = st.number_input("Monte Carlo simulations (10000 recommended)", min_value=1000, max_value=100000, value=10000, step=1000)
    lookback_months = st.number_input("Lookback months for vol/corr/dividends", min_value=1, max_value=120, value=60, step=1)
    iv_maturity_days = st.number_input("Implied vol maturity days (ATM IV, e.g. 30)", min_value=7, max_value=365, value=30, step=7)

    use_implied_vol = st.checkbox("Use Implied Volatility (from options chain)", value=True)
    if not use_implied_vol:
        st.info("Implied volatility is turned OFF → using historical volatility from lookback.")
    
    skew_factor = st.slider("Volatility skew adjustment factor (1.0 = neutral)", min_value=0.70, max_value=1.50, value=1.00, step=0.05)
    equicorr_override = st.slider("Equicorrelation override (0 = historical)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    if product == "FCN (Fixed Coupon Note)":
        fixed_coupon = st.number_input("Fixed coupon rate p.a. (e.g. 0.05 = 5%)", min_value=0.0, max_value=0.20, value=0.05, step=0.005)
        bonus_barrier = 1.0
        bonus_coupon = 0.0
    else:
        bonus_barrier = st.number_input("Bonus barrier (e.g. 1.00 = 100%)", min_value=0.5, max_value=1.5, value=1.00, step=0.05)
        fixed_coupon = st.number_input("Fixed coupon rate p.a. (e.g. 0.05 = 5%)", min_value=0.0, max_value=0.20, value=0.05, step=0.005)
        bonus_coupon = st.number_input("Bonus coupon rate p.a. (e.g. 0.02 = 2%)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

    col1, col2 = st.columns(2)
    with col1:
        show_sensitivity = st.checkbox("Show sensitivity table", value=True)
    with col2:
        sensitivity_param = st.radio("Vary parameter", ["KO barrier", "Put strike"], horizontal=True, disabled=not show_sensitivity)
    
    submitted = st.form_submit_button("Calculate Yield", use_container_width=True)

if submitted:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
    else:
        with st.spinner("Fetching data and running simulations..."):
            try:
                if product == "FCN (Fixed Coupon Note)":
                    results = price_note("FCN", tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf,
                                         sims, lookback_months, iv_maturity_days, use_implied_vol, skew_factor, equicorr_override,
                                         bonus_barrier, fixed_coupon, bonus_coupon)
                else:
                    results = price_note("BCN", tickers, tenor, freq, non_call_periods, ko_barrier, put_strike, rf,
                                         sims, lookback_months, iv_maturity_days, use_implied_vol, skew_factor, equicorr_override,
                                         bonus_barrier, fixed_coupon, bonus_coupon)

                st.success(f"Implied Annualized Yield p.a.: **{results['yield_pa']*100:.2f}%**")

                st.write(f"Vol source: **{'Implied' if use_implied_vol else 'Historical'}** (skew × {skew_factor:.2f})")
                st.write(f"Probability of Autocall: **{results['prob_autocall']*100:.2f}%**")
                st.write(f"Probability of Survival to Maturity: **{results['prob_survival']*100:.2f}%**")
                st.write(f"Probability of Capital Loss (Put hit): **{results['prob_put_hit']*100:.2f}%**")
                if results['prob_put_hit'] > 0:
                    st.write(f"Average Loss Severity when Put hit: **-{results['avg_loss_severity']*100:.2f}%**")
                st.write(f"Expected Coupons Paid: **{results['expected_coupons']:.2f}** (out of {int(tenor * freq)})")

                st.subheader("Simulated Worst-of Paths")
                st.pyplot(results['fig'])
                plt.close(results['fig'])

                if show_sensitivity:
                    st.subheader("Sensitivity Analysis")
                    with st.spinner("Calculating sensitivity table..."):
                        if sensitivity_param == "KO barrier":
                            base = ko_barrier
                            step = 0.05
                            levels = [base + (i - 5) * step for i in range(10)]
                            levels = [max(0.50, min(1.50, level)) for level in levels]
                            param_name = "KO Barrier"
                        else:
                            base = put_strike
                            step = 0.05
                            levels = [base + (i - 5) * step for i in range(10)]
                            levels = [max(0.30, min(1.00, level)) for level in levels]
                            param_name = "Put Strike"

                        table_data = []
                        for level in levels:
                            res = price_note(product, tickers, tenor, freq, non_call_periods, level if sensitivity_param == "KO barrier" else ko_barrier,
                                             level if sensitivity_param == "Put strike" else put_strike, rf,
                                             max(3000, sims // 3), lookback_months, iv_maturity_days, use_implied_vol, skew_factor, equicorr_override,
                                             bonus_barrier, fixed_coupon, bonus_coupon)

                            table_data.append({
                                param_name: f"{level:.0%}",
                                "Yield p.a.": f"{res['yield_pa']*100:.2f}%",
                                "Prob Capital Loss": f"{res['prob_put_hit']*100:.2f}%"
                            })

                        df = pd.DataFrame(table_data)

                        def highlight_middle(row):
                            closest = min(levels, key=lambda x: abs(x - base))
                            if row[param_name] == f"{closest:.0%}":
                                return ['background-color: #e6f3ff'] * len(row)
                            return [''] * len(row)

                        st.table(df.style.apply(highlight_middle, axis=1))

            except Exception as e:
                st.error(f"Error: {str(e)}\n\nTry fewer simulations, different tickers, or untick implied vol.")

st.markdown("---")
st.caption("Worst-of Autocallable Structured Notes • European barriers • GBM Monte Carlo • Yahoo Finance data • Implied vol default • Indication only")
