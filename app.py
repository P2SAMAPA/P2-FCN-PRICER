import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    vols, data_list = [], []
    for t in tickers:
        s = yf.Ticker(t)
        hist = s.history(period="12mo")
        px_curr = hist['Close'].iloc[-1]
        try:
            # Safer Option Scraper
            opts = s.options
            chain = s.option_chain(opts[min(len(opts)-1, 4)])
            calls = chain.calls
            iv = calls.iloc[(calls['strike'] - px_curr).abs().argsort()[:1]]['impliedVolatility'].values[0]
        except:
            iv = hist['Close'].pct_change().std() * np.sqrt(252) # Fallback to HV
        vols.append(iv)
        data_list.append(hist['Close'].pct_change())
    
    corr = pd.concat(data_list, axis=1).dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(vols), corr

def run_simulation(sims, tenor, rf, vols, corr, skew):
    steps = int(tenor * 252)
    dt = 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    adj_vols = vols * (1 + skew)
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    paths = np.exp(np.cumsum(drift + adj_vols * np.sqrt(dt) * epsilon, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

def price_fcn(coupon_pa, paths, r, tenor, strike, ko, freq_m):
    steps_total, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    
    # DISCRETE LOGIC: Only check once per month
    obs_interval = max(1, int((freq_m / 12) * 252))
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    coupons_earned = np.zeros(n_sims)
    cpn_val = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        coupons_earned[active] += 1
        ko_mask = active & (worst_of[d] >= ko)
        if np.any(ko_mask):
            payoffs[ko_mask] = 100 + (coupons_earned[ko_mask] * cpn_val)
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        payoffs[active] = principal + (coupons_earned[active] * cpn_val)
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (worst_of[-1] < strike)) / n_sims) * 100,
        "avg_cpns": np.mean(coupons_earned),
        "cpn_dist": coupons_earned
    }

# --- UI ---
with st.sidebar:
    st.header("1. Assets")
    tk_in = st.text_input("Tickers", "SLV, GLD")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    skew = st.slider("Skew Factor", 0.0, 1.0, 0.5)
    st.header("2. Product")
    tenor_y = st.number_input("Tenor (Y)", 0.5, 2.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Frequency (M)", [1, 3, 6], index=0)

if st.button("Solve & Generate Sensitivities"):
    vols, corr = get_market_data(tickers)
    paths = run_simulation(10000, tenor_y, 0.045, vols, corr, skew)
    
    # Solver for 100 Par
    cpn_rng = np.linspace(0.0, 0.60, 30) # Capped at 60% for sanity
    prices = [price_fcn(c, paths, 0.045, tenor_y, strike_p, ko_p, freq_m)['price'] for c in cpn_rng]
    y_solve = np.interp(100.0, prices, cpn_rng)
    res = price_fcn(y_solve, paths, 0.045, tenor_y, strike_p, ko_p, freq_m)

    # 1. Dashboard
    st.subheader(f"Solved Annualized Yield: {y_solve*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    c2.metric("Prob. Capital Loss", f"{res['prob_loss']:.1f}%")
    c3.metric("Avg Coupons Paid", f"{res['avg_cpns']:.2f}")

    # 2. Probability Chart
    st.write("#### Probability of Total Coupons Received")
    dist = pd.Series(res['cpn_dist']).value_counts(normalize=True).sort_index() * 100
    st.plotly_chart(px.bar(x=dist.index, y=dist.values, labels={'x':'Coupons Paid', 'y':'% Probability'}))

    # 3. Sensitivity with LEGENDS
    st.divider()
    strikes = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    barriers = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    y_grid, l_grid = [], []
    for b in barriers:
        yr, lr = [], []
        for s in strikes:
            # Rapid solve for the grid
            y_c = np.interp(100.0, [price_fcn(c, paths, 0.045, tenor_y, s, b, freq_m)['price'] for c in [0, 0.3, 0.6]], [0, 0.3, 0.6])
            yr.append(y_c * 100)
            lr.append(price_fcn(y_c, paths, 0.045, tenor_y, s, b, freq_m)['prob_loss'])
        y_grid.append(yr); l_grid.append(lr)

    st.write("### 📈 Yield Sensitivity Matrix")
    df_y = pd.DataFrame(y_grid, index=[f"KO Barrier: {b}%" for b in barriers], columns=[f"Put Strike: {s}%" for s in strikes])
    st.dataframe(df_y.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    st.write("### ⚠️ Capital Loss Probability Matrix")
    df_l = pd.DataFrame(l_grid, index=[f"KO Barrier: {b}%" for b in barriers], columns=[f"Put Strike: {s}%" for s in strikes])
    st.dataframe(df_l.style.background_gradient(cmap='Reds').format("{:.1f}"))
