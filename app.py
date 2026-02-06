import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- CORE ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    vols, data_list = [], []
    for t in tickers:
        s = yf.Ticker(t)
        px_curr = s.history(period="1d")['Close'].iloc[-1]
        # Get Implied Vol from ATM options
        chain = s.option_chain(s.options[min(len(s.options)-1, 6)])
        iv = chain.calls.iloc[(chain.calls['strike'] - px_curr).abs().argsort()[:1]]['impliedVolatility'].values[0]
        vols.append(iv)
        data_list.append(s.history(period="12mo")['Close'].pct_change())
    
    corr = pd.concat(data_list, axis=1).dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(vols), corr

def get_simulation(sims, tenor, rf, vols, corr, skew_f):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    adj_vols = vols * (1 + skew_f)
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    paths = np.exp(np.cumsum(drift + adj_vols * np.sqrt(dt) * epsilon, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

def run_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m):
    steps_total, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    
    # Define Discrete Observation Dates (Monthly/Quarterly)
    obs_interval = max(1, int((freq_m / 12) * 252))
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    coupons_paid = np.zeros(n_sims)
    cpn_val = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        # Notes still active get a coupon count
        coupons_paid[active] += 1
        # Auto-call check
        ko_mask = active & (worst_of[d] >= ko)
        if np.any(ko_mask):
            payoffs[ko_mask] = 100 + (coupons_paid[ko_mask] * cpn_val)
            active[ko_mask] = False
            
    # Maturity Handling
    if np.any(active):
        final_px = worst_of[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        payoffs[active] = principal + (coupons_paid[active] * cpn_val)
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (worst_of[-1] < strike)) / n_sims) * 100,
        "avg_cpns": np.mean(coupons_paid),
        "cpn_vector": coupons_paid
    }

# --- UI ---
with st.sidebar:
    st.header("Parameters")
    tickers = [x.strip().upper() for x in st.text_input("Tickers", "SLV").split(",")]
    skew = st.slider("Volatility Skew", 0.0, 1.0, 0.8)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    st.divider()
    tenor = st.number_input("Tenor (Yrs)", 0.5, 3.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Frequency (Months)", [1, 3, 6], index=0)

if st.button("Solve & Generate Sensitivities"):
    vols, corr = get_market_data(tickers)
    paths = get_simulation(10000, tenor, rf, vols, corr, skew)
    
    # Solver
    cpn_rng = np.linspace(0.0, 1.0, 40)
    prices = [run_valuation(c, paths, rf, tenor, strike_p, ko_p, freq_m)['price'] for c in cpn_rng]
    y_solve = np.interp(100.0, prices, cpn_rng)
    res = run_valuation(y_solve, paths, rf, tenor, strike_p, ko_p, freq_m)

    # 1. Dashboard
    st.title(f"Solved Yield: {y_solve*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    c2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    c3.metric("Avg. Coupons Paid", f"{res['avg_cpns']:.2f}")

    # 2. Coupon Distribution
    st.subheader("Probability of Number of Coupons Paid")
    dist = pd.Series(res['cpn_vector']).value_counts(normalize=True).sort_index() * 100
    st.plotly_chart(px.bar(x=dist.index, y=dist.values, labels={'x':'Coupons', 'y':'% Probability'}))

    # 3. Sensitivity Matrices
    st.divider()
    strikes = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    barriers = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    y_grid, l_grid = [], []

    for b in barriers:
        yr, lr = [], []
        for s in strikes:
            y_c = np.interp(100.0, [run_valuation(c, paths, rf, tenor, s, b, freq_m)['price'] for c in cpn_rng], cpn_rng)
            yr.append(y_c * 100)
            lr.append(run_valuation(y_c, paths, rf, tenor, s, b, freq_m)['prob_loss'])
        y_grid.append(yr); l_grid.append(lr)

    # UI Fix: Hard-coded legends in headers
    df_y = pd.DataFrame(y_grid, index=[f"KO: {b}%" for b in barriers], columns=[f"Strike: {s}%" for s in strikes])
    df_l = pd.DataFrame(l_grid, index=[f"KO: {b}%" for b in barriers], columns=[f"Strike: {s}%" for s in strikes])

    st.write("### 📈 Yield Sensitivity (% p.a.)")
    st.caption("Vertical: KO Barrier Level | Horizontal: Put Strike Level")
    st.dataframe(df_y.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    st.write("### ⚠️ Capital Loss Probability (%)")
    st.caption("Vertical: KO Barrier Level | Horizontal: Put Strike Level")
    st.dataframe(df_l.style.background_gradient(cmap='Reds').format("{:.1f}"))
