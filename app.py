import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_raw_market_data(tickers):
    # We fetch both so we can toggle instantly in the app
    ivs, hvs, returns = [], [], []
    for t in tickers:
        s = yf.Ticker(t)
        hist = s.history(period="12mo")
        px_curr = hist['Close'].iloc[-1]
        # HV Calculation
        hv = hist['Close'].pct_change().std() * np.sqrt(252)
        hvs.append(hv)
        returns.append(hist['Close'].pct_change())
        # IV Calculation
        try:
            opts = s.options
            chain = s.option_chain(opts[min(len(opts)-1, 4)])
            calls = chain.calls
            iv = calls.iloc[(calls['strike'] - px_curr).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(iv)
        except:
            ivs.append(hv) # Fallback to HV if options are illiquid
    
    corr = pd.concat(returns, axis=1).dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(ivs), np.array(hvs), corr

# --- CALCULATION ENGINE (RE-LINKED) ---
def run_simulation(sims, tenor, rf, base_vols, corr, skew):
    # FORCE NEW CALCULATION: Apply skew to the specific vol vector selected
    # This is the heart of the price movement
    adj_vols = base_vols * (1 + skew) 
    
    steps = int(tenor * 252)
    dt = 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    
    z = np.random.standard_normal((steps, sims, len(base_vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    
    # Standard GBM: Volatility is the primary driver of the diffusion term
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * epsilon
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    
    return np.vstack([np.ones((1, sims, len(base_vols))), paths]) * 100

def price_fcn(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    coupons_earned = np.zeros(n_sims)
    cpn_val = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        coupons_earned[active] += 1
        if d >= nc_steps:
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

# --- UI (UX PRESERVED) ---
st.title("🛡️ Institutional Fixed Coupon Note (FCN) Pricer")

with st.sidebar:
    st.header("1. Market Configuration")
    tk_in = st.text_input("Underlying Tickers", "TSLA, NVDA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_source = st.radio("Volatility Input", ["Market Implied (IV)", "Historical (HV)"])
    skew_val = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.2)
    
    st.header("2. Note Structure")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_m = st.selectbox("Coupon/Observation Frequency (Months)", [1, 3, 6], index=0)
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    strike_p = st.number_input("Put Strike % (Downside Floor)", 40, 100, 60)
    ko_p = st.number_input("KO Barrier % (Auto-Call Level)", 70, 150, 100)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

if st.button("Generate FCN Valuation"):
    # 1. Fetch data
    ivs, hvs, corr = get_raw_market_data(tickers)
    
    # 2. Select base volatility vector
    base_vols = ivs if vol_source == "Market Implied (IV)" else hvs
    
    # 3. Generate Paths (Directly using UI inputs for every run)
    paths = run_simulation(10000, tenor_y, rf_rate, base_vols, corr, skew_val)
    
    # Solver
    cpn_rng = np.linspace(0.0, 1.0, 50)
    prices = [price_fcn(c, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)['price'] for c in cpn_rng]
    y_solve = np.interp(100.0, prices, cpn_rng)
    res = price_fcn(y_solve, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

    # Dashboard
    st.header(f"Solved Annual Yield: {y_solve*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. of Early KO", f"{res['prob_ko']:.1f}%")
    c2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    c3.metric("Avg. Coupons Earned", f"{res['avg_cpns']:.2f}")

    # Coupon Distribution Chart
    dist = pd.Series(res['cpn_dist']).value_counts(normalize=True).sort_index() * 100
    st.plotly_chart(px.bar(x=dist.index, y=dist.values, labels={'x': 'Coupons Received', 'y': 'Prob (%)'}))

    # Sensitivity Matrices
    st.divider()
    strikes = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    barriers = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    y_grid, l_grid = [], []
    for b in barriers:
        yr, lr = [], []
        for s in strikes:
            # We must resolve the yield for every specific cell
            y_c = np.interp(100.0, [price_fcn(c, paths, rf_rate, tenor_y, s, b, freq_m, nc_m)['price'] for c in [0, 0.4, 0.8]], [0, 0.4, 0.8])
            yr.append(y_c * 100)
            lr.append(price_fcn(y_c, paths, rf_rate, tenor_y, s, b, freq_m, nc_m)['prob_loss'])
        y_grid.append(yr); l_grid.append(lr)

    st.write("### 📈 Yield Sensitivity Matrix (% p.a.)")
    df_y = pd.DataFrame(y_grid, index=[f"KO Barrier: {b}%" for b in barriers], columns=[f"Put Strike: {s}%" for s in strikes])
    st.dataframe(df_y.style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

    st.write("### ⚠️ Capital Loss Probability Matrix (%)")
    df_l = pd.DataFrame(l_grid, index=[f"KO Barrier: {b}%" for b in barriers], columns=[f"Put Strike: {s}%" for s in strikes])
    st.dataframe(df_l.style.background_gradient(cmap='Reds').format("{:.1f}"))
