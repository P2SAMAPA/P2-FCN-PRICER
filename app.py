import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    ivs, hvs = [], []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            chain = s.option_chain(s.options[min(len(s.options)-1, 6)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            ivs.append(iv)
        except: ivs.append(0.35)
        h = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        hvs.append(h)
    
    data = yf.download(tickers, period="12mo", progress=False)['Close']
    corr = data.pct_change().dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    return np.array(ivs), np.array(hvs), corr

# --- ENGINE ---
def get_simulation(sims, tenor, rf, vols, corr, skew_f):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    # Applying Skew: Higher vol for downside protection
    adj_vols = vols * (1 + skew_f)
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    path_returns = np.exp(np.cumsum(drift + adj_vols * np.sqrt(dt) * epsilon, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), path_returns]) * 100

def run_valuation_discrete(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims, _ = paths.shape
    worst_of_paths = np.min(paths, axis=2)
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
    
    # European Observation: Only check on specific monthly/quarterly dates
    obs_dates = np.arange(obs_interval, steps_total, obs_interval)
    obs_dates = obs_dates[obs_dates >= nc_steps]
    
    payoffs, active = np.zeros(n_sims), np.ones(n_sims, dtype=bool)
    cpn_per_period = (coupon_pa * (freq_m / 12)) * 100
    
    for d in obs_dates:
        ko_mask = active & (worst_of_paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = d // obs_interval
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of_paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps_total - 1) // obs_interval
        payoffs[active] = principal + (num_cpns * cpn_per_period)
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (worst_of_paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": (np.sum(~active * (np.arange(n_sims)//obs_interval)) + np.sum(active * (steps_total//obs_interval))) / n_sims # simplified estimate
    }

def solve_yield(paths, r, t, s, k, f, nc):
    coupons = np.linspace(0.0, 0.80, 40)
    prices = [run_valuation_discrete(c, paths, r, t, s, k, f, nc)['price'] for c in coupons]
    return np.interp(100.0, prices, coupons)

# --- UI ---
st.title("🛡️ Institutional FCN Solver")

with st.sidebar:
    st.header("1. Market Inputs")
    tickers = [x.strip().upper() for x in st.text_input("Tickers (CSV)", "TSLA, MSFT").split(",")]
    vol_mode = st.radio("Volatility Source", ["Implied Vol (IV)", "Historical Vol (HV)"])
    skew_val = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.8) # Default 0.8 as requested
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    st.header("2. Product Specs")
    tenor_y = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Coupon/KO Frequency (M)", [1, 3, 6], index=0)
    nc_m = st.number_input("Non-Call Period (M)", 0, 12, 3)

if st.button("Solve & Generate Sensitivities"):
    ivs, hvs, corr = get_market_data(tickers)
    base_vols = ivs if vol_mode == "Implied Vol (IV)" else hvs
    
    # Main Solve
    paths = get_simulation(15000, tenor_y, rf_rate, base_vols, corr, skew_val)
    yield_main = solve_yield(paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)
    res = run_valuation_discrete(yield_main, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

    # TOP OUTPUTS
    st.header(f"Solved Yield: {yield_main*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    c2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    c3.metric("Avg Coupons", f"{res['avg_cpn']:.1f}")

    # SENSITIVITIES
    st.divider()
    ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    y_grid, ki_grid = [], []

    for kv in kk:
        y_r, ki_r = [], []
        for sv in ss:
            yc = solve_yield(paths, rf_rate, tenor_y, sv, kv, freq_m, nc_m)
            rc = run_valuation_discrete(yc, paths, rf_rate, tenor_y, sv, kv, freq_m, nc_m)
            y_r.append(yc * 100); ki_r.append(rc['prob_loss'])
        y_grid.append(y_r); ki_grid.append(ki_r)

    df_y = pd.DataFrame(y_grid, index=kk, columns=ss)
    df_y.index.name = "KO Barrier (%) ↓"
    df_y.columns.name = "Put Strike (%) →"

    st.subheader("Yield Sensitivity (% p.a.)")
    st.dataframe(df_y.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
    
    st.subheader("Capital Loss Probability (%)")
    st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
