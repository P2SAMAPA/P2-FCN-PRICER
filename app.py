import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Institutional FCN Solver", layout="wide")

# --- DATA & SIMULATION ---
@st.cache_data(ttl=3600)
def get_simulated_paths(tickers, tenor, sims, skew_f, rf):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            chain = s.option_chain(s.options[min(len(s.options)-1, 6)])
            vols.append(chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0])
        except: vols.append(0.35)
    
    data = yf.download(tickers, period="12mo", progress=False)['Close']
    corr = data.pct_change().dropna().corr().values if len(tickers) > 1 else np.array([[1.0]])
    
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    adj_vols = np.array(vols) * (1 + skew_f)
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    paths = np.exp(np.cumsum(drift + adj_vols * np.sqrt(dt) * epsilon, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

# --- DISCRETE VALUATION ---
def run_valuation_discrete(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps_total, n_sims, _ = paths.shape
    worst_of_paths = np.min(paths, axis=2)
    obs_interval = max(1, int((freq_m / 12) * 252))
    nc_steps = int((nc_m / 12) * 252)
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
        
    return np.mean(payoffs) * np.exp(-r * tenor)

def solve_yield(paths, r, t, s, k, f, nc):
    coupons = np.linspace(0.0, 0.60, 25)
    prices = [run_valuation_discrete(c, paths, r, t, s, k, f, nc) for c in coupons]
    return np.interp(100.0, prices, coupons)

# --- UI ---
st.title("🛡️ Institutional FCN Solver (Discrete Observation)")

with st.sidebar:
    st.header("Parameters")
    tickers = [x.strip().upper() for x in st.text_input("Tickers (CSV)", "MSFT, GOOGL").split(",")]
    tenor = st.number_input("Tenor (Yrs)", 0.5, 3.0, 1.0)
    freq_m = st.selectbox("Coupon/KO Freq (Months)", [1, 3, 6], index=1)
    nc_m = st.number_input("Non-Call Period (Months)", 0, 12, 3)
    rf = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    skew = st.slider("Vol Skew Factor", 0.0, 0.5, 0.1)
    
if st.button("Calculate Everything"):
    with st.spinner("Simulating Paths..."):
        paths = get_simulated_paths(tickers, tenor, 10000, skew, rf)
    
    # Sensitivity Ranges
    strikes = [50, 55, 60, 65, 70]
    barriers = [110, 105, 100, 95, 90]
    
    results = []
    for ko in barriers:
        row = []
        for strike in strikes:
            y = solve_yield(paths, rf, tenor, strike, ko, freq_m, nc_m)
            row.append(y * 100)
        results.append(row)
    
    # DataFrame with explicit Labels
    df = pd.DataFrame(results, index=barriers, columns=strikes)
    df.index.name = "KO Barrier (%) ↓"
    df.columns.name = "Put Strike (%) →"
    
    st.subheader("Yield Sensitivity Table (% p.a.)")
    st.markdown("Check KO Barrier on the **Left (Rows)** and Put Strike on the **Top (Columns)**.")
    st.dataframe(df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
