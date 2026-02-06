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

# --- PRICING ENGINE ---
def get_simulation(sims, tenor, rf, vols, corr, skew_f):
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    dt, steps = 1/252, int(tenor * 252)
    
    # Apply a global skew adjustment to the base vols
    # This reflects the 'fear' in the market for downside moves
    adj_vols = vols * (1 + skew_f)
    
    z = np.random.standard_normal((steps, sims, len(vols)))
    epsilon = np.einsum('ij,tkj->tki', L, z)
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * epsilon
    path_returns = np.exp(np.cumsum(drift + diffusion, axis=0))
    paths = np.vstack([np.ones((1, sims, len(vols))), path_returns]) * 100
    return np.min(paths, axis=2)

def run_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_obs = (coupon_pa * (freq_m/12)) * 100
    
    payoffs, active, cpn_earned = np.zeros(n_sims), np.ones(n_sims, dtype=bool), np.zeros(n_sims)
    obs_dates = np.arange(nc_step, steps, obs_freq)
    
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num * cpn_per_obs)
            cpn_earned[ko_mask] = num
            active[ko_mask] = False
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num = (steps - 1) // obs_freq
        payoffs[active] = principal + (num * cpn_per_obs)
        cpn_earned[active] = num
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(~active) * 100,
        "prob_loss": (np.sum(active & (paths[-1] < strike)) / n_sims) * 100,
        "avg_cpn": np.mean(cpn_earned)
    }

def solve_for_yield(paths, r, t, s, k, f, nc):
    # Find yield that brings Note Price to 100
    coupons = np.linspace(0.0, 0.60, 25)
    prices = [run_valuation(c, paths, r, t, s, k, f, nc)['price'] for c in coupons]
    return np.interp(100.0, prices, coupons)

# --- UI ---
st.title("🛡️ Institutional Fixed Coupon Note (FCN) Pricer")

with st.sidebar:
    st.header("1. Assets & Volatility")
    tickers = [x.strip().upper() for x in st.text_input("Tickers", "TSLA, MSFT").split(",")]
    vol_choice = st.radio("Volatility Source", ["Market Implied", "Historical"])
    skew_f = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.2)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100
    st.divider()
    st.header("2. Note Parameters")
    tenor_y = st.number_input("Tenor (Years)", 0.1, 5.0, 1.0)
    strike_p = st.number_input("Put Strike %", 40, 100, 60)
    ko_p = st.number_input("KO Barrier %", 70, 150, 100)
    freq_m = st.selectbox("Coupon Freq (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)

if st.button("Calculate FCN Yield"):
    ivs, hvs, corr = get_market_data(tickers)
    base_vols = ivs if vol_choice == "Market Implied" else hvs
    
    # 1. PRIMARY CALCULATION (TOP)
    paths_main = get_simulation(15000, tenor_y, rf_rate, base_vols, corr, skew_f)
    yield_main = solve_for_yield(paths_main, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)
    res = run_valuation(yield_main, paths_main, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m)

    st.header(f"Solved Annualized Yield: {yield_main*100:.2f}% p.a.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob. of KO", f"{res['prob_ko']:.1f}%")
    c2.metric("Prob. of Capital Loss", f"{res['prob_loss']:.1f}%")
    c3.metric("Avg. Coupons Paid", f"{res['avg_cpn']:.2f}")

    # 2. SENSITIVITY MATRICES (BELOW)
    st.divider()
    ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    y_grid, ki_grid = [], []

    for kv in kk:
        y_r, ki_r = [], []
        for sv in ss:
            # We use the same base simulation to ensure logical consistency across strikes
            yc = solve_for_yield(paths_main, rf_rate, tenor_y, sv, kv, freq_m, nc_m)
            rc = run_valuation(yc, paths_main, rf_rate, tenor_y, sv, kv, freq_m, nc_m)
            y_r.append(yc * 100)
            ki_r.append(rc['prob_loss'])
        y_grid.append(y_r)
        ki_grid.append(ki_r)

    # Adding legends/labels to the dataframes
    df_yield = pd.DataFrame(y_grid, index=kk, columns=ss)
    df_yield.index.name = "KO Barrier (%) ↓"
    df_yield.columns.name = "Put Strike (%) →"

    df_risk = pd.DataFrame(ki_grid, index=kk, columns=ss)
    df_risk.index.name = "KO Barrier (%) ↓"
    df_risk.columns.name = "Put Strike (%) →"

    st.subheader("Yield Sensitivity (% p.a.)")
    st.dataframe(df_yield.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
    
    st.subheader("Capital Loss Probability (%)")
    st.dataframe(df_risk.style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
