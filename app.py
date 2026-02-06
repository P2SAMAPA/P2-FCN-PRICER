import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Institutional FCN Pricer", layout="wide")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data = []
    for t in tickers:
        s = yf.Ticker(t)
        h = s.history(period="12mo")['Close']
        data.append(h.rename(t))
    df = pd.concat(data, axis=1).dropna()
    return df, df.pct_change().corr().values

def get_vols(tickers, source):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        hist_v = s.history(period="12mo")['Close'].pct_change().std() * np.sqrt(252)
        if source == "Market Implied (IV)":
            try:
                opts = s.options
                chain = s.option_chain(opts[min(len(opts)-1, 2)])
                px = s.history(period="1d")['Close'].iloc[-1]
                vols.append(chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0])
            except: vols.append(hist_v)
        else: vols.append(hist_v)
    return np.array(vols)

# --- VALUATION ENGINE ---
def run_valuation(coupon_pa, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims, _ = paths.shape
    worst_of = np.min(paths, axis=2)
    obs_dates = np.arange(int((freq_m/12)*252), steps, int((freq_m/12)*252))
    nc_steps = int((nc_m/12)*252)
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    cpn_val = (coupon_pa * (freq_m/12)) * 100
    cpn_accrued = np.zeros(n_sims)
    
    for d in obs_dates:
        cpn_accrued[active] += cpn_val
        if d >= nc_steps:
            ko_mask = active & (worst_of[d] >= ko)
            payoffs[ko_mask] = 100 + cpn_accrued[ko_mask]
            active[ko_mask] = False
            
    if np.any(active):
        final_px = worst_of[-1, active]
        payoffs[active] = np.where(final_px >= strike, 100, final_px) + cpn_accrued[active]
        
    return np.mean(payoffs) * np.exp(-r * tenor)

def generate_paths(sims, tenor, rf, vols, corr, skew):
    # FORCE SENSITIVITY: Skew is applied directly to the diffusion term
    adj_vols = vols * (1 + skew)
    steps = int(tenor * 252)
    dt = 1/252
    L = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-8)
    z = np.random.standard_normal((steps, sims, len(vols)))
    eps = np.einsum('ij,tkj->tki', L, z)
    
    # GBM Path Calculation
    drift = (rf - 0.5 * adj_vols**2) * dt
    diffusion = adj_vols * np.sqrt(dt) * eps
    paths = np.exp(np.cumsum(drift + diffusion, axis=0))
    return np.vstack([np.ones((1, sims, len(vols))), paths]) * 100

# --- UX ---
st.title("Institutional FCN Solver")

with st.sidebar:
    st.header("1. Market Inputs")
    tk_in = st.text_input("Tickers (CSV)", "NVDA, TSLA")
    tickers = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Market Implied (IV)", "Historical (HV)"])
    skew_f = st.slider("Volatility Skew Factor", 0.0, 1.0, 0.20)
    rf_rate = st.number_input("Risk Free Rate %", 0.0, 10.0, 4.5) / 100

    st.header("2. Note Parameters")
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    freq_m = st.selectbox("Coupon Frequency (Months)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (Months)", 0, 12, 3)
    strike_p = st.slider("Put Strike %", 40, 100, 60)
    ko_p = st.slider("KO Barrier %", 80, 150, 100)

if st.button("Solve & Generate Sensitivities"):
    df, corr = get_market_data(tickers)
    vols = get_vols(tickers, vol_src)
    paths = generate_paths(15000, tenor_y, rf_rate, vols, corr, skew_f)
    
    # High-density Solver
    cpn_space = np.linspace(0.0, 1.2, 100)
    prices = [run_valuation(c, paths, rf_rate, tenor_y, strike_p, ko_p, freq_m, nc_m) for c in cpn_space]
    y_solve = np.interp(100.0, prices[::-1], cpn_space[::-1])

    st.subheader(f"Solved Annual Yield: {y_solve*100:.2f}% p.a.")
    
    col1, col2 = st.columns(2)
    worst_final = np.min(paths[-1], axis=1)
    p_loss = (np.sum(worst_final < strike_p) / 15000) * 100
    col1.metric("Prob. of Capital Loss", f"{p_loss:.1f}%")
    
    # Sensitivity Tables
    st.divider()
    st.write("### Yield Sensitivity Matrix (% p.a.)")
    
    stks = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
    bars = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
    
    grid = []
    for b in bars:
        row = []
        for s in stks:
            # Interpolate for each cell
            y_c = np.interp(100.0, [run_valuation(c, paths, rf_rate, tenor_y, s, b, freq_m, nc_m) for c in [0, 0.4, 0.8]][::-1], [0, 0.4, 0.8][::-1])
            row.append(y_c * 100)
        grid.append(row)
        
    df_sens = pd.DataFrame(grid, index=[f"KO {x}%" for x in bars], columns=[f"Stk {x}%" for x in stks])
    st.table(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))

    st.write("### Capital Loss Probability (%)")
    loss_grid = []
    for b in bars:
        row = []
        for s in stks:
            # Re-calculate loss prob for each structural change
            # Note: coupon doesn't affect capital loss prob, only strike/barrier do
            p_l = (np.sum(np.min(paths[-1], axis=1) < s) / 15000) * 100
            row.append(p_l)
        loss_grid.append(row)
    
    df_loss = pd.DataFrame(loss_grid, index=[f"KO {x}%" for x in bars], columns=[f"Stk {x}%" for x in stks])
    st.table(df_loss.style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
