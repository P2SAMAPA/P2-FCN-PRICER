import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Institutional FCN & BCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    vols = []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            opts = s.options
            # Fetch longer-dated IV to capture institutional pricing
            chain = s.option_chain(opts[min(4, len(opts)-1)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(iv)
        except: vols.append(0.35) # Conservative default
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now())['Close']
    corr = h.pct_change().corr().values
    return np.array(vols), corr

# --- ENGINE ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    ko_mask_global = np.zeros(n_sims, dtype=bool)
    cpn_counts = np.zeros(n_sims)
    
    obs_dates = np.arange(nc_step, steps, obs_freq)
    for d in obs_dates:
        ko_now = active & (paths[d] >= ko)
        if np.any(ko_now):
            num_cpns = (d // obs_freq)
            payoffs[ko_now] = 100 + (num_cpns * cpn_per_period)
            cpn_counts[ko_now] = num_cpns
            active[ko_now] = False
            ko_mask_global[ko_now] = True
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps - 1) // obs_freq
        payoffs[active] = principal + (num_cpns * cpn_per_period)
        cpn_counts[active] = num_cpns
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": np.mean(ko_mask_global),
        "prob_ki": np.mean(active & (paths[-1] < strike)),
        "avg_cpn": np.mean(cpn_counts)
    }

def generate_paths(sims, tks, t_yr, rf, vols, corr, strike_val, skew_bps):
    L = np.linalg.cholesky(corr)
    dt, steps = 1/252, int(t_yr * 252)
    # Applying Skew: As strike increases, we simulate higher volatility to force sensitivity
    adj_vols = vols * (1 + (skew_bps/10000) * (100 - strike_val))
    
    raw = np.zeros((steps + 1, sims, len(tks)))
    raw[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((sims, len(tks))) @ L.T
        raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
    return np.min(raw, axis=2)

def solve_yield(paths, rf, t_yr, strike, ko, freq_m, nc_m):
    p1 = run_valuation(0.01, paths, rf, t_yr, strike, ko, freq_m, nc_m)['price']
    p2 = run_valuation(0.40, paths, rf, t_yr, strike, ko, freq_m, nc_m)['price']
    if abs(p2 - p1) < 1e-5: return rf
    return 0.01 + (100 - p1) * (0.40 - 0.01) / (p2 - p1)

# --- UI ---
st.sidebar.title("Configuration")
prod_toggle = st.sidebar.radio("Product Type", ["FCN", "BCN"])
tk_in = st.sidebar.text_input("Tickers", "AAPL, MSFT, TSLA")
tks = [x.strip().upper() for x in tk_in.split(",")]
skew_in = st.sidebar.slider("Volatility Skew (bps per 1% OTM)", 0, 500, 150)
sims_in = st.sidebar.select_slider("Sims", [5000, 10000, 20000], 10000)
rf_in = st.sidebar.number_input("Risk Free %", 0.0, 10.0, 4.5) / 100

st.sidebar.subheader("Parameters")
t_yr = st.sidebar.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
f_m = st.sidebar.selectbox("Freq (M)", [1, 3, 6])
nc_m = st.sidebar.number_input("NC (M)", 0, 12, 3)
strike_p = st.sidebar.number_input("Put Strike %", 40, 100, 60)
ko_p = st.sidebar.number_input("KO Barrier %", 70, 150, 105)

if st.sidebar.button("Calculate"):
    with st.spinner("Solving multi-asset surface..."):
        vols, corr = get_market_data(tks, 12)
        
        ss = [strike_p-10, strike_p-5, strike_p, strike_p+5, strike_p+10]
        kk = [ko_p+10, ko_p+5, ko_p, ko_p-5, ko_p-10]
        
        y_grid, ko_grid, ki_grid, cp_grid = [], [], [], []

        for kv in kk:
            y_r, ko_r, ki_r, cp_r = [], [], [], []
            for sv in ss:
                # Regenerating paths inside the loop is the only way to ensure Strike Sensitivity
                wo = generate_paths(sims_in, tks, t_yr, rf_in, vols, corr, sv, skew_in)
                y = solve_yield(wo, rf_in, t_yr, sv, kv, f_m, nc_m)
                res = run_valuation(y, wo, rf_in, t_yr, sv, kv, f_m, nc_m)
                
                y_r.append(y*100); ko_r.append(res['prob_ko']*100)
                ki_r.append(res['prob_ki']*100); cp_r.append(res['avg_cpn'])
            
            y_grid.append(y_r); ko_grid.append(ko_r); ki_grid.append(ki_r); cp_grid.append(cp_r)

        # STYLED OUTPUTS
        st.subheader("Yield Sensitivity (%)")
        st.dataframe(pd.DataFrame(y_grid, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Prob. of KO (%)**")
            st.dataframe(pd.DataFrame(ko_grid, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None).format("{:.1f}"))
        with c2:
            st.write("**Prob. of Capital Loss (%)**")
            st.dataframe(pd.DataFrame(ki_grid, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None).format("{:.1f}"))
            
        st.write("**Avg. Coupons Paid**")
        st.dataframe(pd.DataFrame(cp_grid, index=kk, columns=ss).style.background_gradient(cmap='YlGn', axis=None).format("{:.2f}"))

    # Chart
    wo_main = generate_paths(sims_in, tks, t_yr, rf_in, vols, corr, strike_p, skew_in)
    fig = go.Figure([go.Scatter(y=wo_main[:, i], mode='lines', opacity=0.2) for i in range(15)])
    fig.add_hline(y=ko_p, line_color="green", line_dash="dash", annotation_text="KO")
    fig.add_hline(y=strike_p, line_color="red", line_dash="dash", annotation_text="Strike")
    st.plotly_chart(fig, use_container_width=True)
