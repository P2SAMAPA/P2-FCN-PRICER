import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Professional FCN & BCN Solver", layout="wide")

# --- DATA LAYER ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, lookback):
    vols, prices = [], []
    for t in tickers:
        s = yf.Ticker(t)
        try:
            px = s.history(period="1d")['Close'].iloc[-1]
            prices.append(px)
            opts = s.options
            chain = s.option_chain(opts[min(2, len(opts)-1)])
            iv = chain.calls.iloc[(chain.calls['strike'] - px).abs().argsort()[:1]]['impliedVolatility'].values[0]
            vols.append(iv)
        except: vols.append(0.30)
    
    start = datetime.now() - pd.DateOffset(months=lookback)
    h = yf.download(tickers, start=start, end=datetime.now())['Close']
    corr = h.pct_change().corr().values
    return np.array(vols), corr

# --- ENGINE ---
def run_valuation(coupon, paths, r, tenor, strike, ko, freq_m, nc_m, is_bcn=False, bonus=0):
    steps, n_sims = paths.shape
    obs_freq = max(1, int(252 * (freq_m/12)))
    nc_step = int(252 * (nc_m/12))
    cpn_per_period = (coupon * (freq_m/12)) * 100
    
    payoffs = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    ko_count = 0
    cpns_paid = np.zeros(n_sims)
    
    obs_dates = np.arange(nc_step, steps, obs_freq)
    for d in obs_dates:
        ko_mask = active & (paths[d] >= ko)
        if np.any(ko_mask):
            num_cpns = (d // obs_freq)
            payoffs[ko_mask] = 100 + (num_cpns * cpn_per_period)
            cpns_paid[ko_mask] = num_cpns
            active[ko_mask] = False
            ko_count += np.sum(ko_mask)
            
    if np.any(active):
        final_px = paths[-1, active]
        principal = np.where(final_px >= strike, 100, final_px)
        num_cpns = (steps - 1) // obs_freq
        b_val = (bonus * 100) if is_bcn else 0
        payoffs[active] = principal + (num_cpns * cpn_per_period) + b_val
        cpns_paid[active] = num_cpns
        
    return {
        "price": np.mean(payoffs) * np.exp(-r * tenor),
        "prob_ko": (ko_count / n_sims),
        "prob_ki": np.sum(active & (paths[-1] < strike)) / n_sims,
        "avg_cpns": np.mean(cpns_paid)
    }

def stable_solve(paths, r, tenor, strike, ko, freq_m, nc_m):
    # Calculate price at 0% and 50% to find the slope
    p_low = run_valuation(0.02, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    p_high = run_valuation(0.40, paths, r, tenor, strike, ko, freq_m, nc_m)['price']
    
    # Linear solve for Par (100)
    if abs(p_high - p_low) < 1e-5: return r
    sol = 0.02 + (100 - p_low) * (0.40 - 0.02) / (p_high - p_low)
    return max(0, sol)

# --- UI ---
st.title("🛡️ Institutional FCN & BCN Solver")

with st.sidebar:
    is_bcn = st.toggle("BCN Mode", value=False)
    tk_in = st.text_input("Tickers", "AAPL, MSFT")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_mode = st.radio("Vol", ["Implied", "Historical"])
    skew = st.slider("Skew (bps)", 0.0, 5.0, 1.2)
    sims = st.select_slider("Sims", [5000, 10000, 20000], 10000)
    rf = st.number_input("Risk Free %", 0.0, 10.0, 4.5) / 100
    
    st.divider()
    t_yr = st.number_input("Tenor (Y)", 0.1, 5.0, 1.0)
    f_m = st.selectbox("Freq (M)", [1, 3, 6])
    nc_m = st.number_input("Non-Call (M)", 0, 12, 3)
    k_in = st.number_input("Strike %", 40, 100, 60)
    ko_in = st.number_input("KO Barrier %", 70, 150, 100)

if st.button("Calculate & Solve"):
    vols, corr = get_market_data(tks, 12)
    adj_vols = vols + (skew/100 * (100 - k_in)/100)
    
    L = np.linalg.cholesky(corr)
    dt, steps = 1/252, int(t_yr * 252)
    raw = np.zeros((steps + 1, sims, len(tks)))
    raw[0] = 100.0
    for t in range(1, steps + 1):
        z = np.random.standard_normal((sims, len(tks))) @ L.T
        raw[t] = raw[t-1] * np.exp((rf - 0.5*adj_vols**2)*dt + adj_vols*np.sqrt(dt)*z)
    wo = np.min(raw, axis=2)

    if not is_bcn:
        target_y = stable_solve(wo, rf, t_yr, k_in, ko_in, f_m, nc_m)
        res = run_valuation(target_y, wo, rf, t_yr, k_in, ko_in, f_m, nc_m)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Yield p.a.", f"{target_y*100:.2f}%")
        c2.metric("Prob. of KO", f"{res['prob_ko']*100:.1f}%")
        c3.metric("Prob. Capital Loss", f"{res['prob_ki']*100:.1f}%")

        # --- SENSITIVITY ---
        st.subheader("Yield & Risk Sensitivity")
        ss = [k_in-10, k_in-5, k_in, k_in+5, k_in+10]
        kk = [ko_in+10, ko_in+5, ko_in, ko_in-5, ko_in-10]
        
        y_data, ko_data, ki_data = [], [], []
        for kv in kk:
            y_r, ko_r, ki_r = [], [], []
            for sv in ss:
                y = stable_solve(wo, rf, t_yr, sv, kv, f_m, nc_m)
                stt = run_valuation(y, wo, rf, t_yr, sv, kv, f_m, nc_m)
                y_r.append(y*100); ko_r.append(stt['prob_ko']*100); ki_r.append(stt['prob_ki']*100)
            y_data.append(y_r); ko_data.append(ko_r); ki_data.append(ki_r)

        st.write("**Annualized Yield (%)**")
        st.dataframe(pd.DataFrame(y_data, index=kk, columns=ss).style.background_gradient(cmap='RdYlGn', axis=None))
        
        st.write("**Probability of KO (%)**")
        st.dataframe(pd.DataFrame(ko_data, index=kk, columns=ss).style.background_gradient(cmap='Blues', axis=None))
        
        st.write("**Probability of Capital Loss (%)**")
        st.dataframe(pd.DataFrame(ki_data, index=kk, columns=ss).style.background_gradient(cmap='Reds', axis=None))
    
    st.plotly_chart(go.Figure([go.Scatter(y=wo[:, i], mode='lines', opacity=0.2) for i in range(15)]))
