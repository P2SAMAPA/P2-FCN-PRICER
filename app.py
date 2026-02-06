import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# --- PRO UI CONFIG ---
st.set_page_config(page_title="Institutional Derivatives Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #d1d9e6; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stTable { background-color: #ffffff; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1a1c23; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- REFINED MARKET DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_rf_rates():
    try:
        # ^IRX = 13-week T-Bill
        val = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        return val if val > 0 else 0.045
    except:
        return 0.045

@st.cache_data(ttl=3600)
def get_mkt_data(tks, src):
    v, p, lp, divs = [], [], [], []
    for t in tks:
        try:
            s = yf.Ticker(t)
            h = s.history(period="12mo")['Close']
            p.append(h.rename(t))
            spot = h.iloc[-1]
            lp.append(spot)
            
            # Fetch Dividend Yield
            dy = s.info.get('dividendYield', 0)
            divs.append(dy if dy is not None else 0.0)
            
            if src == "Market Implied (IV)" and s.options:
                # Find ATM Volatility specifically
                expiry = s.options[min(len(s.options)-1, 2)] # ~3 month expiry
                chain = s.option_chain(expiry)
                calls = chain.calls
                # Get call closest to spot
                atm_vol = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]['impliedVolatility'].values[0]
                v.append(atm_vol)
            else:
                v.append(h.pct_change().std() * np.sqrt(252))
        except:
            v.append(0.35); lp.append(100.0); divs.append(0.0)
    
    df = pd.concat(p, axis=1).dropna() if p else pd.DataFrame()
    corr = df.pct_change().corr().values if not df.empty else np.eye(len(tks))
    return np.array(v), corr, np.array(lp), np.array(divs)

# --- PRECISION MC ENGINE (Antithetic Variates) ---
def run_mc_core(c_g, pths, r, tnr, stk, ko, f_m, nc_m, b_r=0, b_f=0, step_down=0):
    steps, n_s, n_a = pths.shape
    wf = np.min(pths, axis=2)
    obs_idx = np.arange(int((f_m/12)*252), steps, int((f_m/12)*252))
    
    py = np.zeros(n_s)
    act = np.ones(n_s, dtype=bool)
    acc = np.zeros(n_s)
    cpn_count = np.zeros(n_s)
    gv, bv = (c_g*(f_m/12))*100, (b_r*(f_m/12))*100
    
    for i, d in enumerate(obs_idx):
        curr_ko = ko - (i * step_down)
        acc[act] += gv
        if b_r > 0: acc[act & (wf[d] >= b_f)] += bv
        cpn_count[act] += 1
        
        if d >= int((nc_m/12)*252):
            ko_m = act & (wf[d] >= curr_ko)
            py[ko_m] = 100 + acc[ko_m]
            act[ko_m] = False
            
    if np.any(act):
        py[act] = np.where(wf[-1, act] >= stk, 100, wf[-1, act]) + acc[act]
    
    return np.mean(py) * np.exp(-r * tnr), np.mean(cpn_count), (np.sum(wf[-1] < stk)/n_s)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🕹️ Global Controls")
    mode = st.selectbox("Select Product", ["FCN Version 1 (Stable)", "FCN Version 2 (Step-Down)", "BCN Solver"])
    tk_in = st.text_input("Tickers", "NVDA, TSLA")
    tks = [x.strip().upper() for x in tk_in.split(",")]
    vol_src = st.radio("Volatility Source", ["Historical (HV)", "Market Implied (IV)"])
    skew = st.slider("Vol Skew", 0.0, 1.0, 0.2)
    
    st.subheader("🏦 Funding & RF Rate")
    rf_base = get_rf_rates()
    rf_choice = st.selectbox("Base Rate", ["3m T-Bill", "1 Year Treasury", "3m T-Bill + Spread"])
    spread_bps = st.slider("Spread (bps)", 0, 500, 100, step=10) if "Spread" in rf_choice else 0
    rf_rate = rf_base + (spread_bps / 10000)
    if "1 Year" in rf_choice: rf_rate += 0.002
    st.caption(f"Effective Rate: {rf_rate*100:.2f}%")
    
    tenor_y = st.number_input("Tenor (Years)", 0.5, 3.0, 1.0)
    nc_m = st.number_input("Non-Call (M)", 0, 24, 3)

# --- MAIN LOGIC ---
if len(tks) > 0:
    v, corr, lp, divs = get_mkt_data(tks, vol_src)
    av = v * (1 + skew)
    
    # Generate Paths with Antithetic Variates
    n_paths = 10000
    steps = int(tenor_y * 252)
    L = np.linalg.cholesky(corr + np.eye(len(tks))*1e-8)
    
    # Risk-neutral drift: (r - q - 0.5 * sigma^2)
    drift = (rf_rate - divs - 0.5 * av**2) * (1/252)
    diffusion = av * np.sqrt(1/252)
    
    z = np.random.standard_normal((steps, n_paths // 2, len(tks)))
    z_full = np.concatenate([z, -z], axis=1) # The Antithetic part
    
    # Geometric Brownian Motion
    path_returns = drift + diffusion * np.einsum('ij,tkj->tki', L, z_full)
    ps = np.exp(np.cumsum(path_returns, axis=0)) * 100
    ps = np.vstack([np.ones((1, n_paths, len(tks))) * 100, ps])

    if "FCN" in mode:
        st.title(f"🛡️ {mode}")
        ko_val = st.sidebar.slider("Autocall Level (KO) %", 80, 150, 105)
        stk_val = st.sidebar.slider("Put Strike %", 40, 100, 60)
        fq = st.sidebar.selectbox("Payment Frequency", ["Monthly", "Quarterly"])
        fm = {"Monthly": 1, "Quarterly": 3}[fq]
        step_d = st.sidebar.slider("Step-Down % per period", 0.0, 2.0, value=0.5, step=0.5) if "Version 2" in mode else 0.0

        if st.button("Generate Pricing Report"):
            y_solve = brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)
            _, avg_cpn, prob_loss = run_mc_core(y_solve, ps, rf_rate, tenor_y, stk_val, ko_val, fm, nc_m, step_down=step_d)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Solved Yield (p.a.)", f"{y_solve*100:.2f}%")
            m2.metric("Prob. of Capital Loss", f"{prob_loss*100:.1f}%")
            m3.metric("Expected Coupons", f"{avg_cpn:.1f}")
            m4.metric("Strike Barrier", f"{stk_val}%")
            
            # Sensitivity
            st.divider()
            stks, kos = [stk_val-10, stk_val, stk_val+10], [ko_val+10, ko_val, ko_val-10]
            gy, gl = [], []
            for b in kos:
                ry, rl = [], []
                for s in stks:
                    try: ry.append(brentq(lambda x: run_mc_core(x, ps, rf_rate, tenor_y, s, b, fm, nc_m, step_down=step_d)[0] - 100, 0, 5)*100)
                    except: ry.append(0.0)
                    rl.append((np.sum(np.min(ps[-1], 1) < s)/n_paths)*100)
                gy.append(ry); gl.append(rl)
            
            cl, cr = st.columns(2)
            with cl:
                st.write("**Annualized Yield (%)**")
                st.table(pd.DataFrame(gy, index=kos, columns=stks).style.background_gradient(cmap='RdYlGn'))
            with cr:
                st.write("**Loss Probability (%)**")
                st.table(pd.DataFrame(gl, index=kos, columns=stks).style.background_gradient(cmap='Reds'))

    else:
        # BCN logic
        st.title("🛡️ Institutional BCN Solver")
        ko_b = st.sidebar.slider("KO Level %", 80, 150, 105)
        fq_b = st.sidebar.selectbox("Guar Freq", ["Monthly", "Quarterly"])
        fm_b, b_ref = {"Monthly": 1, "Quarterly": 3}[fq_b], st.sidebar.slider("Bonus Ref Strike %", 100, 130, 100)
        ga, ba = st.columns(2)
        g_cpn = ga.number_input("Guar %", 0.0, 20.0, 4.0)/100
        b_rate = ba.number_input("Bonus %", 0.0, 40.0, 8.0)/100
        
        if st.button("Solve BCN KI"):
            try:
                ki = brentq(lambda x: run_mc_core(g_cpn, ps, rf_rate, tenor_y, x, ko_b, fm_b, nc_m, b_rate, b_ref)[0] - 100, 0.01, 150)
                st.metric("Required KI Barrier", f"{ki:.2f}%")
            except: st.error("Solver Error: Note value out of bounds.")
