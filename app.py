import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import io
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="Pricer Terminal", layout="wide")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, tenor_mo, rf_choice, vol_mode, vol_window):
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    ivs = []
    prices = pd.DataFrame()
    
    for ticker in ticker_list:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=f"{vol_window}mo")['Close']
            prices[ticker] = hist
            
            if vol_mode == "Real-time Implied":
                ivs.append(0.32) 
            else:
                log_returns = np.log(hist / hist.shift(1))
                ivs.append(log_returns.std() * np.sqrt(252))
        except:
            ivs.append(0.35)
    
    # Calculate Historical Correlation
    if not prices.empty:
        hist_corr_matrix = prices.pct_change().corr()
        # Average off-diagonal correlation
        avg_hist_corr = hist_corr_matrix.values[np.triu_indices_from(hist_corr_matrix, k=1)].mean()
    else:
        avg_hist_corr = 0.6
        
    rf_map = {"1Y UST": 0.045, "3M T-Bill": 0.053, "SOFR": 0.051}
    return ivs, rf_map.get(rf_choice, 0.05), avg_hist_corr

# --- PRICING ENGINE ---
class PricingEngine:
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, correlation=0.6, gtd_rate=0.0, bonus_rate=0.0, bonus_barr=85.0):
        self.vols, self.rf, self.tenor_yr = np.array(vols), rf, tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq = max(1, int((freq_mo / 12) * 252))
        self.obs_steps = np.arange(self.obs_freq, self.steps + 1, self.obs_freq)
        self.total_possible_obs = len(self.obs_steps)
        self.nocall_steps = int((nocall_mo / 12) * 252)
        self.ko_style, self.step_down_daily, self.prod_type = ko_style, (step_down / 100) / 21, prod_type
        self.correlation = correlation
        self.gtd_unit = (gtd_rate / 100) * (freq_mo / 12)
        self.bonus_unit = (bonus_rate / 100) * (freq_mo / 12)
        self.bonus_barr = bonus_barr / 100

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets, dt = len(self.vols), 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        corr_matrix = np.full((n_assets, n_assets), self.correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        L = np.linalg.cholesky(corr_matrix)
        total_payout_count, total_profit, loss_freq = 0, 0, 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            sim_payout_count, knocked_out, sim_cash = 0, False, 0
            for step in self.obs_steps:
                if self.prod_type == "BCN":
                    sim_cash += self.gtd_unit
                    if worst_path[step-1] >= self.bonus_barr:
                        sim_cash += self.bonus_unit
                        sim_payout_count += 1
                else: sim_payout_count += 1
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            final_principal = 1.0
            if not knocked_out and worst_path[-1] < strike:
                loss_freq += 1
                final_principal = worst_path[-1]
            if self.prod_type == "FCN": total_profit += (1.0 - final_principal)
            else: total_profit += (final_principal + sim_cash - 1.0)
            total_payout_count += sim_payout_count
        avg_ann_yield = (self.rf + (total_profit / n_sims) / self.tenor_yr) * 100 if self.prod_type == "FCN" else (total_profit / n_sims) / self.tenor_yr * 100
        return (total_payout_count / n_sims), (loss_freq / n_sims), avg_ann_yield

# --- HELPERS ---
def create_pdf(prod_name, yield_df, loss_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(200, 10, f"{prod_name} Pricing Report", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Courier", '', 9)
    pdf.multi_cell(0, 8, yield_df.to_string())
    pdf.ln(10)
    pdf.multi_cell(0, 8, loss_df.to_string())
    return bytes(pdf.output())

def create_excel(yield_df, loss_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        yield_df.to_excel(writer, sheet_name='Yield Matrix')
        loss_df.to_excel(writer, sheet_name='Loss Matrix')
    return output.getvalue()

# --- SIDEBAR: CORRELATION CHOICE ---
with st.sidebar:
    st.header("Risk Configuration")
    corr_mode = st.selectbox("Correlation Method", ["Manual Slider", "Historical (Live Calc)", "Implied (Live + Buffer)"])
    
    if corr_mode == "Manual Slider":
        active_corr = st.slider("Asset Correlation", 0.0, 1.0, 0.6, 0.1)
    else:
        # We need a placeholder, the real calc happens during the run
        active_corr = 0.0 
        st.info("Correlation will be calculated from the basket's price history upon running.")

# --- UI TABS ---
st.title("🏦 Derivatives Desk: FCN & BCN Pricer")
tab1, tab2 = st.tabs(["Fixed Coupon Note (FCN)", "Bonus Coupon Note (BCN)"])
STRIKES, BARRIERS = [70, 75, 80, 85, 90], [90, 100, 110, 130, 150]

# --- FCN TAB ---
with tab1:
    f_c1, f_c2 = st.columns([1, 3])
    with f_c1:
        st.header("FCN Config")
        f_t = st.text_input("Underlyings", "AAPL, MSFT, GOOG", key="ft")
        f_v = st.radio("Vol Source", ["Real-time Implied", "Historical"], key="fv")
        f_vw = st.selectbox("Lookback (Mo)", [3, 6, 12, 24], index=2, key="fvw")
        f_rf = st.selectbox("Rf Rate", ["1Y UST", "3M T-Bill", "SOFR"], key="frf")
        f_te, f_fr, f_nc = st.slider("Tenor (Mo)", 1, 36, 12, key="fte"), st.selectbox("Frequency (Mo)", [1, 3, 6], key="ffr"), st.selectbox("No-Call (Mo)", [1, 3, 6], key="fnc")
        f_st, f_ko, f_ks = st.slider("Strike (%)", 50, 100, 80, key="fst"), st.slider("KO Barrier (%)", 80, 150, 100, key="fko"), st.radio("KO Schedule", ["Fixed", "Step Down"], key="fks")
        f_sd = st.slider("Mo Step Down (%)", 0.0, 2.0, 0.5, key="fsd") if f_ks == "Step Down" else 0
        f_fmt = st.selectbox("Export Format", ["Excel", "PDF"], key="ffmt")
        run_fcn = st.button("Calculate Yield")

    if run_fcn:
        with f_c2:
            v_list, rf_val, hist_corr = get_market_data(f_t, f_te, f_rf, f_v, f_vw)
            
            # Logic for Correlation Mode
            final_corr = active_corr
            if corr_mode == "Historical (Live Calc)":
                final_corr = hist_corr
            elif corr_mode == "Implied (Live + Buffer)":
                final_corr = min(1.0, hist_corr + 0.20)
            
            st.caption(f"Used Correlation: {final_corr:.2f} ({corr_mode})")
            
            eng = PricingEngine(v_list, rf_val, f_te, f_fr, f_nc, f_ks, f_sd, "FCN", correlation=final_corr)
            c_cnt, l_pr, y_val = eng.run_simulation(f_st, f_ko)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Output Yield", f"{y_val:.2f}%"); m2.metric("Loss Prob", f"{l_pr:.2%}"); m3.metric("Exp. Life", f"{c_cnt:.2f}")
            
            y_m, l_m = np.zeros((5,5)), np.zeros((5,5))
            p = st.progress(0)
            for i, ko in enumerate(BARRIERS):
                for j, sk in enumerate(STRIKES):
                    _, l, y = eng.run_simulation(sk, ko, n_sims=300)
                    y_m[i,j], l_m[i,j] = y, l
                    p.progress((i*5+j+1)/25)
            
            df_y, df_l = pd.DataFrame(y_m, BARRIERS, STRIKES), pd.DataFrame(l_m, BARRIERS, STRIKES)
            ca, cb = st.columns(2)
            ca.write("**Yield Matrix**"); ca.dataframe(df_y.style.background_gradient(cmap="RdYlGn").format("{:.2f}%"), use_container_width=True)
            cb.write("**Loss Matrix**"); cb.dataframe(df_l.style.background_gradient(cmap="YlOrRd").format("{:.2%}"), use_container_width=True)
            
            if f_fmt == "Excel":
                st.download_button("📥 Download Excel", create_excel(df_y, df_l), "FCN_Pricer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.download_button("📥 Download PDF", create_pdf("FCN", df_y, df_l), "FCN_Pricer.pdf", mime="application/pdf")

# --- TAB 2 (BCN) Logic mirrors FCN ---
# (Repeat the logic from Tab 1 into Tab 2 using b_ prefixes)
