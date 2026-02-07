import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# --- QUANT ENGINE ---
class StructuredProductEngine:
    def __init__(self, vols, rf, tenor_mo, freq_mo, nocall_mo, ko_style, step_down, prod_type, gtd_rate=12.0, bonus_rate=0.0, bonus_barr=85.0):
        self.vols = np.array(vols)
        self.rf = rf
        self.tenor_yr = tenor_mo / 12
        self.steps = int(self.tenor_yr * 252)
        self.obs_freq = int((freq_mo / 12) * 252)
        self.obs_steps = np.arange(self.obs_freq, self.steps + 1, self.obs_freq)
        self.nocall_steps = int((nocall_mo / 12) * 252)
        self.ko_style = ko_style
        self.step_down_daily = (step_down / 100) / 21
        self.prod_type = prod_type
        
        # Dollar coupons per observation for math accuracy
        self.gtd_coupon = (gtd_rate / 100) * (freq_mo / 12)
        self.bonus_coupon = (bonus_rate / 100) * (freq_mo / 12)
        self.bonus_barr = bonus_barr / 100

    def run_simulation(self, strike_pct, ko_pct, n_sims=1000):
        n_assets = len(self.vols)
        dt = 1/252
        strike, ko_barrier = strike_pct / 100, ko_pct / 100
        # Correlation matrix (Assumed 0.6 for equity baskets)
        L = np.linalg.cholesky(np.full((n_assets, n_assets), 0.6) + np.eye(n_assets) * 0.4)
        
        total_payouts, loss_freq, total_final_val = 0, 0, 0
        payout_counts = 0
        
        for _ in range(n_sims):
            Z = np.random.normal(0, 1, (self.steps, n_assets)) @ L.T
            # Geometric Brownian Motion
            paths = np.exp(np.cumsum((self.rf - 0.5 * self.vols**2) * dt + self.vols * np.sqrt(dt) * Z, axis=0))
            worst_path = np.min(paths, axis=1)
            
            sim_payouts = 0
            sim_payout_count = 0
            knocked_out = False
            final_principal = 1.0
            
            for step in self.obs_steps:
                # 1. Payout Logic
                if self.prod_type == "FCN":
                    sim_payouts += self.gtd_coupon
                    sim_payout_count += 1
                else: # BCN
                    sim_payouts += self.gtd_coupon
                    if worst_path[step-1] >= self.bonus_barr:
                        sim_payouts += self.bonus_coupon
                        sim_payout_count += 1
                
                # 2. KO Logic
                curr_ko = ko_barrier
                if self.ko_style == "Step Down" and step > self.nocall_steps:
                    curr_ko -= (self.step_down_daily * (step - self.nocall_steps))
                
                if step >= self.nocall_steps and worst_path[step-1] >= curr_ko:
                    knocked_out = True
                    break
            
            # 3. Maturity Logic (if no KO)
            if not knocked_out:
                if worst_path[-1] < strike:
                    loss_freq += 1
                    final_principal = worst_path[-1] / strike # Physical delivery logic
            
            total_payouts += sim_payouts
            payout_counts += sim_payout_count
            total_final_val += (final_principal + sim_payouts)

        # Annualized Yield = (Total Return - 1) / Tenor
        avg_total_return = total_final_val / n_sims
        ann_yield = ((avg_total_return - 1) / self.tenor_yr) * 100
        
        return (payout_counts / n_sims), (loss_freq / n_sims), ann_yield
