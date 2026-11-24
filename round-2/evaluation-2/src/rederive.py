#!/usr/bin/env python3
"""
Independent re-derivation of headline evaluation numbers.

This script reads raw result files directly and recomputes key metrics
through a different code path than the main eval.py to verify correctness.
"""

import json
import numpy as np
from pathlib import Path

DEP_PATH = Path("/Volumes/aii-s20001/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")

def find_t_obs_cross(t_data, T_data, target_T=50.0):
    """Find observed crossing time of target temperature via linear interpolation."""
    crossed_idx = np.where(T_data <= target_T)[0]
    if len(crossed_idx) == 0:
        return None
    i = crossed_idx[0]
    if i == 0:
        return float(t_data[0])
    t1, T1 = t_data[i-1], T_data[i-1]
    t2, T2 = t_data[i], T_data[i]
    if T1 == T2:
        return float(t1)
    frac = (T1 - target_T) / (T1 - T2)
    return float(t1 + frac * (t2 - t1))

def main():
    print("=== INDEPENDENT RE-DERIVATION ===\n")
    
    # Load raw data
    raw = json.loads((DEP_PATH / "results_raw.json").read_text())
    method = json.loads((DEP_PATH / "full_method_out.json").read_text())
    
    models = ["B0_newton", "B1_piecewise", "C1_coupled_ode", "C2_bayesian_changepoint", "C3_flux_crossover"]
    prefixes = {"B0_newton": "B0", "B1_piecewise": "B1", "C1_coupled_ode": "C1", 
                "C2_bayesian_changepoint": "C2", "C3_flux_crossover": "C3_best"}
    
    # Dataset definitions (from experiment)
    datasets = {
        "kurzweg_1988": {
            "t_min": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float),
            "T_C": np.array([88.0, 80.0, 73.0, 67.0, 62.0, 58.0, 54.0, 51.0, 48.0, 46.0, 44.0, 42.0, 41.0, 39.0, 38.0, 37.0], dtype=float),
            "T_env": 21.0, "T0": 88.0
        },
        "manitoba_workshop": {
            "t_min": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30], dtype=float),
            "T_C": np.array([95.0, 89.0, 84.0, 79.0, 75.0, 71.0, 68.0, 65.0, 62.0, 60.0, 58.0, 55.0, 53.0, 51.0, 49.0, 48.0, 45.0, 43.0], dtype=float),
            "T_env": 21.0, "T0": 95.0
        },
        "ferreira_2024_cup": {
            "t_min": np.array([0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30], dtype=float),
            "T_C": np.array([90.0, 82.0, 75.0, 69.0, 64.0, 60.0, 56.0, 52.0, 49.0, 47.0, 45.0, 44.0, 43.0], dtype=float),
            "T_env": 22.0, "T0": 90.0
        },
        "lomen_lovelock": {
            "t_min": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float),
            "T_C": (np.array([200.0, 185.0, 172.0, 161.0, 152.0, 144.0, 138.0, 132.0, 128.0, 124.0, 121.0, 115.0, 110.0]) - 32.0) * 5.0 / 9.0,
            "T_env": 20.0, "T0": (200.0 - 32.0) * 5.0 / 9.0
        }
    }
    
    # Collect all metrics
    all_rmse = {m: [] for m in models}
    all_bic = {m: [] for m in models}
    all_cross = {m: [] for m in models}
    
    for ds_info in method["datasets"]:
        ds_name = ds_info["dataset"]
        ds_data = datasets[ds_name]
        t_min = ds_data["t_min"]
        T_C = ds_data["T_C"]
        
        # Observed crossing
        t_obs = find_t_obs_cross(t_min, T_C)
        
        for ex in ds_info["examples"]:
            output = json.loads(ex["output"])
            comp = output.get("comparison_row", {})
            
            for m in models:
                p = prefixes[m]
                # RMSE
                if f"oos_rmse_{p}" in comp:
                    all_rmse[m].append(comp[f"oos_rmse_{p}"])
                # BIC
                if f"bic_{p}" in comp:
                    all_bic[m].append(comp[f"bic_{p}"])
                # Crossing
                if f"t_drinkable_{p}_min" in comp:
                    t_pred = comp[f"t_drinkable_{p}_min"]
                    if t_obs is not None and t_pred is not None and not np.isnan(t_pred):
                        all_cross[m].append(abs(t_pred - t_obs))
    
    # Print results
    print("Model Performance (mean across all examples):")
    print(f"{'Model':30s} {'RMSE':>8s} {'BIC':>8s} {'CrossErr':>8s} {'N_RMSE':>6s} {'N_BIC':>6s} {'N_Cross':>6s}")
    print("-" * 80)
    for m in models:
        if all_rmse[m]:
            print(f"{m:30s} {np.mean(all_rmse[m]):8.3f} {np.mean(all_bic[m]):8.1f} {np.mean(all_cross[m]):8.2f} {len(all_rmse[m]):6d} {len(all_bic[m]):6d} {len(all_cross[m]):6d}")
    
    print("\n=== Bootstrap CIs vs Newton (5000 draws, seed=42) ===")
    rng = np.random.default_rng(42)
    ref = "B0_newton"
    
    for m in models:
        if m == ref:
            continue
        n = min(len(all_rmse[m]), len(all_rmse[ref]))
        if n >= 2:
            diffs = np.array(all_rmse[m][:n]) - np.array(all_rmse[ref][:n])
            boots = []
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                boots.append(np.mean(diffs[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "*" if (lo > 0 or hi < 0) else ""
            print(f"RMSE {m:30s} vs Newton: mean={np.mean(diffs):+.3f} CI=[{lo:+.3f}, {hi:+.3f}] {sig}")
    
    print("\n=== Newton Trajectory Error at 5-min intervals ===")
    eval_times = [0, 5, 10, 15, 20, 25, 30]
    
    for t in eval_times:
        errors = []
        for ds_info in method["datasets"]:
            ds_name = ds_info["dataset"]
            ds_data = datasets[ds_name]
            t_min = ds_data["t_min"]
            T_C = ds_data["T_C"]
            T_env = ds_data["T_env"]
            T0 = ds_data["T0"]
            
            for ex in ds_info["examples"]:
                output = json.loads(ex["output"])
                b0 = output.get("B0_newton", {})
                k = b0.get("k")
                if k is not None:
                    T_pred = T_env + (T0 - T_env) * np.exp(-k * t * 60.0)
                    T_obs = np.interp(t, t_min, T_C)
                    errors.append(abs(T_pred - T_obs))
        
        if errors:
            boots = []
            n = len(errors)
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                boots.append(np.mean(np.array(errors)[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  t={t:2d}min: mean={np.mean(errors):.3f}°C CI=[{lo:.3f}, {hi:.3f}]")
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()