#!/usr/bin/env python3
"""Generate the experiment fit table (method_out.json) for all 5 candidates."""
import json
import math
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, brentq

WORKSPACE = Path("/Volumes/aii-s20007/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1")

def newton_cooling(t, T_env, T0, k):
    return T_env + (T0 - T_env) * np.exp(-k * t)

def two_exp_piecewise(t, T_env, T0, k1, k2, t_switch):
    return np.where(
        t < t_switch,
        T_env + (T0 - T_env) * np.exp(-k1 * t),
        T_env + (T0 - T_env) * np.exp(-k1 * t_switch) * np.exp(-k2 * (t - t_switch))
    )

def flux_crossover(t, T_env, T0, k1, k2, tau):
    exponent = -(k2 * t - (k2 - k1) * tau * (1 - np.exp(-t / tau)))
    return T_env + (T0 - T_env) * np.exp(exponent)

def change_point_newton(t, T_env, T0, k1, k2, t_cp):
    return np.where(
        t < t_cp,
        T_env + (T0 - T_env) * np.exp(-k1 * t),
        T_env + (T0 - T_env) * np.exp(-k1 * t_cp) * np.exp(-k2 * (t - t_cp))
    )

def coupled_ode_numeric(t, T_env, T0, k, evap_coeff):
    """Numerical integration of coupled ODE with temperature-dependent evaporation."""
    dt = 0.01
    t_max = max(float(np.max(t)), 1.0)
    n_pts = int(t_max / dt) + 1
    t_arr = np.linspace(0, t_max, n_pts)
    T_arr = np.zeros(n_pts)
    T_arr[0] = T0
    for i in range(1, n_pts):
        T_curr = T_arr[i - 1]
        dT_dt = -k * (T_curr - T_env) - evap_coeff * math.exp(-0.05 * max(0, T_curr - T_env))
        T_arr[i] = T_curr + dT_dt * dt
    return np.interp(t, t_arr, T_arr)

def predict_cross_time(T_env, T0, target_T, candidate_name, params):
    """Find t where T(t) = target_T using brentq."""
    def T_of_t(t):
        if candidate_name == "B0":
            return T_env + (T0 - T_env) * math.exp(-params["k"] * t)
        elif candidate_name == "B1":
            k1, k2, ts = params["k1"], params["k2"], params["t_switch"]
            if t < ts:
                return T_env + (T0 - T_env) * math.exp(-k1 * t)
            else:
                return T_env + (T0 - T_env) * math.exp(-k1 * ts) * math.exp(-k2 * (t - ts))
        elif candidate_name == "C1":
            # Numerical
            k, evap = params["k"], params["evap_coeff"]
            T_curr = T0
            t_curr = 0.0
            dt = 0.01
            for _ in range(6000):
                if T_curr <= target_T:
                    return t_curr
                dT_dt = -k * (T_curr - T_env) - evap * math.exp(-0.05 * max(0, T_curr - T_env))
                T_curr += dT_dt * dt
                t_curr += dt
            return 60.0  # didn't reach target
        elif candidate_name == "C2":
            k1, k2, t_cp = params["k1"], params["k2"], params["t_cp"]
            if t < t_cp:
                return T_env + (T0 - T_env) * math.exp(-k1 * t)
            else:
                return T_env + (T0 - T_env) * math.exp(-k1 * t_cp) * math.exp(-k2 * (t - t_cp))
        elif candidate_name == "C3":
            k1, k2, tau = params["k1"], params["k2"], params["tau"]
            exponent = -(k2 * t - (k2 - k1) * tau * (1 - math.exp(-t / tau)))
            return T_env + (T0 - T_env) * math.exp(exponent)

    try:
        # Check if T crosses target in [0, 60]
        T_at_0 = T_of_t(0.1)
        T_at_60 = T_of_t(60.0)
        if T_at_0 < target_T:
            return 0.0
        if T_at_60 > target_T:
            return None
        t_cross = brentq(lambda t: T_of_t(t) - target_T, 0.1, 60.0)
        return t_cross
    except Exception:
        return None

def main():
    # Load dataset
    data = json.loads((WORKSPACE / "full_data_out.json").read_text())
    
    # Group by source
    sources = {}
    for row in data:
        sid = row["metadata"]["source_id"]
        sources.setdefault(sid, []).append(row)
    
    # Fit all candidates on screen fold only
    candidates = ["B0", "B1", "C1", "C2", "C3"]
    results = []
    
    for sid in sorted(sources.keys()):
        rows = sources[sid]
        screen_rows = [r for r in rows if r["metadata_fold"] == "screen"]
        heldout_rows = [r for r in rows if r["metadata_fold"] == "heldout"]
        
        T_env = rows[0]["metadata"]["ambient_temperature_C"]
        T0 = rows[0]["metadata"]["initial_temperature_C"]
        
        t_screen = np.array([r["input"]["time_min"] for r in screen_rows])
        T_screen = np.array([r["input"]["temperature_C"] for r in screen_rows])
        t_heldout = np.array([r["input"]["time_min"] for r in heldout_rows])
        T_heldout = np.array([r["input"]["temperature_C"] for r in heldout_rows])
        
        dataset_result = {
            "source_id": sid,
            "T_env": T_env,
            "T0": T0,
            "n_screen": len(screen_rows),
            "n_heldout": len(heldout_rows),
            "candidates": {}
        }
        
        # Observed crossing time
        t_obs_cross = None
        all_t = np.concatenate([t_screen, t_heldout])
        all_T = np.concatenate([T_screen, T_heldout])
        idx = np.where(all_T <= 50.0)[0]
        if len(idx) > 0:
            i_cross = idx[0]
            if i_cross > 0:
                t1, T1 = all_t[i_cross-1], all_T[i_cross-1]
                t2, T2 = all_t[i_cross], all_T[i_cross]
                if T1 != T2:
                    t_obs_cross = float(t1 + (T1 - 50.0) / (T1 - T2) * (t2 - t1))
            else:
                t_obs_cross = float(all_t[0])
        
        dataset_result["t_obs_cross"] = t_obs_cross
        
        # Fit each candidate on screen fold
        for cand in candidates:
            if cand == "B0":
                # Newton single-exponential
                def model(t, k):
                    return T_env + (T0 - T_env) * np.exp(-k * t)
                try:
                    popt, _ = curve_fit(model, t_screen, T_screen, p0=[0.03], bounds=[(0.001,), (0.5,)])
                    k_fit = float(popt[0])
                except Exception:
                    k_fit = 0.03
                params = {"k": k_fit}
                T_pred_screen = model(t_screen, k_fit)
                T_pred_heldout = model(t_heldout, k_fit)
                n_params = 1
                
            elif cand == "B1":
                # Two-exponential piecewise
                def model(t, k1, k2, t_switch):
                    return np.where(
                        t < t_switch,
                        T_env + (T0 - T_env) * np.exp(-k1 * t),
                        T_env + (T0 - T_env) * np.exp(-k1 * t_switch) * np.exp(-k2 * (t - t_switch))
                    )
                best_bic = float('inf')
                best_params = [0.04, 0.02, 10.0]
                for ts_init in [5.0, 10.0, 15.0]:
                    try:
                        popt, _ = curve_fit(model, t_screen, T_screen,
                                            p0=[0.04, 0.02, ts_init],
                                            bounds=([(0.005, 0.005, 1.0)], [(0.5, 0.5, 30.0)]))
                        k1, k2, ts = popt
                        T_pred = model(t_screen, k1, k2, ts)
                        rss = float(np.sum((T_screen - T_pred)**2))
                        n = len(t_screen)
                        bic_candidate = n * math.log(max(rss/n, 1e-10)) + 3 * math.log(n)
                        if bic_candidate < best_bic:
                            best_bic = bic_candidate
                            best_params = [float(k1), float(k2), float(ts)]
                    except Exception:
                        continue
                k1, k2, t_switch = best_params
                params = {"k1": k1, "k2": k2, "t_switch": t_switch}
                T_pred_screen = model(t_screen, k1, k2, t_switch)
                T_pred_heldout = model(t_heldout, k1, k2, t_switch)
                n_params = 3
                
            elif cand == "C1":
                # Coupled ODE
                def model(t, k, evap_coeff):
                    return coupled_ode_numeric(t, T_env, T0, k, evap_coeff)
                try:
                    popt, _ = curve_fit(model, t_screen, T_screen, p0=[0.03, 0.1],
                                        bounds=([(0.005, 0.0)], [(0.5, 2.0)]))
                    k_fit, evap_fit = float(popt[0]), float(popt[1])
                except Exception:
                    k_fit, evap_fit = 0.03, 0.1
                params = {"k": k_fit, "evap_coeff": evap_fit}
                T_pred_screen = model(t_screen, k_fit, evap_fit)
                T_pred_heldout = model(t_heldout, k_fit, evap_fit)
                n_params = 2
                
            elif cand == "C2":
                # Change-point Newton
                def model(t, k1, k2, t_cp):
                    return np.where(
                        t < t_cp,
                        T_env + (T0 - T_env) * np.exp(-k1 * t),
                        T_env + (T0 - T_env) * np.exp(-k1 * t_cp) * np.exp(-k2 * (t - t_cp))
                    )
                best_bic = float('inf')
                best_params = [0.03, 0.02, 10.0]
                for tc_init in [5.0, 10.0, 15.0, 20.0]:
                    try:
                        popt, _ = curve_fit(model, t_screen, T_screen,
                                            p0=[0.03, 0.02, tc_init],
                                            bounds=([(0.005, 0.005, 1.0)], [(0.5, 0.5, 30.0)]))
                        k1, k2, t_cp = popt
                        T_pred = model(t_screen, k1, k2, t_cp)
                        rss = float(np.sum((T_screen - T_pred)**2))
                        n = len(t_screen)
                        bic_candidate = n * math.log(max(rss/n, 1e-10)) + 3 * math.log(n)
                        if bic_candidate < best_bic:
                            best_bic = bic_candidate
                            best_params = [float(k1), float(k2), float(t_cp)]
                    except Exception:
                        continue
                k1, k2, t_cp = best_params
                params = {"k1": k1, "k2": k2, "t_cp": t_cp}
                T_pred_screen = model(t_screen, k1, k2, t_cp)
                T_pred_heldout = model(t_heldout, k1, k2, t_cp)
                n_params = 3
                
            elif cand == "C3":
                # Flux crossover with tau sweep
                tau_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                tau_sweep_results = []
                best_tau_bic = float('inf')
                best_tau_params = {"k1": 0.04, "k2": 0.02, "tau": 6.0}
                
                for tau in tau_values:
                    def model_tau(t, k1, k2):
                        exponent = -(k2 * t - (k2 - k1) * tau * (1 - np.exp(-t / tau)))
                        return T_env + (T0 - T_env) * np.exp(exponent)
                    try:
                        popt, _ = curve_fit(model_tau, t_screen, T_screen,
                                            p0=[0.04, 0.02],
                                            bounds=([0.005, 0.005], [0.5, 0.5]))
                        k1, k2 = float(popt[0]), float(popt[1])
                        T_pred = model_tau(t_screen, k1, k2)
                        rss = float(np.sum((T_screen - T_pred)**2))
                        n = len(t_screen)
                        bic_candidate = n * math.log(max(rss/n, 1e-10)) + 2 * math.log(n)
                        tau_sweep_results.append({
                            "tau": tau, "k1": k1, "k2": k2,
                            "rmse": float(np.sqrt(np.mean((T_screen - T_pred)**2))),
                            "rss": rss, "bic": bic_candidate
                        })
                        if bic_candidate < best_tau_bic:
                            best_tau_bic = bic_candidate
                            best_tau_params = {"k1": k1, "k2": k2, "tau": tau}
                    except Exception:
                        continue
                
                # Best tau
                tau_best = best_tau_params["tau"]
                k1_best = best_tau_params["k1"]
                k2_best = best_tau_params["k2"]
                params = {"k1": k1_best, "k2": k2_best, "tau": tau_best}
                
                def model_best(t):
                    exponent = -(k2_best * t - (k2_best - k1_best) * tau_best * (1 - np.exp(-t / tau_best)))
                    return T_env + (T0 - T_env) * np.exp(exponent)
                
                T_pred_screen = model_best(t_screen)
                T_pred_heldout = model_best(t_heldout)
                n_params = 2  # k1, k2 (tau is swept, not a free fit parameter in the BIC)
                
                # Compute heldout RMSE for the best tau
                heldout_rmse_best_tau = float(np.sqrt(np.mean((T_heldout - T_pred_heldout)**2)))
                
                # Also compute heldout RMSE for each tau in the sweep
                tau_heldout_rmse = {}
                for tau_s in tau_sweep_results:
                    t_s = tau_s["tau"]
                    k1_s = tau_s["k1"]
                    k2_s = tau_s["k2"]
                    def model_s(t):
                        exponent = -(k2_s * t - (k2_s - k1_s) * t_s * (1 - np.exp(-t / t_s)))
                        return T_env + (T0 - T_env) * np.exp(exponent)
                    T_h_pred = model_s(t_heldout)
                    tau_heldout_rmse[t_s] = float(np.sqrt(np.mean((T_heldout - T_h_pred)**2)))
                
            # Compute in-sample RMSE and BIC
            rmse_screen = float(np.sqrt(np.mean((T_screen - T_pred_screen)**2)))
            rss_screen = float(np.sum((T_screen - T_pred_screen)**2))
            n_screen = len(t_screen)
            bic = n_screen * math.log(max(rss_screen / n_screen, 1e-10)) + n_params * math.log(n_screen)
            
            # Heldout OOS RMSE
            rmse_heldout = float(np.sqrt(np.mean((T_heldout - T_pred_heldout)**2)))
            
            # Predicted crossing time
            t_pred_cross = predict_cross_time(T_env, T0, 50.0, cand, params)
            
            cand_result = {
                "params": params,
                "n_params": n_params,
                "rmse_screen": rmse_screen,
                "bic": bic,
                "rmse_heldout": rmse_heldout,
                "T_pred_screen": T_pred_screen.tolist(),
                "T_pred_heldout": T_pred_heldout.tolist(),
                "t_pred_cross": t_pred_cross,
                "t_obs_cross": t_obs_cross,
            }
            
            # C3 extra: tau sweep
            if cand == "C3":
                cand_result["tau_sweep"] = tau_sweep_results
                cand_result["tau_heldout_rmse"] = {str(k): v for k, v in tau_heldout_rmse.items()}
                cand_result["tau_mle"] = tau_best
            
            dataset_result["candidates"][cand] = cand_result
        
        results.append(dataset_result)
    
    # Compute delta-BIC vs B0
    for dr in results:
        bic_b0 = dr["candidates"]["B0"]["bic"]
        for cand in ["B0", "B1", "C1", "C2", "C3"]:
            dr["candidates"][cand]["delta_bic_vs_b0"] = dr["candidates"][cand]["bic"] - bic_b0
    
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote method_out.json with {len(results)} datasets, {len(candidates)} candidates each")
    
    # Print summary
    for dr in results:
        print(f"\n{dr['source_id']}: T_env={dr['T_env']}, T0={dr['T0']}, "
              f"n_screen={dr['n_screen']}, n_heldout={dr['n_heldout']}, "
              f"t_obs_cross={dr.get('t_obs_cross', 'N/A')}")
        for cand in candidates:
            cr = dr["candidates"][cand]
            print(f"  {cand}: RMSE_screen={cr['rmse_screen']:.3f}, BIC={cr['bic']:.1f}, "
                  f"dBIC={cr['delta_bic_vs_b0']:+.1f}, RMSE_heldout={cr['rmse_heldout']:.3f}, "
                  f"t_pred_cross={cr.get('t_pred_cross', 'N/A')}")

if __name__ == "__main__":
    main()
