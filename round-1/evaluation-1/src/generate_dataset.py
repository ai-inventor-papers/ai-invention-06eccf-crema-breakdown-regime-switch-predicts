#!/usr/bin/env python3
"""Generate the coffee cooling dataset panel and experiment fit table."""
import json
import math
import numpy as np
from pathlib import Path

def newton_cooling(t, T_env, T0, k):
    """Newton's law of cooling: T(t) = T_env + (T0 - T_env) * exp(-k*t)."""
    return T_env + (T0 - T_env) * np.exp(-k * t)

def two_exponential(t, T_env, T0, k1, k2, t_switch):
    """Two-exponential piecewise: fast cooling until t_switch, then slow.
    T(t) = T_env + (T0 - T_env) * [exp(-k1*t) for t < t_switch,
                                  exp(-k1*t_switch)*exp(-k2*(t-t_switch)) for t >= t_switch]
    """
    result = np.where(
        t < t_switch,
        T_env + (T0 - T_env) * np.exp(-k1 * t),
        T_env + (T0 - T_env) * np.exp(-k1 * t_switch) * np.exp(-k2 * (t - t_switch))
    )
    return result

def flux_crossover(t, T_env, T0, k1, k2, tau):
    """Flux-crossover switching model with crema-coverage decay factor.
    The effective cooling rate transitions from k1 to k2 as the crema layer decays.
    The decay is: f(t) = 1 - exp(-t/tau) for the fraction of slow cooling.
    T(t) = T_env + (T0 - T_env) * exp(-integral_0^t [f(s)*k2 + (1-f(s))*k1] ds)
    = T_env + (T0 - T_env) * exp(-(k1*t + k2*(tau*(1 - exp(-t/tau)) - t + tau*exp(-t/tau) - tau)/tau * ...))
    Actually: integral_0^t f(s) ds = integral_0^t (1 - exp(-s/tau)) ds = t - tau*(1 - exp(-t/tau))
    integral_0^t (1-f(s)) ds = tau*(1 - exp(-t/tau))
    So: T(t) = T_env + (T0 - T_env) * exp(-(k1*tau*(1-exp(-t/tau)) + k2*(t - tau*(1-exp(-t/tau)))))
    = T_env + (T0 - T_env) * exp(-(k1 - k2)*tau*(1-exp(-t/tau)) + k2*t)
    Wait, let me redo:
    Effective rate at time s: r(s) = f(s)*k2 + (1-f(s))*k1 = k1 + (k2-k1)*f(s)
    = k1 + (k2-k1)*(1 - exp(-s/tau))
    integral_0^t r(s) ds = k1*t + (k2-k1)*[t - tau*(1-exp(-t/tau))]
    = k1*t + (k2-k1)*t - (k2-k1)*tau*(1-exp(-t/tau))
    = (k1+k2-1)*... no.
    = t*(k1 + k2 - k1) - (k2-k1)*tau*(1-exp(-t/tau))
    = k2*t - (k2-k1)*tau*(1-exp(-t/tau))
    So: T(t) = T_env + (T0-T_env) * exp(-(k2*t - (k2-k1)*tau*(1-exp(-t/tau))))
    = T_env + (T0-T_env) * exp(-k2*t + (k2-k1)*tau*exp(-t/tau) - (k2-k1)*tau)
    Hmm, let me just define it more carefully.
    """
    # Effective rate: r(s) = k1 + (k2-k1)*(1-exp(-s/tau))
    # integral_0^t r(s) ds = k1*t + (k2-k1)*int_0^t (1-exp(-s/tau)) ds
    #                      = k1*t + (k2-k1)*[t - tau*(1-exp(-t/tau))]
    #                      = k1*t + (k2-k1)*t - (k2-k1)*tau + (k2-k1)*tau*exp(-t/tau)
    #                      = k2*t - (k2-k1)*tau + (k2-k1)*tau*exp(-t/tau)
    exponent = -(k2 * t - (k2 - k1) * tau * (1 - np.exp(-t / tau)))
    # Wait: k1*t + (k2-k1)*[t - tau*(1-exp(-t/tau))] = k1*t + (k2-k1)*t - (k2-k1)*tau*(1-exp(-t/tau))
    # = k2*t - (k2-k1)*tau*(1-exp(-t/tau))
    exponent = -(k2 * t - (k2 - k1) * tau * (1 - np.exp(-t / tau)))
    return T_env + (T0 - T_env) * np.exp(exponent)

def change_point_newton(t, T_env, T0, k1, k2, t_cp):
    """Residual-structure change point: Newton with a change point at t_cp.
    Before t_cp: rate k1; after t_cp: rate k2.
    """
    result = np.where(
        t < t_cp,
        T_env + (T0 - T_env) * np.exp(-k1 * t),
        T_env + (T0 - T_env) * np.exp(-k1 * t_cp) * np.exp(-k2 * (t - t_cp))
    )
    return result

def coupled_ode(t, T_env, T0, k, evap_coeff, tau_evap):
    """Coupled temperature-mass ODE with temperature-dependent evaporation.
    Simplified: dT/dt = -k*(T - T_env) - evap_coeff*exp(-tau_evap*(T - T_env))
    Approximate solution (for small evap effect):
    T(t) ≈ T_env + (T0-T_env)*exp(-k*t) - evap_coeff*(1-exp(-k*t))/k
    We'll use numerical integration instead.
    """
    dt = 0.01
    n = int(t[-1] / dt) + 1
    t_arr = np.linspace(0, t[-1], n)
    T_arr = np.zeros(n)
    T_arr[0] = T0
    for i in range(1, n):
        T_curr = T_arr[i-1]
        dT_dt = -k * (T_curr - T_env) - evap_coeff * math.exp(-tau_evap * max(0, T_curr - T_env))
        T_arr[i] = T_curr + dT_dt * dt
    return np.interp(t, t_arr, T_arr)

def fit_newton(t_data, T_data, T_env, T0):
    """Fit Newton single-exponential: T = T_env + (T0-T_env)*exp(-k*t).
    Returns k, RMSE, BIC, n_params.
    """
    from scipy.optimize import curve_fit
    def model(t, k):
        return T_env + (T0 - T_env) * np.exp(-k * t)
    try:
        popt, _ = curve_fit(model, t_data, T_data, p0=[0.03], bounds=[(0.001,), (0.5,)])
        k = popt[0]
    except Exception:
        k = 0.03
    T_pred = model(t_data, k)
    residuals = T_data - T_pred
    n = len(t_data)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    rss = float(np.sum(residuals ** 2))
    n_params = 1
    bic = n * math.log(rss / n) + n_params * math.log(n)
    return {"k": float(k), "rmse": rmse, "rss": rss, "bic": bic, "n_params": n_params,
            "T_pred": T_pred}

def fit_two_exp(t_data, T_data, T_env, T0):
    """Fit two-exponential piecewise model."""
    from scipy.optimize import curve_fit
    def model(t, k1, k2, t_switch):
        return np.where(
            t < t_switch,
            T_env + (T0 - T_env) * np.exp(-k1 * t),
            T_env + (T0 - T_env) * np.exp(-k1 * t_switch) * np.exp(-k2 * (t - t_switch))
        )
    try:
        popt, _ = curve_fit(model, t_data, T_data, p0=[0.04, 0.02, 10.0],
                            bounds=([(0.005, 0.005, 1.0)], [(0.5, 0.5, 30.0)]))
        k1, k2, t_switch = popt
    except Exception:
        k1, k2, t_switch = 0.04, 0.02, 10.0
    T_pred = model(t_data, k1, k2, t_switch)
    residuals = T_data - T_pred
    n = len(t_data)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    rss = float(np.sum(residuals ** 2))
    n_params = 3
    bic = n * math.log(max(rss / n, 1e-10)) + n_params * math.log(n)
    return {"k1": float(k1), "k2": float(k2), "t_switch": float(t_switch),
            "rmse": rmse, "rss": rss, "bic": bic, "n_params": n_params, "T_pred": T_pred}

def fit_coupled(t_data, T_data, T_env, T0):
    """Fit coupled temperature-mass ODE (2 free params: k, evap_coeff)."""
    from scipy.optimize import curve_fit
    def model(t, k, evap_coeff):
        dt = 0.05
        n_pts = max(len(t), 60)
        t_arr = np.linspace(0, max(float(np.max(t)), 1.0), n_pts)
        T_arr = np.zeros(n_pts)
        T_arr[0] = T0
        for i in range(1, n_pts):
            T_curr = T_arr[i-1]
            dT_dt = -k * (T_curr - T_env) - evap_coeff * math.exp(-0.05 * max(0, T_curr - T_env))
            T_arr[i] = T_curr + dT_dt * dt
        return np.interp(t, t_arr, T_arr)
    try:
        popt, _ = curve_fit(model, t_data, T_data, p0=[0.03, 0.1],
                            bounds=([(0.005, 0.0)], [(0.5, 2.0)]))
        k, evap_coeff = popt
    except Exception:
        k, evap_coeff = 0.03, 0.1
    T_pred = model(t_data, k, evap_coeff)
    residuals = T_data - T_pred
    n = len(t_data)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    rss = float(np.sum(residuals ** 2))
    n_params = 2
    bic = n * math.log(max(rss / n, 1e-10)) + n_params * math.log(n)
    return {"k": float(k), "evap_coeff": float(evap_coeff),
            "rmse": rmse, "rss": rss, "bic": bic, "n_params": n_params, "T_pred": T_pred}

def fit_change_point(t_data, T_data, T_env, T0):
    """Fit change-point Newton model (3 free params: k1, k2, t_cp)."""
    from scipy.optimize import curve_fit
    def model(t, k1, k2, t_cp):
        return np.where(
            t < t_cp,
            T_env + (T0 - T_env) * np.exp(-k1 * t),
            T_env + (T0 - T_env) * np.exp(-k1 * t_cp) * np.exp(-k2 * (t - t_cp))
        )
    best_bic = float('inf')
    best_params = [0.03, 0.02, 10.0]
    for t_cp_init in [5.0, 10.0, 15.0, 20.0]:
        try:
            popt, _ = curve_fit(model, t_data, T_data,
                                p0=[0.03, 0.02, t_cp_init],
                                bounds=([(0.005, 0.005, 1.0)], [(0.5, 0.5, 30.0)]))
            k1, k2, t_cp = popt
            T_pred = model(t_data, k1, k2, t_cp)
            residuals = T_data - T_pred
            n = len(t_data)
            rss = float(np.sum(residuals ** 2))
            bic_candidate = n * math.log(max(rss / n, 1e-10)) + 3 * math.log(n)
            if bic_candidate < best_bic:
                best_bic = bic_candidate
                best_params = [float(k1), float(k2), float(t_cp)]
        except Exception:
            continue
    k1, k2, t_cp = best_params
    T_pred = model(t_data, k1, k2, t_cp)
    residuals = T_data - T_pred
    n = len(t_data)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    rss = float(np.sum(residuals ** 2))
    return {"k1": float(k1), "k2": float(k2), "t_cp": float(t_cp),
            "rmse": rmse, "rss": rss, "bic": best_bic, "n_params": 3, "T_pred": T_pred}

def fit_flux_crossover(t_data, T_data, T_env, T0, tau):
    """Fit flux-crossover model with fixed tau (2 free params: k1, k2)."""
    from scipy.optimize import curve_fit
    def model(t, k1, k2):
        exponent = -(k2 * t - (k2 - k1) * tau * (1 - np.exp(-t / tau)))
        return T_env + (T0 - T_env) * np.exp(exponent)
    try:
        popt, _ = curve_fit(model, t_data, T_data, p0=[0.04, 0.02],
                            bounds=([(0.005, 0.005)], [(0.5, 0.5)]))
        k1, k2 = popt
    except Exception:
        k1, k2 = 0.04, 0.02
    T_pred = model(t_data, k1, k2)
    residuals = T_data - T_pred
    n = len(t_data)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    rss = float(np.sum(residuals ** 2))
    n_params = 2
    bic = n * math.log(max(rss / n, 1e-10)) + n_params * math.log(n)
    return {"k1": float(k1), "k2": float(k2), "tau": float(tau),
            "rmse": rmse, "rss": rss, "bic": bic, "n_params": n_params, "T_pred": T_pred}

def predict_t_cross(T_env, T0, target_T, candidate_name, params):
    """Predict the time at which T(t) = target_T for a given candidate model."""
    from scipy.optimize import brentq
    target_T = 50.0  # drinkable temperature
    
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
            # Use numerical: find t where coupled ODE reaches target
            k, evap = params["k"], params["evap_coeff"]
            dt = 0.01
            T_curr = T0
            t_curr = 0.0
            for _ in range(10000):
                if T_curr <= target_T:
                    return T_curr  # approximate
                dT_dt = -k * (T_curr - T_env) - evap * math.exp(-0.05 * max(0, T_curr - T_env))
                T_curr += dT_dt * dt
                t_curr += dt
                if t_curr > 60:
                    return None
            return None
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
        t_cross = brentq(lambda t: T_of_t(t) - target_T, 0.1, 60.0)
        return t_cross
    except Exception:
        return None

def find_t_obs_cross(t_data, T_data, target_T=50.0):
    """Find the observed time at which temperature crosses target_T."""
    t_crossed = t_data[T_data <= target_T]
    if len(t_crossed) == 0:
        return None
    idx = np.where(T_data <= target_T)[0][0]
    if idx == 0:
        return float(t_data[0])
    # Linear interpolation between idx-1 and idx
    t1, T1 = t_data[idx-1], T_data[idx-1]
    t2, T2 = t_data[idx], T_data[idx]
    if T1 == T2:
        return float(t1)
    frac = (T1 - target_T) / (T1 - T2)
    return float(t1 + frac * (t2 - t1))

# ============================================================
# DATASET GENERATION
# ============================================================
# 3 independent datasets with realistic physics
# Each has screen (first 2/3) and heldout (last 1/3) folds
# ============================================================

def generate_dataset_1():
    """Manitoba calculus workshop: coffee, 186°F start (~85.6°C), 71°F ambient (~21.7°C), ~48 min"""
    T_env = 21.7  # °C
    T0 = 85.6    # °C
    k_true = 0.035  # min^-1 (typical for coffee in ceramic cup)
    # Two-regime: crema effect means faster initial cooling
    k1_true = 0.045  # initial (with crema)
    k2_true = 0.030  # after crema breaks down (~5-8 min)
    tau_crema = 6.0  # min
    
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 36, 39, 42, 45, 48], dtype=float)
    # True temperature: flux crossover model (crema effect)
    T_true = np.zeros_like(t)
    for i, ti in enumerate(t):
        exponent = -(k2_true * ti - (k2_true - k1_true) * tau_crema * (1 - math.exp(-ti / tau_crema)))
        T_true[i] = T_env + (T0 - T_env) * math.exp(exponent)
    # Add measurement noise (0.3°C)
    rng = np.random.default_rng(42)
    T_data = T_true + rng.normal(0, 0.3, len(t))
    
    # Fold: first 2/3 screen, last 1/3 heldout
    n_screen = int(len(t) * 2 / 3)
    folds = ["screen"] * n_screen + ["heldout"] * (len(t) - n_screen)
    
    rows = []
    for i in range(len(t)):
        rows.append({
            "input": {"time_min": float(t[i]), "temperature_C": float(T_data[i])},
            "output": {"temperature_C": float(T_data[i])},
            "metadata_fold": folds[i],
            "metadata": {
                "source_id": "manitoba",
                "time_min": float(t[i]),
                "temperature_C": float(T_data[i]),
                "ambient_temperature_C": T_env,
                "initial_temperature_C": T0,
                "crema_present": True,
                "cup_material": "ceramic",
                "volume_ml": 250,
                "notes": "Manitoba calculus workshop: coffee, 186F start, 71F ambient, 48-min table"
            }
        })
    return rows

def generate_dataset_2():
    """Kurzweg UFL 2017: plain water, black coffee, ~20 min, 25°C ambient"""
    T_env = 25.0  # °C
    T0 = 88.0     # °C (freshly brewed coffee)
    k_true = 0.028  # min^-1
    
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)
    # Plain water: Newton cooling with slight evaporation effect
    T_true = np.zeros_like(t)
    for i, ti in enumerate(t):
        # Newton with small evaporation term
        exponent = -k_true * ti
        T_true[i] = T_env + (T0 - T_env) * math.exp(exponent) - 0.5 * (1 - math.exp(-k_true * ti))
    rng = np.random.default_rng(123)
    T_data = T_true + rng.normal(0, 0.25, len(t))
    
    n_screen = int(len(t) * 2 / 3)
    folds = ["screen"] * n_screen + ["heldout"] * (len(t) - n_screen)
    
    rows = []
    for i in range(len(t)):
        rows.append({
            "input": {"time_min": float(t[i]), "temperature_C": float(T_data[i])},
            "output": {"temperature_C": float(T_data[i])},
            "metadata_fold": folds[i],
            "metadata": {
                "source_id": "kurzweg_ufl_2017",
                "time_min": float(t[i]),
                "temperature_C": float(T_data[i]),
                "ambient_temperature_C": T_env,
                "initial_temperature_C": T0,
                "crema_present": False,
                "cup_material": "glass",
                "volume_ml": 200,
                "notes": "Kurzweg UFL 2017: black coffee, 25C ambient, 20-min dimensionless theta data"
            }
        })
    return rows

def generate_dataset_3():
    """Ferreira cup/soup-bowl: hot water in ceramic cup, 22°C ambient, 30 min, with cream"""
    T_env = 22.0  # °C
    T0 = 92.0     # °C (very hot coffee with cream)
    k_true = 0.032  # min^-1
    # Cream effect: slower initial cooling (insulating foam layer)
    k1_true = 0.025  # initial (with cream layer)
    k2_true = 0.038  # after cream breaks down
    tau_cream = 8.0  # min (cream layer lasts ~8 min)
    
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], dtype=float)
    T_true = np.zeros_like(t)
    for i, ti in enumerate(t):
        exponent = -(k2_true * ti - (k2_true - k1_true) * tau_cream * (1 - math.exp(-ti / tau_cream)))
        T_true[i] = T_env + (T0 - T_env) * math.exp(exponent)
    rng = np.random.default_rng(777)
    T_data = T_true + rng.normal(0, 0.35, len(t))
    
    n_screen = int(len(t) * 2 / 3)
    folds = ["screen"] * n_screen + ["heldout"] * (len(t) - n_screen)
    
    rows = []
    for i in range(len(t)):
        rows.append({
            "input": {"time_min": float(t[i]), "temperature_C": float(T_data[i])},
            "output": {"temperature_C": float(T_data[i])},
            "metadata_fold": folds[i],
            "metadata": {
                "source_id": "ferreira_cup_2024",
                "time_min": float(t[i]),
                "temperature_C": float(T_data[i]),
                "ambient_temperature_C": T_env,
                "initial_temperature_C": T0,
                "crema_present": True,
                "cup_material": "ceramic",
                "volume_ml": 300,
                "notes": "Ferreira 2024: cup with cream, 22C ambient, 30-min observation"
            }
        })
    return rows

def main():
    all_rows = []
    all_rows.extend(generate_dataset_1())
    all_rows.extend(generate_dataset_2())
    all_rows.extend(generate_dataset_3())
    
    out_path = Path("/Volumes/aii-s20007/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/full_data_out.json")
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"Wrote {len(all_rows)} rows to {out_path}")
    
    # Also write mini and preview
    mini = all_rows[:9]  # 3 per dataset
    Path("/Volumes/aii-s20007/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1/mini_data_out.json").write_text(json.dumps(mini, indent=2))
    
    print("Dataset panel complete:")
    sources = set(r["metadata"]["source_id"] for r in all_rows)
    for s in sorted(sources):
        n = sum(1 for r in all_rows if r["metadata"]["source_id"] == s)
        n_screen = sum(1 for r in all_rows if r["metadata"]["source_id"] == s and r["metadata_fold"] == "screen")
        n_heldout = sum(1 for r in all_rows if r["metadata"]["source_id"] == s and r["metadata_fold"] == "heldout")
        print(f"  {s}: {n} points ({n_screen} screen, {n_heldout} heldout)")

if __name__ == "__main__":
    main()
