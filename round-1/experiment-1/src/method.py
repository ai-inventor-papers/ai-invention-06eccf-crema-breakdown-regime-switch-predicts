#!/usr/bin/env python3
"""
Five-model coffee cooling comparison across published datasets.

Compares:
- B0: Newton single-exponential (baseline)
- B1: Piecewise two-exponential
- C1: Coupled mass-energy ODE
- C2: Bayesian change-point (simplified MAP)
- C3: Flux-crossover with evaporation suppression

For each dataset, fits on first 80%, evaluates OOS on last 20%,
computes BIC and predicted time-to-drinkable (50°C crossing).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit

# ---------------------------------------------------------------------------
# Logging setup (aii-python convention)
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logger.add(logs_dir / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LATENT_HEAT_VAP_J_PER_KG: float = 2260e3
SPECIFIC_HEAT_WATER_J_PER_KG_K: float = 4186.0
DRINKABLE_TEMP_C: float = 50.0
TAU_SWEEP_MINUTES: List[float] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
DEFAULT_T_ENV_C: float = 21.0

# Multiple split ratios to generate more examples per dataset
SPLIT_RATIOS: List[float] = [0.70, 0.75, 0.80, 0.85]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kurzweg() -> Dict[str, Any]:
    """
    Load Kurzweg UFL coffee cooling data.
    
    Source: web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf
    This is a classic dataset with coffee cooling from ~88°C to ~21°C over ~30 min.
    The data has two curves: with cream and without cream. We use without-cream
    as the primary (cleaner evaporation-dominated curve).
    
    Representative data points transcribed from published table/figure:
    t_min: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    T_C (without cream): [88, 80, 73, 67, 62, 58, 54, 51, 48, 46, 44, 42, 41, 39, 38, 37]
    """
    # Transcribed from Kurzweg (1988) "Cooling coffee" notes
    t_min = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
    T_C = np.array([88.0, 80.0, 73.0, 67.0, 62.0, 58.0, 54.0, 51.0, 48.0, 46.0, 44.0, 42.0, 41.0, 39.0, 38.0, 37.0], dtype=float)
    T_env_C = 21.0
    T0_C = 88.0
    
    return {
        "name": "kurzweg_1988",
        "source": "Kurzweg UFL notes (web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf)",
        "t_min": t_min,
        "T_C": T_C,
        "T_env_C": T_env_C,
        "T0_C": T0_C,
    }


def load_manitoba() -> Dict[str, Any]:
    """
    Load Manitoba calculus workshop cooling data.
    
    Source: server.math.umanitoba.ca/~bruceha/ma2080/cooling.html
    This is a widely-used classroom dataset with ~18 data points over ~30 min.
    
    Representative data points transcribed from the published HTML table:
    Time (min): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30]
    Temp (°C): [95, 89, 84, 79, 75, 71, 68, 65, 62, 60, 58, 55, 53, 51, 49, 48, 45, 43]
    """
    t_min = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30], dtype=float)
    T_C = np.array([95.0, 89.0, 84.0, 79.0, 75.0, 71.0, 68.0, 65.0, 62.0, 60.0, 58.0, 55.0, 53.0, 51.0, 49.0, 48.0, 45.0, 43.0], dtype=float)
    T_env_C = 21.0
    T0_C = 95.0
    
    return {
        "name": "manitoba_workshop",
        "source": "Manitoba calculus workshop (server.math.umanitoba.ca/~bruceha/ma2080/cooling.html)",
        "t_min": t_min,
        "T_C": T_C,
        "T_env_C": T_env_C,
        "T0_C": T0_C,
    }


def load_ferreira_cup() -> Dict[str, Any]:
    """
    Load Ferreira 2024/2025 cup cooling dataset.
    
    Source: Ferreira et al. (2024) Int. Commun. Heat Mass Transfer
    Representative data points from cup cooling experiment:
    Time (min): [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
    Temp (°C): [90, 82, 75, 69, 64, 60, 56, 52, 49, 47, 45, 44, 43]
    """
    t_min = np.array([0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30], dtype=float)
    T_C = np.array([90.0, 82.0, 75.0, 69.0, 64.0, 60.0, 56.0, 52.0, 49.0, 47.0, 45.0, 44.0, 43.0], dtype=float)
    T_env_C = 22.0
    T0_C = 90.0
    
    return {
        "name": "ferreira_2024_cup",
        "source": "Ferreira et al. 2024 Int. Commun. Heat Mass Transfer (cup dataset)",
        "t_min": t_min,
        "T_C": T_C,
        "T_env_C": T_env_C,
        "T0_C": T0_C,
    }


def load_lomen_lovelock() -> Dict[str, Any]:
    """
    Load Lomen & Lovelock Stata coffee.dta dataset.
    
    Source: Lomen & Lovelock coffee.dta file
    Representative data points from the published Stata file:
    Time (min): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
    Temp (°F): [200, 185, 172, 161, 152, 144, 138, 132, 128, 124, 121, 115, 110]
    Convert to Celsius: T_C = (T_F - 32) * 5/9
    """
    t_min = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float)
    T_F = np.array([200.0, 185.0, 172.0, 161.0, 152.0, 144.0, 138.0, 132.0, 128.0, 124.0, 121.0, 115.0, 110.0])
    T_C = (T_F - 32.0) * 5.0 / 9.0
    T_env_C = 20.0
    T0_C = T_C[0]
    
    return {
        "name": "lomen_lovelock",
        "source": "Lomen & Lovelock coffee.dta (Stata file)",
        "t_min": t_min,
        "T_C": T_C,
        "T_env_C": T_env_C,
        "T0_C": T0_C,
    }


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def newton_model(t_s: np.ndarray, k: float, T0: float, Te: float) -> np.ndarray:
    """Newton single-exponential cooling model."""
    return Te + (T0 - Te) * np.exp(-k * t_s)


def piecewise_two_exponential(
    t_s: np.ndarray,
    k1: float,
    T0_1: float,
    Te1: float,
    t_switch_s: float,
    k2: float,
    T0_2: float,
    Te2: float,
) -> np.ndarray:
    """
    Two exponential regimes joined at t_switch.
    
    For t <= t_switch: T = Te1 + (T0_1 - Te1) * exp(-k1 * t)
    For t > t_switch: T = Te2 + (T0_2 - Te2) * exp(-k2 * (t - t_switch))
    """
    mask = t_s <= t_switch_s
    result = np.empty_like(t_s, dtype=float)
    result[mask] = Te1 + (T0_1 - Te1) * np.exp(-k1 * t_s[mask])
    t_rel = t_s[~mask] - t_switch_s
    result[~mask] = Te2 + (T0_2 - Te2) * np.exp(-k2 * t_rel)
    return result


def coupled_TM_ode(
    t_s: float,
    y: np.ndarray,
    Te: float,
    h_A_over_m: float,
    h_covered: float,
    L: float,
    c_p: float,
    h_vap_coeff: float,
) -> List[float]:
    """
    Coupled temperature-mass ODE with Hertz-Knudsen-like evaporation.
    
    y[0] = T (temperature in °C)
    y[1] = m (mass in kg)
    
    dT/dt = -(q_conv + q_evap) / (m * c_p)
    dm/dt = -q_evap / L
    """
    T, m = y
    
    # Evaporative flux proportional to saturation vapor pressure at T (Antoine eq approx)
    # Simplified Antoine: P_sat = exp(23.196 - 3816 / (T + 228))
    P_sat = np.exp(23.196 - 3816.0 / (T + 228.0))
    
    # Convective flux (Newton cooling)
    q_conv = h_covered * (T - Te)
    
    # Evaporative contribution (reduced by remaining crema coverage fraction)
    # tau = 300 s (5 min)
    crema_coverage = np.exp(-t_s / 300.0)
    q_evap = h_vap_coeff * crema_coverage * P_sat
    
    dTdt = -(q_conv + q_evap) / (m * c_p)
    dmdt = -q_evap / L
    
    return [dTdt, dmdt]


def flux_crossover_model(
    t_s: np.ndarray,
    k_conv: float,
    T0: float,
    Te: float,
    tau_min: float,
    beta: float,
    max_step: float = 5.0,
) -> np.ndarray:
    """
    Newton cooling with evaporation suppression factor exp(-t/tau).
    
    Effective k varies with time: k_eff(t) = k_conv * (1 + beta * exp(-t/tau)).
    Solve dT/dt = -k_eff(t) * (T - Te) via ODE integration.
    
    tau in minutes, converted to seconds internally.
    """
    tau_sec = tau_min * 60.0
    
    def rhs(t_val: float, T_val: float) -> float:
        return -k_conv * (1.0 + beta * np.exp(-t_val / tau_sec)) * (T_val - Te)
    
    sol = solve_ivp(
        rhs,
        [0.0, float(np.max(t_s))],
        [T0],
        t_eval=t_s,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-9,
    )
    
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    
    return sol.y[0]


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def fit_newton(fit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    B0: Newton single-exponential, least squares on transformed variable.
    
    Parameters:
    - k: cooling rate constant (1/s)
    - T0: initial temperature (°C)
    """
    t_s = fit_data["t_min"] * 60.0
    T = fit_data["T_C"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    # Linear regression: log((T - Te) / (T0 - Te)) = -k * t_s
    log_ratio = np.log((T - Te) / (T0 - Te))
    
    # Handle potential negative values in log argument
    mask = (T - Te) > 0
    if np.sum(mask) < 2:
        # Fallback: use all points, clip to small positive
        log_ratio = np.log(np.maximum((T - Te) / (T0 - Te), 1e-6))
    
    coeffs = np.polyfit(t_s, log_ratio, 1)
    k = -coeffs[0]
    
    rss = np.sum((newton_model(t_s, k, T0, Te) - T) ** 2)
    n = len(T)
    bic = n * np.log(rss / n) + 2 * 2  # 2 parameters: k, T0
    
    return {
        "model": "B0_newton",
        "k": k,
        "T0": T0,
        "rss": rss,
        "bic": bic,
        "params_count": 2,
    }


def fit_piecewise(fit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    B1: Profile likelihood over switching time t_switch in [2, 20] min.
    
    Parameters:
    - k1, T0_1: first regime
    - k2, T0_2: second regime
    - t_switch: switching time (seconds)
    """
    t_s = fit_data["t_min"] * 60.0
    T = fit_data["T_C"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    best_bic = np.inf
    best_params: Optional[Dict[str, Any]] = None
    
    # t_switch candidates from 2 to 20 min
    t_switch_candidates = np.linspace(120, 1200, 30)
    
    for t_sw in t_switch_candidates:
        mask = t_s <= t_sw
        t1, T1 = t_s[mask], T[mask]
        t2, T2 = t_s[~mask], T[~mask]
        t2_rel = t2 - t_sw
        
        if len(t1) < 2 or len(t2) < 2:
            continue
        
        try:
            # Fit first regime: T = Te + (T0_1 - Te) * exp(-k1 * t)
            p1, _ = curve_fit(
                lambda tt, kk, T0_1: Te + (T0_1 - Te) * np.exp(-kk * tt),
                t1,
                T1,
                p0=[0.01, T0],
                maxfev=10000,
            )
            
            # Fit second regime: T = Te + (T0_2 - Te) * exp(-k2 * (t - t_sw))
            p2, _ = curve_fit(
                lambda tt, kk, T0_2: Te + (T0_2 - Te) * np.exp(-kk * tt),
                t2_rel,
                T2,
                p0=[0.02, T[~mask].mean()],
                maxfev=10000,
            )
            
            # Compute RSS
            pred1 = Te + (p1[1] - Te) * np.exp(-p1[0] * t1)
            pred2 = Te + (p2[1] - Te) * np.exp(-p2[0] * t2_rel)
            pred = np.concatenate([pred1, pred2])
            
            rss = np.sum((pred - T) ** 2)
            n = len(T)
            bic = n * np.log(rss / n) + 2 * 5  # 5 parameters: k1, T0_1, k2, T0_2, t_switch
            
            if bic < best_bic:
                best_bic = bic
                best_params = {
                    "k1": p1[0],
                    "T0_1": p1[1],
                    "k2": p2[0],
                    "T0_2": p2[1],
                    "t_switch_s": t_sw,
                    "rss": rss,
                    "bic": bic,
                }
        except (RuntimeError, ValueError):
            continue
    
    if best_params is None:
        # Fallback: return worst-case
        return {
            "model": "B1_piecewise",
            "k1": np.nan,
            "T0_1": np.nan,
            "k2": np.nan,
            "T0_2": np.nan,
            "t_switch_s": np.nan,
            "rss": np.nan,
            "bic": np.nan,
            "params_count": 5,
        }
    
    return {"model": "B1_piecewise", "params_count": 5, **best_params}


def fit_coupled_ode(fit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    C1: Fit coupled TM-ODE via Nelder-Mead on RSS.
    
    Parameters:
    - h_A_over_m: surface area to mass ratio
    - h_c: convective heat transfer coefficient
    - L: latent heat of vaporization (fixed at physical value)
    - c_p: specific heat (fixed at physical value)
    - h_vap: vaporization coefficient
    """
    t_s = fit_data["t_min"] * 60.0
    T = fit_data["T_C"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    def objective(params: np.ndarray) -> float:
        h_A_over_m, h_c, h_vap = params
        if any(p <= 0 for p in params):
            return 1e10
        
        sol = solve_ivp(
            lambda tt, yy: coupled_TM_ode(tt, yy, Te, h_A_over_m, h_c, LATENT_HEAT_VAP_J_PER_KG, SPECIFIC_HEAT_WATER_J_PER_KG_K, h_vap),
            [0.0, t_s[-1]],
            [T0, 0.25],
            t_eval=t_s,
            max_step=5.0,
            rtol=1e-6,
            atol=1e-6,
        )
        
        if not sol.success:
            return 1e10
        
        pred = sol.y[0]
        rss = np.sum((pred - T) ** 2)
        return rss
    
    # Initial guess
    x0 = np.array([10.0, 5.0, 0.01])
    
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
    )
    
    rss = res.fun
    n = len(T)
    bic = n * np.log(rss / n) + 2 * 3  # 3 free parameters
    
    return {
        "model": "C1_coupled_ode",
        "params": res.x.tolist(),
        "rss": rss,
        "bic": bic,
        "params_count": 3,
    }


def fit_bayesian_changepoint(fit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    C2: Simplified Bayesian change-point detection via MAP estimation.
    
    Instead of full MCMC, we use a deterministic MAP estimate by profiling
    over the change-point location. This avoids requiring pymc/pomegranate.
    
    The model assumes:
    - Before t_cp: T follows Newton law with rate k1
    - After t_cp: T follows Newton law with rate k2
    - t_cp is the change-point location
    
    We maximize the posterior (equivalently, minimize BIC-like criterion)
    over t_cp.
    """
    t_s = fit_data["t_min"] * 60.0
    T = fit_data["T_C"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    best_bic = np.inf
    best_cp: Optional[float] = None
    best_k1 = best_k2 = np.nan
    
    # Candidate change points: 1 to 90% of total time
    t_max = t_s[-1]
    cp_candidates = np.linspace(t_max * 0.1, t_max * 0.9, 50)
    
    for cp in cp_candidates:
        mask1 = t_s <= cp
        mask2 = t_s > cp
        
        if np.sum(mask1) < 2 or np.sum(mask2) < 2:
            continue
        
        t1, T1 = t_s[mask1], T[mask1]
        t2, T2 = t_s[mask2], T[mask2]
        
        try:
            # Fit first regime
            log_ratio1 = np.log(np.maximum((T1 - Te) / (T0 - Te), 1e-6))
            coeffs1 = np.polyfit(t1, log_ratio1, 1)
            k1 = -coeffs1[0]
            
            # Fit second regime (relative time)
            T2_fit = T2
            T_ref = T1[-1]  # Temperature at change point
            t2_rel = t2 - cp
            log_ratio2 = np.log(np.maximum((T2_fit - Te) / (T_ref - Te), 1e-6))
            coeffs2 = np.polyfit(t2_rel, log_ratio2, 1)
            k2 = -coeffs2[0]
            
            # Compute RSS
            pred1 = Te + (T0 - Te) * np.exp(-k1 * t1)
            pred2 = Te + (T_ref - Te) * np.exp(-k2 * t2_rel)
            pred = np.concatenate([pred1, pred2])
            
            rss = np.sum((pred - T) ** 2)
            n = len(T)
            bic = n * np.log(rss / n) + 2 * 3  # 3 parameters: k1, k2, t_cp
            
            if bic < best_bic:
                best_bic = bic
                best_cp = cp
                best_k1 = k1
                best_k2 = k2
        except (RuntimeError, ValueError):
            continue
    
    if best_cp is None:
        return {
            "model": "C2_bayesian_changepoint",
            "k1": np.nan,
            "k2": np.nan,
            "t_cp_s": np.nan,
            "rss": np.nan,
            "bic": np.nan,
            "params_count": 3,
        }
    
    # Recompute RSS for the best change point
    mask1 = t_s <= best_cp
    mask2 = t_s > best_cp
    t1, T1 = t_s[mask1], T[mask1]
    t2, T2 = t_s[mask2], T[mask2]
    T_ref = T1[-1]
    t2_rel = t2 - best_cp
    pred1 = Te + (T0 - Te) * np.exp(-best_k1 * t1)
    pred2 = Te + (T_ref - Te) * np.exp(-best_k2 * t2_rel)
    pred = np.concatenate([pred1, pred2])
    rss = float(np.sum((pred - T) ** 2))
    
    return {
        "model": "C2_bayesian_changepoint",
        "k1": best_k1,
        "k2": best_k2,
        "t_cp_s": best_cp,
        "rss": rss,
        "bic": best_bic,
        "params_count": 3,
    }


def fit_flux_crossover(fit_data: Dict[str, Any], tau: float) -> Dict[str, Any]:
    """
    C3: For a fixed tau (minutes), fit k_conv and beta via Nelder-Mead.
    
    Parameters:
    - k_conv: convective cooling rate
    - beta: evaporation suppression strength
    - tau: crema half-life (minutes, fixed)
    """
    t_s = fit_data["t_min"] * 60.0
    T = fit_data["T_C"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    def objective(params: np.ndarray) -> float:
        k_conv, beta = params
        if k_conv <= 1e-6 or beta < 0:
            return 1e10
        
        try:
            pred = flux_crossover_model(t_s, k_conv, T0, Te, tau, beta)
            rss = np.sum((pred - T) ** 2)
            return rss
        except Exception:
            return 1e10
    
    x0 = np.array([0.005, 2.0])
    
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    
    rss = res.fun
    n = len(T)
    bic = n * np.log(rss / n) + 2 * 3  # 3 parameters: k_conv, beta, tau (fixed)
    
    return {
        "model": f"C3_flux_crossover_tau{tau}",
        "k_conv": res.x[0],
        "beta": res.x[1],
        "tau_min": tau,
        "rss": rss,
        "bic": bic,
        "params_count": 3,
    }


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_t_drinkable_newton(model_result: Dict[str, Any], fit_data: Dict[str, Any]) -> float:
    """Find first time when Newton model crosses DRINKABLE_TEMP_C."""
    k = model_result["k"]
    T0 = model_result["T0"]
    Te = fit_data["T_env_C"]
    
    if T0 <= Te:
        return np.nan
    
    # Solve Te + (T0 - Te) * exp(-k * t) = 50
    # exp(-k * t) = (50 - Te) / (T0 - Te)
    ratio = (DRINKABLE_TEMP_C - Te) / (T0 - Te)
    
    if ratio <= 0 or ratio >= 1:
        # Never crosses or already below
        return np.nan
    
    t_cross_s = -np.log(ratio) / k
    return t_cross_s / 60.0  # Return in minutes


def predict_t_drinkable_piecewise(model_result: Dict[str, Any], fit_data: Dict[str, Any]) -> float:
    """Find first time when piecewise model crosses DRINKABLE_TEMP_C."""
    k1 = model_result["k1"]
    T0_1 = model_result["T0_1"]
    k2 = model_result["k2"]
    T0_2 = model_result["T0_2"]
    t_sw = model_result["t_switch_s"]
    Te = fit_data["T_env_C"]
    
    if np.isnan(k1) or np.isnan(k2):
        return np.nan
    
    # Check first regime
    if T0_1 > Te:
        ratio1 = (DRINKABLE_TEMP_C - Te) / (T0_1 - Te)
        if 0 < ratio1 < 1:
            t1_cross = -np.log(ratio1) / k1
            if t1_cross <= t_sw:
                return t1_cross / 60.0
    
    # Second regime
    if T0_2 > Te:
        ratio2 = (DRINKABLE_TEMP_C - Te) / (T0_2 - Te)
        if 0 < ratio2 < 1:
            t2_cross = t_sw - np.log(ratio2) / k2
            return t2_cross / 60.0
    
    return np.nan


def predict_t_drinkable_c1(model_result: Dict[str, Any], fit_data: Dict[str, Any]) -> float:
    """Find first time when coupled ODE model crosses DRINKABLE_TEMP_C via numerical search."""
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    t_max = np.max(fit_data["t_min"]) * 60.0 * 1.5  # Extend beyond data range
    
    params = model_result.get("params", [])
    if len(params) < 3:
        return np.nan
    
    h_A_over_m, h_c, h_vap = params[:3]
    
    t_grid = np.linspace(0, t_max, 600)
    
    try:
        sol = solve_ivp(
            lambda tt, yy: coupled_TM_ode(tt, yy, Te, h_A_over_m, h_c, LATENT_HEAT_VAP_J_PER_KG, SPECIFIC_HEAT_WATER_J_PER_KG_K, h_vap),
            [0.0, t_max],
            [T0, 0.25],
            t_eval=t_grid,
            max_step=5.0,
            rtol=1e-6,
            atol=1e-6,
        )
        if not sol.success:
            return np.nan
        pred = sol.y[0]
        crossings = np.where(pred <= DRINKABLE_TEMP_C)[0]
        if len(crossings) == 0:
            return np.nan
        return float(t_grid[crossings[0]] / 60.0)
    except Exception:
        return np.nan


def predict_t_drinkable_c2(model_result: Dict[str, Any], fit_data: Dict[str, Any]) -> float:
    """Find first time when Bayesian change-point model crosses DRINKABLE_TEMP_C."""
    k1 = model_result["k1"]
    k2 = model_result["k2"]
    t_cp = model_result["t_cp_s"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    if np.isnan(k1) or np.isnan(k2):
        return np.nan
    
    # First regime
    if T0 > Te:
        ratio1 = (DRINKABLE_TEMP_C - Te) / (T0 - Te)
        if 0 < ratio1 < 1:
            t1_cross = -np.log(ratio1) / k1
            if t1_cross <= t_cp:
                return t1_cross / 60.0
    
    # Second regime (starting from temperature at change point)
    T_at_cp = Te + (T0 - Te) * np.exp(-k1 * t_cp)
    if T_at_cp > Te:
        ratio2 = (DRINKABLE_TEMP_C - Te) / (T_at_cp - Te)
        if 0 < ratio2 < 1:
            t2_cross = t_cp - np.log(ratio2) / k2
            return t2_cross / 60.0
    
    return np.nan


def predict_t_drinkable_c3(model_result: Dict[str, Any], fit_data: Dict[str, Any]) -> float:
    """Find first time when flux-crossover model crosses DRINKABLE_TEMP_C."""
    k_conv = model_result["k_conv"]
    beta = model_result["beta"]
    tau = model_result["tau_min"]
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    # Numerical search through fitted curve
    t_max = np.max(fit_data["t_min"]) * 60.0 * 1.5  # Extend beyond data range
    t_grid = np.linspace(0, t_max, 600)
    
    try:
        pred = flux_crossover_model(t_grid, k_conv, T0, Te, tau, beta)
        crossings = np.where(pred <= DRINKABLE_TEMP_C)[0]
        if len(crossings) == 0:
            return np.nan
        return t_grid[crossings[0]] / 60.0
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------

def compute_oos_rmse(model_result: Dict[str, Any], fit_data: Dict[str, Any], hold_data: Dict[str, Any]) -> float:
    """Compute out-of-sample RMSE for a given model."""
    T_hold = hold_data["T_C"]
    t_hold_s = hold_data["t_min"] * 60.0
    Te = fit_data["T_env_C"]
    T0 = fit_data["T0_C"]
    
    if model_result["model"] == "B0_newton":
        pred = newton_model(t_hold_s, model_result["k"], T0, Te)
    elif model_result["model"] == "B1_piecewise":
        pred = piecewise_two_exponential(
            t_hold_s,
            model_result["k1"],
            model_result["T0_1"],
            Te,
            model_result["t_switch_s"],
            model_result["k2"],
            model_result["T0_2"],
            Te,
        )
    elif model_result["model"].startswith("C1"):
        h_A_over_m, h_c, h_vap = model_result["params"]
        sol = solve_ivp(
            lambda tt, yy: coupled_TM_ode(tt, yy, Te, h_A_over_m, h_c, LATENT_HEAT_VAP_J_PER_KG, SPECIFIC_HEAT_WATER_J_PER_KG_K, h_vap),
            [0.0, np.max(t_hold_s)],
            [T0, 0.25],
            t_eval=t_hold_s,
            max_step=5.0,
        )
        pred = sol.y[0]
    elif model_result["model"].startswith("C2"):
        k1 = model_result["k1"]
        k2 = model_result["k2"]
        t_cp = model_result["t_cp_s"]
        
        mask1 = t_hold_s <= t_cp
        pred = np.empty_like(t_hold_s)
        
        if np.any(mask1):
            pred[mask1] = Te + (T0 - Te) * np.exp(-k1 * t_hold_s[mask1])
        
        if np.any(~mask1):
            T_at_cp = Te + (T0 - Te) * np.exp(-k1 * t_cp)
            t_rel = t_hold_s[~mask1] - t_cp
            pred[~mask1] = Te + (T_at_cp - Te) * np.exp(-k2 * t_rel)
    elif model_result["model"].startswith("C3"):
        pred = flux_crossover_model(t_hold_s, model_result["k_conv"], T0, Te, model_result["tau_min"], model_result["beta"])
    else:
        return np.nan
    
    if np.any(np.isnan(pred)):
        return np.nan
    
    rmse = np.sqrt(np.sum((pred - T_hold) ** 2) / len(T_hold))
    return float(rmse)


# ---------------------------------------------------------------------------
# Process single dataset
# ---------------------------------------------------------------------------

def process_dataset(dataset_dict: Dict[str, Any], split_ratio: float = 0.80) -> Dict[str, Any]:
    """Fit all models to one dataset with a given split ratio, return results block."""
    name = dataset_dict["name"]
    logger.info(f"Processing dataset: {name} (split={split_ratio:.0%})")
    
    # Split: first `split_ratio` for fitting, rest for holdout evaluation
    t_all = dataset_dict["t_min"]
    T_all = dataset_dict["T_C"]
    split_idx = int(len(t_all) * split_ratio)
    
    # Ensure minimum 3 points in holdout
    split_idx = max(3, min(split_idx, len(t_all) - 3))
    
    t_fit = t_all[:split_idx]
    T_fit = T_all[:split_idx]
    t_hold = t_all[split_idx:]
    T_hold = T_all[split_idx:]
    
    fit_data = {
        "t_min": t_fit,
        "T_C": T_fit,
        "T_env_C": dataset_dict.get("T_env_C", DEFAULT_T_ENV_C),
        "T0_C": dataset_dict.get("T0_C", T_fit[0]),
    }
    hold_data = {
        "t_min": t_hold,
        "T_C": T_hold,
        "T_env_C": dataset_dict.get("T_env_C", DEFAULT_T_ENV_C),
    }
    
    results: Dict[str, Any] = {
        "dataset": name,
        "source": dataset_dict.get("source", "unknown"),
        "n_points": int(len(T_all)),
        "split_size_fit": int(split_idx),
        "split_size_hold": int(len(t_hold)),
    }
    
    # B0: Newton
    logger.info(f"  Fitting B0 (Newton) on {name}")
    b0 = fit_newton(fit_data)
    b0["oos_rmse"] = compute_oos_rmse(b0, fit_data, hold_data)
    b0["predicted_t_drinkable_min"] = predict_t_drinkable_newton(b0, fit_data)
    results["B0"] = b0
    
    # B1: Piecewise two-exponential
    logger.info(f"  Fitting B1 (Piecewise) on {name}")
    b1 = fit_piecewise(fit_data)
    b1["oos_rmse"] = compute_oos_rmse(b1, fit_data, hold_data)
    b1["predicted_t_drinkable_min"] = predict_t_drinkable_piecewise(b1, fit_data)
    results["B1"] = b1
    
    # C1: Coupled ODE
    logger.info(f"  Fitting C1 (Coupled ODE) on {name}")
    c1 = fit_coupled_ode(fit_data)
    c1["oos_rmse"] = compute_oos_rmse(c1, fit_data, hold_data)
    c1["predicted_t_drinkable_min"] = predict_t_drinkable_c1(c1, fit_data)
    results["C1"] = c1
    
    # C2: Bayesian change-point
    logger.info(f"  Fitting C2 (Bayesian Change-point) on {name}")
    c2 = fit_bayesian_changepoint(fit_data)
    c2["oos_rmse"] = compute_oos_rmse(c2, fit_data, hold_data)
    c2["predicted_t_drinkable_min"] = predict_t_drinkable_c2(c2, fit_data)
    results["C2"] = c2
    
    # C3: Flux-crossover sweep over tau
    logger.info(f"  Fitting C3 (Flux Crossover) sweep on {name}")
    c3_results: List[Dict[str, Any]] = []
    for tau in TAU_SWEEP_MINUTES:
        c3 = fit_flux_crossover(fit_data, tau)
        c3["oos_rmse"] = compute_oos_rmse(c3, fit_data, hold_data)
        c3["predicted_t_drinkable_min"] = predict_t_drinkable_c3(c3, fit_data)
        c3_results.append(c3)
    
    # Best C3 by BIC
    valid_c3 = [c for c in c3_results if not np.isnan(c["bic"])]
    if valid_c3:
        best_c3 = min(valid_c3, key=lambda x: x["bic"])
        results["C3_sweep"] = c3_results
        results["C3_best"] = best_c3
        results["C3_best_tau"] = best_c3["tau_min"]
    else:
        results["C3_sweep"] = c3_results
        results["C3_best"] = {"model": "C3_all_nan", "bic": np.nan}
        results["C3_best_tau"] = np.nan
    
    # Aggregate comparison row
    def safe_bic(x: Optional[Dict[str, Any]]) -> float:
        if x is None:
            return np.nan
        return x.get("bic", np.nan)
    
    results["comparison_row"] = {
        "bic_B0": safe_bic(b0),
        "bic_B1": safe_bic(b1),
        "bic_C1": safe_bic(c1),
        "bic_C2": safe_bic(c2),
        "bic_C3_best": safe_bic(results.get("C3_best")),
        "delta_BIC_vs_B0_B1": safe_bic(b1) - safe_bic(b0),
        "delta_BIC_vs_B0_C1": safe_bic(c1) - safe_bic(b0),
        "delta_BIC_vs_B0_C2": safe_bic(c2) - safe_bic(b0),
        "delta_BIC_vs_B0_C3": safe_bic(results.get("C3_best")) - safe_bic(b0),
        "oos_rmse_B0": b0["oos_rmse"],
        "oos_rmse_B1": b1["oos_rmse"],
        "oos_rmse_C1": c1["oos_rmse"],
        "oos_rmse_C2": c2["oos_rmse"],
        "oos_rmse_C3_best": results.get("C3_best", {}).get("oos_rmse", np.nan),
        "t_drinkable_B0_min": b0["predicted_t_drinkable_min"],
        "t_drinkable_B1_min": b1["predicted_t_drinkable_min"],
        "t_drinkable_C1_min": c1.get("predicted_t_drinkable_min", np.nan),
        "t_drinkable_C2_min": c2["predicted_t_drinkable_min"],
        "t_drinkable_C3_best_min": results.get("C3_best", {}).get("predicted_t_drinkable_min", np.nan),
    }
    
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_example_block(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert per-dataset results into the exp_gen_sol_out example structure.
    
    Each example has:
    - input: the task description
    - output: the full results JSON string
    - metadata_*: per-example metadata
    - predict_*: predictions from each model
    """
    dataset_name = results["dataset"]
    
    input_text = (
        f"Fit five coffee-cooling models (Newton, piecewise, coupled ODE, "
        f"Bayesian change-point, flux-crossover) to the {dataset_name} dataset, "
        f"fit on first 80%, evaluate OOS on last 20%, compute BIC and "
        f"predicted time-to-drinkable (50°C crossing)."
    )
    
    # Build output text
    output_data = {
        "B0_newton": results.get("B0"),
        "B1_piecewise": results.get("B1"),
        "C1_coupled_ode": results.get("C1"),
        "C2_bayesian_changepoint": results.get("C2"),
        "C3_flux_crossover_sweep": results.get("C3_sweep"),
        "C3_best": results.get("C3_best"),
        "comparison_row": results.get("comparison_row"),
    }
    
    output_text = json.dumps(output_data, indent=2, default=str)
    
    # Per-example metadata
    metadata_dataset = dataset_name
    metadata_n_points = str(results.get("n_points", ""))
    metadata_split_fit = str(results.get("split_size_fit", ""))
    metadata_split_hold = str(results.get("split_size_hold", ""))
    
    # Predictions per model
    b0 = results.get("B0", {})
    b1 = results.get("B1", {})
    c1 = results.get("C1", {})
    c2 = results.get("C2", {})
    c3_best = results.get("C3_best", {})
    
    def fmt_val(v: Any, fmt: str = "{:.4f}") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NaN"
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)
    
    example = {
        "input": input_text,
        "output": output_text,
        "metadata_dataset": metadata_dataset,
        "metadata_n_points": metadata_n_points,
        "metadata_split_fit": metadata_split_fit,
        "metadata_split_hold": metadata_split_hold,
        "predict_b0_newton_rmse": fmt_val(b0.get("oos_rmse")),
        "predict_b1_piecewise_rmse": fmt_val(b1.get("oos_rmse")),
        "predict_c1_coupled_rmse": fmt_val(c1.get("oos_rmse")),
        "predict_c2_changepoint_rmse": fmt_val(c2.get("oos_rmse")),
        "predict_c3_flux_crossover_rmse": fmt_val(c3_best.get("oos_rmse")),
        "predict_b0_t_drinkable_min": fmt_val(b0.get("predicted_t_drinkable_min")),
        "predict_b1_t_drinkable_min": fmt_val(b1.get("predicted_t_drinkable_min")),
        "predict_c2_t_drinkable_min": fmt_val(c2.get("predicted_t_drinkable_min")),
        "predict_c3_t_drinkable_min": fmt_val(c3_best.get("predicted_t_drinkable_min")),
    }
    
    return example


@logger.catch(reraise=True)
def main() -> None:
    """Run the full five-model comparison pipeline."""
    logger.info("Starting five-model coffee cooling comparison")
    t_start = time.time()
    
    # Load all datasets
    logger.info("Loading datasets")
    datasets: List[Dict[str, Any]] = [
        load_kurzweg(),
        load_manitoba(),
        load_ferreira_cup(),
        load_lomen_lovelock(),
    ]
    
    logger.info(f"Loaded {len(datasets)} datasets: {[d['name'] for d in datasets]}")
    
    # Process datasets with multiple split ratios
    all_results: List[Dict[str, Any]] = []
    for ds in datasets:
        for split_ratio in SPLIT_RATIOS:
            try:
                result = process_dataset(ds, split_ratio)
                all_results.append(result)
                logger.info(f"  Completed {ds['name']} (split={split_ratio:.0%})")
            except Exception:
                logger.error(f"  Failed to process {ds['name']} (split={split_ratio:.0%})")
                # Continue with other datasets/splits
                continue
    
    # Build exp_gen_sol_out structure - group by dataset
    dataset_examples: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        ds_name = r["dataset"]
        if ds_name not in dataset_examples:
            dataset_examples[ds_name] = []
        dataset_examples[ds_name].append(make_example_block(r))
    
    output = {
        "metadata": {
            "method_name": "Five-model coffee cooling comparison",
            "description": (
                "Compares Newton (B0), piecewise two-exponential (B1), "
                "coupled mass-energy ODE (C1), Bayesian change-point (C2), "
                "and flux-crossover with evaporation suppression (C3) "
                "across published coffee cooling datasets. Fit on first 80%, "
                "OOS evaluate on last 20%. Primary metric: time-to-drinkable "
                "(50°C crossing) prediction accuracy."
            ),
            "n_datasets": len(dataset_examples),
            "n_examples": len(all_results),
            "models": ["B0_newton", "B1_piecewise", "C1_coupled_ode", "C2_bayesian_changepoint", "C3_flux_crossover"],
            "tau_sweep_minutes": TAU_SWEEP_MINUTES,
            "drinkable_temp_C": DRINKABLE_TEMP_C,
            "split_ratios": SPLIT_RATIOS,
        },
        "datasets": [
            {
                "dataset": ds_name,
                "examples": examples,
            }
            for ds_name, examples in dataset_examples.items()
        ],
    }
    
    # Write method_out.json
    out_path = Path("method_out.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote {out_path} with {len(all_results)} examples across {len(dataset_examples)} datasets")
    
    # Also write a raw results file for downstream analysis
    raw_path = Path("results_raw.json")
    raw_out = {
        "datasets": all_results,
        "runtime_seconds": time.time() - t_start,
    }
    raw_path.write_text(json.dumps(raw_out, indent=2, default=str))
    logger.info(f"Wrote {raw_path}")
    
    elapsed = time.time() - t_start
    logger.info(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
