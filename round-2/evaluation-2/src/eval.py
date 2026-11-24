#!/usr/bin/env python3
"""
Evaluate the parsimonious crema-breakdown cooling model against Newton's law
and other baselines using the pre-registered protocol.

Metrics per artifact plan:
1. Held-out out-of-sample (OOS) RMSE
2. Bayesian Information Criterion (BIC)
3. Drinkable-temperature crossing error (|t_pred - t_obs| for 50°C)
4. Newton 30-minute trajectory error at 5-min intervals with bootstrap CIs
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings("ignore")

from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logger.add(logs_dir / "run.log", rotation="30 MB", level="DEBUG")

# Constants
DEPENDENCY_PATH = Path("/Volumes/aii-s20001/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
DRINKABLE_TEMP_C = 50.0
TAU_FIXED_MIN = 5.5  # Fixed tau from literature (Dunnett et al. 2025)
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42


def paired_bootstrap_ci(diffs, n_resamples=N_BOOTSTRAP, confidence=0.95, seed=BOOTSTRAP_SEED):
    """Paired bootstrap CI for mean of differences.
    diffs: array of per-dataset differences (e.g., RMSE_cand - RMSE_newton)
    Returns (lower, upper, mean) of the percentile CI.
    """
    if len(diffs) < 2:
        return None, None, float(np.mean(diffs)) if len(diffs) > 0 else None
    
    rng = np.random.default_rng(seed)
    n = len(diffs)
    bootstrap_means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        bootstrap_means.append(np.mean(np.array(diffs)[idx]))
    
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_means, 100 * alpha))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))
    mean_diff = float(np.mean(diffs))
    return lower, upper, mean_diff


def find_t_obs_cross(t_data, T_data, target_T=DRINKABLE_TEMP_C):
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


def newton_model(t_min, k, T0, Te):
    """Newton single-exponential cooling model."""
    return Te + (T0 - Te) * np.exp(-k * t_min * 60.0)  # t in seconds


def compute_bic(rss, n, k):
    """Compute BIC = n * ln(RSS/n) + k * ln(n)"""
    if n <= 0 or rss <= 0:
        return np.inf
    return n * math.log(rss / n) + k * math.log(n)


@logger.catch(reraise=True)
def main():
    logger.info("Starting parsimonious crema-breakdown cooling evaluation")
    
    # Load data from dependency
    method_out_path = DEPENDENCY_PATH / "full_method_out.json"
    logger.info(f"Loading method results from {method_out_path}")
    method_data = json.loads(method_out_path.read_text())
    
    # Load raw results for trajectory evaluation
    results_raw_path = DEPENDENCY_PATH / "results_raw.json"
    logger.info(f"Loading raw results from {results_raw_path}")
    raw_data = json.loads(results_raw_path.read_text())
    
    # Extract datasets and models
    datasets_info = method_data["datasets"]
    n_datasets = method_data["metadata"]["n_datasets"]
    split_ratios = method_data["metadata"]["split_ratios"]
    
    logger.info(f"Found {n_datasets} datasets with {len(split_ratios)} split ratios each")
    
    # Model names (matching comparison_row keys)
    models = ["B0_newton", "B1_piecewise", "C1_coupled_ode", "C2_bayesian_changepoint", "C3_flux_crossover"]
    # Comparison row uses short prefixes
    model_prefixes = {
        "B0_newton": "B0",
        "B1_piecewise": "B1",
        "C1_coupled_ode": "C1",
        "C2_bayesian_changepoint": "C2",
        "C3_flux_crossover": "C3_best",
    }
    
    # Per-dataset storage
    per_dataset_results = []
    
    # Aggregate metrics storage
    heldout_rmse = {m: [] for m in models}
    bic_values = {m: [] for m in models}
    cross_time_error = {m: [] for m in models}
    t_obs_list = []
    
    # Newton trajectory error at 5-min intervals (0, 5, 10, 15, 20, 25, 30)
    eval_times = [0, 5, 10, 15, 20, 25, 30]
    newton_traj_error = {t: [] for t in eval_times}
    
    # Process each dataset and split
    for ds_info in datasets_info:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]
        
        # Get raw data for this dataset
        raw_ds = next((d for d in raw_data["datasets"] if d["dataset"] == ds_name), None)
        if raw_ds is None:
            logger.warning(f"No raw data found for {ds_name}, skipping")
            continue
        
        # Extract full time and temperature arrays from raw data
        # The raw data has multiple splits; we need the original full data
        # Reconstruct from the experiment's data loading functions
        # We'll use the raw data's first entry to get the original points
        # Actually, let's load the original data directly using the same functions
        
        # For trajectory evaluation, we need the full temperature-time data
        # Let's load the dataset directly
        if ds_name == "kurzweg_1988":
            t_min = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
            T_C = np.array([88.0, 80.0, 73.0, 67.0, 62.0, 58.0, 54.0, 51.0, 48.0, 46.0, 44.0, 42.0, 41.0, 39.0, 38.0, 37.0], dtype=float)
            T_env = 21.0
            T0 = 88.0
        elif ds_name == "manitoba_workshop":
            t_min = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30], dtype=float)
            T_C = np.array([95.0, 89.0, 84.0, 79.0, 75.0, 71.0, 68.0, 65.0, 62.0, 60.0, 58.0, 55.0, 53.0, 51.0, 49.0, 48.0, 45.0, 43.0], dtype=float)
            T_env = 21.0
            T0 = 95.0
        elif ds_name == "ferreira_2024_cup":
            t_min = np.array([0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30], dtype=float)
            T_C = np.array([90.0, 82.0, 75.0, 69.0, 64.0, 60.0, 56.0, 52.0, 49.0, 47.0, 45.0, 44.0, 43.0], dtype=float)
            T_env = 22.0
            T0 = 90.0
        elif ds_name == "lomen_lovelock":
            t_min = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float)
            T_F = np.array([200.0, 185.0, 172.0, 161.0, 152.0, 144.0, 138.0, 132.0, 128.0, 124.0, 121.0, 115.0, 110.0])
            T_C = (T_F - 32.0) * 5.0 / 9.0
            T_env = 20.0
            T0 = T_C[0]
        else:
            logger.warning(f"Unknown dataset {ds_name}, skipping")
            continue
        
        # Find observed 50°C crossing time on full dataset
        t_obs_cross = find_t_obs_cross(t_min, T_C)
        t_obs_list.append(t_obs_cross)
        
        # Process each split example
        for ex in examples:
            # Parse output JSON
            output = json.loads(ex["output"])
            comparison = output.get("comparison_row", {})
            
            # Extract metrics per model
            for m in models:
                prefix = model_prefixes[m]
                
                # Heldout RMSE
                rmse_key = f"oos_rmse_{prefix}"
                if rmse_key in comparison:
                    heldout_rmse[m].append(comparison[rmse_key])
                
                # BIC
                bic_key = f"bic_{prefix}"
                if bic_key in comparison:
                    bic_values[m].append(comparison[bic_key])
                
                # Drinkable crossing time
                t_drink_key = f"t_drinkable_{prefix}_min"
                if t_drink_key in comparison:
                    t_pred = comparison[t_drink_key]
                    if t_obs_cross is not None and t_pred is not None and not (isinstance(t_pred, float) and np.isnan(t_pred)):
                        cross_time_error[m].append(abs(t_pred - t_obs_cross))
            
            # Newton trajectory error at evaluation times
            # Get Newton's fitted k for this split
            b0 = output.get("B0_newton", {})
            k = b0.get("k", None)
            if k is not None:
                for t in eval_times:
                    T_pred = newton_model(t, k, T0, T_env)
                    # Find observed temperature at this time via interpolation
                    T_obs = np.interp(t, t_min, T_C)
                    newton_traj_error[t].append(abs(T_pred - T_obs))
        
        # Store per-dataset summary
        per_dataset_results.append({
            "dataset": ds_name,
            "n_points": len(t_min),
            "t_env": T_env,
            "t0": T0,
            "t_obs_cross_min": t_obs_cross,
            "n_splits": len(examples),
        })
    
    # Filter out None values for crossing time error
    for m in models:
        cross_time_error[m] = [e for e in cross_time_error[m] if e is not None and not (isinstance(e, float) and np.isnan(e))]
    
    logger.info("=== Per-model aggregate metrics ===")
    for m in models:
        n_rmse = len(heldout_rmse[m])
        n_bic = len(bic_values[m])
        n_cross = len(cross_time_error[m])
        if n_rmse > 0:
            logger.info(f"{m}: RMSE={np.mean(heldout_rmse[m]):.3f}±{np.std(heldout_rmse[m]):.3f} (n={n_rmse}), "
                       f"BIC={np.mean(bic_values[m]):.1f} (n={n_bic}), "
                       f"CrossErr={np.mean(cross_time_error[m]):.2f}±{np.std(cross_time_error[m]):.2f} (n={n_cross})")
    
    # Bootstrap CIs for pairwise differences vs Newton (B0_newton)
    ref_model = "B0_newton"
    bootstrap_results = {}
    
    for m in models:
        if m == ref_model:
            continue
        
        # RMSE differences: RMSE_m - RMSE_newton (positive means Newton better)
        min_len = min(len(heldout_rmse[m]), len(heldout_rmse[ref_model]))
        if min_len >= 2:
            rmse_diffs = [heldout_rmse[m][i] - heldout_rmse[ref_model][i] for i in range(min_len)]
            lower, upper, mean_diff = paired_bootstrap_ci(rmse_diffs)
            bootstrap_results[f"rmse_{m}_vs_{ref_model}"] = {
                "mean_diff": mean_diff,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_pairs": min_len
            }
            logger.info(f"  RMSE {m} - {ref_model}: mean={mean_diff:.3f}, 95% CI=[{lower:.3f}, {upper:.3f}]")
        
        # Crossing time error differences
        min_len = min(len(cross_time_error[m]), len(cross_time_error[ref_model]))
        if min_len >= 2:
            cross_diffs = [cross_time_error[m][i] - cross_time_error[ref_model][i] for i in range(min_len)]
            lower_c, upper_c, mean_c = paired_bootstrap_ci(cross_diffs)
            bootstrap_results[f"cross_{m}_vs_{ref_model}"] = {
                "mean_diff": mean_c,
                "ci_lower": lower_c,
                "ci_upper": upper_c,
                "n_pairs": min_len
            }
            logger.info(f"  Cross error {m} - {ref_model}: mean={mean_c:.2f}, 95% CI=[{lower_c:.2f}, {upper_c:.2f}]")
    
    # Newton trajectory error bootstrap CIs at each evaluation time
    newton_traj_ci = {}
    for t in eval_times:
        errors = newton_traj_error[t]
        if len(errors) >= 2:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            n = len(errors)
            bootstrap_means = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, n, n)
                bootstrap_means.append(np.mean(np.array(errors)[idx]))
            
            alpha = 0.025
            lower = float(np.percentile(bootstrap_means, 100 * alpha))
            upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))
            mean_err = float(np.mean(errors))
            
            newton_traj_ci[t] = {
                "mean": mean_err,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": n
            }
            logger.info(f"  Newton traj error at {t}min: mean={mean_err:.3f}°C, 95% CI=[{lower:.3f}, {upper:.3f}]")
    
    # Per-candidate summary for metadata
    per_candidate_summary = {}
    for m in models:
        per_candidate_summary[m] = {
            "mean_heldout_rmse": float(np.mean(heldout_rmse[m])) if heldout_rmse[m] else None,
            "std_heldout_rmse": float(np.std(heldout_rmse[m])) if heldout_rmse[m] else None,
            "mean_bic": float(np.mean(bic_values[m])) if bic_values[m] else None,
            "std_bic": float(np.std(bic_values[m])) if bic_values[m] else None,
            "mean_cross_time_error": float(np.mean(cross_time_error[m])) if cross_time_error[m] else None,
            "std_cross_time_error": float(np.std(cross_time_error[m])) if cross_time_error[m] else None,
            "n_rmse": len(heldout_rmse[m]),
            "n_bic": len(bic_values[m]),
            "n_cross_valid": len(cross_time_error[m]),
        }
    
    # Aggregate metrics
    metrics_agg = {}
    for m in models:
        if heldout_rmse[m]:
            metrics_agg[f"mean_heldout_rmse_{m}"] = float(np.mean(heldout_rmse[m]))
        if bic_values[m]:
            metrics_agg[f"mean_bic_{m}"] = float(np.mean(bic_values[m]))
        if cross_time_error[m]:
            metrics_agg[f"mean_cross_time_error_{m}"] = float(np.mean(cross_time_error[m]))
    
    # Newton trajectory aggregate
    for t in eval_times:
        if newton_traj_error[t]:
            metrics_agg[f"newton_traj_error_{t}min_mean"] = float(np.mean(newton_traj_error[t]))
            if t in newton_traj_ci:
                metrics_agg[f"newton_traj_error_{t}min_ci_lower"] = newton_traj_ci[t]["ci_lower"]
                metrics_agg[f"newton_traj_error_{t}min_ci_upper"] = newton_traj_ci[t]["ci_upper"]
    
    # Build output in exp_eval_sol_out format
    datasets_out = []
    
    for ds_info in datasets_info:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]
        
        examples_out = []
        for ex in examples:
            output = json.loads(ex["output"])
            comparison = output.get("comparison_row", {})
            
            ex_out = {
                "input": ex["input"],
                "output": ex["output"],
                "metadata_dataset": ex["metadata_dataset"],
                "metadata_n_points": ex["metadata_n_points"],
                "metadata_split_fit": ex["metadata_split_fit"],
                "metadata_split_hold": ex["metadata_split_hold"],
            }
            
            # Add predictions and eval metrics
            for m in models:
                prefix = model_prefixes[m]
                # Predictions from original
                rmse_pred_key = f"predict_{m.lower().replace('_', '_')}_rmse"
                if rmse_pred_key in ex:
                    ex_out[f"predict_{m}_rmse"] = ex[rmse_pred_key]
                
                t_drink_key = f"predict_{m.lower().replace('_', '_')}_t_drinkable_min"
                if t_drink_key in ex:
                    ex_out[f"predict_{m}_t_drinkable_min"] = ex[t_drink_key]
                
                # Eval metrics
                rmse_key = f"oos_rmse_{prefix}"
                if rmse_key in comparison:
                    ex_out[f"eval_{m}_oos_rmse"] = float(comparison[rmse_key])
                
                bic_key = f"bic_{prefix}"
                if bic_key in comparison:
                    ex_out[f"eval_{m}_bic"] = float(comparison[bic_key])
                
                t_drink_comp_key = f"t_drinkable_{prefix}_min"
                if t_drink_comp_key in comparison:
                    t_pred = comparison[t_drink_comp_key]
                    # Find observed crossing for this dataset
                    t_obs = next((d["t_obs_cross_min"] for d in per_dataset_results if d["dataset"] == ds_name), None)
                    if t_obs is not None and t_pred is not None and not (isinstance(t_pred, float) and np.isnan(t_pred)):
                        ex_out[f"eval_{m}_cross_time_error"] = float(abs(t_pred - t_obs))
            
            examples_out.append(ex_out)
        
        datasets_out.append({
            "dataset": ds_name,
            "examples": examples_out
        })
    
    eval_out = {
        "metadata": {
            "evaluation_name": "Parsimonious Crema-Breakdown Cooling Evaluation",
            "description": "Evaluates crema-breakdown switching model with fixed tau (5.5 min) on coffee cooling datasets, comparing against Newton baseline and free-transition models using out-of-sample RMSE, BIC, and drinkable-temperature crossing error with bootstrap CIs.",
            "models": models,
            "reference_model": ref_model,
            "n_datasets": n_datasets,
            "drinkable_temp_C": DRINKABLE_TEMP_C,
            "tau_fixed_min": TAU_FIXED_MIN,
            "n_bootstrap": N_BOOTSTRAP,
            "per_candidate_summary": per_candidate_summary,
            "bootstrap_cis": bootstrap_results,
            "newton_trajectory_error": newton_traj_ci,
            "per_dataset": per_dataset_results,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out
    }
    
    # Write eval_out.json
    out_path = Path("eval_out.json")
    out_path.write_text(json.dumps(eval_out, indent=2, default=str))
    logger.info(f"Wrote eval_out.json to {out_path}")
    
    # ============================================================
    # FIGURE GENERATION
    # ============================================================
    
    # Figure 1: Heldout OOS RMSE grouped bar chart
    fig1_spec = {
        "type": "bar",
        "title": "Heldout OOS RMSE by Dataset and Model",
        "xlabel": "Dataset",
        "ylabel": "Heldout OOS RMSE (°C)",
        "aspect": "16:9",
        "categories": [d["dataset"] for d in per_dataset_results],
        "series": []
    }
    
    # We need to get per-dataset per-model RMSE values
    # The per_dataset_results doesn't have the split-level data; we need to reconstruct
    # Let's compute mean RMSE per dataset per model
    ds_rmse = {d["dataset"]: {m: [] for m in models} for d in per_dataset_results}
    
    for ds_info in datasets_info:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]
        for ex in examples:
            output = json.loads(ex["output"])
            comparison = output.get("comparison_row", {})
            for m in models:
                prefix = model_prefixes[m]
                rmse_key = f"oos_rmse_{prefix}"
                if rmse_key in comparison:
                    ds_rmse[ds_name][m].append(comparison[rmse_key])
    
    for m in models:
        values = []
        for ds in per_dataset_results:
            vals = ds_rmse[ds["dataset"]][m]
            if vals:
                values.append(float(np.mean(vals)))
            else:
                values.append(0.0)
        fig1_spec["series"].append({
            "label": m.replace("_", " ").title(),
            "values": values,
            "errors": None
        })
    
    # Figure 2: BIC comparison
    fig2_spec = {
        "type": "bar",
        "title": "BIC by Dataset and Model (Lower is Better)",
        "xlabel": "Dataset",
        "ylabel": "BIC",
        "aspect": "16:9",
        "categories": [d["dataset"] for d in per_dataset_results],
        "series": []
    }
    
    ds_bic = {d["dataset"]: {m: [] for m in models} for d in per_dataset_results}
    for ds_info in datasets_info:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]
        for ex in examples:
            output = json.loads(ex["output"])
            comparison = output.get("comparison_row", {})
            for m in models:
                prefix = model_prefixes[m]
                bic_key = f"bic_{prefix}"
                if bic_key in comparison:
                    ds_bic[ds_name][m].append(comparison[bic_key])
    
    for m in models:
        values = []
        for ds in per_dataset_results:
            vals = ds_bic[ds["dataset"]][m]
            if vals:
                values.append(float(np.mean(vals)))
            else:
                values.append(0.0)
        fig2_spec["series"].append({
            "label": m.replace("_", " ").title(),
            "values": values,
            "errors": None
        })
    
    # Figure 3: Drinkable crossing time error
    fig3_spec = {
        "type": "bar",
        "title": "Drinkable Temperature (50°C) Crossing Time Error",
        "xlabel": "Dataset",
        "ylabel": "Absolute Crossing Time Error (min)",
        "aspect": "16:9",
        "categories": [d["dataset"] for d in per_dataset_results],
        "series": []
    }
    
    ds_cross = {d["dataset"]: {m: [] for m in models} for d in per_dataset_results}
    for ds_info in datasets_info:
        ds_name = ds_info["dataset"]
        examples = ds_info["examples"]
        for ex in examples:
            output = json.loads(ex["output"])
            comparison = output.get("comparison_row", {})
            t_obs = next((d["t_obs_cross_min"] for d in per_dataset_results if d["dataset"] == ds_name), None)
            if t_obs is not None:
                for m in models:
                    prefix = model_prefixes[m]
                    t_drink_key = f"t_drinkable_{prefix}_min"
                    if t_drink_key in comparison:
                        t_pred = comparison[t_drink_key]
                        if t_pred is not None and not (isinstance(t_pred, float) and np.isnan(t_pred)):
                            ds_cross[ds_name][m].append(abs(t_pred - t_obs))
    
    for m in models:
        values = []
        for ds in per_dataset_results:
            vals = ds_cross[ds["dataset"]][m]
            if vals:
                values.append(float(np.mean(vals)))
            else:
                values.append(None)
        # Filter out None for plotting
        valid_values = [v for v in values if v is not None]
        if valid_values:
            fig3_spec["series"].append({
                "label": m.replace("_", " ").title(),
                "values": [v if v is not None else 0.0 for v in values],
                "errors": None
            })
    
    # Figure 4: Newton trajectory error with bootstrap CIs
    fig4_spec = {
        "type": "line",
        "title": "Newton's Law 30-min Trajectory Error with 95% Bootstrap CIs",
        "xlabel": "Time (min)",
        "ylabel": "Absolute Temperature Error (°C)",
        "aspect": "16:9",
        "series": []
    }
    
    x_vals = eval_times
    means = [newton_traj_ci[t]["mean"] if t in newton_traj_ci else 0.0 for t in eval_times]
    # Use band = half-width of CI for symmetric error band
    bands = []
    for t in eval_times:
        if t in newton_traj_ci:
            ci_half = (newton_traj_ci[t]["ci_upper"] - newton_traj_ci[t]["ci_lower"]) / 2.0
            bands.append(ci_half)
        else:
            bands.append(0.0)
    
    fig4_spec["series"].append({
        "label": "Newton Trajectory Error",
        "x": x_vals,
        "values": means,
        "band": bands[0] if len(set(bands)) == 1 else bands,  # Use scalar if all same, else array
        "marker": "o",
        "linestyle": "-",
    })
    
    # Figure 5: Model comparison - RMSE vs BIC tradeoff
    fig5_spec = {
        "type": "scatter",
        "title": "Model Trade-off: Heldout RMSE vs BIC",
        "xlabel": "Mean BIC (Lower Better)",
        "ylabel": "Mean Heldout OOS RMSE (°C, Lower Better)",
        "aspect": "16:9",
        "series": []
    }
    
    for m in models:
        mean_bic = np.mean(bic_values[m]) if bic_values[m] else 0
        mean_rmse = np.mean(heldout_rmse[m]) if heldout_rmse[m] else 0
        fig5_spec["series"].append({
            "label": m.replace("_", " ").title(),
            "x": [float(mean_bic)],
            "y": [float(mean_rmse)],
            "marker": "o"
        })
    
    # Render figures using aii-data-fig-gen
    import subprocess
    import tempfile
    
    figures = [
        ("fig1_heldout_rmse", fig1_spec),
        ("fig2_bic", fig2_spec),
        ("fig3_cross_error", fig3_spec),
        ("fig4_newton_trajectory", fig4_spec),
        ("fig5_rmse_vs_bic", fig5_spec),
    ]
    
    for fig_name, fig_spec in figures:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(fig_spec, f)
            spec_path = f.name
        
        fig_out = Path(fig_name)
        result = subprocess.run([
            sys.executable, 
            "/Volumes/aii-s20001/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
            "--spec", spec_path, "--out", str(fig_out), "--format", "pdf,png"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Figure {fig_name} render failed: {result.stderr}")
        else:
            logger.info(f"Rendered {fig_name}.pdf")
        
        Path(spec_path).unlink()
    
    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()