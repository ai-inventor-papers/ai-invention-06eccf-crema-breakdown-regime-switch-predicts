#!/usr/bin/env python3
"""Evaluate the cooling model screen results per the pre-registered protocol."""
import json
import math
import numpy as np
from pathlib import Path
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings("ignore")

WORKSPACE = Path("/Volumes/aii-s20007/work/run_qZEiMJexNbcK/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1")

def paired_bootstrap_ci(diffs, n_resamples=5000, confidence=0.95, seed=42):
    """Paired bootstrap CI for mean of differences.
    diffs: array of per-dataset differences (e.g., RMSE_cand - RMSE_survivor)
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
    # Load data
    dataset = json.loads((WORKSPACE / "full_data_out.json").read_text())
    method_results = json.loads((WORKSPACE / "method_out.json").read_text())
    
    # Group dataset by source
    source_data = {}
    for row in dataset:
        sid = row["metadata"]["source_id"]
        source_data.setdefault(sid, []).append(row)
    
    candidates = ["B0", "B1", "C1", "C2", "C3"]
    n_datasets = len(method_results)
    
    # Extract per-candidate heldout metrics
    heldout_rmse = {c: [] for c in candidates}
    delta_bic = {c: [] for c in candidates}
    cross_time_error = {c: [] for c in candidates}
    t_obs_cross_list = []
    t_pred_cross = {c: [] for c in candidates}
    
    # Also collect C3 tau sweep data
    c3_tau_sweep = {sid: {} for sid in sorted(source_data.keys())}
    
    for dr in method_results:
        sid = dr["source_id"]
        rows = source_data[sid]
        T_env = dr["T_env"]
        T0 = dr["T0"]
        
        # Observed crossing time on full dataset
        all_t = np.array([r["input"]["time_min"] for r in rows])
        all_T = np.array([r["input"]["temperature_C"] for r in rows])
        t_obs = find_t_obs_cross(all_t, all_T, 50.0)
        t_obs_cross_list.append(t_obs)
        
        for c in candidates:
            cr = dr["candidates"][c]
            # Heldout OOS RMSE
            heldout_rmse[c].append(cr["rmse_heldout"])
            # Delta BIC vs B0
            delta_bic[c].append(cr["delta_bic_vs_b0"])
            # Crossing time error
            t_pred = cr.get("t_pred_cross")
            t_pred_cross[c].append(t_pred)
            if t_pred is not None and t_obs is not None:
                cross_time_error[c].append(abs(t_pred - t_obs))
            else:
                cross_time_error[c].append(None)
        
        # C3 tau sweep
        c3 = dr["candidates"]["C3"]
        if "tau_heldout_rmse" in c3:
            c3_tau_sweep[sid] = c3["tau_heldout_rmse"]
    
    # Filter out None values for crossing time error
    for c in candidates:
        cross_time_error[c] = [e for e in cross_time_error[c] if e is not None]
    
    # Print summary
    print("=== Per-candidate heldout metrics ===")
    for c in candidates:
        print(f"{c}: heldout_RMSE={np.mean(heldout_rmse[c]):.3f}±{np.std(heldout_rmse[c]):.3f}, "
              f"mean_deltaBIC={np.mean(delta_bic[c]):.1f}, "
              f"cross_error={np.mean(cross_time_error[c]):.2f}±{np.std(cross_time_error[c]):.2f} "
              f"({len(cross_time_error[c])}/{n_datasets} datasets)")
    
    # Survival rule: lowest heldout OOS RMSE on >=80% of datasets AND mean delta-BIC > 10
    # Find which candidate has lowest RMSE on each dataset
    lowest_rmse_wins = {c: 0 for c in candidates}
    for i in range(n_datasets):
        best_c = min(candidates, key=lambda c: heldout_rmse[c][i])
        lowest_rmse_wins[best_c] += 1
    
    win_rates = {c: lowest_rmse_wins[c] / n_datasets for c in candidates}
    mean_delta_bic = {c: np.mean(delta_bic[c]) for c in candidates}
    
    print("\n=== Survival Rule Check ===")
    print(f"N datasets = {n_datasets}")
    print(f"80% threshold = {0.80 * n_datasets:.1f} wins required (i.e., {int(np.ceil(0.80 * n_datasets))} out of {n_datasets})")
    survivors = []
    for c in candidates:
        survives = (lowest_rmse_wins[c] >= 0.80 * n_datasets) and (mean_delta_bic[c] > 10.0)
        print(f"  {c}: wins={lowest_rmse_wins[c]}, win_rate={win_rates[c]:.2f}, "
              f"mean_dBIC={mean_delta_bic[c]:.1f}, survives={survives}")
        if survives:
            survivors.append(c)
    
    if not survivors:
        print("\nNO SURVIVORS - honest null result")
        # Find closest candidate (best heldout RMSE among those with positive delta-BIC)
        eligible = [c for c in candidates if mean_delta_bic[c] > 0]
        if eligible:
            closest = min(eligible, key=lambda c: np.mean(heldout_rmse[c]))
            print(f"Closest candidate: {closest} (mean heldout RMSE={np.mean(heldout_rmse[closest]):.3f})")
        else:
            closest = min(candidates, key=lambda c: np.mean(heldout_rmse[c]))
            print(f"Closest candidate (all negative delta-BIC): {closest}")
        survivors = []  # empty
    else:
        closest = survivors[0] if survivors else None
    
    # Bootstrap CIs for pairwise differences vs survivor (or closest if no survivor)
    ref_candidate = survivors[0] if survivors else closest
    print(f"\n=== Bootstrap CIs (5000 draws) vs {ref_candidate} ===")
    
    bootstrap_results = {}
    for c in candidates:
        if c == ref_candidate:
            continue
        # RMSE differences: RMSE_c - RMSE_ref (so positive means ref is better)
        rmse_diffs = [heldout_rmse[c][i] - heldout_rmse[ref_candidate][i] for i in range(n_datasets)]
        lower, upper, mean_diff = paired_bootstrap_ci(rmse_diffs)
        bootstrap_results[f"rmse_{c}_vs_{ref_candidate}"] = {
            "mean_diff": mean_diff,
            "ci_lower": lower,
            "ci_upper": upper,
            "n_datasets": n_datasets
        }
        print(f"  RMSE {c} - {ref_candidate}: mean={mean_diff:.3f}, 95% CI=[{lower:.3f}, {upper:.3f}]")
        
        # Crossing time error differences
        if len(cross_time_error[c]) == len(cross_time_error[ref_candidate]) == n_datasets:
            cross_diffs = [cross_time_error[c][i] - cross_time_error[ref_candidate][i] for i in range(n_datasets)]
            lower_c, upper_c, mean_c = paired_bootstrap_ci(cross_diffs)
            bootstrap_results[f"cross_{c}_vs_{ref_candidate}"] = {
                "mean_diff": mean_c,
                "ci_lower": lower_c,
                "ci_upper": upper_c,
                "n_datasets": n_datasets
            }
            print(f"  Cross error {c} - {ref_candidate}: mean={mean_c:.2f}, 95% CI=[{lower_c:.2f}, {upper_c:.2f}]")
    
    # C3 tau ablation
    print("\n=== C3 Tau Ablation ===")
    c3_ablation = {}
    for dr in method_results:
        sid = dr["source_id"]
        c3 = dr["candidates"]["C3"]
        tau_mle = c3.get("tau_mle", 6.0)
        
        # Heldout RMSE at tau_mle
        tau_heldout = c3.get("tau_heldout_rmse", {})
        rmse_at_mle = tau_heldout.get(str(tau_mle), None)
        
        # Best sweep heldout RMSE
        if tau_heldout:
            best_tau = min(tau_heldout, key=tau_heldout.get)
            best_rmse = tau_heldout[best_tau]
        else:
            best_tau = tau_mle
            best_rmse = rmse_at_mle
        
        ablation_effect = (rmse_at_mle - best_rmse) if rmse_at_mle is not None and best_rmse is not None else None
        c3_ablation[sid] = {
            "tau_mle": tau_mle,
            "heldout_rmse_at_mle": rmse_at_mle,
            "best_sweep_tau": float(best_tau),
            "best_sweep_heldout_rmse": best_rmse,
            "ablation_effect": ablation_effect
        }
        print(f"  {sid}: tau_mle={tau_mle}, RMSE@MLE={rmse_at_mle:.3f}, "
              f"best_sweep_tau={best_tau} RMSE={best_rmse:.3f}, "
              f"ablation={ablation_effect:.3f}")
    
    # Overall ablation effect
    ablation_effects = [c3_ablation[s]["ablation_effect"] for s in c3_ablation if c3_ablation[s]["ablation_effect"] is not None]
    mean_ablation = np.mean(ablation_effects) if ablation_effects else None
    
    # Power note
    min_wins_needed = int(np.ceil(0.80 * n_datasets))
    print(f"\n=== Power Note ===")
    print(f"N datasets = {n_datasets}")
    print(f"Survival rule: >=80% wins => {min_wins_needed} out of {n_datasets} required")
    if n_datasets == 3:
        print("  At N=3, 80% rule requires 3/3 = unanimity (100%), not majority.")
        print("  This is a known limitation: a candidate best on 2/3 fails the screen.")
    elif n_datasets == 4:
        print("  At N=4, 80% rule requires 4/4 = unanimity (100%).")
    elif n_datasets == 5:
        print("  At N=5, 80% rule requires 4/5 = 80% (majority).")
    
    # Minimum detectable effect size
    # Using bootstrap CI width for RMSE differences
    rmse_cis = [v for k, v in bootstrap_results.items() if k.startswith("rmse_")]
    if rmse_cis:
        ci_widths = [v["ci_upper"] - v["ci_lower"] for v in rmse_cis]
        avg_ci_width = np.mean(ci_widths)
        # MDE at 95% power ≈ 2.8 * SEM (for paired test)
        # SEM ≈ CI_width / (2 * 1.96) for 95% CI
        mde = 2.8 * avg_ci_width / (2 * 1.96)
        print(f"  Average 95% CI width on RMSE diff: {avg_ci_width:.3f} °C")
        print(f"  Approx MDE at 95% power (paired): {mde:.3f} °C")
    
    # Build output in exp_eval_sol_out schema format
    # Required: metrics_agg (object of numbers), datasets (array of {dataset, examples})
    
    # Aggregate metrics
    metrics_agg = {
        "n_datasets": n_datasets,
        "n_survivors": len(survivors),
        "closest_candidate_rank": 1 if closest else 0,
        "mean_heldout_rmse_b0": float(np.mean(heldout_rmse["B0"])),
        "mean_heldout_rmse_b1": float(np.mean(heldout_rmse["B1"])),
        "mean_heldout_rmse_c1": float(np.mean(heldout_rmse["C1"])),
        "mean_heldout_rmse_c2": float(np.mean(heldout_rmse["C2"])),
        "mean_heldout_rmse_c3": float(np.mean(heldout_rmse["C3"])),
        "mean_delta_bic_b0": 0.0,
        "mean_delta_bic_b1": float(np.mean(delta_bic["B1"])),
        "mean_delta_bic_c1": float(np.mean(delta_bic["C1"])),
        "mean_delta_bic_c2": float(np.mean(delta_bic["C2"])),
        "mean_delta_bic_c3": float(np.mean(delta_bic["C3"])),
        "mean_cross_error_b0": float(np.mean(cross_time_error["B0"])) if cross_time_error["B0"] else 0.0,
        "mean_cross_error_b1": float(np.mean(cross_time_error["B1"])) if cross_time_error["B1"] else 0.0,
        "mean_cross_error_c1": float(np.mean(cross_time_error["C1"])) if cross_time_error["C1"] else 0.0,
        "mean_cross_error_c2": float(np.mean(cross_time_error["C2"])) if cross_time_error["C2"] else 0.0,
        "mean_cross_error_c3": float(np.mean(cross_time_error["C3"])) if cross_time_error["C3"] else 0.0,
        "best_candidate_mean_rmse": float(min(np.mean(heldout_rmse[c]) for c in candidates)),
        "c3_mean_ablation": float(mean_ablation) if mean_ablation is not None else 0.0,
        "bootstrap_ci_width_rmse": float(np.mean(ci_widths)) if rmse_cis else 0.0,
    }
    
    # Build datasets array with per-example eval_* and predict_* fields
    datasets_out = []
    for dr in method_results:
        sid = dr["source_id"]
        examples = []
        for row in source_data[sid]:
            ex = {
                "input": json.dumps(row["input"]),
                "output": json.dumps(row["output"]),
                "metadata_fold": row["metadata_fold"],
                "metadata_source_id": sid,
                "metadata_ambient_temperature_C": row["metadata"]["ambient_temperature_C"],
            }
            # Add predict_* and eval_* fields for each candidate
            # Find the index of this row within the dataset
            t_val = row["input"]["time_min"]
            T_obs = row["input"]["temperature_C"]
            fold = row["metadata_fold"]
            
            # Get the index within this dataset's rows
            idx_in_dataset = None
            for ri, r in enumerate(source_data[sid]):
                if r["input"]["time_min"] == t_val and r["metadata_fold"] == fold:
                    idx_in_dataset = ri
                    break
            
            if idx_in_dataset is not None:
                for c in candidates:
                    cr = dr["candidates"][c]
                    # Get predicted temperature at this time point
                    if fold == "screen":
                        # Find index in screen rows
                        screen_rows = [r for r in source_data[sid] if r["metadata_fold"] == "screen"]
                        screen_idx = None
                        for si, sr in enumerate(screen_rows):
                            if sr["input"]["time_min"] == t_val:
                                screen_idx = si
                                break
                        if screen_idx is not None and screen_idx < len(cr["T_pred_screen"]):
                            T_pred = cr["T_pred_screen"][screen_idx]
                        else:
                            T_pred = None
                    else:
                        # heldout
                        heldout_rows = [r for r in source_data[sid] if r["metadata_fold"] == "heldout"]
                        heldout_idx = None
                        for hi, hr in enumerate(heldout_rows):
                            if hr["input"]["time_min"] == t_val:
                                heldout_idx = hi
                                break
                        if heldout_idx is not None and heldout_idx < len(cr["T_pred_heldout"]):
                            T_pred = cr["T_pred_heldout"][heldout_idx]
                        else:
                            T_pred = None
                    
                    if T_pred is not None:
                        ex[f"predict_{c.lower()}"] = str(round(T_pred, 4))
                        ex[f"eval_{c.lower()}_abs_error"] = round(abs(T_obs - T_pred), 4)
                        ex[f"eval_{c.lower()}_squared_error"] = round((T_obs - T_pred) ** 2, 6)
            
            examples.append(ex)
        datasets_out.append({
            "dataset": sid,
            "examples": examples
        })
    
    eval_out = {
        "metadata": {
            "evaluation_name": "Pre-registered model-screen evaluation with reserved-fold confirmation",
            "description": "Evaluation of five cooling model candidates (B0-B1-C1-C2-C3) on a panel of 3 independent published datasets. Applies pre-registered survival rule: lowest heldout OOS RMSE on >=80% of datasets AND mean delta-BIC > 10 over Newton baseline B0. Paired 5000-draw bootstrap 95% CIs on pairwise RMSE and crossing-time differences. C3 tau-ablation at in-sample MLE. Power note on N=3 panel.",
            "candidates": candidates,
            "survivors": survivors,
            "closest_candidate": closest,
            "n_datasets": n_datasets,
            "per_candidate_summary": {
                c: {
                    "mean_heldout_rmse": float(np.mean(heldout_rmse[c])),
                    "std_heldout_rmse": float(np.std(heldout_rmse[c])),
                    "mean_delta_bic_vs_b0": float(np.mean(delta_bic[c])),
                    "mean_cross_time_error": float(np.mean(cross_time_error[c])) if cross_time_error[c] else None,
                    "std_cross_time_error": float(np.std(cross_time_error[c])) if cross_time_error[c] else None,
                    "n_cross_valid": len(cross_time_error[c])
                } for c in candidates
            },
            "survival_rule": {
                "threshold_win_rate": 0.80,
                "threshold_delta_bic": 10.0,
                "wins_per_candidate": lowest_rmse_wins,
                "win_rates": {k: float(v) for k, v in win_rates.items()},
                "mean_delta_bic": {k: float(v) for k, v in mean_delta_bic.items()},
                "survives": {k: (lowest_rmse_wins[k] >= 0.80 * n_datasets and mean_delta_bic[k] > 10.0) for k in candidates}
            },
            "bootstrap_cis": bootstrap_results,
            "c3_tau_ablation": c3_ablation,
            "c3_ablation_mean_effect": float(mean_ablation) if mean_ablation is not None else 0.0,
            "power_note": {
                "n_datasets": n_datasets,
                "min_wins_for_80pct": min_wins_needed,
                "effective_rule": "unanimity" if n_datasets <= 4 else ">=80%",
                "avg_bootstrap_ci_width_rmse": float(np.mean(ci_widths)) if rmse_cis else 0.0,
                "approx_mde_95pct_power": float(mde) if rmse_cis else 0.0
            },
            "per_dataset": [
                {
                    "source_id": method_results[idx]["source_id"],
                    "heldout_rmse": {c: float(heldout_rmse[c][idx]) for c in candidates},
                    "delta_bic_vs_b0": {c: float(delta_bic[c][idx]) for c in candidates},
                    "t_pred_cross": {c: method_results[idx]["candidates"][c].get("t_pred_cross") for c in candidates},
                    "t_obs_cross": t_obs_cross_list[idx]
                } for idx in range(n_datasets)
            ]
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out
    }
    
    # Write eval_out.json
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(eval_out, indent=2))
    print(f"\nWrote eval_out.json to {out_path}")
    
    # ============================================================
    # FIGURE GENERATION
    # ============================================================
    
    # Figure 1: Grouped bar of heldout OOS RMSE per candidate with 95% CI
    # Use aii-data-fig-gen
    
    # Prepare data for bar chart
    categories = candidates
    series = []
    colors = ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#d55e00"]
    
    for i, c in enumerate(candidates):
        series.append({
            "label": c,
            "values": [float(np.mean(heldout_rmse[c]))],
            "errors": [float(1.96 * np.std(heldout_rmse[c]) / np.sqrt(n_datasets))]  # SEM * 1.96 for 95% CI
        })
    
    # For a grouped bar we need one value per candidate with error bars
    # Actually the figure should show per-dataset RMSEs or mean with CI
    # Let's do: grouped bar with one group per dataset, bars per candidate
    # That's more informative
    
    # Figure 1: Per-dataset heldout RMSE grouped bar
    fig1_spec = {
        "type": "bar",
        "title": "Heldout OOS RMSE by Dataset and Candidate",
        "xlabel": "Dataset",
        "ylabel": "Heldout OOS RMSE (°C)",
        "aspect": "16:9",
        "categories": [dr["source_id"] for dr in method_results],
        "series": []
    }
    
    for i, c in enumerate(candidates):
        fig1_spec["series"].append({
            "label": c,
            "values": [float(heldout_rmse[c][j]) for j in range(n_datasets)],
            "errors": None  # single point per dataset, no error
        })
    
    # Figure 2: C3 tau sweep profile (in-sample RMSE vs tau)
    fig2_spec = {
        "type": "line",
        "title": "C3 In-Sample OOS RMSE vs Tau Sweep",
        "xlabel": "Tau (min)",
        "ylabel": "In-Sample OOS RMSE (°C)",
        "aspect": "16:9",
        "series": []
    }
    
    for dr in method_results:
        sid = dr["source_id"]
        c3 = dr["candidates"]["C3"]
        sweep = c3.get("tau_sweep", [])
        if sweep:
            taus = [s["tau"] for s in sweep]
            rmses = [s["rmse"] for s in sweep]
            fig2_spec["series"].append({
                "label": sid,
                "values": rmses,
                "x": taus,
                "marker": "o"
            })
    
    # Also mark MLE
    mle_series = {
        "label": "MLE",
        "values": [c3["tau_mle"] for c3 in [dr["candidates"]["C3"] for dr in method_results]],
        "x": [dr["candidates"]["C3"]["tau_mle"] for dr in method_results],
        "marker": "s"
    }
    # Actually add MLE markers as a separate series
    
    # Write figure specs and render
    import sys
    
    # Render Figure 1 using CLI
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(fig1_spec, f)
        spec1_path = f.name
    
    fig1_out = WORKSPACE / "fig1_heldout_rmse"
    result = subprocess.run([
        sys.executable, "/Volumes/aii-s20007/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
        "--spec", spec1_path, "--out", str(fig1_out), "--format", "pdf,png"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Figure 1 render failed: {result.stderr}")
    else:
        print(f"Rendered Figure 1 to {fig1_out}.pdf")
    Path(spec1_path).unlink()
    
    # Render Figure 2
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(fig2_spec, f)
        spec2_path = f.name
    
    fig2_out = WORKSPACE / "fig2_c3_tau_sweep"
    result = subprocess.run([
        sys.executable, "/Volumes/aii-s20007/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py",
        "--spec", spec2_path, "--out", str(fig2_out), "--format", "pdf,png"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Figure 2 render failed: {result.stderr}")
    else:
        print(f"Rendered Figure 2 to {fig2_out}.pdf")
    Path(spec2_path).unlink()
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()