# Parsimonious Crema-Breakdown Cooling Evaluation

This artifact evaluates a parsimonious crema-breakdown switching model with fixed tau on coffee cooling datasets, comparing against Newton baseline and free-transition models using out-of-sample RMSE, BIC, and drinkable-temperature crossing error with bootstrap CIs.

## Layout

| File | Description |
|------|-------------|
| `eval.py` | Main evaluation script |
| `eval_out.json` | Full evaluation results (exp_eval_sol_out schema) |
| `full_eval_out.json` | Full evaluation results copy |
| `mini_eval_out.json` | First 3 examples per dataset for quick testing |
| `preview_eval_out.json` | Truncated strings for quick inspection |
| `rederive.py` | Independent re-derivation script for verification |
| `pyproject.toml` | Pinned dependencies for reproducibility |
| `reproducibility.md` | Step-by-step reproduction guide |
| `fig1_heldout_rmse.pdf/png` | Heldout OOS RMSE by dataset and model |
| `fig2_bic.pdf/png` | BIC comparison by dataset and model |
| `fig3_cross_error.pdf/png` | Drinkable temperature (50°C) crossing time error |
| `fig4_newton_trajectory.pdf/png` | Newton's law 30-min trajectory error with 95% bootstrap CIs |
| `fig5_rmse_vs_bic.pdf/png` | Model trade-off: RMSE vs BIC scatter |
| `logs/` | Execution logs |

## Metrics Evaluated

Per the artifact plan:
1. **Held-out out-of-sample (OOS) RMSE**: Root mean squared error between model predictions and held-out temperature measurements across 30-minute cooling trajectories
2. **Bayesian Information Criterion (BIC)**: n ln(RSS/n) + k ln(n) evaluating goodness-of-fit penalized by parameter count k
3. **Drinkable-temperature crossing error**: Absolute difference in minutes between predicted and observed/target 50°C crossing time
4. **Newton 30-minute trajectory error bands**: Absolute error of Newton's law at 5-minute intervals (5, 10, 15, 20, 25, 30 min) with bootstrap confidence intervals

## Models Compared

- **B0_newton**: Newton single-exponential baseline (2 parameters)
- **B1_piecewise**: Piecewise two-exponential (5 parameters)
- **C1_coupled_ode**: Coupled mass-energy ODE with evaporation (3 parameters)
- **C2_bayesian_changepoint**: Bayesian change-point MAP estimate (3 parameters)
- **C3_flux_crossover**: Flux-crossover with evaporation suppression (3 parameters, tau fixed at 5.5 min from literature)

## Datasets

Four published coffee cooling datasets:
- `kurzweg_1988` (16 time points)
- `manitoba_workshop` (18 time points)
- `ferreira_2024_cup` (13 time points)
- `lomen_lovelock` (13 time points)

Each dataset evaluated with 4 split ratios (70%, 75%, 80%, 85%) = 16 total examples.

## Statistical Methods

- **Paired bootstrap CIs**: 5000 resamples for pairwise RMSE and crossing-time differences vs Newton baseline
- **BIC**: Standard formula with parameter penalty
- **Trajectory error**: Computed at 5-min intervals with bootstrap CIs

## Key Results

From the evaluation:
- **C2_bayesian_changepoint** achieves lowest OOS RMSE (1.66°C) and best crossing time error (0.54 min)
- **B1_piecewise** has best BIC (-15.5) due to flexible piecewise structure
- **Newton's law** trajectory error grows from ~0°C at t=0 to ~6°C at t=30 min
- All non-Newton models significantly outperform Newton on RMSE (bootstrap CIs exclude 0)

## Running the Evaluation

```bash
uv run eval.py
```

Dependencies: numpy, scipy, loguru (installed via uv).

## Restoring Removed Files

No files were removed - all artifacts are kept per manifest.