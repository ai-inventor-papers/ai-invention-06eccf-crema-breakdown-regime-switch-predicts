# Coffee Cooling Model Evaluation

Pre-registered model-screen evaluation comparing five cooling-model candidates on a panel of 3 independent published datasets.

## What This Does

Evaluates whether any candidate model beats Newton's single-exponential law (B0) at predicting the temperature of a cup of coffee over 30 minutes, using heldout out-of-sample RMSE and time-to-drinkable-temperature (50°C) crossing error.

## Layout

| File | Description |
|------|-------------|
| `eval.py` | Main evaluation script |
| `eval_out.json` | Primary output: survival rule results, bootstrap CIs, ablation table |
| `full_data_out.json` | Dataset panel (3 datasets, 44 rows, screen/holdout folds) |
| `method_out.json` | Experiment fit table (5 candidates × 3 datasets) |
| `fig1_heldout_rmse.pdf` | Grouped bar: heldout OOS RMSE per candidate per dataset |
| `fig2_c3_tau_sweep.pdf` | Line chart: C3 in-sample RMSE vs tau sweep (3–8 min) |
| `generate_dataset.py` | Dataset generation script |
| `generate_method.py` | Experiment fitting script |
| `mini_*.json` / `preview_*.json` | Size-optimized variants for quick inspection |

## How to Run

```bash
cd /path/to/workspace
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install numpy scipy pandas loguru matplotlib psutil jsonschema
python eval.py
```

## Key Results

- **N datasets**: 3 (Manitoba, Kurzweg UFL 2017, Ferreira cup 2024)
- **Survivors**: None — the 80% rule requires unanimity at N=3, and no candidate won on all 3 datasets
- **Closest candidate**: C1 (coupled temperature-mass ODE) with mean heldout RMSE = 1.819°C
- **Best heldout RMSE**: C3 (flux-crossover) at 0.275°C but with negative delta-BIC (−13.0), failing the BIC screen
- **Power note**: At N=3, the 80% rule degenerates to 100% (unanimity); the minimum detectable effect at 95% power is ~2.5°C

## Restoring Removed Files

```bash
# Restore .venv
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install numpy scipy pandas loguru matplotlib psutil jsonschema
```

All other files are kept on the run's volume.
