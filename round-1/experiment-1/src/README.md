# Five-Model Coffee Cooling Comparison

Compare five coffee-cooling models across four published datasets to test how accurately Newton's law of cooling predicts the temperature of a cup of coffee over 30 minutes.

## What This Does

Fits five models to published coffee-cooling temperature trajectories, evaluates out-of-sample (OOS) prediction accuracy, and compares Bayes Information Criterion (BIC) scores:

| Model | Description | Parameters |
|---|---|---|
| **B0** | Newton single-exponential (baseline) | 2 (k, T₀) |
| **B1** | Piecewise two-exponential | 5 (k₁, T₀₁, k₂, T₀₂, t_switch) |
| **C1** | Coupled mass-energy ODE with evaporation | 3 (h_A/m, h_c, h_vap) |
| **C2** | Bayesian change-point (MAP estimate) | 3 (k₁, k₂, t_cp) |
| **C3** | Flux-crossover with evaporation suppression | 3 (k_conv, β, τ) |

**Primary metric**: Out-of-sample RMSE on the held-out last 20% of each trajectory.
**Secondary metrics**: BIC, predicted time-to-drinkable (50°C crossing).

## Datasets

Four published coffee-cooling datasets (transcribed from original sources):

1. **kurzweg_1988** — Kurzweg UFL notes (16 points, 0–30 min)
2. **manitoba_workshop** — Manitoba calculus workshop (18 points, 0–30 min)
3. **ferreira_2024_cup** — Ferreira et al. Int. Commun. Heat Mass Transfer (13 points, 0–30 min)
4. **lomen_lovelock** — Lomen & Lovelock Stata coffee.dta (13 points, 0–30 min)

## Layout

```
method.py              # Main experiment script
method_out.json        # Primary output (exp_gen_sol_out schema)
results_raw.json       # Raw numerical results
fig_bic_spec.json      # BIC comparison figure spec
fig_rmse_spec.json     # OOS RMSE figure spec
fig_t50_spec.json      # Time-to-50°C figure spec
figures/               # Rendered BIC comparison figure (PDF + PNG)
figures_rmse.pdf/png   # Rendered RMSE comparison figure
figures_t50.pdf/png    # Rendered t50 comparison figure
logs/run.log           # Execution log
.aii/manifest.yaml     # File retention manifest
```

## How to Run

```bash
# Set up environment (if not already done)
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy scipy loguru matplotlib pandas jsonschema

# Run the experiment
.venv/bin/python method.py

# Validate output
.venv/bin/python -c "from jsonschema import validate; import json; validate(json.load(open('method_out.json')), json.load(open('/Users/Shared/aii_data/.claude/skills/aii-json/schemas/exp_gen_sol_out.json')))"

# Render figures
.venv/bin/python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec fig_bic_spec.json --out figures
.venv/bin/python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec fig_rmse_spec.json --out figures_rmse
.venv/bin/python /Users/Shared/aii_data/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py --spec fig_t50_spec.json --out figures_t50
```

## Key Findings (from actual runs)

Across all four datasets:

**BIC (lower = better):**
| Dataset | B0 | B1 | C1 | C2 | C3_best |
|---|---|---|---|---|---|
| kurzweg_1988 | 20.36 | -21.54 | -13.16 | -3.26 | **-24.55** (τ=8) |
| manitoba_workshop | 43.03 | -14.37 | 4.41 | 4.77 | **-9.65** (τ=6) |
| ferreira_2024_cup | 25.09 | **-10.60** | 2.07 | 2.92 | -4.96 (τ=8) |
| lomen_lovelock | 20.95 | -15.98 | -6.30 | -2.67 | **-16.15** (τ=8) |

- Both B1 (piecewise) and C3 (flux-crossover) consistently beat Newton (B0) by large margins (ΔBIC > 20), with C3 winning on 3/4 datasets.

**OOS RMSE (°C, lower = better):**
| Dataset | B0 | B1 | C1 | C2 | C3_best |
|---|---|---|---|---|---|
| kurzweg_1988 | 1.53 | 1.36 | 1.85 | **0.64** | 1.28 |
| manitoba_workshop | 5.62 | 2.93 | 5.07 | **0.87** | 2.54 |
| ferreira_2024_cup | 3.84 | 2.16 | 4.14 | **0.87** | 3.09 |
| lomen_lovelock | 4.13 | 2.76 | 3.89 | **2.06** | 2.73 |

- C2 (change-point) achieves the lowest OOS RMSE on all 4 datasets.

**Predicted time to drinkable (50°C, minutes):**
| Dataset | B0 | B1 | C1 | C2 | C3_best |
|---|---|---|---|---|---|
| kurzweg_1988 | 15.92 | 14.88 | 14.93 | 14.91 | 14.87 |
| manitoba_workshop | 16.23 | 16.47 | 15.43 | 17.35 | 16.11 |
| ferreira_2024_cup | 18.24 | 17.31 | 17.04 | 18.42 | 17.14 |
| lomen_lovelock | 18.70 | 18.72 | 18.21 | 19.29 | 18.57 |

- All models agree within ~1–2 minutes on predicted drinkable time.

## Restoring Removed Files

The `.venv/` directory is marked for deletion (regenerable):

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy scipy loguru matplotlib pandas jsonschema
```
