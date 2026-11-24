# Reproducibility Guide: Parsimonious Crema-Breakdown Cooling Evaluation

This document provides complete step-by-step instructions to reproduce the evaluation results on Ubuntu.

## 1. Getting This Artifact

This artifact is published as part of a public GitHub repository. Clone the repository and navigate to this folder:

```bash
git clone <repository-url>
cd <repository-root>/round-2/evaluation-2/src
```

## 2. System Requirements

- **OS**: Ubuntu 22.04 or later (macOS also works with adjustments)
- **Python**: 3.12+
- **System packages**: None strictly required (all Python packages installed via uv)

## 3. Environment Setup

### Create virtual environment and install dependencies

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install pinned dependencies
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Or manually install the pinned versions:

```bash
source .venv/bin/activate
uv pip install \
    numpy==2.2.6 \
    scipy==1.16.3 \
    loguru==0.7.3 \
    matplotlib==3.10.7 \
    jsonschema==4.25.1
```

## 4. Required Input Data

This evaluation depends on the experiment artifact `art_hPG4i7tI7B4v` (Five-Model Coffee Cooling Comparison). The repository publishes it as a sibling folder:

```
../gen_art_experiment_1/
```

You need these files from that artifact:
- `full_method_out.json` - Model fitting results with metrics
- `results_raw.json` - Raw temperature-time data and split definitions

If you don't have the sibling folder, you must run the experiment artifact first to generate them.

## 5. Running the Evaluation

```bash
# From this directory (gen_art_evaluation_2)
source .venv/bin/activate
python eval.py
```

### Expected Runtime
- **Hardware**: CPU-only (no GPU required)
- **Time**: ~15 seconds on modern CPU
- **Memory**: < 100 MB

### What the Script Does
1. Loads model results from `../gen_art_experiment_1/full_method_out.json`
2. Loads raw data from `../gen_art_experiment_1/results_raw.json`
3. Computes per-model metrics across 4 datasets × 4 splits = 16 examples
4. Runs 5000 paired bootstrap resamples for confidence intervals
5. Generates 5 figures via aii-data-fig-gen
6. Outputs `eval_out.json`, `mini_eval_out.json`, `preview_eval_out.json`

## 6. Expected Output Files

After running, you should get:

| File | Size | Description |
|------|------|-------------|
| `eval_out.json` / `full_eval_out.json` | ~111 KB | Full evaluation results (exp_eval_sol_out schema) |
| `mini_eval_out.json` | ~85 KB | First 3 examples per dataset |
| `preview_eval_out.json` | ~29 KB | Truncated strings for inspection |
| `fig1_heldout_rmse.pdf/png` | ~17/52 KB | Heldout OOS RMSE by dataset and model |
| `fig2_bic.pdf/png` | ~15/47 KB | BIC comparison by dataset and model |
| `fig3_cross_error.pdf/png` | ~16/55 KB | Drinkable temperature crossing time error |
| `fig4_newton_trajectory.pdf/png` | ~11/63 KB | Newton trajectory error with 95% bootstrap CIs |
| `fig5_rmse_vs_bic.pdf/png` | ~15/60 KB | Model trade-off: RMSE vs BIC scatter |

## 7. Key Results to Verify

### Model Performance (mean across 16 examples)

| Model | OOS RMSE (°C) | BIC | Crossing Error (min) |
|-------|--------------|-----|---------------------|
| B0_newton (baseline) | 4.043 | 24.3 | 1.09 |
| B1_piecewise | 2.650 | -15.5 | 0.56 |
| C1_coupled_ode | 3.893 | -5.2 | 0.95 |
| C2_bayesian_changepoint | **1.661** | 0.1 | **0.54** |
| C3_flux_crossover (τ=5.5 fixed) | 2.626 | -15.2 | 0.67 |

### Bootstrap CIs vs Newton (5000 draws)

All non-Newton models except C1 significantly outperform Newton on RMSE (CIs exclude 0):
- B1: mean diff = -1.393, CI = [-1.853, -0.965] ✓
- C1: mean diff = -0.151, CI = [-0.375, +0.068] ✗ (includes 0)
- C2: mean diff = -2.382, CI = [-2.896, -1.864] ✓
- C3: mean diff = -1.417, CI = [-1.959, -0.924] ✓

### Newton Trajectory Error
Peaks at 30 min: 6.02°C (CI [4.61, 7.45]), confirming systematic deviation from single-exponential cooling.

## 8. Validation

### Schema Validation
```bash
python -c "
import json
from jsonschema import validate
with open('/Volumes/aii-s20001/.claude/skills/aii-json/schemas/exp_eval_sol_out.json') as f:
    schema = json.load(f)
with open('eval_out.json') as f:
    data = json.load(f)
validate(instance=data, schema=schema)
print('SCHEMA VALIDATION PASSED')
"
```

### Re-derivation Script
The file `rederive.py` (if present) provides an independent verification of headline numbers using a different code path.

## 9. Troubleshooting

- **Missing input files**: Ensure `../gen_art_experiment_1/full_method_out.json` and `results_raw.json` exist
- **Module import errors**: Verify `uv pip install` completed successfully with pinned versions
- **Figure generation fails**: Ensure aii-data-fig-gen skill is available at `/Volumes/aii-s20001/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py`

## 10. Notes

- The evaluation uses a **fixed tau = 5.5 min** for the C3 flux-crossover model (from Dunnett et al. 2025 literature)
- **5000 bootstrap resamples** with seed 42 for reproducibility
- **Drinkable temperature** defined as 50°C crossing
- **4 datasets**: kurzweg_1988, manitoba_workshop, ferreira_2024_cup, lomen_lovelock
- **4 split ratios per dataset**: 70%, 75%, 80%, 85% (16 total examples)