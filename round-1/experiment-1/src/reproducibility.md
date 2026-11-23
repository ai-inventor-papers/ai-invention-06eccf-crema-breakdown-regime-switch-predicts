# Reproducibility Instructions

## 1. Getting This Artifact

This workspace is published as one folder of a public GitHub repository. Clone the repository and navigate to this folder:

```bash
git clone <repository-url>
cd <repository-root>/round-1/experiment-1/src
```

## 2. Environment Setup

### System Requirements
- **OS**: Ubuntu 22.04+ (or any Linux/macOS with Python 3.12)
- **Python**: 3.12
- **Hardware**: CPU-only, 2 vCPUs, 16 GB RAM (Apple Silicon M-series or x86_64)
- **GPU**: Not required (no GPU used)

### Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies (pinned versions from pyproject.toml)

```bash
pip install loguru==0.7.3 matplotlib==3.10.7 numpy==2.2.6 pandas==2.3.3 scipy==1.16.3 jsonschema==4.25.1
```

## 3. Data and External Dependencies

**No external data downloads required.** All four datasets (Kurzweg 1988, Manitoba workshop, Ferreira 2024 cup, Lomen & Lovelock) are hardcoded in `method.py` as transcribed from their published sources. No API keys, environment variables, or checkpoint downloads are needed.

The data sources are:
1. Kurzweg UFL notes: `web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf`
2. Manitoba calculus workshop: `server.math.umanitoba.ca/~bruceha/ma2080/cooling.html`
3. Ferreira et al. 2024: Int. Commun. Heat Mass Transfer
4. Lomen & Lovelock: Stata coffee.dta file

## 4. Exact Commands Run

### Run the experiment

```bash
source .venv/bin/activate
python method.py
```

**Approximate runtime**: 25 seconds on 2 vCPUs (Apple Silicon).

**Hardware used**: CPU-only (Apple Silicon M-series, 2 vCPUs shared, 16 GB RAM). No GPU.

### Generate figure specs and render figures (optional)

```bash
# Figure specs are generated inline in the reproducibility verification step
# To render figures, you need the aii-data-fig-gen skill's chart_gen.py
# If available:
python /path/to/chart_gen.py --spec fig_bic_spec.json --out figures
python /path/to/chart_gen.py --spec fig_rmse_spec.json --out figures_rmse
python /path/to/chart_gen.py --spec fig_t50_spec.json --out figures_t50
```

### Validate output

```bash
python -c "from jsonschema import validate; import json; validate(json.load(open('method_out.json')), json.load(open('schemas/exp_gen_sol_out.json')))"
```

(Note: The schema file is at `/Users/Shared/aii_data/.claude/skills/aii-json/schemas/exp_gen_sol_out.json` on the original machine. For independent validation, install `jsonschema` and check the structure manually.)

## 5. Expected Output

### Files produced
| File | Size | Description |
|------|------|-------------|
| `method_out.json` | ~75 KB | Primary output (exp_gen_sol_out schema, 16 examples) |
| `full_method_out.json` | ~75 KB | Identical to method_out.json |
| `mini_method_out.json` | ~75 KB | First 3 datasets only |
| `preview_method_out.json` | ~4 KB | Mini with truncated strings |
| `results_raw.json` | ~75 KB | Raw numerical results |
| `logs/run.log` | ~1 KB | Execution log |
| `figures/` | ~100 KB | Rendered BIC comparison (PDF + PNG) |
| `figures_rmse.pdf/png` | ~100 KB | Rendered RMSE comparison |
| `figures_t50.pdf/png` | ~100 KB | Rendered t50 comparison |

### Key numbers a reader should get

**BIC scores** (lower = better; per dataset, 80% split):
- kurzweg_1988: B0=20.4, B1=-21.5, C1=-13.2, C2=-3.3, C3=-24.6
- manitoba_workshop: B0=43.0, B1=-14.4, C1=4.4, C2=4.8, C3=-9.6
- ferreira_2024_cup: B0=25.1, B1=-10.6, C1=2.1, C2=2.9, C3=-5.0
- lomen_lovelock: B0=21.0, B1=-16.0, C1=-6.3, C2=-2.7, C3=-16.1

**OOS RMSE** (°C, lower = better; 80% split):
- kurzweg_1988: B0=1.53, B1=1.36, C1=1.85, C2=0.64, C3=1.28
- manitoba_workshop: B0=5.62, B1=2.93, C1=5.06, C2=2.05, C3=2.54
- ferreira_2024_cup: B0=3.84, B1=2.16, C1=4.14, C2=0.87, C3=3.09
- lomen_lovelock: B0=4.13, B1=2.76, C1=3.89, C2=2.06, C3=2.73

**Predicted time to 50°C** (min; 80% split):
- kurzweg_1988: B0=15.9, B1=14.9, C1=14.9, C2=14.9, C3=14.9
- manitoba_workshop: B0=16.2, B1=16.5, C1=15.4, C2=17.4, C3=16.1
- ferreira_2024_cup: B0=18.2, B1=17.3, C1=17.0, C2=18.4, C3=17.1
- lomen_lovelock: B0=18.7, B1=18.7, C1=18.2, C2=19.3, C3=18.6

These numbers appear in `results_raw.json` under each dataset's `comparison_row`, and in `method_out.json` under each dataset's `examples[0]` predict_* fields (for the 80% split).

### Re-derivation

Run `rederive.py` to independently recompute all headline numbers from the raw data in `results_raw.json`:

```bash
python rederive.py
```

This script reads `results_raw.json`, recomputes BIC, OOS RMSE, and t(50°C) predictions through a completely different code path (direct numpy operations, not the model fitting functions), and prints whether each number matches.

The script now verifies all 16 examples (4 datasets × 4 split ratios).
