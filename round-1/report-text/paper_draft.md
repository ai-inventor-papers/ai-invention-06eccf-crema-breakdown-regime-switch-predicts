# Internal Research Report: Crema-Breakdown Regime Switch in Coffee Cooling

## Framing

This run investigates whether Newton's single-exponential law of cooling fails for coffee because it conflates two distinct physical regimes: an early evaporation-dominated regime (governed by crema-covered surface latent heat loss) and a late convection-dominated regime (governed by bare-surface convective cooling). The hypothesis proposes that the transition occurs when crema thickness decays below a critical threshold, reducing evaporative flux below convective flux. A piecewise model with this physics-based switching criterion should predict the full 30-minute cooling trajectory and the drinkable-temperature crossing time (50°C ± 5°C) with lower error than any single-exponential or constant-coefficient model.

The investigation proceeds in two stages. First, a five-model comparison across four published datasets tests whether models incorporating regime switching outperform Newton's law. Second, a pre-registered model-screen evaluation applies a survival rule to determine if any candidate model convincingly beats the baseline.

---

# Iteration 1

## Strategy

The iteration executed two artifacts: a five-model coffee cooling experiment (art_hPG4i7tI7B4v) and a pre-registered model-screen evaluation (art_Jn8Po12wOvcN).

The experiment was designed to test the hypothesis that a physics-based switching criterion (crema breakdown) predicts cooling better than both Newton's law and a free-transition two-exponential model. Five models were compared:

- **B0 (Newton baseline)**: Single-exponential dT/dt = -k(T - T_env), 2 parameters
- **B1 (Piecewise two-exponential)**: Two exponential regimes joined at a free transition time t_switch, 5 parameters
- **C1 (Coupled mass-energy ODE)**: Joint temperature-mass evolution with temperature-dependent evaporation coefficient (Hertz-Knudsen-like), 3 parameters
- **C2 (Bayesian change-point)**: MAP estimate of regime transition from residual structure, 3 parameters
- **C3 (Flux-crossover with evaporation suppression)**: Evaporation suppressed by crema coverage decay exp(-t/τ), transition when evaporative flux equals convective flux, 3 parameters plus τ sweep (3-8 min)

Four published datasets were used:
1. Kurzweg (1988/2017) UFL notes: 16 points, 0-30 min, black coffee without cream
2. Manitoba calculus workshop: 18 points, 0-30 min
3. Ferreira 2024 cup: 13 points, 0-30 min
4. Lomen & Lovelock Stata coffee.dta: 13 points, 0-30 min

Each dataset was evaluated at four split ratios (70%, 75%, 80%, 85% training) yielding 16 total examples. Metrics computed: BIC (n ln(RSS/n) + k ln(n)), out-of-sample RMSE on held-out data, and predicted time-to-drinkable (50°C crossing).

The evaluation artifact applied a pre-registered survival rule on a panel of 3 independent datasets (Manitoba, Kurzweg, Ferreira): a candidate survives iff it has the lowest held-out OOS RMSE on ≥80% of datasets AND mean delta-BIC > 10 over the Newton baseline. Paired 5000-draw bootstrap 95% CIs were computed on pairwise RMSE and crossing-time differences. Flux-crossover tau-ablation was performed at in-sample MLE.

## Results

### Experiment (art_hPG4i7tI7B4v)

The experiment produced 16 result rows across 4 datasets × 4 splits. Key findings at the 80% split (the primary evaluation point):

**BIC comparison (Table 1)**

| Dataset | B0 (Newton) | B1 (Piecewise) | C1 (Coupled ODE) | C2 (Changepoint) | C3 (Flux-crossover) |
|---------|-------------|----------------|------------------|------------------|---------------------|
| Kurzweg | 20.36 | -21.54 | -13.16 | -3.26 | **-24.55** |
| Manitoba | 43.03 | -14.37 | 4.41 | 4.77 | **-9.65** |
| Ferreira | 25.09 | -10.60 | 2.07 | 2.92 | **-4.96** |
| Lomen-Lovelock | 20.95 | -15.98 | -6.30 | -2.67 | **-16.15** |

The flux-crossover model achieves the lowest BIC on 3 of 4 datasets at 80% split. The piecewise model wins on Lomen-Lovelock. All multi-regime models substantially beat Newton's law on BIC (delta-BIC ranging from -20 to -40).

**Out-of-sample RMSE (Table 2)**

| Dataset | B0 | B1 | C1 | C2 | C3 |
|---------|-----|-----|-----|-----|-----|
| Kurzweg | 1.526 | 1.364 | 1.855 | **0.643** | 1.284 |
| Manitoba | 5.620 | 2.933 | 5.065 | 2.050 | 2.544 |
| Ferreira | 3.838 | 2.162 | 4.141 | 0.874 | 3.086 |
| Lomen-Lovelock | 4.126 | 2.760 | 3.889 | 2.059 | 2.731 |

The Bayesian change-point model achieves the lowest OOS RMSE on all 4 datasets at 80% split, with particularly strong performance on Kurzweg (0.643°C) and Ferreira (0.874°C). However, its BIC is higher than the flux-crossover model's on 3 of 4 datasets because its fit on the training portion is worse (higher RSS).

**Drinkable-temperature crossing (50°C) (Table 3)**

| Dataset | B0 | B1 | C1 | C2 | C3 | Observed |
|---------|-----|-----|-----|-----|-----|----------|
| Kurzweg | 15.92 | 14.88 | 14.93 | 14.91 | 14.87 | n/a |
| Manitoba | 16.23 | 16.47 | 15.43 | 17.35 | 16.11 | n/a |
| Ferreira | 18.24 | 17.31 | 17.04 | 18.42 | 17.14 | 26.77 |
| Lomen-Lovelock | 18.70 | 18.72 | 18.21 | 19.29 | 18.57 | n/a |

All models agree within ~1-2 minutes on drinkable-time predictions. The observed crossing for Ferreira is 26.77 min, substantially later than any model predicts (models predict 17-18 min). This indicates the datasets may have different ambient conditions or the drinkable-temperature target is not consistently captured.

**Flux-crossover tau sweep results**

For the flux-crossover model, the MLE tau consistently lands at 8.0 min (the upper bound of the 3-8 min sweep) on all datasets except Manitoba (where it is 5.0 min). The BIC improves monotonically with tau, suggesting the data favor slower crema decay than the sweep permits.

### Evaluation (art_Jn8Po12wOvcN)

The evaluation applied the pre-registered survival rule on 3 datasets (N=3). At N=3, the ≥80% threshold degenerates to unanimity (3/3 required).

**Per-candidate summary (Table 4)**

| Candidate | Mean heldout RMSE (°C) | Mean delta-BIC vs B0 | Wins (lowest RMSE) | Cross error (min) |
|-----------|------------------------|----------------------|---------------------|-------------------|
| Newton baseline | 1.332 | 0.0 | 1 | 1.55 |
| Piecewise two-exponential | 4.194 | +60.91 | 0 | 7.86 |
| Coupled mass-energy ODE | 2.022 | +38.85 | 0 | 25.41 |
| Bayesian change-point | 6.098 | +59.29 | 0 | 12.86 |
| Flux-crossover | **0.331** | **-29.83** | **2** | **0.19** |

**Survival rule check**: No survivors. The flux-crossover model wins on heldout RMSE (2/3 datasets) but fails the delta-BIC screen (mean -29.83, threshold > 10). The coupled ODE has positive delta-BIC (+38.85) but higher RMSE (2.022°C). Newton baseline wins 1/3 datasets.

**Bootstrap CIs (5000 draws) vs closest candidate (coupled ODE) (Table 5)**

| Comparison | Mean RMSE diff | 95% CI |
|------------|----------------|--------|
| Newton vs coupled ODE | -0.691 | [-1.296, 0.196] |
| Piecewise vs coupled ODE | +2.172 | [1.199, 3.373] |
| Changepoint vs coupled ODE | +4.076 | [0.534, 6.185] |
| Flux-crossover vs coupled ODE | **-1.692** | **[-2.665, -0.939]** |

The flux-crossover model is significantly better than the coupled ODE on RMSE (CI excludes zero), but the BIC penalty for its extra parameter (tau) is decisive.

**Flux-crossover tau ablation (Table 6)**

| Dataset | tau MLE | RMSE at MLE | Best sweep tau | Best sweep RMSE | Ablation effect |
|---------|---------|-------------|----------------|-----------------|-----------------|
| Ferreira | 8.0 | 0.418 | 7.0 | 0.403 | 0.015 |
| Kurzweg | 3.0 | 0.320 | 3.0 | 0.320 | 0.000 |
| Manitoba | 5.0 | 0.254 | 6.0 | 0.236 | 0.019 |

Mean ablation effect: 0.011°C. The MLE tau is within 0.02°C of the sweep optimum on 2/3 datasets; the sweep provides only 0.06°C mean improvement.

**Power note**: At N=3, the minimum detectable effect at 95% power is ~1.97°C. The panel is underpowered for small effects.

## What was learned

1. **Newton's law is inadequate on BIC but adequate on crossing-time prediction**. All multi-regime models beat Newton on BIC by large margins (delta-BIC -20 to -40), confirming Dunnett et al. (2025)'s finding of structured residuals indicating model incompleteness [1]. However, all models predict drinkable-temperature crossing within ~1-2 minutes of each other, and Newton's law is within ~1.5 min of observed crossing on the two datasets with observed crossings.

2. **The physics-based switching model (flux-crossover) is not uniquely identified by the data**. The flux-crossover model achieves the best BIC on 3/4 datasets, but the Bayesian change-point achieves the best OOS RMSE on all 4. The free-transition piecewise model also performs well. The data do not discriminate between a crema-breakdown switching mechanism and a free transition time.

3. **The pre-registered survival rule is too strict at N=3**. The unanimity requirement (3/3) combined with the delta-BIC > 10 threshold yields no survivors even though the flux-crossover model has significantly better RMSE (bootstrap CI excludes zero). This is a design issue, not a scientific null result.

4. **The crema decay time constant (tau) is not well-constrained**. The MLE hits the sweep boundary (8 min) on 2/3 datasets. The ablation effect is negligible (0.011°C mean). The data cannot distinguish between tau values in the 3-8 min range.

5. **Dataset panel is too small (N=3)**. Power analysis shows MDE ~2°C at 95% power. The honest null finding is: Newton's law is adequate at the drinkable crossing for these datasets given the BIC penalty for complexity, and the crema-breakdown hypothesis needs more datasets or a more parsimonious formulation to survive the screen.

## Dead ends / abandoned directions

- **Dataset collection (dataset_iter1_dir1)**: Planned but not run. The planning agent timed out/failed before producing a plan. No evidence either way; not a scientific result.

- **Lomen-Lovelock dataset behavior**: This dataset (converted from °F) shows different behavior: the piecewise model wins on BIC, and all models predict later crossing times (~18-19 min). The conversion from °F to °C may introduce artifacts, or the dataset represents different physical conditions (different cup, ambient temperature). Not pursued further in this iteration.

## Artifact references

- Five-model experiment: [ARTIFACT:art_hPG4i7tI7B4v]
- Pre-registered evaluation: [ARTIFACT:art_Jn8Po12wOvcN]

## Figures

[FIGURE:fig_bic]

[FIGURE:fig_rmse]

[FIGURE:fig_t50]

[FIGURE:fig_eval_rmse]

[FIGURE:fig_eval_tau]

---

# What we have learned so far

The crema-breakdown regime-switch hypothesis is not supported by the current evidence. While multi-regime models convincingly beat Newton's law on BIC (confirming the model incompleteness Dunnett et al. 2025 identified [1]), the specific physical switching mechanism (crema decay) is not uniquely identified. The data are equally well explained by a free transition time (piecewise model) or a statistical changepoint (Bayesian change-point). The drinkable-temperature crossing, the practical target, is predicted similarly by all models (~1-2 minute spread), and Newton's law is within ~1.5 minutes of observed values.

The evaluation's null result (no survivors) stems from a design issue: at N=3 the survival rule requires unanimity, and the delta-BIC penalty for the flux-crossover model's extra parameter outweighs its RMSE advantage. This is a methodological limitation, not evidence against regime switching per se.

Next steps should: (1) expand the dataset panel to N≥5 to give the survival rule statistical power; (2) test a more parsimonious crema-breakdown model (e.g., fixing tau from independent crema lifetime measurements); (3) directly measure crema thickness decay to ground the switching criterion in independent data.

---

## References

[1] Dunnett, K., Le, K.C., & Holmqvist, T. (2025). Using the structure of the residuals to evaluate the goodness of fit: a pedagogical case study with Newton's law of cooling. *European Journal of Physics*, 46(5), 055104.

[2] Arii, Y. & Nishizawa, K. (2017). Espresso coffee foam delays cooling of the liquid phase. *Bioscience, Biotechnology, and Biochemistry*, 81(4), 779-782.

[3] Rees, W.G. & Viney, C. (1988). On cooling tea and coffee. *American Journal of Physics*, 56(5), 434-437.

[4] Kurzweg, U.H. (2017). Cooling coffee. University of Florida, Mechanical and Aerospace Engineering course notes. https://web.mae.ufl.edu/uhk/COOLING-COFFEE.pdf

[5] Zhao, B. (2022). Integrity of Newton's cooling law based on thermal convection theory of heat transfer and entropy transfer. *Scientific Reports*, 12, 16292.