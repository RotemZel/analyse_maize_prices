# AGRICAF — Stage 2: Predictor Screening for Maize Prices

This repository contains the code for **Stage 2 (Predictor Screening)** of the [AGRICAF](https://github.com/rotemzelingher/AGRICAF) pipeline, applied to **maize**. Stage 2 identifies which predictors matter for forecasting maize prices and discards the rest, before any forecasting takes place (Stage 3).

---

## What Stage 2 does

Commodity price datasets contain many candidate predictors — market indicators, supply-side variables, and their lags — relative to a limited number of yearly observations. Fitting a forecasting model directly on such a dataset risks overfitting. Stage 2 addresses this by running a retrospective analysis: models are fitted to historical data and evaluated out-of-sample, and variable-importance scores are extracted to identify which predictors contribute most to accurate predictions.

**Procedure for each combination of calendar month `m` and forecast horizon `h` (both ranging from 1 to 12):**

1. Load the processed input dataset (`analyse_data.RData`) containing candidate predictors at lags up to 12 months.
2. For parametric models (LM, GAM), remove predictors with pairwise correlations > 0.9 to prevent collinearity from destabilising estimates.
3. Fit five models — Random Forest (RF), Gradient Boosted Machine (GBM), CART, Linear Regression (LM), and Generalized Additive Model (GAM) — using **leave-one-out cross-validation (LOO-CV)**: each year from 2000 onward is held out in turn; the model is trained on the remaining years; and the absolute prediction error for the held-out year is recorded.
4. Within each LOO-CV fold, perform **hyperparameter tuning via grid search** (inside the loop to avoid data leakage). For GBM, a two-phase procedure first does a grid search over tree count, interaction depth, and learning rate, then calls `gbm.perf` with the test method to refine the boosting iteration count.
5. Extract **variable-importance scores** from the best-performing hyperparameter configuration in each fold. Each model uses its own importance metric: RF uses %IncMSE (permutation importance), GBM uses relative influence from split-gain reduction, CART uses reduction in residual sum of squares, LM uses the absolute *t*-statistic |β̂/SE(β̂)|, and GAM uses an analogous smoothing-parameter-based measure.
6. Normalise importance scores to [0, 1] within each fold (replacing negative values from RF with 0) to make scores comparable across models.
7. Save all results for downstream aggregation in Stage 3.

The full procedure is run separately for **4 configurations** of maize:

| Configuration | Geographic level | Supply variable |
|---|---|---|
| `maize_country/analyse_production` | Country | Production |
| `maize_country/analyse_yield` | Country | Yield |
| `maize_region/analyse_production` | Region | Production |
| `maize_region/analyse_yield` | Region | Yield |

Each configuration generates **720 model runs** (12 months × 12 horizons × 5 models), parallelised on a SLURM cluster as a job array.

---

## Repository structure

```
analyse_maize_prices/
│
├── setup_analyse_maize_country_production.sh   # Generates parameter file & submits SLURM job
├── setup_analyse_maize_country_yield.sh
├── setup_analyse_maize_region_production.sh
├── setup_analyse_maize_region_yield.sh
│
├── task_analyse_maize_country_production.slurm  # SLURM job array script
├── task_analyse_maize_country_yield.slurm
├── task_analyse_maize_region_production.slurm
├── task_analyse_maize_region_yield.slurm
│
└── maize_prices/
    ├── maize_country/
    │   ├── analyse_processed/
    │   │   └── analyse_data.RData              # Processed input data (country level)
    │   ├── analyse_production/
    │   │   ├── 2_regression_analyse_rf.R       # Random Forest screening
    │   │   ├── 3_regression_analyse_gbm.R      # Gradient Boosted Machine screening
    │   │   ├── 4_regression_analyse_cart.R     # CART screening
    │   │   ├── 5_regression_analyse_lm.R       # Linear regression (stepwise AIC) screening
    │   │   └── 6_regression_analyse_gam.R      # GAM screening
    │   └── analyse_yield/
    │       └── [same 5 R scripts as above]
    └── maize_region/
        ├── analyse_processed/
        │   └── analyse_data.RData              # Processed input data (regional level)
        ├── analyse_production/
        │   └── [same 5 R scripts]
        └── analyse_yield/
            └── [same 5 R scripts]
```

---

## How to run

### Prerequisites

- R ≥ 4.5 with packages: `caret`, `randomForest`, `gbm`, `rpart`, `broom`, `car`, `mgcv`, `tidyverse`
- A SLURM-based HPC cluster
- The AGRICAF framework installed at `~/price_forecasting/` with commodity master script `master_cmaf_maize.R`

### Running a configuration

From the `price_forecasting/` root directory, run the corresponding setup script:

```bash
bash analyse_maize_prices/setup_analyse_maize_country_production.sh
```

This will:
1. Generate a `params_maize_analyse_country_production.txt` file listing all 720 month–horizon–model combinations.
2. Submit a SLURM job array (up to 100 concurrent tasks) that runs each combination independently.

Each SLURM task calls the relevant R script with three arguments passed as inline `Rscript -e` assignments:
- `pmonth`: prediction month (1–12)
- `lags`: forecast horizon in months (1–12)
- `c`: commodity name (`"maize"`)

Repeat for the other three configurations (`country_yield`, `region_production`, `region_yield`).

### Output

Results are saved to `analyse_processed/` within each configuration folder. These outputs feed directly into Stage 3 (forecasting) of the AGRICAF pipeline.

---

## Part of AGRICAF

This code is part of the **AGRICAF** (Agricultural Commodity price Analytical and Forecasting) framework. AGRICAF is a four-stage pipeline:

| Stage | Task |
|---|---|
| 1 | Assemble and harmonise raw data |
| **2** | **Screen predictors (this repository)** |
| 3 | Generate 1–12 month ahead forecasts |
| 4 | Explain forecasts (variable importance, interpretability) |

---

## Citation

If you use this code, please cite:

> Zelingher, R. (*in preparation*). *AGRICAF: An open-source framework for forecasting global agricultural commodity prices*. 

---

## License

[MIT License](LICENSE)
