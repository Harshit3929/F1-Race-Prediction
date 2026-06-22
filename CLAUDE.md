# CLAUDE.md — F1 Race Prediction

## Project Overview

**Goal:** Predict F1 race finishing positions using qualifying data, historical stats, and enriched features.
**Primary metric:** MAE (current best: 2.12), Spearman correlation (current best: 0.83)
**Test set:** 2025 season

---

## Folder Structure

```
f1_prediction/
└── f1_race_prediciton/           ← typo in name — do NOT rename, paths are hardcoded
    ├── src/
    │   ├── collect_data.py           # FastF1 API → raw results (seasons 2022–2025)
    │   ├── clean_data.py             # raw → clean_race_data.csv
    │   ├── feature_engineering.py    # rolling windows, ELO, form features
    │   ├── collect_extra_features.py # quali gap-to-pole, weather, tire compound → enriched CSV
    │   ├── train_model.py            # trains all models + ensemble, saves best_model.pkl
    │   └── use_saved_model.py        # loads best_model.pkl, evaluates on 2025
    ├── data/
    │   ├── raw/                      # FastF1 cache directory
    │   └── processed/
    │       ├── clean_race_data.csv
    │       ├── featured_race_data.csv
    │       └── enriched_race_data.csv   ← primary training input (preferred)
    ├── models/
    │   ├── best_model.pkl            # Ensemble/RankOptimized — current best
    │   ├── rf_model.pkl
    │   ├── xgb_model.pkl
    │   ├── feature_cols.json         # feature list saved at training time
    │   └── metrics_summary.json      # evaluation results for all models
    ├── outputs/
    │   └── predictions_2025.csv      # predicted vs actual for 2025 races
    ├── web/                          # empty — future dashboard placeholder
    ├── notebooks/                    # exploration notebooks
    └── .venv/                        # virtual environment
```

---

## Pipeline — Run Order

```
collect_data.py
  → clean_data.py
    → feature_engineering.py
      → collect_extra_features.py
        → train_model.py
          → use_saved_model.py
```

Scripts use hardcoded absolute paths prefixed with `D:\Coding\f1_prediction\f1_race_prediciton\`.

---

## Current Model Performance

| Model | MAE | RMSE | Spearman | Within ±2 |
|---|---|---|---|---|
| **Ensemble/RankOptimized** (best) | 2.12 | 3.05 | 0.830 | 67.5% |
| CB/Optuna (best single model) | 2.22 | 3.14 | 0.819 | 65.2% |
| Baseline (grid position only) | 2.80 | — | — | — |

Improvement over baseline: **−0.68 MAE**.

---

## Tech Stack

- **Data:** FastF1 API, pandas, numpy
- **Models:** CatBoost, XGBoost, LightGBM (LambdaRank), RandomForest, ExtraTrees, Ridge/ElasticNet stacking
- **Tuning:** Optuna (200 trials), Bayesian ensemble weight optimisation
- **Serialisation:** joblib `.pkl`
- **Environment:** `.venv` inside `f1_race_prediciton/`

---

## Key Rules

- **Never rename** `f1_race_prediciton/` — the typo is baked into every hardcoded path
- `enriched_race_data.csv` is preferred; scripts fall back to `featured_race_data.csv` automatically
- Models are saved as `.pkl` (joblib) — keep this format
- Run scripts from the `f1_race_prediciton/` directory or use the full absolute paths already in each file

---

## Multi-agent coordination (claim-mode)
Before editing any file in this project, check D:\Claude Vault\Tasks\f1_race_prediciton.md for the task board. Claim the specific file(s) you're about to touch first, using the same atomic lock as D:\Claude Vault\Tasks\f1_race_prediciton-sessions.md. If already claimed by an in-progress session, wait or pick a different open task instead. Stay only within your claimed scope. Mark done and release the claim when finished. This is a followed convention, not an OS-level lock — it only works if checked every time.
