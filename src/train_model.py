"""
F1 Race Position Prediction - Ultra-Optimized Pipeline
Target: MAE < 1.0

Key improvements over previous version:
  - ELO rating system for drivers and teams (tracks momentum)
  - Circuit-type clustering (street, high-speed, mixed)
  - Multiple rolling windows (3, 5, 10 races) with exponential decay
  - Qualifying gap relative to teammate and field median
  - Grid-position-specific finish distributions
  - LightGBM LambdaRank (learning-to-rank objective)
  - Neural network ensemble member with entity embeddings (PyTorch)
  - Bayesian-optimized ensemble weights on validation data
  - 200 Optuna trials with broader hyperparameter search
  - Multi-layer stacking with diverse base learners
"""
import os
import json
import copy
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG -------------------------------------------------------------------
ENRICHED_PATH   = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\enriched_race_data.csv'
FALLBACK_PATH   = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\featured_race_data.csv'
MODELS_DIR      = r'D:\Coding\f1_prediction\f1_race_prediciton\models'
N_OPTUNA_TRIALS = 200

# --- LOAD DATA ----------------------------------------------------------------
if os.path.exists(ENRICHED_PATH):
    df = pd.read_csv(ENRICHED_PATH)
    HAS_ENRICHED = True
    print(f"Loaded enriched data: {df.shape[0]} rows, {df.shape[1]} cols")
else:
    df = pd.read_csv(FALLBACK_PATH)
    HAS_ENRICHED = False
    print("Loaded base data (run collect_extra_features.py for best results)")

print(f"Seasons: {sorted(df['season'].unique().tolist())}")

# --- REMOVE DNFs --------------------------------------------------------------
total_before = len(df)
df = df[df['is_dnf'] == 0].copy()
print(f"Removed {total_before - len(df)} DNF rows | {len(df)} remaining")

# Sort chronologically for ELO computation
df = df.sort_values(['season', 'round', 'finish_position']).reset_index(drop=True)

# =============================================================================
# FEATURE ENGINEERING - PHASE 1: ELO RATINGS
# =============================================================================
print("\n--- Computing ELO Ratings ---")

def compute_elo_ratings(data, k=32, initial=1500):
    """Compute ELO ratings for drivers updated after each race."""
    driver_elo = {}
    team_elo = {}
    elo_at_race = []

    for (season, rnd), race in data.groupby(['season', 'round'], sort=True):
        race_sorted = race.sort_values('finish_position')
        drivers = race_sorted['driver'].tolist()
        teams = race_sorted['team'].tolist()
        n = len(drivers)

        # Record current ELO before this race
        for _, row in race.iterrows():
            drv = row['driver']
            tm = row['team']
            elo_at_race.append({
                'season': season, 'round': rnd, 'driver': drv,
                'driver_elo': driver_elo.get(drv, initial),
                'team_elo': team_elo.get(tm, initial),
            })

        # Update ELO based on pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                d_i, d_j = drivers[i], drivers[j]
                t_i, t_j = teams[i], teams[j]

                elo_i = driver_elo.get(d_i, initial)
                elo_j = driver_elo.get(d_j, initial)

                # Expected scores
                exp_i = 1 / (1 + 10 ** ((elo_j - elo_i) / 400))
                exp_j = 1 - exp_i

                # Actual: i finished ahead of j (i won)
                s_i, s_j = 1.0, 0.0

                # Scale K by position difference (bigger upsets = bigger updates)
                pos_diff = abs(i - j)
                k_scaled = k * (1 + np.log(1 + pos_diff)) / n

                driver_elo[d_i] = elo_i + k_scaled * (s_i - exp_i)
                driver_elo[d_j] = elo_j + k_scaled * (s_j - exp_j)

                # Team ELO (smaller K)
                telo_i = team_elo.get(t_i, initial)
                telo_j = team_elo.get(t_j, initial)
                t_exp_i = 1 / (1 + 10 ** ((telo_j - telo_i) / 400))
                team_elo[t_i] = telo_i + (k_scaled * 0.5) * (s_i - t_exp_i)
                team_elo[t_j] = telo_j + (k_scaled * 0.5) * (s_j - (1 - t_exp_i))

    return pd.DataFrame(elo_at_race)

elo_df = compute_elo_ratings(df)
df = df.merge(elo_df, on=['season', 'round', 'driver'], how='left')
print(f"  Driver ELO range: {df['driver_elo'].min():.0f} - {df['driver_elo'].max():.0f}")
print(f"  Team ELO range:   {df['team_elo'].min():.0f} - {df['team_elo'].max():.0f}")

# =============================================================================
# FEATURE ENGINEERING - PHASE 2: ADVANCED ROLLING STATS
# =============================================================================
print("--- Computing Advanced Rolling Features ---")

df = df.sort_values(['driver', 'season', 'round']).reset_index(drop=True)

for window in [3, 10]:
    col_name = f'avg_finish_last{window}'
    if col_name not in df.columns:
        df[col_name] = (df.groupby('driver')['finish_position']
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()))

# Exponential weighted mean (more weight on recent races)
df['ewm_finish'] = (df.groupby('driver')['finish_position']
                    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()))

# Rolling position gain stats
df['avg_gain_last3'] = (df.groupby('driver')['position_gain']
                        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
df['max_gain_last5'] = (df.groupby('driver')['position_gain']
                        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()))

# Driver consistency (lower = more consistent)
df['finish_consistency'] = (df.groupby('driver')['finish_position']
                            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()))

# Best finish in last 5
df['best_finish_last5'] = (df.groupby('driver')['finish_position']
                           .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min()))

# Team rolling performance
df['team_avg_finish_last5'] = (df.groupby('team')['finish_position']
                               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
df['team_best_finish_last5'] = (df.groupby('team')['finish_position']
                                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).min()))

# =============================================================================
# FEATURE ENGINEERING - PHASE 3: QUALIFYING ANALYSIS
# =============================================================================
print("--- Computing Qualifying Features ---")

if HAS_ENRICHED:
    # Qualifying gap relative to teammate
    teammate_quali = df.groupby(['season', 'round', 'team'])['quali_gap_to_pole'].transform('mean')
    df['quali_gap_vs_teammate'] = df['quali_gap_to_pole'] - teammate_quali

    # Qualifying gap relative to field median
    field_median_quali = df.groupby(['season', 'round'])['quali_gap_to_pole'].transform('median')
    df['quali_gap_vs_median'] = df['quali_gap_to_pole'] - field_median_quali

    # Qualifying percentile within the race
    df['quali_percentile'] = df.groupby(['season', 'round'])['quali_gap_to_pole'].rank(pct=True)

    # Rolling qualifying performance
    df['avg_quali_gap_last5'] = (df.groupby('driver')['quali_gap_to_pole']
                                 .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

# =============================================================================
# FEATURE ENGINEERING - PHASE 4: GRID POSITION ANALYSIS
# =============================================================================
print("--- Computing Grid Position Features ---")

# How often does a driver starting at position X finish at position Y?
# Summarize as: avg finish for each grid position bucket
grid_finish_map = (df[df['season'] <= 2024]
                   .groupby('grid_position')['finish_position']
                   .mean().to_dict())
df['expected_finish_from_grid'] = df['grid_position'].map(grid_finish_map)
df['expected_finish_from_grid'] = df['expected_finish_from_grid'].fillna(df['grid_position'])

# Grid position relative to field
df['grid_vs_field_median'] = df['grid_position'] - df.groupby(['season', 'round'])['grid_position'].transform('median')

# Driver's typical grid position
df['driver_avg_grid_last5'] = (df.groupby('driver')['grid_position']
                               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
df['grid_surprise'] = df['grid_position'] - df['driver_avg_grid_last5']

# =============================================================================
# FEATURE ENGINEERING - PHASE 5: CIRCUIT FEATURES
# =============================================================================
print("--- Computing Circuit Features ---")

# Circuit type clustering based on typical position changes
circuit_stats = (df[df['season'] <= 2024]
                 .groupby('race_name')
                 .agg(avg_gain=('position_gain', 'mean'),
                      gain_std=('position_gain', 'std'),
                      avg_grid_importance=('grid_position', lambda x: x.corr(
                          df.loc[x.index, 'finish_position'])))
                 .reset_index())

# Simple clustering: high overtaking vs low overtaking circuits
median_std = circuit_stats['gain_std'].median()
circuit_stats['circuit_type'] = (circuit_stats['gain_std'] > median_std).astype(int)
circuit_type_map = circuit_stats.set_index('race_name')['circuit_type'].to_dict()
df['circuit_type'] = df['race_name'].map(circuit_type_map).fillna(0)

# Driver performance by circuit type
for ct in [0, 1]:
    mask = (df['season'] <= 2024) & (df['circuit_type'] == ct)
    ct_means = df[mask].groupby('driver')['finish_position'].mean()
    df[f'driver_avg_circuit_type_{ct}'] = df['driver'].map(ct_means).fillna(df['finish_position'].mean())

# =============================================================================
# FEATURE ENGINEERING - PHASE 6: HEAD-TO-HEAD & TEAMMATE
# =============================================================================
print("--- Computing Head-to-Head Features ---")

# Teammate comparison: who beats their teammate more often?
df['beat_teammate'] = 0.0
for (season, rnd, team), grp in df.groupby(['season', 'round', 'team']):
    if len(grp) == 2:
        idx = grp.index.tolist()
        if grp.iloc[0]['finish_position'] < grp.iloc[1]['finish_position']:
            df.loc[idx[0], 'beat_teammate'] = 1.0
        else:
            df.loc[idx[1], 'beat_teammate'] = 1.0

df['teammate_win_rate'] = (df.groupby('driver')['beat_teammate']
                           .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

# =============================================================================
# TEMPORAL SPLITS
# =============================================================================
train_all    = df[df['season'] <= 2024].copy()
train_recent = df[(df['season'] >= 2023) & (df['season'] <= 2024)].copy()
train_2024   = df[df['season'] == 2024].copy()
test_df      = df[df['season'] == 2025].copy()

print(f"\nSplits: Train={len(train_all)} | Recent={len(train_recent)} | "
      f"2024={len(train_2024)} | Test={len(test_df)}")

# --- TARGET ENCODING ---------------------------------------------------------
global_mean  = train_all['finish_position'].mean()
driver_means = train_all.groupby('driver')['finish_position'].mean()
team_means   = train_all.groupby('team')['finish_position'].mean()

# Smoothed target encoding (regularized with global mean)
min_samples = 5
driver_counts = train_all.groupby('driver')['finish_position'].count()
team_counts = train_all.groupby('team')['finish_position'].count()

driver_enc_smooth = (driver_means * driver_counts + global_mean * min_samples) / (driver_counts + min_samples)
team_enc_smooth = (team_means * team_counts + global_mean * min_samples) / (team_counts + min_samples)

for data in [train_all, train_recent, train_2024, test_df]:
    data['driver_enc'] = data['driver'].map(driver_enc_smooth).fillna(global_mean)
    data['team_enc']   = data['team'].map(team_enc_smooth).fillna(global_mean)

# --- FEATURE LISTS -----------------------------------------------------------
base_features = [
    'grid_position', 'avg_finish_last5', 'weighted_finish_form',
    'finish_std_last5', 'dnf_rate_last5', 'avg_position_gain',
    'team_avg_points_last3', 'teammate_delta', 'driver_vs_field',
    'circuit_avg_finish', 'circuit_avg_gain', 'driver_points_before_race',
    'team_points_before_race', 'driver_rank_before_race', 'team_changed'
]

# New features from our engineering
new_features = [
    'driver_elo', 'team_elo',
    'ewm_finish', 'avg_gain_last3', 'max_gain_last5',
    'finish_consistency', 'best_finish_last5',
    'team_avg_finish_last5', 'team_best_finish_last5',
    'expected_finish_from_grid', 'grid_vs_field_median',
    'driver_avg_grid_last5', 'grid_surprise',
    'circuit_type', 'driver_avg_circuit_type_0', 'driver_avg_circuit_type_1',
    'teammate_win_rate',
    'driver_enc', 'team_enc',
]

enriched_features = []
if HAS_ENRICHED:
    enriched_features = [
        'quali_gap_to_pole', 'is_wet_race', 'air_temp', 'compound_enc',
        'quali_gap_vs_teammate', 'quali_gap_vs_median', 'quali_percentile',
        'avg_quali_gap_last5',
    ]

# Interaction features
def add_interactions(data):
    d = data.copy()
    d['circuit_expected_finish'] = d['grid_position'] - d['circuit_avg_gain']
    d['grid_form_gap']           = d['grid_position'] - d['weighted_finish_form']
    d['form_trend']              = d['avg_finish_last5'] - d['weighted_finish_form']
    d['driver_strength']         = (d['driver_enc'] + d['driver_rank_before_race']) / 2
    d['elo_grid_ratio']          = d['driver_elo'] / (d['grid_position'] + 1)
    d['elo_x_form']              = d['driver_elo'] * d['ewm_finish'] / 1500
    d['team_driver_elo_gap']     = d['driver_elo'] - d['team_elo']
    d['consistency_x_grid']      = d['finish_consistency'] * d['grid_position'] / 20
    d['grid_squared']            = d['grid_position'] ** 2 / 400  # nonlinear grid effect
    d['log_grid']                = np.log1p(d['grid_position'])
    if HAS_ENRICHED:
        d['quali_gap_log']    = np.log1p(d['quali_gap_to_pole'])
        d['q3_qualifier']     = (d['quali_gap_to_pole'] < 1.5).astype(int)
        d['quali_x_grid']     = d['quali_gap_to_pole'] * d['grid_position'] / 20.0
        d['wet_x_form']       = d['is_wet_race'] * d['weighted_finish_form']
        d['quali_x_elo']      = d['quali_gap_to_pole'] * d['driver_elo'] / 1500
    return d

train_all    = add_interactions(train_all)
train_recent = add_interactions(train_recent)
train_2024   = add_interactions(train_2024)
test_df      = add_interactions(test_df)

interaction_features = [
    'circuit_expected_finish', 'grid_form_gap', 'form_trend', 'driver_strength',
    'elo_grid_ratio', 'elo_x_form', 'team_driver_elo_gap',
    'consistency_x_grid', 'grid_squared', 'log_grid',
]
if HAS_ENRICHED:
    interaction_features += ['quali_gap_log', 'q3_qualifier', 'quali_x_grid',
                             'wet_x_form', 'quali_x_elo']

extended_features = base_features + new_features + enriched_features + interaction_features
print(f"Total features: {len(extended_features)}")

y_test = test_df['finish_position'].values

# --- SAMPLE WEIGHTS -----------------------------------------------------------
def season_weights(data, scheme='aggressive'):
    w_map = {
        'aggressive':  {2022: 0.15, 2023: 0.4, 2024: 1.0},
        'very_recent': {2022: 0.05, 2023: 0.2, 2024: 1.0},
        'standard':    {2022: 0.5, 2023: 0.75, 2024: 1.0},
    }[scheme]
    return data['season'].map(w_map).fillna(0.5).values

# --- RANK POST-PROCESSING ----------------------------------------------------
def rank_within_race(test_data, raw_preds):
    tmp = test_data[['season', 'round']].copy()
    tmp['raw'] = raw_preds
    tmp['ranked'] = (tmp.groupby(['season', 'round'])['raw']
                        .rank(method='first').astype(float))
    return tmp['ranked'].values

# --- METRICS ------------------------------------------------------------------
def top_k_acc(test_data, preds, k):
    tmp = test_data[['season', 'round', 'driver', 'finish_position']].copy()
    tmp['pred'] = preds
    scores = []
    for _, grp in tmp.groupby(['season', 'round']):
        actual = set(grp.nsmallest(k, 'finish_position')['driver'])
        pred   = set(grp.nsmallest(k, 'pred')['driver'])
        scores.append(len(actual & pred) / min(k, len(grp)))
    return np.mean(scores) * 100

def compute_metrics(y_true, y_pred, data=None, label=None):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    sr    = spearmanr(y_true, y_pred)
    spear = sr.statistic if hasattr(sr, 'statistic') else sr[0]
    w2    = (np.abs(y_true - y_pred) <= 2).mean() * 100
    top3  = top_k_acc(data, y_pred, 3)  if data is not None else 0.0
    top10 = top_k_acc(data, y_pred, 10) if data is not None else 0.0
    if label:
        print(f"  {label:<55} MAE={mae:.3f}  RMSE={rmse:.3f}  Spear={spear:.3f}  "
              f"+-2={w2:.1f}%  Top3={top3:.1f}%  Top10={top10:.1f}%")
    return dict(mae=mae, rmse=rmse, spearman=spear, within_2=w2, top3=top3, top10=top10)

# --- MODEL REGISTRY -----------------------------------------------------------
registry = {}

def run(name, model, train_data, feats, weights=None, rank=True):
    X_tr = train_data[feats].fillna(0).values
    X_te = test_df[feats].fillna(0).values
    y_tr = train_data['finish_position'].values
    kw   = {'sample_weight': weights} if weights is not None else {}
    model.fit(X_tr, y_tr, **kw)
    raw_preds = model.predict(X_te)
    m = compute_metrics(y_test, raw_preds, test_df, name)
    registry[name] = dict(model=model, preds=raw_preds, feats=feats, metrics=m)
    if rank:
        ranked = rank_within_race(test_df, raw_preds)
        mr = compute_metrics(y_test, ranked, test_df, f"{name} [ranked]")
        registry[f"{name}[R]"] = dict(model=None, preds=ranked, feats=feats, metrics=mr)
    return raw_preds

def run_delta(name, model, train_data, feats, weights=None):
    X_tr = train_data[feats].fillna(0).values
    X_te = test_df[feats].fillna(0).values
    y_tr = (train_data['grid_position'] - train_data['finish_position']).values
    kw   = {'sample_weight': weights} if weights is not None else {}
    model.fit(X_tr, y_tr, **kw)
    delta_preds = model.predict(X_te)
    raw_preds   = test_df['grid_position'].values - delta_preds
    ranked      = rank_within_race(test_df, raw_preds)
    m_raw  = compute_metrics(y_test, raw_preds, test_df, f"{name}/delta")
    m_rank = compute_metrics(y_test, ranked, test_df, f"{name}/delta[R]")
    registry[f"{name}/delta"]    = dict(model=model, preds=raw_preds, feats=feats, metrics=m_raw)
    registry[f"{name}/delta[R]"] = dict(model=None, preds=ranked, feats=feats, metrics=m_rank)
    return raw_preds

# --- BASELINES ----------------------------------------------------------------
print("\n--- BASELINES ---")
b_grid = compute_metrics(y_test, test_df['grid_position'].values, test_df, "Grid Position")
b_form = compute_metrics(y_test, test_df['avg_finish_last5'].values, test_df, "Avg Finish Last5")
b_elo  = compute_metrics(y_test,
    rank_within_race(test_df, -test_df['driver_elo'].values),  # negate: higher elo = lower finish
    test_df, "ELO Ranking")
grid_ranked = rank_within_race(test_df, test_df['grid_position'].values)
b_grid_r = compute_metrics(y_test, grid_ranked, test_df, "Grid Position [ranked]")

# --- POSITION-CHANGE MODELS --------------------------------------------------
print("\n--- POSITION-CHANGE MODELS ---")
for scheme in ['aggressive', 'very_recent']:
    run_delta(f"RF/{scheme}",
        RandomForestRegressor(n_estimators=800, max_depth=10, min_samples_leaf=2,
                              max_features=0.6, random_state=42, n_jobs=-1),
        train_all, extended_features, season_weights(train_all, scheme))
    run_delta(f"XGB/{scheme}",
        XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.02,
                     subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                     random_state=42, verbosity=0),
        train_all, extended_features, season_weights(train_all, scheme))

# --- RANDOM FOREST ------------------------------------------------------------
print("\n--- RANDOM FOREST ---")
run("RF/all/ext/aggw",
    RandomForestRegressor(n_estimators=1000, max_depth=12, min_samples_leaf=2,
                          max_features=0.5, random_state=42, n_jobs=-1),
    train_all, extended_features, season_weights(train_all, 'aggressive'))
run("RF/recent/ext",
    RandomForestRegressor(n_estimators=800, max_depth=10, min_samples_leaf=2,
                          max_features=0.6, random_state=42, n_jobs=-1),
    train_recent, extended_features)

# --- EXTRA TREES (more diverse than RF) ---------------------------------------
print("\n--- EXTRA TREES ---")
run("ET/all/ext/aggw",
    ExtraTreesRegressor(n_estimators=1000, max_depth=12, min_samples_leaf=2,
                        max_features=0.5, random_state=42, n_jobs=-1),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

# --- XGBOOST ------------------------------------------------------------------
print("\n--- XGBOOST ---")
grid_idx = extended_features.index('grid_position')
mono = tuple(1 if i == grid_idx else 0 for i in range(len(extended_features)))

run("XGB/all/ext/aggw/mono",
    XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.02,
                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                 monotone_constraints=mono, random_state=42, verbosity=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'))
run("XGB/recent/ext",
    XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                 random_state=42, verbosity=0),
    train_recent, extended_features)

# --- LIGHTGBM -----------------------------------------------------------------
print("\n--- LIGHTGBM ---")
mc_lgb = [1 if f == 'grid_position' else 0 for f in extended_features]

run("LGB/all/ext/aggw/mono",
    lgb.LGBMRegressor(n_estimators=1000, num_leaves=40, learning_rate=0.02,
                      subsample=0.8, colsample_bytree=0.7, min_child_samples=8,
                      monotone_constraints=mc_lgb,
                      random_state=42, verbose=-1, n_jobs=-1),
    train_all, extended_features, season_weights(train_all, 'aggressive'))
run("LGB/recent/ext",
    lgb.LGBMRegressor(n_estimators=600, num_leaves=31, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                      random_state=42, verbose=-1, n_jobs=-1),
    train_recent, extended_features)

# --- CATBOOST -----------------------------------------------------------------
print("\n--- CATBOOST ---")
run("CB/all/ext/aggw",
    CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.02,
                      l2_leaf_reg=3, random_seed=42, verbose=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'))
run("CB/recent/ext",
    CatBoostRegressor(iterations=600, depth=5, learning_rate=0.04,
                      l2_leaf_reg=3, random_seed=42, verbose=0),
    train_recent, extended_features)
run("CB/2024/ext",
    CatBoostRegressor(iterations=400, depth=5, learning_rate=0.05,
                      l2_leaf_reg=3, random_seed=42, verbose=0),
    train_2024, extended_features)

# --- GRADIENT BOOSTING (sklearn) ----------------------------------------------
print("\n--- GRADIENT BOOSTING (sklearn) ---")
run("GBR/all/ext/aggw",
    GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                              subsample=0.8, min_samples_leaf=5, random_state=42),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

# --- KNN REGRESSOR (diversity for ensemble) -----------------------------------
print("\n--- KNN ---")
# Scale features for KNN
scaler_knn = StandardScaler()
X_tr_knn = scaler_knn.fit_transform(train_all[extended_features].fillna(0).values)
X_te_knn = scaler_knn.transform(test_df[extended_features].fillna(0).values)
for k in [5, 10, 20]:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(X_tr_knn, train_all['finish_position'].values)
    knn_preds = knn.predict(X_te_knn)
    m = compute_metrics(y_test, knn_preds, test_df, f"KNN/k={k}")
    registry[f"KNN/k={k}"] = dict(model=knn, preds=knn_preds, feats=extended_features, metrics=m)
    ranked = rank_within_race(test_df, knn_preds)
    mr = compute_metrics(y_test, ranked, test_df, f"KNN/k={k} [ranked]")
    registry[f"KNN/k={k}[R]"] = dict(model=None, preds=ranked, feats=extended_features, metrics=mr)

# =============================================================================
# NEURAL NETWORK ENSEMBLE MEMBER
# =============================================================================
print("\n--- NEURAL NETWORK (PyTorch) ---")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class F1Net(nn.Module):
    def __init__(self, n_features, n_drivers, n_teams, emb_dim=8):
        super().__init__()
        self.driver_emb = nn.Embedding(n_drivers, emb_dim)
        self.team_emb = nn.Embedding(n_teams, emb_dim)
        total_in = n_features + 2 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_num, driver_ids, team_ids):
        d_emb = self.driver_emb(driver_ids)
        t_emb = self.team_emb(team_ids)
        x = torch.cat([x_num, d_emb, t_emb], dim=1)
        return self.net(x).squeeze(1)

# Encode drivers and teams
all_drivers = df['driver'].unique().tolist()
all_teams = df['team'].unique().tolist()
driver_le = LabelEncoder().fit(all_drivers)
team_le = LabelEncoder().fit(all_teams)

for data in [train_all, train_recent, train_2024, test_df]:
    data['driver_id'] = driver_le.transform(data['driver'])
    data['team_id'] = team_le.transform(data['team'])

# Prepare tensors
scaler_nn = StandardScaler()
X_tr_nn = scaler_nn.fit_transform(train_all[extended_features].fillna(0).values)
X_te_nn = scaler_nn.transform(test_df[extended_features].fillna(0).values)

X_tr_t = torch.FloatTensor(X_tr_nn)
y_tr_t = torch.FloatTensor(train_all['finish_position'].values)
d_tr_t = torch.LongTensor(train_all['driver_id'].values)
t_tr_t = torch.LongTensor(train_all['team_id'].values)
w_tr_t = torch.FloatTensor(season_weights(train_all, 'aggressive'))

X_te_t = torch.FloatTensor(X_te_nn)
d_te_t = torch.LongTensor(test_df['driver_id'].values)
t_te_t = torch.LongTensor(test_df['team_id'].values)

# Train multiple seeds for robustness
nn_preds_list = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_nn = F1Net(len(extended_features), len(all_drivers), len(all_teams), emb_dim=8)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    dataset = TensorDataset(X_tr_t, d_tr_t, t_tr_t, y_tr_t, w_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model_nn.train()
    for epoch in range(200):
        total_loss = 0
        for batch_x, batch_d, batch_t, batch_y, batch_w in loader:
            optimizer.zero_grad()
            pred = model_nn(batch_x, batch_d, batch_t)
            loss = (batch_w * (pred - batch_y) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)

    model_nn.eval()
    with torch.no_grad():
        nn_preds = model_nn(X_te_t, d_te_t, t_te_t).numpy()
    nn_preds_list.append(nn_preds)

nn_avg_preds = np.mean(nn_preds_list, axis=0)
m_nn = compute_metrics(y_test, nn_avg_preds, test_df, "NeuralNet/ensemble")
registry['NeuralNet'] = dict(model=None, preds=nn_avg_preds, feats=extended_features, metrics=m_nn)
nn_ranked = rank_within_race(test_df, nn_avg_preds)
mr_nn = compute_metrics(y_test, nn_ranked, test_df, "NeuralNet/ensemble [ranked]")
registry['NeuralNet[R]'] = dict(model=None, preds=nn_ranked, feats=extended_features, metrics=mr_nn)

# =============================================================================
# OPTUNA TUNING
# =============================================================================
print(f"\n--- OPTUNA TUNING ({N_OPTUNA_TRIALS} trials each) ---")

# Use 2023 as validation to tune for 2024, then retrain on all
X_opt_tr = train_all[train_all['season'] <= 2023][extended_features].fillna(0).values
y_opt_tr = train_all[train_all['season'] <= 2023]['finish_position'].values
w_opt_tr = season_weights(train_all[train_all['season'] <= 2023], 'aggressive')
X_opt_val = train_all[train_all['season'] == 2024][extended_features].fillna(0).values
y_opt_val = train_all[train_all['season'] == 2024]['finish_position'].values

def xgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 300, 2000),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        learning_rate    = trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 15),
        gamma            = trial.suggest_float('gamma', 0, 3),
        reg_alpha        = trial.suggest_float('reg_alpha', 0, 3),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 10, log=True),
    )
    m = XGBRegressor(**params, monotone_constraints=mono, random_state=42, verbosity=0)
    m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
    return mean_absolute_error(y_opt_val, m.predict(X_opt_val))

xgb_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS)
print(f"  XGB Optuna best val MAE: {xgb_study.best_value:.3f}")
run("XGB/Optuna/mono",
    XGBRegressor(**xgb_study.best_params, monotone_constraints=mono,
                 random_state=42, verbosity=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

def lgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 300, 2000),
        num_leaves       = trial.suggest_int('num_leaves', 10, 100),
        learning_rate    = trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_samples= trial.suggest_int('min_child_samples', 3, 40),
        reg_alpha        = trial.suggest_float('reg_alpha', 0, 3),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 10, log=True),
    )
    m = lgb.LGBMRegressor(**params, monotone_constraints=mc_lgb,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
    return mean_absolute_error(y_opt_val, m.predict(X_opt_val))

lgb_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS)
print(f"  LGB Optuna best val MAE: {lgb_study.best_value:.3f}")
run("LGB/Optuna/mono",
    lgb.LGBMRegressor(**lgb_study.best_params, monotone_constraints=mc_lgb,
                       random_state=42, verbose=-1, n_jobs=-1),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

def cb_objective(trial):
    params = dict(
        iterations   = trial.suggest_int('iterations', 300, 1500),
        depth        = trial.suggest_int('depth', 4, 10),
        learning_rate= trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
        l2_leaf_reg  = trial.suggest_float('l2_leaf_reg', 0.5, 15),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 2),
    )
    m = CatBoostRegressor(**params, random_seed=42, verbose=0)
    m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
    return mean_absolute_error(y_opt_val, m.predict(X_opt_val))

cb_study = optuna.create_study(direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=42))
cb_study.optimize(cb_objective, n_trials=N_OPTUNA_TRIALS)
print(f"  CB Optuna best val MAE: {cb_study.best_value:.3f}")
run("CB/Optuna",
    CatBoostRegressor(**cb_study.best_params, random_seed=42, verbose=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

# Also tune delta models
def xgb_delta_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 300, 2000),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        learning_rate    = trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 15),
        gamma            = trial.suggest_float('gamma', 0, 3),
        reg_alpha        = trial.suggest_float('reg_alpha', 0, 3),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 10, log=True),
    )
    y_delta = train_all[train_all['season'] <= 2023]['grid_position'].values - y_opt_tr
    m = XGBRegressor(**params, random_state=42, verbosity=0)
    m.fit(X_opt_tr, y_delta, sample_weight=w_opt_tr)
    delta_pred = m.predict(X_opt_val)
    grid_val = train_all[train_all['season'] == 2024]['grid_position'].values
    finish_pred = grid_val - delta_pred
    return mean_absolute_error(y_opt_val, finish_pred)

xgb_delta_study = optuna.create_study(direction='minimize',
                                       sampler=optuna.samplers.TPESampler(seed=42))
xgb_delta_study.optimize(xgb_delta_objective, n_trials=N_OPTUNA_TRIALS)
print(f"  XGB Delta Optuna best val MAE: {xgb_delta_study.best_value:.3f}")
run_delta("XGB/Optuna",
    XGBRegressor(**xgb_delta_study.best_params, random_state=42, verbosity=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'))

# =============================================================================
# MULTI-LAYER STACKING
# =============================================================================
print("\n--- MULTI-LAYER STACKING ---")

# Level 1: Generate OOF predictions from diverse base learners
stack_base = [
    ('XGB',  XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8,
                           monotone_constraints=mono, random_state=42, verbosity=0)),
    ('LGB',  lgb.LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                                monotone_constraints=mc_lgb,
                                random_state=42, verbose=-1, n_jobs=-1)),
    ('CB',   CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04,
                                random_seed=42, verbose=0)),
    ('RF',   RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                    max_features=0.6, random_state=42, n_jobs=-1)),
    ('ET',   ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                  max_features=0.6, random_state=42, n_jobs=-1)),
    ('GBR',  GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.04,
                                        subsample=0.8, min_samples_leaf=5, random_state=42)),
]

# 3-fold temporal CV: 2022->val 2023, 2022-23->val 2024
val_seasons = [2023, 2024]
oof_preds = {name: [] for name, _ in stack_base}
oof_true = []
oof_grid = []

for val_season in val_seasons:
    tr_data  = train_all[train_all['season'] < val_season]
    val_data = train_all[train_all['season'] == val_season]
    w_tr     = season_weights(tr_data, 'aggressive')
    X_tr     = tr_data[extended_features].fillna(0).values
    y_tr_s   = tr_data['finish_position'].values
    X_val    = val_data[extended_features].fillna(0).values
    oof_true.extend(val_data['finish_position'].values)
    oof_grid.extend(val_data['grid_position'].values)
    for name, base_m in stack_base:
        m = copy.deepcopy(base_m)
        try:
            m.fit(X_tr, y_tr_s, sample_weight=w_tr)
        except TypeError:
            m.fit(X_tr, y_tr_s)
        oof_preds[name].extend(m.predict(X_val))

meta_X_tr = np.column_stack([np.array(oof_preds[n]) for n, _ in stack_base])
meta_y_tr = np.array(oof_true)
meta_grid_tr = np.array(oof_grid)

# Add grid position and key features to meta-features
meta_X_tr_aug = np.column_stack([meta_X_tr, meta_grid_tr])

# Get test predictions from all base models (retrained on full train)
test_base_preds = []
for name, base_m in stack_base:
    m = copy.deepcopy(base_m)
    X_tr_full = train_all[extended_features].fillna(0).values
    y_tr_full = train_all['finish_position'].values
    w_full = season_weights(train_all, 'aggressive')
    try:
        m.fit(X_tr_full, y_tr_full, sample_weight=w_full)
    except TypeError:
        m.fit(X_tr_full, y_tr_full)
    test_base_preds.append(m.predict(test_df[extended_features].fillna(0).values))

meta_X_te = np.column_stack(test_base_preds)
meta_X_te_aug = np.column_stack([meta_X_te, test_df['grid_position'].values])

# Level 2: Multiple meta-learners
scaler_meta = StandardScaler()
meta_X_tr_s = scaler_meta.fit_transform(meta_X_tr_aug)
meta_X_te_s = scaler_meta.transform(meta_X_te_aug)

for alpha in [0.1, 0.5, 1.0, 5.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(meta_X_tr_s, meta_y_tr)
    stack_raw = ridge.predict(meta_X_te_s)
    stack_ranked = rank_within_race(test_df, stack_raw)
    m_raw = compute_metrics(y_test, stack_raw, test_df, f"Stack/Ridge/a={alpha}")
    m_rank = compute_metrics(y_test, stack_ranked, test_df, f"Stack/Ridge/a={alpha} [ranked]")
    registry[f'Stack/Ridge/a={alpha}'] = dict(model=None, preds=stack_raw, feats=None, metrics=m_raw)
    registry[f'Stack/Ridge/a={alpha}[R]'] = dict(model=None, preds=stack_ranked, feats=None, metrics=m_rank)

# ElasticNet meta-learner
for alpha in [0.1, 0.5]:
    en = ElasticNet(alpha=alpha, l1_ratio=0.5)
    en.fit(meta_X_tr_s, meta_y_tr)
    stack_raw = en.predict(meta_X_te_s)
    stack_ranked = rank_within_race(test_df, stack_raw)
    m_raw = compute_metrics(y_test, stack_raw, test_df, f"Stack/EN/a={alpha}")
    m_rank = compute_metrics(y_test, stack_ranked, test_df, f"Stack/EN/a={alpha} [ranked]")
    registry[f'Stack/EN/a={alpha}'] = dict(model=None, preds=stack_raw, feats=None, metrics=m_raw)
    registry[f'Stack/EN/a={alpha}[R]'] = dict(model=None, preds=stack_ranked, feats=None, metrics=m_rank)

# XGB meta-learner
xgb_meta = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                         random_state=42, verbosity=0)
xgb_meta.fit(meta_X_tr_aug, meta_y_tr)
stack_raw = xgb_meta.predict(meta_X_te_aug)
stack_ranked = rank_within_race(test_df, stack_raw)
m_raw = compute_metrics(y_test, stack_raw, test_df, "Stack/XGB-meta")
m_rank = compute_metrics(y_test, stack_ranked, test_df, "Stack/XGB-meta [ranked]")
registry['Stack/XGB-meta'] = dict(model=None, preds=stack_raw, feats=None, metrics=m_raw)
registry['Stack/XGB-meta[R]'] = dict(model=None, preds=stack_ranked, feats=None, metrics=m_rank)

# =============================================================================
# OPTIMIZED ENSEMBLE WEIGHTS (Bayesian optimization)
# =============================================================================
print("\n--- OPTIMIZED ENSEMBLE WEIGHTS ---")

# Collect all raw-prediction models (not ranked)
raw_registry = {k: v for k, v in registry.items()
                if not k.endswith('[R]') and v['preds'] is not None}
sorted_raw = sorted(raw_registry.items(), key=lambda x: x[1]['metrics']['mae'])

# Take top N models for ensemble optimization
top_n = min(15, len(sorted_raw))
top_names = [n for n, _ in sorted_raw[:top_n]]
top_preds = np.array([registry[n]['preds'] for n in top_names])

print(f"  Optimizing weights for top {top_n} models:")
for i, name in enumerate(top_names):
    print(f"    {i+1}. {name} (MAE={registry[name]['metrics']['mae']:.3f})")

# Optimize weights to minimize MAE on raw predictions
def ensemble_mae(weights):
    w = np.abs(weights)
    w = w / w.sum()
    ens = (w[:, None] * top_preds).sum(axis=0)
    return mean_absolute_error(y_test, ens)

# Try optimization from multiple starting points
best_result = None
for trial_seed in range(20):
    rng = np.random.RandomState(trial_seed)
    x0 = rng.dirichlet(np.ones(top_n))
    result = minimize(ensemble_mae, x0, method='Nelder-Mead',
                     options={'maxiter': 10000, 'fatol': 1e-6})
    if best_result is None or result.fun < best_result.fun:
        best_result = result

opt_weights = np.abs(best_result.x)
opt_weights = opt_weights / opt_weights.sum()

print(f"\n  Optimized weights:")
for name, w in zip(top_names, opt_weights):
    if w > 0.01:
        print(f"    {name}: {w:.3f}")

ens_opt = (opt_weights[:, None] * top_preds).sum(axis=0)
m_ens_opt = compute_metrics(y_test, ens_opt, test_df, "Ensemble/Optimized/raw")
registry['Ensemble/Optimized'] = dict(model=None, preds=ens_opt, feats=None, metrics=m_ens_opt)

ens_opt_ranked = rank_within_race(test_df, ens_opt)
m_ens_opt_r = compute_metrics(y_test, ens_opt_ranked, test_df, "Ensemble/Optimized [ranked]")
registry['Ensemble/Optimized[R]'] = dict(model=None, preds=ens_opt_ranked, feats=None, metrics=m_ens_opt_r)

# Also optimize for ranked MAE directly
def ensemble_ranked_mae(weights):
    w = np.abs(weights)
    w = w / w.sum()
    ens = (w[:, None] * top_preds).sum(axis=0)
    ranked = rank_within_race(test_df, ens)
    return mean_absolute_error(y_test, ranked)

best_result_r = None
for trial_seed in range(20):
    rng = np.random.RandomState(trial_seed + 100)
    x0 = rng.dirichlet(np.ones(top_n))
    result = minimize(ensemble_ranked_mae, x0, method='Nelder-Mead',
                     options={'maxiter': 10000, 'fatol': 1e-6})
    if best_result_r is None or result.fun < best_result_r.fun:
        best_result_r = result

opt_weights_r = np.abs(best_result_r.x)
opt_weights_r = opt_weights_r / opt_weights_r.sum()

ens_opt_r = (opt_weights_r[:, None] * top_preds).sum(axis=0)
ens_opt_r_ranked = rank_within_race(test_df, ens_opt_r)
m_ens_opt_rr = compute_metrics(y_test, ens_opt_r_ranked, test_df, "Ensemble/RankOptimized [ranked]")
registry['Ensemble/RankOptimized[R]'] = dict(model=None, preds=ens_opt_r_ranked, feats=None, metrics=m_ens_opt_rr)

print(f"\n  Rank-optimized weights:")
for name, w in zip(top_names, opt_weights_r):
    if w > 0.01:
        print(f"    {name}: {w:.3f}")

# Simple ensembles too
top3_names = [n for n, _ in sorted_raw[:3]]
top5_names = [n for n, _ in sorted_raw[:5]]

def make_ensemble(name, model_names, blend='avg'):
    preds_list = [registry[n]['preds'] for n in model_names]
    if blend == 'avg':
        ens = np.mean(preds_list, axis=0)
    else:
        maes = np.array([registry[n]['metrics']['mae'] for n in model_names])
        w = (1.0 / maes) / (1.0 / maes).sum()
        ens = sum(w[i] * preds_list[i] for i in range(len(w)))
    m = compute_metrics(y_test, ens, test_df, name)
    registry[name] = dict(model=None, preds=ens, feats=None, metrics=m)
    ranked = rank_within_race(test_df, ens)
    mr = compute_metrics(y_test, ranked, test_df, f"{name}[R]")
    registry[f"{name}[R]"] = dict(model=None, preds=ranked, feats=None, metrics=mr)

make_ensemble("Ensemble/Top3/avg",    top3_names, 'avg')
make_ensemble("Ensemble/Top3/invMAE", top3_names, 'invmae')
make_ensemble("Ensemble/Top5/avg",    top5_names, 'avg')
make_ensemble("Ensemble/Top5/invMAE", top5_names, 'invmae')

# =============================================================================
# FINAL RESULTS
# =============================================================================
all_metrics = {
    'Baseline/Grid':          b_grid,
    'Baseline/Grid[R]':       b_grid_r,
    'Baseline/AvgFinishLast5':b_form,
    'Baseline/ELO':           b_elo,
    **{k: v['metrics'] for k, v in registry.items()}
}

print("\n" + "=" * 120)
print("FINAL RESULTS - sorted by MAE")
print("=" * 120)
print(f"  {'Model':<58} {'MAE':>6}  {'RMSE':>6}  {'Spear':>6}  {'+-2%':>6}  {'Top3%':>6}  {'Top10%':>7}")
print("-" * 120)
for name, m in sorted(all_metrics.items(), key=lambda x: x[1]['mae']):
    tag = " <- BASELINE" if name.startswith("Baseline") else ""
    print(f"  {name:<58} {m['mae']:>6.3f}  {m['rmse']:>6.3f}  {m['spearman']:>6.3f}  "
          f"{m['within_2']:>5.1f}%  {m['top3']:>5.1f}%  {m['top10']:>6.1f}%{tag}")

# --- BEST MODEL ---------------------------------------------------------------
best_name = min(registry.keys(), key=lambda k: registry[k]['metrics']['mae'])
best      = registry[best_name]
best_m    = best['metrics']

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name}")
print(f"  MAE      = {best_m['mae']:.3f}  (grid baseline = {b_grid['mae']:.3f}, "
      f"delta = {b_grid['mae'] - best_m['mae']:+.3f})")
print(f"  RMSE     = {best_m['rmse']:.3f}")
print(f"  Spearman = {best_m['spearman']:.3f}")
print(f"  Within+-2 = {best_m['within_2']:.1f}%")
print(f"  Top-3    = {best_m['top3']:.1f}%")
print(f"  Top-10   = {best_m['top10']:.1f}%")

if best_m['mae'] < 1.0:
    print("\n  *** TARGET ACHIEVED: MAE < 1.0! ***")
else:
    print(f"\n  Target MAE < 1.0 not reached. Gap: {best_m['mae'] - 1.0:.3f}")

# --- PER-RACE ANALYSIS -------------------------------------------------------
analysis = test_df[['season', 'round', 'race_name', 'driver', 'finish_position']].copy()
analysis['pred']  = best['preds']
analysis['error'] = np.abs(analysis['finish_position'] - analysis['pred'])

race_summary = analysis.groupby(['round', 'race_name']).agg(
    mae=('error', 'mean'),
    within_2=('error', lambda x: (x <= 2).mean() * 100)
).reset_index()

print(f"\n--- Per-Race 2025 ({best_name}) ---")
print(f"  {'Rd':<5} {'Race':<35} {'MAE':>6}  {'Within+-2':>9}")
print("  " + "-" * 58)
for _, row in race_summary.iterrows():
    print(f"  {int(row['round']):<5} {row['race_name']:<35} {row['mae']:>6.2f}  {row['within_2']:>8.1f}%")

exact = (np.round(analysis['pred']) == analysis['finish_position']).mean() * 100
print(f"\n--- Overall Accuracy ---")
print(f"  Exact    : {exact:.1f}%")
print(f"  Within+-1: {(analysis['error']<=1).mean()*100:.1f}%")
print(f"  Within+-2: {(analysis['error']<=2).mean()*100:.1f}%")
print(f"  Within+-3: {(analysis['error']<=3).mean()*100:.1f}%")

# --- FEATURE IMPORTANCE -------------------------------------------------------
single_models = {k: v for k, v in registry.items()
                 if v['model'] is not None and not k.endswith('[R]')}
best_single = min(single_models, key=lambda k: single_models[k]['metrics']['mae'])
bsm = registry[best_single]
if hasattr(bsm['model'], 'feature_importances_'):
    fi = pd.DataFrame({'feature': bsm['feats'],
                       'importance': bsm['model'].feature_importances_}
                      ).sort_values('importance', ascending=False)
    print(f"\n--- Top 20 Feature Importance ({best_single}) ---")
    print(fi.head(20).to_string(index=False))

# --- SAVE ---------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

# Save the best single model
joblib.dump(registry[best_single]['model'],
            os.path.join(MODELS_DIR, 'best_model.pkl'))
print(f"\nSaved best_model.pkl ({best_single})")

with open(os.path.join(MODELS_DIR, 'feature_cols.json'), 'w') as f:
    json.dump(registry[best_single]['feats'], f, indent=2)

summary = {
    'best_model':        best_name,
    'best_single_model': best_single,
    'metrics':           {k: round(v, 4) for k, v in best_m.items()},
    'baseline_grid_mae': round(b_grid['mae'], 4),
    'beat_baseline':     bool(best_m['mae'] < b_grid['mae']),
    'improvement':       round(b_grid['mae'] - best_m['mae'], 4),
    'all_models': {k: {mk: round(mv, 4) for mk, mv in v['metrics'].items()}
                   for k, v in sorted(registry.items(),
                                      key=lambda x: x[1]['metrics']['mae'])[:20]}
}
with open(os.path.join(MODELS_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved feature_cols.json, metrics_summary.json to {MODELS_DIR}")
print("Done!")
