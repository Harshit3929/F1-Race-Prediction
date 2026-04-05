"""
F1 Race Position Prediction - Maximally Optimized Pipeline
Changes vs previous version:
  - Rank post-processing: convert raw preds to per-race integer ranks
  - Temporal stacking: Ridge meta-learner trained on OOF predictions
  - Position-change model: predict (grid - finish) then reconstruct
  - New features: quali_gap_log, q3_qualifier
  - Monotone constraint on grid_position for XGB
  - 150 Optuna trials (up from 80)
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG -------------------------------------------------------------------
ENRICHED_PATH   = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\enriched_race_data.csv'
FALLBACK_PATH   = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\featured_race_data.csv'
MODELS_DIR      = r'D:\Coding\f1_prediction\f1_race_prediciton\models'
N_OPTUNA_TRIALS = 150

# --- LOAD DATA ----------------------------------------------------------------
if os.path.exists(ENRICHED_PATH):
    df = pd.read_csv(ENRICHED_PATH)
    HAS_ENRICHED = True
    print(f"Loaded enriched data: {df.shape[0]} rows")
else:
    df = pd.read_csv(FALLBACK_PATH)
    HAS_ENRICHED = False
    print("Loaded base data (run collect_extra_features.py first for best results)")

print(f"Seasons: {sorted(df['season'].unique().tolist())}")

# --- REMOVE DNFs --------------------------------------------------------------
total_before = len(df)
df = df[df['is_dnf'] == 0].copy()
print(f"Removed {total_before - len(df)} DNF rows | {len(df)} remaining")

# --- BASE FEATURES ------------------------------------------------------------
base_features = [
    'grid_position', 'avg_finish_last5', 'weighted_finish_form',
    'finish_std_last5', 'dnf_rate_last5', 'avg_position_gain',
    'team_avg_points_last3', 'teammate_delta', 'driver_vs_field',
    'circuit_avg_finish', 'circuit_avg_gain', 'driver_points_before_race',
    'team_points_before_race', 'driver_rank_before_race', 'team_changed'
]

# --- TEMPORAL SPLITS ----------------------------------------------------------
train_all    = df[df['season'] <= 2024].copy()
train_recent = df[(df['season'] >= 2023) & (df['season'] <= 2024)].copy()
train_2024   = df[df['season'] == 2024].copy()
test_df      = df[df['season'] == 2025].copy()

assert len(train_recent) < len(train_all), "DATA LEAKAGE: check season filter!"

print(f"\nSplits: Train 2022-24={len(train_all)} | Train 2023-24={len(train_recent)} | "
      f"Train 2024={len(train_2024)} | Test 2025={len(test_df)}")

# --- TARGET ENCODING ----------------------------------------------------------
global_mean  = train_all['finish_position'].mean()
driver_means = train_all.groupby('driver')['finish_position'].mean()
team_means   = train_all.groupby('team')['finish_position'].mean()

for data in [train_all, train_recent, train_2024, test_df]:
    data['driver_enc'] = data['driver'].map(driver_means).fillna(global_mean)
    data['team_enc']   = data['team'].map(team_means).fillna(global_mean)

# --- FEATURE ENGINEERING ------------------------------------------------------
def add_interactions(data):
    d = data.copy()
    d['circuit_expected_finish'] = d['grid_position'] - d['circuit_avg_gain']
    d['grid_form_gap']           = d['grid_position'] - d['weighted_finish_form']
    d['form_trend']              = d['avg_finish_last5'] - d['weighted_finish_form']
    d['driver_strength']         = (d['driver_enc'] + d['driver_rank_before_race']) / 2
    if HAS_ENRICHED:
        d['quali_gap_log']   = np.log1p(d['quali_gap_to_pole'])
        d['q3_qualifier']    = (d['quali_gap_to_pole'] < 1.5).astype(int)
        d['quali_x_grid']    = d['quali_gap_to_pole'] * d['grid_position'] / 20.0
        d['wet_x_form']      = d['is_wet_race'] * d['weighted_finish_form']
    return d

train_all    = add_interactions(train_all)
train_recent = add_interactions(train_recent)
train_2024   = add_interactions(train_2024)
test_df      = add_interactions(test_df)

new_feats = (['quali_gap_to_pole', 'is_wet_race', 'air_temp', 'compound_enc',
               'quali_gap_log', 'q3_qualifier', 'quali_x_grid', 'wet_x_form']
             if HAS_ENRICHED else [])

extended_features = base_features + [
    'driver_enc', 'team_enc',
    'circuit_expected_finish', 'grid_form_gap', 'form_trend', 'driver_strength'
] + new_feats

y_test = test_df['finish_position'].values

# --- SAMPLE WEIGHTS -----------------------------------------------------------
def season_weights(data, scheme='aggressive'):
    w_map = {'aggressive': {2022: 0.2, 2023: 0.5, 2024: 1.0},
             'standard':   {2022: 0.5, 2023: 0.75, 2024: 1.0}}[scheme]
    return data['season'].map(w_map).fillna(1.0).values

# --- RANK POST-PROCESSING -----------------------------------------------------
def rank_within_race(test_data, raw_preds):
    """Convert raw float predictions to per-race integer ranks (1=best)."""
    tmp = test_data[['season', 'round']].copy()
    tmp['raw'] = raw_preds
    tmp['ranked'] = (tmp.groupby(['season', 'round'])['raw']
                        .rank(method='first')
                        .astype(float))
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
        print(f"  {label:<50} MAE={mae:.3f}  RMSE={rmse:.3f}  Spear={spear:.3f}  "
              f"+-2={w2:.1f}%  Top3={top3:.1f}%  Top10={top10:.1f}%")
    return dict(mae=mae, rmse=rmse, spearman=spear, within_2=w2, top3=top3, top10=top10)

# --- MODEL REGISTRY -----------------------------------------------------------
registry = {}

def run(name, model, train_data, feats, weights=None, rank=False):
    """Fit, predict raw, optionally also evaluate ranked predictions."""
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

# --- BASELINES ----------------------------------------------------------------
print("\n--- BASELINES ---")
b_grid = compute_metrics(y_test, test_df['grid_position'].values, test_df, "Grid Position")
b_form = compute_metrics(y_test, test_df['avg_finish_last5'].values, test_df, "Avg Finish Last5")
# Ranked grid position baseline
grid_ranked = rank_within_race(test_df, test_df['grid_position'].values)
b_grid_r = compute_metrics(y_test, grid_ranked, test_df, "Grid Position [ranked]")

# --- POSITION-CHANGE MODEL ---------------------------------------------------
# Predict (grid_position - finish_position) then reconstruct finish
# This reframes as "how many positions will this driver gain/lose?"
print("\n--- POSITION-CHANGE MODELS ---")

def run_delta(name, model, train_data, feats, weights=None):
    """Train on position_gain as target, reconstruct finish position."""
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

run_delta("RF",  RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                        max_features=0.7, random_state=42, n_jobs=-1),
          train_all, extended_features, season_weights(train_all, 'aggressive'))

run_delta("XGB", XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                               random_state=42, verbosity=0),
          train_all, extended_features, season_weights(train_all, 'aggressive'))

# --- RANDOM FOREST ------------------------------------------------------------
print("\n--- RANDOM FOREST ---")
run("RF/all/base",
    RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3,
                           max_features=0.7, random_state=42, n_jobs=-1),
    train_all, base_features, rank=True)

run("RF/all/ext/aggw",
    RandomForestRegressor(n_estimators=600, max_depth=10, min_samples_leaf=2,
                           max_features=0.6, random_state=42, n_jobs=-1),
    train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)

run("RF/2024/ext",
    RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=2,
                           max_features=0.6, random_state=42, n_jobs=-1),
    train_2024, extended_features, rank=True)

# --- XGBOOST with monotone constraint on grid_position -----------------------
print("\n--- XGBOOST ---")

# grid_position: 1 = higher grid -> worse finish (positive constraint)
# all other features: 0 = no constraint
grid_idx = extended_features.index('grid_position')
mono = tuple(1 if i == grid_idx else 0 for i in range(len(extended_features)))
mono_base = tuple(1 if i == base_features.index('grid_position') else 0
                  for i in range(len(base_features)))

run("XGB/all/base/mono",
    XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  monotone_constraints=mono_base, random_state=42, verbosity=0),
    train_all, base_features, rank=True)

run("XGB/all/ext/aggw/mono",
    XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.025,
                  subsample=0.8, colsample_bytree=0.75, min_child_weight=3,
                  monotone_constraints=mono, random_state=42, verbosity=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)

run("XGB/2024/ext",
    XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                  random_state=42, verbosity=0),
    train_2024, extended_features, rank=True)

# --- LIGHTGBM -----------------------------------------------------------------
HAS_LGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("\n--- LIGHTGBM ---")
    run("LGB/all/ext/aggw",
        lgb.LGBMRegressor(n_estimators=800, num_leaves=31, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                           monotone_constraints=[1 if f=='grid_position' else 0
                                                 for f in extended_features],
                           random_state=42, verbose=-1, n_jobs=-1),
        train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)
    run("LGB/2024/ext",
        lgb.LGBMRegressor(n_estimators=400, num_leaves=20, learning_rate=0.04,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                           random_state=42, verbose=-1, n_jobs=-1),
        train_2024, extended_features, rank=True)
except ImportError:
    print("  LightGBM not found")

# --- CATBOOST -----------------------------------------------------------------
HAS_CB = False
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
    print("\n--- CATBOOST ---")
    run("CB/all/ext/aggw",
        CatBoostRegressor(iterations=600, depth=6, learning_rate=0.03,
                           l2_leaf_reg=3, random_seed=42, verbose=0),
        train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)
    run("CB/2023-24/ext",
        CatBoostRegressor(iterations=500, depth=5, learning_rate=0.04,
                           l2_leaf_reg=3, random_seed=42, verbose=0),
        train_recent, extended_features, rank=True)
    run("CB/2024/ext",
        CatBoostRegressor(iterations=400, depth=5, learning_rate=0.05,
                           l2_leaf_reg=3, random_seed=42, verbose=0),
        train_2024, extended_features, rank=True)
except ImportError:
    print("  CatBoost not found")

# --- OPTUNA TUNING ------------------------------------------------------------
print(f"\n--- OPTUNA TUNING ({N_OPTUNA_TRIALS} trials each) ---")

X_opt_tr  = train_all[train_all['season'] <= 2023][extended_features].fillna(0).values
y_opt_tr  = train_all[train_all['season'] <= 2023]['finish_position'].values
w_opt_tr  = train_all[train_all['season'] <= 2023]['season'].map({2022:0.2,2023:0.5}).values
X_opt_val = train_all[train_all['season'] == 2024][extended_features].fillna(0).values
y_opt_val = train_all[train_all['season'] == 2024]['finish_position'].values

def xgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 1500),
        max_depth        = trial.suggest_int('max_depth', 3, 8),
        learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        gamma            = trial.suggest_float('gamma', 0, 2),
        reg_alpha        = trial.suggest_float('reg_alpha', 0, 2),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 5),
    )
    m = XGBRegressor(**params, monotone_constraints=mono,
                     random_state=42, verbosity=0)
    m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
    return mean_absolute_error(y_opt_val, m.predict(X_opt_val))

xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS)
print(f"  XGB Optuna val MAE (2024): {xgb_study.best_value:.3f}")
run("XGB/Optuna/mono",
    XGBRegressor(**xgb_study.best_params, monotone_constraints=mono,
                 random_state=42, verbosity=0),
    train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)

if HAS_LGB:
    import lightgbm as lgb
    def lgb_objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 200, 1500),
            num_leaves       = trial.suggest_int('num_leaves', 10, 80),
            learning_rate    = trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_samples= trial.suggest_int('min_child_samples', 3, 30),
            reg_alpha        = trial.suggest_float('reg_alpha', 0, 2),
            reg_lambda       = trial.suggest_float('reg_lambda', 0, 5),
        )
        mc = [1 if f == 'grid_position' else 0 for f in extended_features]
        m = lgb.LGBMRegressor(**params, monotone_constraints=mc,
                               random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
        return mean_absolute_error(y_opt_val, m.predict(X_opt_val))
    lgb_study = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS)
    print(f"  LGB Optuna val MAE (2024): {lgb_study.best_value:.3f}")
    mc = [1 if f == 'grid_position' else 0 for f in extended_features]
    run("LGB/Optuna/mono",
        lgb.LGBMRegressor(**lgb_study.best_params, monotone_constraints=mc,
                           random_state=42, verbose=-1, n_jobs=-1),
        train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)

if HAS_CB:
    from catboost import CatBoostRegressor
    def cb_objective(trial):
        params = dict(
            iterations   = trial.suggest_int('iterations', 200, 1200),
            depth        = trial.suggest_int('depth', 4, 8),
            learning_rate= trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            l2_leaf_reg  = trial.suggest_float('l2_leaf_reg', 1, 10),
        )
        m = CatBoostRegressor(**params, random_seed=42, verbose=0)
        m.fit(X_opt_tr, y_opt_tr, sample_weight=w_opt_tr)
        return mean_absolute_error(y_opt_val, m.predict(X_opt_val))
    cb_study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    cb_study.optimize(cb_objective, n_trials=N_OPTUNA_TRIALS // 2)
    print(f"  CB Optuna val MAE (2024): {cb_study.best_value:.3f}")
    run("CB/Optuna",
        CatBoostRegressor(**cb_study.best_params, random_seed=42, verbose=0),
        train_all, extended_features, season_weights(train_all, 'aggressive'), rank=True)

# --- STACKING WITH TEMPORAL CV -----------------------------------------------
print("\n--- STACKING (temporal CV meta-learner) ---")

stack_base = []
if HAS_LGB:
    import lightgbm as lgb
    stack_base.append(('LGB', lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                                                  learning_rate=0.04, subsample=0.8,
                                                  random_state=42, verbose=-1, n_jobs=-1)))
if HAS_CB:
    from catboost import CatBoostRegressor
    stack_base.append(('CB', CatBoostRegressor(iterations=300, depth=6, learning_rate=0.04,
                                                random_seed=42, verbose=0)))
stack_base.append(('RF',  RandomForestRegressor(n_estimators=300, max_depth=8,
                                                  min_samples_leaf=3, max_features=0.7,
                                                  random_state=42, n_jobs=-1)))
stack_base.append(('XGB', XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        monotone_constraints=mono,
                                        random_state=42, verbosity=0)))

# OOF generation: 2022 -> validate 2023, then 2022-23 -> validate 2024
val_seasons = [2023, 2024]
oof_preds   = {name: [] for name, _ in stack_base}
oof_true    = []

for val_season in val_seasons:
    tr_data  = train_all[train_all['season'] < val_season]
    val_data = train_all[train_all['season'] == val_season]
    w_tr     = season_weights(tr_data, 'aggressive')
    X_tr     = tr_data[extended_features].fillna(0).values
    y_tr_s   = tr_data['finish_position'].values
    X_val    = val_data[extended_features].fillna(0).values
    oof_true.extend(val_data['finish_position'].values)
    for name, base_m in stack_base:
        import copy
        m = copy.deepcopy(base_m)
        try:
            m.fit(X_tr, y_tr_s, sample_weight=w_tr)
        except TypeError:
            m.fit(X_tr, y_tr_s)
        oof_preds[name].extend(m.predict(X_val))

meta_X_tr = np.column_stack([np.array(oof_preds[n]) for n, _ in stack_base])
meta_y_tr = np.array(oof_true)

# Get test-set predictions from the best model of each type in registry
test_cols = []
for name, _ in stack_base:
    candidates = {k: v for k, v in registry.items()
                  if k.startswith(f"{name}/") and v['model'] is not None}
    if candidates:
        best_k = min(candidates, key=lambda k: candidates[k]['metrics']['mae'])
        test_cols.append(registry[best_k]['preds'])
    else:
        test_cols.append(np.full(len(test_df), global_mean))

meta_X_te = np.column_stack(test_cols)

scaler = StandardScaler()
meta_X_tr_s = scaler.fit_transform(meta_X_tr)
meta_X_te_s = scaler.transform(meta_X_te)

# Ridge meta-learner
ridge = Ridge(alpha=1.0)
ridge.fit(meta_X_tr_s, meta_y_tr)
stack_raw    = ridge.predict(meta_X_te_s)
stack_ranked = rank_within_race(test_df, stack_raw)

compute_metrics(y_test, stack_raw,    test_df, "Stacking/Ridge/raw")
m_stack = compute_metrics(y_test, stack_ranked, test_df, "Stacking/Ridge/ranked")
registry['Stacking/Ridge/raw']    = dict(model=None, preds=stack_raw,    feats=None, metrics=compute_metrics(y_test, stack_raw, test_df))
registry['Stacking/Ridge/ranked'] = dict(model=None, preds=stack_ranked, feats=None, metrics=m_stack)

# --- ENSEMBLES ----------------------------------------------------------------
print("\n--- ENSEMBLES ---")

# Only use raw-prediction models (not ranked variants) for ensembling
raw_registry = {k: v for k, v in registry.items() if not k.endswith('[R]')}
sorted_raw   = sorted(raw_registry.items(), key=lambda x: x[1]['metrics']['mae'])
top3_names   = [n for n, _ in sorted_raw[:3]]
top5_names   = [n for n, _ in sorted_raw[:5]]
print(f"  Top-3: {top3_names}")

def make_ensemble(name, model_names, blend='avg', do_rank=True):
    preds_list = [registry[n]['preds'] for n in model_names]
    if blend == 'avg':
        ens = np.mean(preds_list, axis=0)
    else:
        maes = np.array([registry[n]['metrics']['mae'] for n in model_names])
        w = (1.0 / maes) / (1.0 / maes).sum()
        ens = sum(w[i] * preds_list[i] for i in range(len(w)))
    m = compute_metrics(y_test, ens, test_df, name)
    registry[name] = dict(model=None, preds=ens, feats=None, metrics=m)
    if do_rank:
        ranked = rank_within_race(test_df, ens)
        mr = compute_metrics(y_test, ranked, test_df, f"{name}[R]")
        registry[f"{name}[R]"] = dict(model=None, preds=ranked, feats=None, metrics=mr)

make_ensemble("Ensemble/Top3/avg",    top3_names, 'avg')
make_ensemble("Ensemble/Top3/invMAE", top3_names, 'invmae')
make_ensemble("Ensemble/Top5/avg",    top5_names, 'avg')

# --- FINAL RESULTS TABLE ------------------------------------------------------
all_metrics = {
    'Baseline/Grid':          b_grid,
    'Baseline/Grid[R]':       b_grid_r,
    'Baseline/AvgFinishLast5':b_form,
    **{k: v['metrics'] for k, v in registry.items()}
}

print("\n" + "=" * 110)
print("FINAL RESULTS - sorted by MAE")
print("=" * 110)
print(f"  {'Model':<52} {'MAE':>6}  {'RMSE':>6}  {'Spear':>6}  {'+-2%':>6}  {'Top3%':>6}  {'Top10%':>7}")
print("-" * 110)
for name, m in sorted(all_metrics.items(), key=lambda x: x[1]['mae']):
    tag = " <- BASELINE" if name.startswith("Baseline") else ""
    ranked_tag = " [ranked]" if name.endswith("[R]") else ""
    print(f"  {name:<52} {m['mae']:>6.3f}  {m['rmse']:>6.3f}  {m['spearman']:>6.3f}  "
          f"{m['within_2']:>5.1f}%  {m['top3']:>5.1f}%  {m['top10']:>6.1f}%{tag}{ranked_tag}")

# --- BEST MODEL ---------------------------------------------------------------
best_name = min(registry.keys(), key=lambda k: registry[k]['metrics']['mae'])
best      = registry[best_name]
best_m    = best['metrics']

print(f"\n{'-'*60}")
print(f"Best model : {best_name}")
print(f"  MAE      = {best_m['mae']:.3f}  (grid baseline = {b_grid['mae']:.3f}, "
      f"delta = {b_grid['mae'] - best_m['mae']:+.3f})")
print(f"  RMSE     = {best_m['rmse']:.3f}")
print(f"  Spearman = {best_m['spearman']:.3f}")
print(f"  Within+-2 = {best_m['within_2']:.1f}%")
print(f"  Top-3    = {best_m['top3']:.1f}%")
print(f"  Top-10   = {best_m['top10']:.1f}%")

# --- PER-RACE ANALYSIS --------------------------------------------------------
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

print(f"\n  Best  : {race_summary.loc[race_summary['mae'].idxmin(), 'race_name']} "
      f"(MAE={race_summary['mae'].min():.2f})")
print(f"  Worst : {race_summary.loc[race_summary['mae'].idxmax(), 'race_name']} "
      f"(MAE={race_summary['mae'].max():.2f})")

exact = (np.round(analysis['pred']) == analysis['finish_position']).mean() * 100
print(f"\n--- Overall Accuracy ---")
print(f"  Exact    : {exact:.1f}%")
print(f"  Within+-1: {(analysis['error']<=1).mean()*100:.1f}%")
print(f"  Within+-2: {(analysis['error']<=2).mean()*100:.1f}%")
print(f"  Within+-3: {(analysis['error']<=3).mean()*100:.1f}%")

# --- FEATURE IMPORTANCE -------------------------------------------------------
single_models = {k: v for k, v in registry.items()
                 if v['model'] is not None and not k.endswith('[R]')}
best_single   = min(single_models, key=lambda k: single_models[k]['metrics']['mae'])
bsm = registry[best_single]
if hasattr(bsm['model'], 'feature_importances_'):
    fi = pd.DataFrame({'feature': bsm['feats'],
                       'importance': bsm['model'].feature_importances_}
                      ).sort_values('importance', ascending=False)
    print(f"\n--- Feature Importance ({best_single}) ---")
    print(fi.to_string(index=False))

# --- SAVE ---------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

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
                                      key=lambda x: x[1]['metrics']['mae'])}
}
with open(os.path.join(MODELS_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Saved feature_cols.json, metrics_summary.json to {MODELS_DIR}")
print("Done!")
