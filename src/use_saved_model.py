import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

ENRICHED_PATH = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\enriched_race_data.csv'
FALLBACK_PATH = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\featured_race_data.csv'
MODELS_DIR    = r'D:\Coding\f1_prediction\f1_race_prediciton\models'

# Load data
if os.path.exists(ENRICHED_PATH):
    df = pd.read_csv(ENRICHED_PATH)
else:
    df = pd.read_csv(FALLBACK_PATH)

# Filter test set (2025) and remove DNFs
df = df[(df['season'] == 2025) & (df['is_dnf'] == 0)].copy()

# Load model and features
model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
feature_path = os.path.join(MODELS_DIR, 'feature_cols.json')

if not os.path.exists(model_path):
    print("No saved model found!")
    exit(1)

model = joblib.load(model_path)
with open(feature_path, 'r') as f:
    features = json.load(f)

print(f"Loaded model from {model_path}")
print(f"Loaded {len(features)} features for prediction")

# Target Encoding
train_all = pd.read_csv(ENRICHED_PATH if os.path.exists(ENRICHED_PATH) else FALLBACK_PATH)
train_all = train_all[(train_all['season'] <= 2024) & (train_all['is_dnf'] == 0)]
global_mean  = train_all['finish_position'].mean()
driver_means = train_all.groupby('driver')['finish_position'].mean()
team_means   = train_all.groupby('team')['finish_position'].mean()

df['driver_enc'] = df['driver'].map(driver_means).fillna(global_mean)
df['team_enc']   = df['team'].map(team_means).fillna(global_mean)

# Interactions
df['circuit_expected_finish'] = df['grid_position'] - df['circuit_avg_gain']
df['grid_form_gap']           = df['grid_position'] - df['weighted_finish_form']
df['form_trend']              = df['avg_finish_last5'] - df['weighted_finish_form']
df['driver_strength']         = (df['driver_enc'] + df['driver_rank_before_race']) / 2

if 'quali_gap_to_pole' in df.columns:
    df['quali_gap_log']   = np.log1p(df['quali_gap_to_pole'])
    df['q3_qualifier']    = (df['quali_gap_to_pole'] < 1.5).astype(int)
    df['quali_x_grid']    = df['quali_gap_to_pole'] * df['grid_position'] / 20.0
    df['wet_x_form']      = df['is_wet_race'] * df['weighted_finish_form']

X_test = df[features].fillna(0)

# Predict delta -> reconstruct raw finish
delta_preds = model.predict(X_test)
raw_preds = df['grid_position'].values - delta_preds

# Rank within race
df['pred_raw'] = raw_preds
df['pred_ranked'] = df.groupby(['season', 'round'])['pred_raw'].rank(method='first').astype(int)

# Evaluate
y_true = df['finish_position'].values
y_pred = df['pred_ranked'].values
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
sr = spearmanr(y_true, y_pred)
spear = sr.statistic if hasattr(sr, 'statistic') else sr[0]
within_2 = (np.abs(y_true - y_pred) <= 2).mean() * 100

print("\n--- 2025 Test Evaluation ---")
print(f"MAE:       {mae:.3f}")
print(f"RMSE:      {rmse:.3f}")
print(f"Spearman:  {spear:.3f}")
print(f"Within +-2:{within_2:.1f}%")

print("\n--- Sample Predictions (Next Race) ---")
first_2025_race = df[df['round'] == df['round'].min()]
sample = first_2025_race[['driver', 'team', 'grid_position', 'pred_ranked', 'finish_position']].sort_values('pred_ranked')
sample.columns = ['Driver', 'Team', 'Grid', 'Predicted Finish', 'Actual Finish']
print(sample.to_string(index=False))

# Export outputs
output_path = r'D:\Coding\f1_prediction\f1_race_prediciton\outputs\predictions_2025.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df[['season', 'round', 'race_name', 'driver', 'team', 'grid_position', 'pred_ranked', 'finish_position']].to_csv(output_path, index=False)
print(f"\nSaved full predictions to {output_path}")
