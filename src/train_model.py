import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import keras
from keras import layers

# Load featured data
df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\featured_race_data.csv')

# Define features and target
feature_cols = [
    'grid_position',
    'avg_finish_last5',
    'weighted_finish_form',
    'finish_std_last5',
    'dnf_rate_last5',
    'avg_position_gain',
    'team_avg_points_last3',
    'teammate_delta',
    'driver_vs_field',
    'circuit_avg_finish',
    'circuit_avg_gain',
    'driver_points_before_race',
    'team_points_before_race',
    'driver_rank_before_race',
    # 'team_changed'
]

# Target variable
target = 'finish_position'

# Temporal split - train on 2022-2024, test on 2025
train = df[df['season'] == 2024]
test = df[df['season'] == 2025]

X_train = train[feature_cols]
y_train = train[target]
X_test = test[feature_cols]
y_test = test[target]

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

#  BASELINES

# Baseline 1 - predict grid position as finish position
baseline1_preds = test['grid_position']
mae_b1 = mean_absolute_error(y_test, baseline1_preds)

# Baseline 2 - predict avg finish last 5 as finish position
baseline2_preds = test['avg_finish_last5']
mae_b2 = mean_absolute_error(y_test, baseline2_preds)

print(f"\nBaseline 1 (Grid Position) MAE: {mae_b1:.2f}")
print(f"Baseline 2 (Avg Finish Last 5) MAE: {mae_b2:.2f}")

#  RANDOM FOREST

# Train random forest regressor
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.7,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_preds)
print(f"\nRandom Forest MAE: {mae_rf:.2f}")

# Give more weight to recent seasons
weights = train['season'].map({
    2022: 0.5,   # least recent - half weight
    2023: 0.75,  # medium weight
    2024: 1.0    # most recent - full weight
})

# Train RF with sample weights
rf_weighted = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.7,
    random_state=42
)

rf_weighted.fit(X_train, y_train, sample_weight=weights)
rf_weighted_preds = rf_weighted.predict(X_test)
mae_rf_weighted = mean_absolute_error(y_test, rf_weighted_preds)
print(f"RF Weighted MAE: {mae_rf_weighted:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# XGBOOST

# Train XGBoost regressor
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, xgb_preds)
print(f"\nXGBoost MAE: {mae_xgb:.2f}")

# XGBOOST WEIGHTED

xgb_weighted = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42
)

xgb_weighted.fit(X_train, y_train, sample_weight=weights)
xgb_weighted_preds = xgb_weighted.predict(X_test)
mae_xgb_weighted = mean_absolute_error(y_test, xgb_weighted_preds)
print(f"XGBoost Weighted MAE: {mae_xgb_weighted:.2f}")

#  NEURAL NETWORK

# Normalize features - essential for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network - Input → Dense(ReLU) → Dense(ReLU) → Linear Output
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation - regression output
])

# Compile with MSE loss
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

nn_preds = model.predict(X_test_scaled).flatten()
mae_nn = mean_absolute_error(y_test, nn_preds)
print(f"\nNeural Network MAE: {mae_nn:.2f}")

# ENSEMBLE

# Average predictions from all 3 models
ensemble_preds = (rf_preds + xgb_preds + nn_preds) / 3
mae_ensemble = mean_absolute_error(y_test, ensemble_preds)
print(f"\nEnsemble MAE: {mae_ensemble:.2f}")

ensemble_weighted_preds = (rf_weighted_preds + xgb_weighted_preds) / 2
mae_ensemble_weighted = mean_absolute_error(y_test, ensemble_weighted_preds)
print(f"Ensemble Weighted MAE: {mae_ensemble_weighted:.2f}")



# Best ensemble - combine our two best models
best_ensemble_preds = (rf_preds + rf_weighted_preds) / 2
mae_best_ensemble = mean_absolute_error(y_test, best_ensemble_preds)
print(f"Best Ensemble (RF2024+RFWeighted): {mae_best_ensemble:.2f}")

# SUMMARY

print("\n─── Model Comparison ───")
print(f"Baseline 1 (Grid Position) : {mae_b1:.2f}")
print(f"Baseline 2 (Avg Finish L5) : {mae_b2:.2f}")
print(f"Random Forest              : {mae_rf:.2f}")
print(f"XGBoost                    : {mae_xgb:.2f}")
print(f"Neural Network             : {mae_nn:.2f}")
print(f"Ensemble                   : {mae_ensemble:.2f}")
print(f"RF Weighted                : {mae_rf_weighted:.2f}")
print(f"XGBoost Weighted           : {mae_xgb_weighted:.2f}")
print(f"Ensemble Weighted          : {mae_ensemble_weighted:.2f}")
print(f"Best Ensemble (RF2024+RFWeighted): {mae_best_ensemble:.2f}")




# Save models and artifacts

import joblib
import os
import json

os.makedirs('models', exist_ok=True)

# Save all models
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(rf_weighted, 'models/rf_weighted.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(xgb_weighted, 'models/xgb_weighted.pkl')
joblib.dump(model, 'models/nn_model.keras')

# Save scaler for neural network
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature columns list
with open('models/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print("All models saved!")
print(os.listdir('models'))



# FULL 2025 SEASON PREDICTION ANALYSIS

# Add predictions to test dataframe
test_results = test.copy()
test_results['predicted_position'] = rf_weighted_preds
test_results['error'] = np.abs(test_results['finish_position'] - test_results['predicted_position'])

# Convert predicted positions to rankings per race
test_results['predicted_rank'] = test_results.groupby(['season', 'round'])['predicted_position'].rank(method='min')
test_results['actual_rank'] = test_results.groupby(['season', 'round'])['finish_position'].rank(method='min')

# Per race accuracy
race_summary = test_results.groupby(['round', 'race_name']).agg(
    mae=('error', 'mean'),
    within_2=('error', lambda x: (x <= 2).mean() * 100)
).reset_index()

print("─── Per Race Accuracy (2025 Season) ───")
print(f"{'Round':<8} {'Race':<35} {'MAE':>6} {'Within±2':>10}")
print("-" * 62)
for _, row in race_summary.iterrows():
    print(f"{int(row['round']):<8} {row['race_name']:<35} {row['mae']:>6.2f} {row['within_2']:>9.1f}%")

print(f"\nOverall 2025 MAE        : {test_results['error'].mean():.2f}")
print(f"Overall Within ±2       : {(test_results['error'] <= 2).mean() * 100:.1f}%")

# Best and worst predicted races
print(f"\nBest predicted race  : {race_summary.loc[race_summary['mae'].idxmin(), 'race_name']} (MAE: {race_summary['mae'].min():.2f})")
print(f"Worst predicted race : {race_summary.loc[race_summary['mae'].idxmax(), 'race_name']} (MAE: {race_summary['mae'].max():.2f})")




# Overall accuracy percentage
exact_match = (np.round(test_results['predicted_position']) == test_results['finish_position']).mean() * 100
within_1 = (test_results['error'] <= 1).mean() * 100
within_2 = (test_results['error'] <= 2).mean() * 100
within_3 = (test_results['error'] <= 3).mean() * 100

print("\n─── Overall Accuracy ───")
print(f"Exact position match  : {exact_match:.1f}%")
print(f"Within ±1 position    : {within_1:.1f}%")
print(f"Within ±2 positions   : {within_2:.1f}%")
print(f"Within ±3 positions   : {within_3:.1f}%")