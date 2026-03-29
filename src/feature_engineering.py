import pandas as pd
import os

df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\cleaned_race_data.csv')

# Sort first - very important!
df = df.sort_values(['driver', 'season', 'round']).reset_index(drop=True)

# avg finish last 5 races per driver
df['avg_finish_last5'] = (
    df.groupby('driver')['finish_position']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Weighted finish form
df['weighted_finish_form'] = (
    df.groupby('driver')['finish_position']
    .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
)

# Finish std last 5
df['finish_std_last5'] = (
    df.groupby('driver')['finish_position']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
)

# DNF flag
finished_statuses = ['Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+6 Laps', 'Lapped']
df['is_dnf'] = (~df['status'].isin(finished_statuses)).astype(int)

# DNF rate last 5
df['dnf_rate_last5'] = (
    df.groupby('driver')['is_dnf']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Position gain
df['position_gain'] = df['grid_position'] - df['finish_position']

# Avg position gain last 5
df['avg_position_gain'] = (
    df.groupby('driver')['position_gain']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Team avg points last 3
df['team_avg_points_last3'] = (
    df.groupby('team')['points']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Teammate delta
team_avg_finish = df.groupby(['season', 'round', 'team'])['finish_position'].transform('mean')
df['teammate_delta'] = df['finish_position'] - team_avg_finish
df['teammate_delta'] = (
    df.groupby('driver')['teammate_delta']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Driver vs field
race_avg_finish = df.groupby(['season', 'round'])['finish_position'].transform('mean')
df['driver_vs_field'] = df['finish_position'] - race_avg_finish
df['driver_vs_field'] = (
    df.groupby('driver')['driver_vs_field']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Circuit avg finish
df['circuit_avg_finish'] = (
    df.groupby(['driver', 'race_name'])['finish_position']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Circuit avg gain
df['circuit_avg_gain'] = (
    df.groupby(['driver', 'race_name'])['position_gain']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# Driver points before race
df['driver_points_before_race'] = (
    df.groupby('driver')['points']
    .transform(lambda x: x.shift(1).cumsum())
)

# Team points before race
df['team_points_before_race'] = (
    df.groupby('team')['points']
    .transform(lambda x: x.shift(1).cumsum())
)

# Driver rank before race
df['driver_rank_before_race'] = (
    df.groupby(['season', 'round'])['driver_points_before_race']
    .rank(ascending=False, method='min')
)


# Get previous season team for each driver
df['prev_team'] = df.groupby('driver')['team'].shift(1)

# Flag if team changed
df['team_changed'] = (df['team'] != df['prev_team']).astype(int)

# print(df[df['team_changed'] == 1][['driver', 'season', 'round', 'team', 'prev_team']].head(10))





# Fill nulls
df['avg_finish_last5'] = df.groupby('driver')['avg_finish_last5'].transform(lambda x: x.fillna(x.mean()))
df['circuit_avg_finish'] = df.groupby('race_name')['circuit_avg_finish'].transform(lambda x: x.fillna(x.mean()))
df['circuit_avg_gain'] = df.groupby('race_name')['circuit_avg_gain'].transform(lambda x: x.fillna(x.mean()))
# Fill NaN team_changed with 1
# (first appearance = no history = treat as team change)
df['team_changed'] = df['team_changed'].fillna(1).astype(int)
df.fillna(df.median(numeric_only=True), inplace=True)
#
# Save
os.makedirs('data/processed', exist_ok=True)
df.to_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\featured_race_data.csv', index=False)
# print(f"Done! {df.shape}")
# print(df.isnull().sum())



