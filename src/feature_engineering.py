# import pandas as pd
#
# df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\clean_race_data.csv')
#
# # Sort first - very important!
# df = df.sort_values(['driver', 'season', 'round']).reset_index(drop=True)
#
# # avg finish last 5 races per driver
# df['avg_finish_last5'] = (
#     df.groupby('driver')['finish_position']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
# )
#
# # print(df[['driver', 'season', 'round', 'finish_position', 'avg_finish_last5']].head(20))
#
# #Weighted_finish_form
# df['weighted_finish_form'] = (
#     df.groupby('driver')['finish_position']
#     .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
# )
# # print(df[df['driver'] == 'VER'][['round', 'finish_position', 'avg_finish_last5', 'weighted_finish_form']].head(20))
#
# # finish position std last 5
# df['finish_std_last5'] = (
#     df.groupby('driver')['finish_position']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
# )
#
# # print(df['status'].unique())
#
# #if status is not in finished_statuses, then it's a DNF
# finished_statuses = ['Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+6 Laps', 'Lapped']
#
# df['is_dnf'] = (~df['status'].isin(finished_statuses)).astype(int)
#
# # print(df['is_dnf'].value_counts())
#
#
# #DNF rate last 5
# df['dnf_rate_last5'] = (
#     df.groupby('driver')['is_dnf']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
# )
#
# # print(df[df['is_dnf'] == 1][['driver', 'season', 'round', 'is_dnf', 'dnf_rate_last5']].head(10))
#
#
# # avg position gain last 5
# df['position_gain'] = df['grid_position'] - df['finish_position']
#
# df['avg_position_gain'] = (
#     df.groupby('driver')['position_gain']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
# )
#
# #avg team points last 3
# df['team_avg_points_last3'] = (
#     df.groupby(['team'])['points']
#     .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
# )
#
#
# #Calculate teammate delta - how much better/worse a driver did compared to their teammate(s) in the same race
# # First calculate average finish per team per race
# team_avg_finish = df.groupby(['season', 'round', 'team'])['finish_position'].transform('mean')
# # Then teammate delta is driver's finish minus team average
# df['teammate_delta'] = df['finish_position'] - team_avg_finish
#
# df['teammate_delta'] = (
#     df.groupby('driver')['teammate_delta']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
# )
#
# # print(df[['driver', 'season', 'round', 'team', 'finish_position', 'teammate_delta']].head(20))
#
# # Driver vs field - how much better/worse a driver did compared to the average of the entire field in that race
# race_avg_finish = df.groupby(['season', 'round'])['finish_position'].transform('mean')
#
# df['driver_vs_field'] = df['finish_position'] - race_avg_finish
#
# df['driver_vs_field'] = (
#     df.groupby('driver')['driver_vs_field']
#     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
# )
#
# # circuit_avg_finish - average finish position for a driver at a specific circuit (race_name) over the last 3(as data of 3 years so used 3) times they've raced there
# df['circuit_avg_finish'] = (
#     df.groupby(['driver', 'race_name'])['finish_position']
#     .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
# )
#
# # circuit_avg_gain
# df['circuit_avg_gain'] = (
#     df.groupby(['driver', 'race_name'])['position_gain']
#     .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
# )
#
# # driver points before race
# df['driver_points_before_race'] = (
#     df.groupby('driver')['points']
#     .transform(lambda x: x.shift(1).cumsum())
# )
#
# # team points before race
# df['team_points_before_race'] = (
#     df.groupby('team')['points']
#     .transform(lambda x: x.shift(1).cumsum())
# )
#
# # driver rank before race
# df['driver_rank_before_race'] = (
#     df.groupby(['season', 'round'])['driver_points_before_race']
#     .rank(ascending=False, method='min')
# )
#
# # print(df[['driver', 'season', 'round', 'driver_points_before_race', 'driver_rank_before_race']].head(20))
#
# # import os
# # os.makedirs('data/processed', exist_ok=True)
# # df.to_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\featured_race_data.csv', index=False)
# # print(f"Done! {df.shape}")
# # print(df.columns.tolist())
#
# df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\featured_race_data.csv')
#
# # For rolling features - fill with overall driver average
# df['avg_finish_last5'] = df.groupby('driver')['avg_finish_last5'].transform(lambda x: x.fillna(x.mean()))
#
# # For circuit features - fill with overall circuit average
# df['circuit_avg_finish'] = df.groupby('race_name')['circuit_avg_finish'].transform(lambda x: x.fillna(x.mean()))
# df['circuit_avg_gain'] = df.groupby('race_name')['circuit_avg_gain'].transform(lambda x: x.fillna(x.mean()))
#
# # Fill any remaining nulls with median
# df.fillna(df.median(numeric_only=True), inplace=True)
#
#
# print(df.shape)
# print(df.isnull().sum())
#
# df.to_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\featured_race_data.csv', index=False)
# print(f"Done! {df.shape}")

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



