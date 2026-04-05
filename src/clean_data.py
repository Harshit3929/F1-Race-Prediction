import os
import pandas as pd
df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\raw\\raw_race_data.csv')



df['grid_position'] = df['grid_position'].fillna(0)
print(df.isnull().sum())


df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')

df = df.sort_values(['driver', 'season', 'round']).reset_index(drop=True)
df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
drivers_per_race = df.groupby(['season', 'round'])['driver'].transform('count')
df['finish_position'] = df['finish_position'].fillna(drivers_per_race + 1)
#
os.makedirs('data/processed', exist_ok=True)
df.to_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\cleaned_race_data.csv', index=False)
print(f"Done! {len(df)} rows saved.")
print(df.dtypes)
#
# import pandas as pd
# df = pd.read_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\processed\\clean_race_data.csv')
# print(df.shape)
# print(df.dtypes)
# print(df.isnull().sum())
# print(df.head())
