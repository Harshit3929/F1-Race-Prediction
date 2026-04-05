import fastf1
import time
import pandas as pd
import os

fastf1.Cache.enable_cache('data/raw')

seasons = [2022, 2023, 2024, 2025]
all_races = []
#
for season in seasons:
   schedule = fastf1.get_event_schedule(season, include_testing=False)

   for _, event in schedule.iterrows():
        round_num = event['RoundNumber']
        race_name = event['EventName']

        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)

            results = session.results

            for _, driver in results.iterrows():
                all_races.append({
                    'season': season,
                    'round': round_num,
                    'race_name': race_name,
                    'driver': driver['Abbreviation'],
                    'team': driver['TeamName'],
                    'grid_position': driver['GridPosition'],
                    'finish_position': driver['ClassifiedPosition'],
                    'points': driver['Points'],
                    'status': driver['Status'],
                    'laps': driver['Laps']
                })

        except Exception as e:
            print(f"Skipping {season} Round {round_num}: {e}")
            continue

        time.sleep(3)

df = pd.DataFrame(all_races)
os.makedirs('data/raw', exist_ok=True)
df.to_csv('D:\\Coding\\f1_prediction\\f1_race_prediciton\\data\\raw\\raw_race_data.csv', index=False)
print(f"Done! {len(df)} rows saved.")



