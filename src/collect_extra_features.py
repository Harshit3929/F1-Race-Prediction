"""
Fetches qualifying gap-to-pole, race weather, and starting tire compound
for every race in featured_race_data.csv using fastf1.
Output: data/processed/enriched_race_data.csv

Run from project root: python src/collect_extra_features.py
"""
import fastf1
import pandas as pd
import numpy as np
import os

CACHE_DIR = r'D:\Coding\f1_prediction\f1_race_prediciton\data\raw'
DATA_IN    = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\featured_race_data.csv'
DATA_OUT   = r'D:\Coding\f1_prediction\f1_race_prediciton\data\processed\enriched_race_data.csv'

fastf1.Cache.enable_cache(CACHE_DIR)

# ── helpers ───────────────────────────────────────────────────────────────────
def to_sec(t):
    """Timedelta / NaT → float seconds, else NaN."""
    try:
        if pd.isna(t):
            return np.nan
        s = t.total_seconds() if hasattr(t, 'total_seconds') else float(t)
        return s if s > 0 else np.nan
    except Exception:
        return np.nan

def best_quali_sec(row):
    """Driver's best qualifying time in seconds (Q3 > Q2 > Q1 priority)."""
    for col in ['Q3', 'Q2', 'Q1']:
        s = to_sec(row.get(col, np.nan))
        if not np.isnan(s):
            return s
    return np.nan

# ── load base data ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)
races = (df[['season', 'round', 'race_name']]
         .drop_duplicates()
         .sort_values(['season', 'round'])
         .reset_index(drop=True))

print(f"Fetching extra features for {len(races)} races across seasons "
      f"{sorted(df['season'].unique().tolist())}...\n")

extra_rows = []

for _, race_row in races.iterrows():
    season    = int(race_row['season'])
    round_num = int(race_row['round'])
    race_name = race_row['race_name']
    print(f"  {season} R{round_num:02d}: {race_name}")

    quali_gaps     = {}   # driver_abbr -> gap_to_pole (sec)
    is_wet         = 0
    air_temp       = np.nan
    tire_compounds = {}   # driver_abbr -> compound string

    # ── QUALIFYING ────────────────────────────────────────────────────────────
    try:
        qsess = fastf1.get_session(season, round_num, 'Q')
        qsess.load(telemetry=False, weather=False, messages=False, laps=False)
        qres = qsess.results

        if qres is not None and len(qres) > 0:
            qres = qres[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
            qres['best_sec'] = qres.apply(best_quali_sec, axis=1)
            pole_time = qres['best_sec'].min()

            for _, dr in qres.iterrows():
                bs = dr['best_sec']
                gap = (bs - pole_time) if not np.isnan(bs) else np.nan
                quali_gaps[dr['Abbreviation']] = gap
    except Exception as e:
        print(f"    Quali error: {e}")

    # ── RACE: WEATHER + TIRES ────────────────────────────────────────────────
    try:
        rsess = fastf1.get_session(season, round_num, 'R')
        rsess.load(telemetry=False, weather=True, messages=False, laps=True)

        # Weather
        wd = rsess.weather_data
        if wd is not None and len(wd) > 0:
            is_wet   = int(bool(wd['Rainfall'].any()))
            air_temp = float(wd['AirTemp'].mean())

        # Starting tire compound: first compound seen per driver
        laps = rsess.laps
        if laps is not None and len(laps) > 0:
            lap1 = (laps.sort_values('LapNumber')
                        .groupby('Driver')
                        .first()
                        .reset_index()[['Driver', 'Compound']]
                        .dropna(subset=['Compound']))
            for _, lr in lap1.iterrows():
                tire_compounds[lr['Driver']] = str(lr['Compound']).upper()
    except Exception as e:
        print(f"    Race error: {e}")

    # ── BUILD ROWS ────────────────────────────────────────────────────────────
    race_drivers = df[(df['season'] == season) & (df['round'] == round_num)]['driver'].unique()
    for driver in race_drivers:
        extra_rows.append({
            'season':           season,
            'round':            round_num,
            'driver':           driver,
            'quali_gap_to_pole': quali_gaps.get(driver, np.nan),
            'is_wet_race':      is_wet,
            'air_temp':         air_temp,
            'starting_compound': tire_compounds.get(driver, 'UNKNOWN'),
        })

# ── BUILD EXTRA DATAFRAME ─────────────────────────────────────────────────────
extra_df = pd.DataFrame(extra_rows)

# Ordinal-encode compound: SOFT=1, MEDIUM=2, HARD=3, INTERMEDIATE=4, WET=5
compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
extra_df['compound_enc'] = extra_df['starting_compound'].map(compound_map).fillna(2).astype(int)

# ── MERGE WITH BASE DATA ──────────────────────────────────────────────────────
enriched = df.merge(
    extra_df[['season', 'round', 'driver',
              'quali_gap_to_pole', 'is_wet_race', 'air_temp', 'compound_enc']],
    on=['season', 'round', 'driver'],
    how='left'
)

# Fill missing quali gap with per-race median, then global median
enriched['quali_gap_to_pole'] = (
    enriched.groupby(['season', 'round'])['quali_gap_to_pole']
            .transform(lambda x: x.fillna(x.median()))
)
enriched['quali_gap_to_pole'] = enriched['quali_gap_to_pole'].fillna(
    enriched['quali_gap_to_pole'].median()
)
enriched['air_temp']     = enriched['air_temp'].fillna(enriched['air_temp'].median())
enriched['is_wet_race']  = enriched['is_wet_race'].fillna(0).astype(int)
enriched['compound_enc'] = enriched['compound_enc'].fillna(2).astype(int)

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
enriched.to_csv(DATA_OUT, index=False)

print(f"\nSaved {DATA_OUT}")
print(f"Shape: {enriched.shape}")
print(f"New columns: quali_gap_to_pole, is_wet_race, air_temp, compound_enc")
print(f"\nSample:")
print(enriched[['season', 'round', 'driver', 'quali_gap_to_pole',
                'is_wet_race', 'air_temp', 'compound_enc']].head(10).to_string(index=False))

wet_races = enriched[enriched['is_wet_race'] == 1][['season', 'round', 'race_name']].drop_duplicates()
print(f"\nWet races detected ({len(wet_races)}):")
print(wet_races.to_string(index=False))
