import json, pyarrow.dataset as ds
from pathlib import Path

DATA_PATH = Path('/kaggle/input/datasets/yashag03/data-final-parquet/data_final.parquet')
TARGET = 'kwh_norm'

# Alle Spots analysieren
meta = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])

# Die 406 bereits genutzten Spots
selected = stats.sort_values(['clean', 'total'], ascending=False).iloc[:406]['spot_uuid'].tolist()

# Alle anderen Spots — nach Sauberkeit und Vollständigkeit sortiert
unseen_stats = stats[~stats['spot_uuid'].isin(selected)].copy()
unseen_stats = unseen_stats.sort_values(['clean', 'total'], ascending=False)

# Top 100 sauberste und vollständigste ungesehene Spots
oos_spots = unseen_stats.iloc[:100]['spot_uuid'].tolist()

print(f"OOS Spots: {len(oos_spots)}")
print(f"Durchschnittliche Sauberkeit: {unseen_stats.iloc[:100]['clean'].mean():.4f}")
print(f"Durchschnittliche Länge: {unseen_stats.iloc[:100]['total'].mean():.0f} Zeilen")

with open('oos_spots.json', 'w') as f:
    json.dump(oos_spots, f)

print("oos_spots.json gespeichert — als Kaggle Dataset hochladen!")