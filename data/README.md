# Data

Die Rohdaten sind nicht im Repository enthalten (zu groß / proprietär).
Bitte die Dateien in die entsprechenden Ordner legen:

## data/raw/
- `grid_feed_in_15min_202603271142.csv` — PV-Produktionsdaten (15-min, alle Spots)
- `grid_feed_in_15min_202603251621.csv` — PV-Produktionsdaten (ältere Version)
- `location_meta_202603250933.csv` — Anlagen-Metadaten (Standort, Kapazität)

## data/raw/2026_03_10/
- `helios_location_energy_15min.csv` — Verbrauchsdaten (für UMBRA)
- `helios_location_users.csv` — Zuordnung Location → Spot
- `day_ahead_prices.csv` — Day-Ahead Strompreise
- `spt_pv_c_forecast.csv` — SPT Referenzprognose

## data/processed/
Wird automatisch durch `01_data_prep.ipynb` generiert:
- `data_final.parquet` — Aufbereiteter Datensatz mit allen Features (67M Zeilen)
- `unseen_data.parquet` — 100 OOS-Spots (Hold-out)
