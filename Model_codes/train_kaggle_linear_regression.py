import gc, json, time, numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

DATA_PATH = Path('/kaggle/input/datasets/yashag03/data-final-parquet/data_final.parquet')  # FIXED
OOS_PATH  = Path('/kaggle/input/datasets/yashagrawal511/unseen/oos_spots.parquet')
TARGET = 'kwh_norm'
FEATURES = ['ghi', 'dhi', 'dni', 'tsi', 'kt', 'dni_frac', 'dhi_frac',
            'solar_elevation', 'solar_azimuth', 'temperature_2m', 'precipitation_mm',
            'wind_speed_10m', 'relative_humidity_2m', 'snowfall', 'visibility',
            'weather_regime', 'snow_roll6h', 'snow_roll24h',
            'precip_roll6h', 'temp_roll24h', 'hour_sin', 'hour_cos', 'month_sin',
            'month_cos', 'doy_sin', 'doy_cos', 'latitude', 'longitude', 'panel_tilt',
            'max_kwh_est', 'kwp_est', 'panel_unimodal_azimuth', 'panel_tilt_azi_confidence',
            'panel_bimodal_east_azimuth', 'panel_bimodal_east_fraction',
            'panel_bimodal_west_azimuth', 'panel_is_ew_split', 'irr_mismatch',
            'analog_knn_mean_norm', 'scaled_analog_kwh_1', 'scaled_analog_knn_mean',
            'analog_dist_1', 'analog_dist_gap', 'analog_knn_std', 'analog_age_hours']

print("Analysiere Spots...")
meta = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])
selected = stats.sort_values(['clean', 'total'], ascending=False).iloc[:406]['spot_uuid'].tolist()
del meta, stats; gc.collect()

print("Lade Daten...")
df = ds.dataset(str(DATA_PATH)).to_table(
    columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
    filter=pc.field('spot_uuid').isin(selected)
).to_pandas()
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df = df[df[TARGET] <= 2.0].reset_index(drop=True)
df[FEATURES] = df[FEATURES].ffill().fillna(0.0)
print(f"Datensatzgröße: {len(df):,} Zeilen")

df['ts'] = pd.to_datetime(df['ts'])
train_end = pd.Timestamp('2025-10-31 23:00:00')
val_end   = pd.Timestamp('2025-12-31 23:00:00')
_split = np.zeros(len(df), dtype='int8')
_split[df['ts'] > train_end] = 1
_split[df['ts'] > val_end]   = 2

scaler = StandardScaler()
X_tr  = scaler.fit_transform(df.loc[_split==0, FEATURES].values.astype(np.float32))
y_tr  = df.loc[_split==0, TARGET].values.astype(np.float32)
X_te  = scaler.transform(df.loc[_split==2, FEATURES].values.astype(np.float32))
y_te  = df.loc[_split==2, TARGET].values.astype(np.float32)
ts_te = df.loc[_split==2, 'ts'].values
print(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")
del df; gc.collect()

print("Trainiere LinearRegression...")
t0 = time.time()
model = LinearRegression()
model.fit(X_tr, y_tr)
train_time = time.time() - t0
print(f"Training in {train_time:.1f}s")

y_p  = np.maximum(0, model.predict(X_te))
mae  = float(mean_absolute_error(y_te, y_p))
rmse = float(np.sqrt(mean_squared_error(y_te, y_p)))
r2   = float(r2_score(y_te, y_p))
print(f"\n=== TEST ===\nMAE:  {mae:.6f}\nRMSE: {rmse:.6f}\nR²:   {r2:.6f}")

print("\nOOS Evaluation...")
oos_spots = pd.read_parquet(OOS_PATH)['spot_uuid'].tolist()
df_oos = ds.dataset(str(DATA_PATH)).to_table(
    columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
    filter=pc.field('spot_uuid').isin(oos_spots)
).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos[FEATURES] = df_oos[FEATURES].ffill().fillna(0.0)

X_oos  = scaler.transform(df_oos[FEATURES].values.astype(np.float32))
y_oos  = df_oos[TARGET].values.astype(np.float32)
ts_oos = pd.to_datetime(df_oos['ts']).values
del df_oos; gc.collect()

y_oos_p  = np.maximum(0, model.predict(X_oos))
oos_mae  = float(mean_absolute_error(y_oos, y_oos_p))
oos_rmse = float(np.sqrt(mean_squared_error(y_oos, y_oos_p)))
oos_r2   = float(r2_score(y_oos, y_oos_p))
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f}")

coef_importance = dict(sorted(
    zip(FEATURES, [float(c) for c in model.coef_]),
    key=lambda x: abs(x[1]), reverse=True))

np.save('linreg_preds.npy',           y_p)
np.save('linreg_targets.npy',         y_te)
np.save('linreg_test_timestamps.npy', ts_te)
np.save('linreg_oos_preds.npy',       y_oos_p)
np.save('linreg_oos_targets.npy',     y_oos)
np.save('linreg_oos_timestamps.npy',  ts_oos)
joblib.dump(model, 'linreg_model.pkl')

with open('linreg_results.json', 'w') as f:
    json.dump({
        'model':              'LinearRegression',
        'n_features':         len(FEATURES),
        'features':           FEATURES,
        'mae':                mae,
        'rmse':               rmse,
        'r2':                 r2,
        'train_rows':         int(len(X_tr)),
        'test_rows':          int(len(X_te)),
        'train_time':         round(train_time, 2),
        'feature_importance': coef_importance,
        'oos': {
            'mae':    oos_mae,
            'rmse':   oos_rmse,
            'r2':     oos_r2,
            'n_spots': len(oos_spots),
            'n_rows': int(len(y_oos))
        }
    }, f, indent=2)
print("Fertig! Ergebnisse gespeichert.")