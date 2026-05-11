import gc, json, time, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, lightgbm as lgb, optuna
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path('/kaggle/input/datasets/yashag03/data-final-parquet/data_final.parquet')
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

print("Analysiere Daten...")
meta = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])
selected = stats.sort_values(['clean', 'total'], ascending=False).iloc[:406]['spot_uuid'].tolist()
del meta, stats; gc.collect()

print("Lade Daten...")
df = ds.dataset(str(DATA_PATH)).to_table(columns=FEATURES + ['spot_uuid', TARGET, 'ts'], filter=pc.field('spot_uuid').isin(selected)).to_pandas()
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
X_tr = scaler.fit_transform(df.loc[_split==0, FEATURES].values.astype(np.float32))
y_tr = df.loc[_split==0, TARGET].values.astype(np.float32)
X_va = scaler.transform(df.loc[_split==1, FEATURES].values.astype(np.float32))
y_va = df.loc[_split==1, TARGET].values.astype(np.float32)
X_te = scaler.transform(df.loc[_split==2, FEATURES].values.astype(np.float32))
y_te = df.loc[_split==2, TARGET].values.astype(np.float32)
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
del df; gc.collect()

def objective(trial):
    params = {'objective': 'regression', 'metric': 'mae', 'verbosity': -1, 'num_threads': -1,
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
              'num_leaves': trial.suggest_int('num_leaves', 31, 255),
              'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
              'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
              'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
              'bagging_freq': 5,
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True)}
    dtr  = lgb.Dataset(X_tr[:2_000_000], label=y_tr[:2_000_000])
    dval = lgb.Dataset(X_va, label=y_va)
    m = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return mean_absolute_error(y_va, m.predict(X_va))

print("Starte Optuna (30 Trials)...")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Finales Training...")
t0 = time.time()
best_params = {**study.best_params, 'objective': 'regression', 'metric': 'mae',
               'verbosity': -1, 'num_threads': -1, 'bagging_freq': 5}
dtr_full = lgb.Dataset(X_tr, label=y_tr)
dval     = lgb.Dataset(X_va, label=y_va)
final_model = lgb.train(best_params, dtr_full, num_boost_round=3000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

y_p = np.maximum(0, final_model.predict(X_te))
mae  = float(mean_absolute_error(y_te, y_p))
rmse = float(np.sqrt(mean_squared_error(y_te, y_p)))
r2   = float(r2_score(y_te, y_p))
print(f"\n=== ERGEBNISSE ===\nMAE:  {mae:.6f}\nRMSE: {rmse:.6f}\nR²:   {r2:.6f}")

feature_importance = dict(sorted(zip(FEATURES, final_model.feature_importance(importance_type='gain')), key=lambda x: x[1], reverse=True))

print("\nOOS Evaluation...")
oos_spots = pd.read_parquet(OOS_PATH)['spot_uuid'].tolist()
df_oos = ds.dataset(str(DATA_PATH)).to_table(columns=FEATURES + ['spot_uuid', TARGET], filter=pc.field('spot_uuid').isin(oos_spots)).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos[FEATURES] = df_oos[FEATURES].ffill().fillna(0.0)
X_oos = scaler.transform(df_oos[FEATURES].values.astype(np.float32))
y_oos = df_oos[TARGET].values.astype(np.float32)
del df_oos; gc.collect()

y_oos_p = np.maximum(0, final_model.predict(X_oos))
oos_mae  = float(mean_absolute_error(y_oos, y_oos_p))
oos_rmse = float(np.sqrt(mean_squared_error(y_oos, y_oos_p)))
oos_r2   = float(r2_score(y_oos, y_oos_p))
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f}")

final_model.save_model('lgbm_model.txt')
np.save('lgbm_preds.npy', y_p); np.save('lgbm_targets.npy', y_te)
np.save('lgbm_oos_preds.npy', y_oos_p); np.save('lgbm_oos_targets.npy', y_oos)

with open('lgbm_results.json', 'w') as f:
    json.dump({'model': 'LightGBM', 'mae': mae, 'rmse': rmse, 'r2': r2,
               'best_params': study.best_params, 'best_val_mae': float(study.best_value),
               'n_estimators': final_model.num_trees(),
               'train_rows': int(len(X_tr)), 'val_rows': int(len(X_va)), 'test_rows': int(len(X_te)),
               'train_time': round(time.time() - t0, 2),
               'feature_importance': {k: int(v) for k, v in feature_importance.items()},
               'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2,
                       'n_spots': len(oos_spots), 'n_rows': int(len(y_oos))}}, f, indent=2)
print("Fertig! lgbm_results.json gespeichert.")