import gc, json, time, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, optuna
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH  = Path('/kaggle/input/datasets/yashagrawal511/data-final-parquet/data_final.parquet')
OOS_PATH   = Path('/kaggle/input/datasets/yashagrawal511/unseen/oos_spots.parquet')
TARGET     = 'kwh_norm'
FEATURES   = ['ghi', 'dhi', 'dni', 'tsi', 'kt', 'dni_frac', 'dhi_frac',
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

SPOTS_BATCH     = 400
TREES_PER_BATCH = 20
train_end       = pd.Timestamp('2025-10-31 23:00:00')
val_end         = pd.Timestamp('2025-12-31 23:00:00')

print("Lade OOS Spots...")
oos_spots = set(pd.read_parquet(OOS_PATH)['spot_uuid'].tolist())
print(f"OOS Spots: {len(oos_spots)}")

print("Analysiere Spots...")
meta  = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])
stats = stats.sort_values(['clean', 'total'], ascending=False).reset_index(drop=True)
all_spots = [s for s in stats['spot_uuid'].tolist() if s not in oos_spots]
del meta, stats; gc.collect()

spot_batches = [all_spots[i:i+SPOTS_BATCH] for i in range(0, len(all_spots), SPOTS_BATCH)]
print(f"Training Spots: {len(all_spots)} | Batches: {len(spot_batches)}")

scaler = StandardScaler()

def load_batch(spots, fit_scaler=False):
    df = ds.dataset(str(DATA_PATH)).to_table(
        columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
        filter=pc.field('spot_uuid').isin(spots)
    ).to_pandas()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df[df[TARGET] <= 2.0].reset_index(drop=True)
    df[FEATURES] = df[FEATURES].ffill().fillna(0.0)
    df['ts'] = pd.to_datetime(df['ts'])
    _split = np.zeros(len(df), dtype='int8')
    _split[df['ts'] > train_end] = 1
    _split[df['ts'] > val_end]   = 2
    X_tr_raw = df.loc[_split==0, FEATURES].values.astype(np.float32)
    if fit_scaler: scaler.fit(X_tr_raw)
    X_tr  = scaler.transform(X_tr_raw)
    y_tr  = df.loc[_split==0, TARGET].values.astype(np.float32)
    X_va  = scaler.transform(df.loc[_split==1, FEATURES].values.astype(np.float32))
    y_va  = df.loc[_split==1, TARGET].values.astype(np.float32)
    X_te  = scaler.transform(df.loc[_split==2, FEATURES].values.astype(np.float32))
    y_te  = df.loc[_split==2, TARGET].values.astype(np.float32)
    ts_te = df.loc[_split==2, 'ts'].values
    del df; gc.collect()
    return X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te

def predict_quantiles(model, X, quantiles=[0.1, 0.5, 0.9]):
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    return np.quantile(tree_preds, quantiles, axis=0)

def pinball(y_true, y_pred, alpha):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1) * e)))

print("Fitte Scaler auf Batch 0...")
X_tr0, y_tr0, X_va0, y_va0, X_te0, y_te0, ts_te0 = load_batch(spot_batches[0], fit_scaler=True)

print("\nStarte Optuna (30 Trials auf Batch 0)...")
def objective(trial):
    params = dict(
        n_estimators=50,
        max_depth=trial.suggest_int('max_depth', 10, 40),
        max_features=trial.suggest_float('max_features', 0.2, 0.8),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 100),
        max_samples=trial.suggest_float('max_samples', 0.05, 0.3),
    )
    m = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    m.fit(X_tr0, y_tr0)
    return mean_absolute_error(y_va0, m.predict(X_va0))

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)
best_params = study.best_params
print(f"Beste Params: {best_params} | Val-MAE: {study.best_value:.6f}")

print("\nInitialisiere Random Forest mit besten Parametern...")
model = RandomForestRegressor(
    n_estimators=TREES_PER_BATCH,
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_samples=best_params['max_samples'],
    n_jobs=-1, warm_start=True, random_state=42
)

t0 = time.time()
print(f"\nTrainiere über {len(spot_batches)} Batches...")
all_te_preds, all_te_targets, all_ts_te = [], [], []

for i, batch_spots in enumerate(spot_batches):
    print(f"Batch {i+1}/{len(spot_batches)} ({len(batch_spots)} Spots)...")
    if i == 0:
        X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te = X_tr0, y_tr0, X_va0, y_va0, X_te0, y_te0, ts_te0
    else:
        X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te = load_batch(batch_spots)
    model.n_estimators = (i + 1) * TREES_PER_BATCH
    model.fit(X_tr, y_tr)
    y_va_p = np.maximum(0, model.predict(X_va))
    print(f"  Val MAE: {mean_absolute_error(y_va, y_va_p):.6f} | Bäume gesamt: {model.n_estimators}")
    all_te_preds.append(np.maximum(0, model.predict(X_te)))
    all_te_targets.append(y_te); all_ts_te.append(ts_te)
    if i > 0: del X_tr, y_tr, X_va, y_va, X_te, y_te; gc.collect()

preds_te   = np.concatenate(all_te_preds)
y_te_glob  = np.concatenate(all_te_targets)
ts_te_glob = np.concatenate(all_ts_te)

mae  = float(mean_absolute_error(y_te_glob, preds_te))
rmse = float(np.sqrt(mean_squared_error(y_te_glob, preds_te)))
r2   = float(r2_score(y_te_glob, preds_te))
print(f"\n=== TEST ERGEBNISSE ===\nMAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

print("\nBerechne Test-Quantile...")
all_q_preds = []
for i, batch_spots in enumerate(spot_batches):
    X_te_q = X_te0 if i == 0 else load_batch(batch_spots)[4]
    all_q_preds.append(predict_quantiles(model, X_te_q))
    if i > 0: del X_te_q; gc.collect()

q_preds_te = np.concatenate(all_q_preds, axis=1)
preds_q10  = np.maximum(0, q_preds_te[0])
preds_q50  = np.maximum(0, q_preds_te[1])
preds_q90  = np.maximum(0, q_preds_te[2])

pinball_scores = {
    'q10': pinball(y_te_glob, preds_q10, 0.1),
    'q50': pinball(y_te_glob, preds_q50, 0.5),
    'q90': pinball(y_te_glob, preds_q90, 0.9),
}
coverage = float(np.mean((preds_q10 <= y_te_glob) & (y_te_glob <= preds_q90)))
print(f"Pinball: {pinball_scores} | 80%-Coverage: {coverage:.3f}")

print("\nOOS Evaluation...")
df_oos = ds.dataset(str(DATA_PATH)).to_table(
    columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
    filter=pc.field('spot_uuid').isin(list(oos_spots))
).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos = df_oos[df_oos[TARGET] <= 2.0].reset_index(drop=True)
df_oos[FEATURES] = df_oos[FEATURES].ffill().fillna(0.0)
ts_oos = df_oos['ts'].values
X_oos  = scaler.transform(df_oos[FEATURES].values.astype(np.float32))
y_oos  = df_oos[TARGET].values.astype(np.float32)
del df_oos; gc.collect()

y_oos_p  = np.maximum(0, model.predict(X_oos))
oos_q    = predict_quantiles(model, X_oos)
oos_q10  = np.maximum(0, oos_q[0]); oos_q50 = np.maximum(0, oos_q[1]); oos_q90 = np.maximum(0, oos_q[2])

oos_mae      = float(mean_absolute_error(y_oos, y_oos_p))
oos_rmse     = float(np.sqrt(mean_squared_error(y_oos, y_oos_p)))
oos_r2       = float(r2_score(y_oos, y_oos_p))
oos_coverage = float(np.mean((oos_q10 <= y_oos) & (y_oos <= oos_q90)))
oos_pinball  = {
    'q10': pinball(y_oos, oos_q10, 0.1),
    'q50': pinball(y_oos, oos_q50, 0.5),
    'q90': pinball(y_oos, oos_q90, 0.9),
}
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f} | Coverage={oos_coverage:.3f}")

feat_imp = dict(sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True))

joblib.dump(model, 'rf_model.pkl')
np.save('rf_point_preds.npy', preds_te); np.save('rf_q10_preds.npy', preds_q10)
np.save('rf_q50_preds.npy', preds_q50);  np.save('rf_q90_preds.npy', preds_q90)
np.save('rf_targets.npy', y_te_glob);    np.save('rf_test_timestamps.npy', ts_te_glob)
np.save('rf_oos_preds.npy', y_oos_p);    np.save('rf_oos_q10.npy', oos_q10)
np.save('rf_oos_q50.npy', oos_q50);      np.save('rf_oos_q90.npy', oos_q90)
np.save('rf_oos_targets.npy', y_oos);    np.save('rf_oos_timestamps.npy', ts_oos)

with open('rf_results.json', 'w') as f:
    json.dump({
        'model': 'RandomForest (warm_start, alle Spots)',
        'split': {'train': f"bis {train_end}", 'val': 'Nov–Dez 2025', 'test': 'Jan 2026 – Ende'},
        'n_train_spots': len(all_spots), 'n_oos_spots': len(oos_spots), 'n_batches': len(spot_batches),
        'trees_per_batch': TREES_PER_BATCH, 'total_trees': model.n_estimators,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'best_params': best_params, 'best_val_mae': float(study.best_value),
        'train_time': round(time.time() - t0, 2),
        'quantiles': {'alphas': [0.1, 0.5, 0.9], 'pinball_loss': pinball_scores, 'coverage_80pct': coverage},
        'feature_importance': {k: float(v) for k, v in feat_imp.items()},
        'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2, 'coverage_80pct': oos_coverage,
                'pinball_loss': oos_pinball, 'n_spots': len(oos_spots), 'n_rows': int(len(y_oos))},
    }, f, indent=2)
print("\nFertig! rf_results.json gespeichert.")