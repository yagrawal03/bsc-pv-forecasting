import gc, json, time, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, lightgbm as lgb, optuna
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH  = Path('/kaggle/input/datasets/yashagrawal511/data-final-parquet/data_final.parquet')
OOS_PATH   = Path('/kaggle/input/datasets/yashagrawal511/unseen/oos_spots.parquet')
TARGET     = 'kwh_norm'
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
QUANTILES   = [0.1, 0.5, 0.9]
SPOTS_BATCH = 400
ROUNDS_PER_BATCH = 200

print("Lade OOS Spots...")
oos_spots = set(pd.read_parquet(OOS_PATH)['spot_uuid'].tolist())
print(f"OOS Spots (ausgeschlossen): {len(oos_spots)}")

print("Analysiere alle Spots...")
meta = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])
stats = stats.sort_values(['clean', 'total'], ascending=False).reset_index(drop=True)
all_spots = [s for s in stats['spot_uuid'].tolist() if s not in oos_spots]
del meta, stats; gc.collect()
print(f"Training Spots gesamt: {len(all_spots)}")

spot_batches = [all_spots[i:i+SPOTS_BATCH] for i in range(0, len(all_spots), SPOTS_BATCH)]
print(f"Batches: {len(spot_batches)} à ~{SPOTS_BATCH} Spots")

train_end = pd.Timestamp('2025-10-31 23:00:00')
val_end   = pd.Timestamp('2025-12-31 23:00:00')

def load_batch(spots):
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
    X_tr = df.loc[_split==0, FEATURES].values.astype(np.float32)
    y_tr = df.loc[_split==0, TARGET].values.astype(np.float32)
    X_va = df.loc[_split==1, FEATURES].values.astype(np.float32)
    y_va = df.loc[_split==1, TARGET].values.astype(np.float32)
    X_te = df.loc[_split==2, FEATURES].values.astype(np.float32)
    y_te = df.loc[_split==2, TARGET].values.astype(np.float32)
    del df; gc.collect()
    return X_tr, y_tr, X_va, y_va, X_te, y_te

print("\nStarte Optuna (30 Trials auf Batch 0)...")
X_tr0, y_tr0, X_va0, y_va0, _, _ = load_batch(spot_batches[0])
dtr0 = lgb.Dataset(X_tr0, label=y_tr0, free_raw_data=True)
dva0 = lgb.Dataset(X_va0, label=y_va0, free_raw_data=True, reference=dtr0)

def objective(trial):
    params = {
        'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1, 'num_threads': -1,
        'bagging_freq': 5, 'feature_pre_filter': False, 'device': 'gpu',
        'gpu_platform_id': 0, 'gpu_device_id': 0,
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves':        trial.suggest_int('num_leaves', 31, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
    }
    m = lgb.train(params, dtr0, num_boost_round=300, valid_sets=[dva0],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    return m.best_score['valid_0']['l1']

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"Beste Params: {study.best_params} | Val-MAE: {study.best_value:.6f}")
del X_tr0, y_tr0, X_va0, y_va0, dtr0, dva0; gc.collect()

best_base = {
    **study.best_params,
    'verbosity': -1, 'num_threads': -1, 'bagging_freq': 5,
    'feature_pre_filter': False, 'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
}

def train_all_batches(objective_name, alpha=None):
    name = f'q{int(alpha*100)}' if alpha else 'point'
    print(f"\n=== Trainiere {name} über alle {len(spot_batches)} Batches ===")
    params = {**best_base, 'objective': objective_name,
              'metric': 'mae' if not alpha else 'quantile'}
    if alpha: params['alpha'] = alpha
    model = None; dtr_ref = None; all_preds = []; all_targets = []
    for i, batch_spots in enumerate(spot_batches):
        print(f"  Batch {i+1}/{len(spot_batches)} ({len(batch_spots)} Spots)...")
        X_tr, y_tr, X_va, y_va, X_te, y_te = load_batch(batch_spots)
        if dtr_ref is None:
            dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True); dtr_ref = dtr
        else:
            dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True, reference=dtr_ref)
        dva = lgb.Dataset(X_va, label=y_va, free_raw_data=True, reference=dtr_ref)
        del X_tr, y_tr, X_va, y_va; gc.collect()
        model = lgb.train(params, dtr, num_boost_round=ROUNDS_PER_BATCH, valid_sets=[dva],
                          init_model=model, callbacks=[lgb.log_evaluation(50)])
        del dtr, dva; gc.collect()
        all_preds.append(np.maximum(0, model.predict(X_te)))
        all_targets.append(y_te)
        del X_te, y_te; gc.collect()
    return model, np.concatenate(all_preds), np.concatenate(all_targets)

t0 = time.time()
all_results = {}

model_point, preds_te, y_te = train_all_batches('regression_l1')
mae  = float(mean_absolute_error(y_te, preds_te))
rmse = float(np.sqrt(mean_squared_error(y_te, preds_te)))
r2   = float(r2_score(y_te, preds_te))
print(f"\n[point] MAE={mae:.6f} | RMSE={rmse:.6f} | R²={r2:.6f}")
model_point.save_model('lgbm_point.txt')
all_results['point'] = {'model': model_point, 'preds': preds_te, 'mae': mae, 'rmse': rmse, 'r2': r2}

for q in QUANTILES:
    name = f'q{int(q*100)}'
    m, preds, _ = train_all_batches('quantile', alpha=q)
    mae_q  = float(mean_absolute_error(y_te, preds))
    rmse_q = float(np.sqrt(mean_squared_error(y_te, preds)))
    r2_q   = float(r2_score(y_te, preds))
    print(f"[{name}] MAE={mae_q:.6f} | RMSE={rmse_q:.6f} | R²={r2_q:.6f}")
    m.save_model(f'lgbm_{name}.txt')
    all_results[name] = {'model': m, 'preds': preds, 'mae': mae_q, 'rmse': rmse_q, 'r2': r2_q}

def pinball(y_true, y_pred, alpha):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1) * e)))

pinball_scores = {f'q{int(q*100)}': pinball(y_te, all_results[f'q{int(q*100)}']['preds'], q) for q in QUANTILES}
coverage       = float(np.mean((all_results['q10']['preds'] <= y_te) & (y_te <= all_results['q90']['preds'])))
print(f"\nPinball: {pinball_scores} | 80%-Coverage: {coverage:.3f}")

print("\nOOS Evaluation...")
df_oos = ds.dataset(str(DATA_PATH)).to_table(
    columns=FEATURES + ['spot_uuid', TARGET],
    filter=pc.field('spot_uuid').isin(list(oos_spots))
).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos = df_oos[df_oos[TARGET] <= 2.0].reset_index(drop=True)
df_oos[FEATURES] = df_oos[FEATURES].ffill().fillna(0.0)
X_oos = df_oos[FEATURES].values.astype(np.float32)
y_oos = df_oos[TARGET].values.astype(np.float32)
del df_oos; gc.collect()

y_oos_p      = np.maximum(0, all_results['point']['model'].predict(X_oos))
oos_mae      = float(mean_absolute_error(y_oos, y_oos_p))
oos_rmse     = float(np.sqrt(mean_squared_error(y_oos, y_oos_p)))
oos_r2       = float(r2_score(y_oos, y_oos_p))
oos_q_preds  = {f'q{int(q*100)}': np.maximum(0, all_results[f'q{int(q*100)}']['model'].predict(X_oos)) for q in QUANTILES}
oos_coverage = float(np.mean((oos_q_preds['q10'] <= y_oos) & (y_oos <= oos_q_preds['q90'])))
oos_pinball  = {f'q{int(q*100)}': pinball(y_oos, oos_q_preds[f'q{int(q*100)}'], q) for q in QUANTILES}
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f} | Coverage={oos_coverage:.3f}")

for name, d in all_results.items(): np.save(f'lgbm_{name}_preds.npy', d['preds'])
np.save('lgbm_targets.npy', y_te); np.save('lgbm_oos_preds.npy', y_oos_p); np.save('lgbm_oos_targets.npy', y_oos)
for qname, arr in oos_q_preds.items(): np.save(f'lgbm_{qname}_oos_preds.npy', arr)

feat_imp = dict(sorted(zip(FEATURES, all_results['point']['model'].feature_importance(importance_type='gain')),
                       key=lambda x: x[1], reverse=True))

with open('lgbm_results.json', 'w') as f:
    json.dump({
        'model': 'LightGBM (init_model alle Spots)',
        'split': {'train': f"bis {train_end}", 'val': 'Nov–Dez 2025', 'test': 'Jan 2026 – Ende'},
        'n_train_spots': len(all_spots), 'n_oos_spots': len(oos_spots),
        'n_batches': len(spot_batches), 'rounds_per_batch': ROUNDS_PER_BATCH,
        'mae': all_results['point']['mae'], 'rmse': all_results['point']['rmse'], 'r2': all_results['point']['r2'],
        'best_params': study.best_params, 'best_val_mae': float(study.best_value),
        'n_estimators': all_results['point']['model'].num_trees(),
        'train_time': round(time.time() - t0, 2),
        'quantiles': {'alphas': QUANTILES, 'pinball_loss': pinball_scores, 'coverage_80pct': coverage},
        'feature_importance': {k: int(v) for k, v in feat_imp.items()},
        'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2, 'coverage_80pct': oos_coverage,
                'pinball_loss': oos_pinball, 'n_spots': len(oos_spots), 'n_rows': int(len(y_oos))},
    }, f, indent=2)
print("\nFertig! lgbm_results.json gespeichert.")