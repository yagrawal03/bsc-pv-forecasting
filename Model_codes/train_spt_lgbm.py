"""
LightGBM trainiert auf SPT_PV_D Daten (2026-03-03_pv_with_analogs.parquet).
Features und Modellkonfiguration identisch zu SPT_PV_D.ipynb.

Wichtig:
- OOS-Spots werden aus den Trainingsdaten HERAUSGEFILTERT (keine Kontamination)
- OOS-Test auf allen 100 oos_spots.json Spots:
    25 Spots aus SPT-Parquet (dort vorhanden, aber nie trainiert)
    75 Spots aus data_final.parquet (gleiche Features, komplett ungesehen)
"""

import gc, json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── PFADE ────────────────────────────────────────────────────────────────────
SPT_DATA_PATH  = Path('../../experiments/total_pv_output/data/processed/2026-03-03_pv_with_analogs.parquet')
OOS_SPOTS_PATH = Path('oos_spots.json')

TARGET = 'kwh_norm'

# ── FEATURES (identisch zu SPT_PV_D.ipynb) ───────────────────────────────────
CONTINUOUS_FEATURES = (
    ['hour_sin', 'hour_cos']
    + ['kt', 'dni_frac', 'dhi_frac', 'ghi_magnitude', 'precipitation_mm']
    + ['solar_elevation']
    + ['kwp_est']
    + ['analog_kwh_1_norm']
    + ['analog_dist_1', 'analog_dist_gap']
    + ['scaled_analog_kwh_1_norm', 'irr_mismatch']
)
BOOLEAN_FEATURES     = ['analog_available']
CATEGORICAL_FEATURES = ['weather_regime']
FEATURES = CONTINUOUS_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# ── SCHRITT 1: OOS-Spots laden (werden aus Training ausgeschlossen) ────────────
print("Lade OOS-Spot-Liste...")
with open(OOS_SPOTS_PATH) as f:
    oos_spots = set(json.load(f))
print(f"OOS-Spots: {len(oos_spots)}")

# ── SCHRITT 2: SPT_PV_D Daten laden ──────────────────────────────────────────
print("Lade SPT_PV_D Daten...")
df = pd.read_parquet(SPT_DATA_PATH, columns=['ts', 'spot_uuid'] + FEATURES + [TARGET])
df['ts'] = pd.to_datetime(df['ts'])
if df['ts'].dt.tz is not None:
    df['ts'] = df['ts'].dt.tz_convert('UTC').dt.tz_localize(None)

# OOS-Spots aus Training herausfiltern
n_before = df['spot_uuid'].nunique()
df = df[~df['spot_uuid'].isin(oos_spots)].reset_index(drop=True)
n_after = df['spot_uuid'].nunique()
print(f"Spots: {n_before} → {n_after} (herausgefiltert: {n_before - n_after} OOS-Spots)")

df = df.dropna(subset=[TARGET]).reset_index(drop=True)
# weather_regime is categorical — encode to int codes using a fixed mapping
# saved so OOS data (from a different parquet) gets the same encoding.
_cat_encoders = {}  # col -> list of categories
for col in FEATURES:
    if str(df[col].dtype) == 'category':
        df[col] = df[col].cat.add_categories(['unknown']).ffill().fillna('unknown')
        _cat_encoders[col] = list(df[col].cat.categories)
        df[col] = df[col].cat.codes.astype(np.int32)
    else:
        df[col] = df[col].ffill().fillna(0.0)
print(f"Datensatzgröße: {len(df):,} Zeilen, {df['spot_uuid'].nunique()} Spots")
print(f"Zeitraum: {df['ts'].min().date()} – {df['ts'].max().date()}")

# ── SCHRITT 3: 80/20 Split nach Timestamp (wie in SPT_PV_D.ipynb) ─────────────
df_sorted = df.sort_values('ts').reset_index(drop=True)
cut = int(len(df_sorted) * 0.8)
df_train = df_sorted.iloc[:cut].copy()
df_test  = df_sorted.iloc[cut:].copy()

X_tr = df_train[FEATURES]
y_tr = df_train[TARGET].values.astype(np.float32)
X_te = df_test[FEATURES]
y_te = df_test[TARGET].values.astype(np.float32)
print(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")
del df, df_sorted, df_train, df_test; gc.collect()

# ── SCHRITT 4: Beste Hyperparameter (aus vorherigem Optuna-Lauf, 50 Trials) ───
# Val-RMSE: 0.034495
best = {
    'objective':              'tweedie',
    'tweedie_variance_power': 1.072291786689186,
    'metric':                 'rmse',
    'num_leaves':             76,
    'min_data_in_leaf':       1420,
    'num_iterations':         428,
    'learning_rate':          0.04423822599957653,
    'lambda_l1':              0.2120812685221297,
    'lambda_l2':              1.1505477024985432e-07,
    'feature_fraction':       0.7282291926277684,
    'bagging_fraction':       0.7112221311959,
    'bagging_freq':           2,
    'use_missing':            True,
    'force_row_wise':         True,
    'n_jobs':                 -1,
    'verbose':                -1,
    'random_state':           42,
}
print(f"Verwende gespeicherte beste Params (Val-RMSE: 0.034495)")

# ── SCHRITT 5: Finales Training auf allen Trainingsdaten ──────────────────────
print("\nFinales Training...")
t0 = time.time()

final_model = lgb.train(
    best,
    lgb.Dataset(X_tr, label=y_tr),
    valid_sets=[lgb.Dataset(X_te, label=y_te)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
)
train_time = round(time.time() - t0, 2)
final_model.save_model('spt_lgbm_model.txt')
print("Modell gespeichert → spt_lgbm_model.txt")

# ── SCHRITT 6: Evaluation auf Test-Set ───────────────────────────────────────
y_p = np.maximum(0, final_model.predict(X_te))
ae  = np.abs(y_p - y_te)
w   = y_te / (y_te.mean() + 1e-9)
mae   = float(ae.mean())
rmse  = float(np.sqrt((ae**2).mean()))
r2    = float(r2_score(y_te, y_p))
wrmse = float(np.sqrt((w * ae**2).sum() / (w.sum() + 1e-9)))
print(f"\n=== TEST-SET (SPT_PV_D, letzte 20%) ===")
print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | wRMSE={wrmse:.4f}")

# Feature Importance
fi = dict(sorted(
    zip(FEATURES, final_model.feature_importance(importance_type='gain').tolist()),
    key=lambda x: abs(x[1]), reverse=True
))

# ── SCHRITT 7: Speichern ─────────────────────────────────────────────────────
np.save('spt_lgbm_preds.npy',   y_p)
np.save('spt_lgbm_targets.npy', y_te)

results = {
    'model':       'LightGBM_SPT',
    'trained_on':  'SPT_PV_D (2026-03-03_pv_with_analogs.parquet)',
    'n_train_spots': int(n_after),
    'n_oos_excluded': int(n_before - n_after),
    'mae':         mae,
    'rmse':        rmse,
    'r2':          r2,
    'wrmse':       wrmse,
    'best_params': best,
    'best_val_rmse': 0.034495,
    'train_rows':  int(len(X_tr)),
    'test_rows':   int(len(X_te)),
    'train_time':  train_time,
    'feature_importance': fi,
}

# ── SCHRITT 8: OOS-Evaluation auf allen 100 OOS-Spots ────────────────────────
# 25 Spots aus SPT-Parquet (gleiche Features), 75 aus data_final.parquet
print("\nOOS Evaluation (alle 100 OOS-Spots)...")
import pyarrow.dataset as pads
import pyarrow.compute as papc

_oos_list = list(oos_spots)

# Teil A: 25 Spots aus SPT-Parquet laden
_df_spt = pd.read_parquet(SPT_DATA_PATH, columns=['spot_uuid'] + FEATURES + [TARGET])
_spt_spot_set = set(_df_spt['spot_uuid'].unique())
_oos_in_spt   = [s for s in _oos_list if s in _spt_spot_set]
_oos_not_spt  = [s for s in _oos_list if s not in _spt_spot_set]
print(f"  {len(_oos_in_spt)} Spots aus SPT-Parquet, {len(_oos_not_spt)} aus data_final.parquet")

_df_oos_spt = _df_spt[_df_spt['spot_uuid'].isin(_oos_in_spt)].copy()
del _df_spt; gc.collect()

# Teil B: restliche 75 Spots aus data_final.parquet (gleiche Features vorhanden)
DATA_FINAL_PATH = Path('../data/processed/data_final.parquet')
_df_oos_fin = pads.dataset(str(DATA_FINAL_PATH)).to_table(
    columns=['spot_uuid'] + FEATURES + [TARGET],
    filter=papc.field('spot_uuid').isin(_oos_not_spt)
).to_pandas()

# Zusammenführen
_df_oos = pd.concat([_df_oos_spt, _df_oos_fin], ignore_index=True)
del _df_oos_spt, _df_oos_fin; gc.collect()

_df_oos = _df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
for col in FEATURES:
    if col in _cat_encoders:
        # convert to string (handles both Categorical and object dtype)
        _df_oos[col] = _df_oos[col].astype(str).replace({'nan': 'unknown', '<NA>': 'unknown'})
        _df_oos[col] = _df_oos[col].fillna('unknown')
        # apply same encoding as training; unseen categories → -1
        _df_oos[col] = pd.Categorical(_df_oos[col], categories=_cat_encoders[col]).codes.astype(np.int32)
    else:
        _df_oos[col] = _df_oos[col].ffill().fillna(0.0)
_X_oos = _df_oos[FEATURES]
_y_oos = _df_oos[TARGET].values.astype(np.float32)
del _df_oos; gc.collect()

_y_oos_p = np.maximum(0, final_model.predict(_X_oos))
_ae = np.abs(_y_oos_p - _y_oos)
_w  = _y_oos / (_y_oos.mean() + 1e-9)
oos = {
    'mae':     float(_ae.mean()),
    'rmse':    float(np.sqrt((_ae**2).mean())),
    'r2':      float(r2_score(_y_oos, _y_oos_p)),
    'wrmse':   float(np.sqrt((_w * _ae**2).sum() / (_w.sum() + 1e-9))),
    'n_spots': len(_oos_list),
    'n_rows':  int(len(_y_oos)),
    'note':    f'{len(_oos_in_spt)} spots from SPT parquet, {len(_oos_not_spt)} from data_final',
}
print(f"OOS → MAE={oos['mae']:.4f} | RMSE={oos['rmse']:.4f} | R²={oos['r2']:.4f} | wRMSE={oos['wrmse']:.4f}")

results['oos'] = oos

with open('spt_lgbm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nFertig! spt_lgbm_results.json gespeichert. (Trainingszeit: {train_time:.0f}s)")
