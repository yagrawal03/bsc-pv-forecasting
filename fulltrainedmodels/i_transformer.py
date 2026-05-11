import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DATA_PATH = Path('/kaggle/input/datasets/yashagrawal511/data-final-parquet/data_final.parquet')
OOS_PATH  = Path('/kaggle/input/datasets/yashagrawal511/unseen/oos_spots.parquet')

TARGET = 'kwh_norm'

FEATURES = [
    'ghi', 'dhi', 'dni', 'tsi', 'kt', 'dni_frac', 'dhi_frac',
    'solar_elevation', 'solar_azimuth', 'temperature_2m', 'precipitation_mm',
    'wind_speed_10m', 'relative_humidity_2m', 'snowfall', 'visibility',
    'weather_regime', 'snow_roll6h', 'snow_roll24h',
    'precip_roll6h', 'temp_roll24h', 'hour_sin', 'hour_cos', 'month_sin',
    'month_cos', 'doy_sin', 'doy_cos', 'latitude', 'longitude', 'panel_tilt',
    'max_kwh_est', 'kwp_est', 'panel_unimodal_azimuth', 'panel_tilt_azi_confidence',
    'panel_bimodal_east_azimuth', 'panel_bimodal_east_fraction',
    'panel_bimodal_west_azimuth', 'panel_is_ew_split', 'irr_mismatch',
    'analog_knn_mean_norm', 'scaled_analog_kwh_1', 'scaled_analog_knn_mean',
    'analog_dist_1', 'analog_dist_gap', 'analog_knn_std', 'analog_age_hours'
]

QUANTILES        = [0.1, 0.5, 0.9]
N_FEATURES       = len(FEATURES)
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY       = DEVICE.type == 'cuda'
SPOTS_BATCH      = 200
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE  = 1024
TRAIN_STRIDE     = 48
train_end        = pd.Timestamp('2025-10-31 23:00:00')
val_end          = pd.Timestamp('2025-12-31 23:00:00')

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"Device: {DEVICE}")

oos_spots = set(pd.read_parquet(OOS_PATH)['spot_uuid'].tolist())
print(f"OOS Spots ausgeschlossen: {len(oos_spots)}")

print("Analysiere alle Spots...")
dataset = ds.dataset(str(DATA_PATH))
meta = dataset.to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = (
    meta.groupby('spot_uuid', observed=True)[TARGET]
    .agg(total='count', nans=lambda x: x.isna().sum())
    .reset_index()
)
stats['clean'] = 1.0 - stats['nans'] / stats['total']
stats = stats.sort_values(['clean', 'total'], ascending=False).reset_index(drop=True)
all_spots = [s for s in stats['spot_uuid'].tolist() if s not in oos_spots]
del meta, stats
gc.collect()

spot_batches = [all_spots[i:i + SPOTS_BATCH] for i in range(0, len(all_spots), SPOTS_BATCH)]
print(f"Training Spots: {len(all_spots)} | Batches: {len(spot_batches)}")

scaler = StandardScaler()

def clean_df(df):
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df[df[TARGET] <= 2.0].reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values(['spot_uuid', 'ts']).reset_index(drop=True)
    df[FEATURES] = df.groupby('spot_uuid', observed=True)[FEATURES].ffill().fillna(0.0)
    return df.reset_index(drop=True)

def build_segments(df_part):
    segs, curr = [], 0
    for _, g in df_part.groupby('spot_uuid', observed=True, sort=False):
        n = len(g); segs.append((curr, n)); curr += n
    return segs

def load_batch(spots, fit_scaler=False):
    df = dataset.to_table(
        columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
        filter=pc.field('spot_uuid').isin(spots),
    ).to_pandas()
    df = clean_df(df)
    split = np.zeros(len(df), dtype=np.int8)
    split[df['ts'] > train_end] = 1
    split[df['ts'] > val_end]   = 2
    df_tr = df.loc[split == 0].copy(); df_va = df.loc[split == 1].copy(); df_te = df.loc[split == 2].copy()
    segs_tr = build_segments(df_tr); segs_va = build_segments(df_va); segs_te = build_segments(df_te)
    X_tr_raw = df_tr[FEATURES].values.astype(np.float32)
    if fit_scaler: scaler.fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    y_tr = df_tr[TARGET].values.astype(np.float32)
    X_va = scaler.transform(df_va[FEATURES].values.astype(np.float32)).astype(np.float32)
    y_va = df_va[TARGET].values.astype(np.float32)
    X_te = scaler.transform(df_te[FEATURES].values.astype(np.float32)).astype(np.float32)
    y_te = df_te[TARGET].values.astype(np.float32)
    ts_te = df_te['ts'].values
    del df, df_tr, df_va, df_te; gc.collect()
    return X_tr, y_tr, X_va, y_va, X_te, y_te, ts_te, segs_tr, segs_va, segs_te

print("Fitte Scaler auf Batch 0...")
X_tr0, y_tr0, X_va0, y_va0, X_te0, y_te0, ts_te0, segs_tr0, segs_va0, segs_te0 = load_batch(
    spot_batches[0], fit_scaler=True
)
print(f"Batch 0 — Train: {len(X_tr0):,} | Val: {len(X_va0):,} | Test: {len(X_te0):,}")

class SpotSeqDS(Dataset):
    def __init__(self, X, y, segs, slen, stride):
        self.X, self.y, self.slen = X, y, slen
        self.starts = []
        for off, n in segs:
            if n >= slen:
                self.starts.extend(off + i for i in range(0, n - slen + 1, stride))
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(self.X[s:s + self.slen]), torch.tensor(self.y[s + self.slen - 1], dtype=torch.float32)

def extract_targets(y, segs, slen, stride=1):
    out = []
    for off, n in segs:
        if n >= slen:
            out.extend(y[off + slen - 1 + i] for i in range(0, n - slen + 1, stride))
    return np.asarray(out, dtype=np.float32)

def extract_timestamps(ts, segs, slen, stride=1):
    out = []
    for off, n in segs:
        if n >= slen:
            out.extend(ts[off + slen - 1 + i] for i in range(0, n - slen + 1, stride))
    return np.asarray(out)

class ITransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        res = x; x, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + res); res = x
        x = self.norm2(self.ff(x) + res)
        return x

class ITransformerMulti(nn.Module):
    def __init__(self, seq_len, n_features, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([ITransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(n_features * d_model, d_model), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model, 4))
    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x).flatten(1))

def pinball_loss(pred, target, alpha):
    e = target - pred
    return torch.mean(torch.where(e >= 0, alpha * e, (alpha - 1.0) * e))

def combined_loss(pred, target):
    return (
        nn.L1Loss()(pred[:, 0], target)
        + pinball_loss(pred[:, 1], target, 0.1)
        + pinball_loss(pred[:, 2], target, 0.5)
        + pinball_loss(pred[:, 3], target, 0.9)
    )

def pinball_np(y_true, y_pred, alpha):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1.0) * e)))

def predict_loader(model, loader):
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE, non_blocking=True)).cpu().numpy())
    if not preds: return np.empty((0, 4), dtype=np.float32)
    return np.maximum(0.0, np.concatenate(preds, axis=0))

def objective(trial):
    slen    = trial.suggest_categorical('slen', [48, 96, 168])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    d_ff    = trial.suggest_categorical('d_ff', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr      = trial.suggest_float('lr', 5e-4, 3e-3, log=True)
    if d_model % n_heads != 0: return float('inf')
    m = ITransformerMulti(slen, N_FEATURES, d_model, n_heads, n_layers, d_ff, dropout).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    dl  = DataLoader(SpotSeqDS(X_tr0, y_tr0, segs_tr0, slen, stride=192), batch_size=512, shuffle=True)
    dva = DataLoader(SpotSeqDS(X_va0, y_va0, segs_va0, slen, stride=slen), batch_size=1024)
    best_v = float('inf')
    for _ in range(10):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); combined_loss(m(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
        m.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in dva: vl += combined_loss(m(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        best_v = min(best_v, vl / len(dva.dataset))
    return best_v

print("Starte Optuna (30 Trials auf Batch 0)...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)
best = study.best_params
print(f"Beste Params: {best} | Val-Loss: {study.best_value:.6f}")

VAL_STRIDE = best['slen']
TEST_STRIDE = 1
OOS_STRIDE  = 1

final_model = ITransformerMulti(best['slen'], N_FEATURES, best['d_model'], best['n_heads'],
                                 best['n_layers'], best['d_ff'], best['dropout']).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
print(f"Parameter: {sum(p.numel() for p in final_model.parameters()):,}")

best_val, no_improve, best_state = float('inf'), 0, None
train_losses, val_losses = [], []
t0 = time.time()

print(f"\nFinales Training über {len(spot_batches)} Batches, max. 8 Epochen...")
for epoch in range(8):
    ep_loss = ep_n = ep_val_loss = ep_val_n = 0
    for i, batch_spots in enumerate(spot_batches):
        if i == 0:
            X_tr, y_tr, X_va, y_va = X_tr0, y_tr0, X_va0, y_va0
            segs_tr, segs_va = segs_tr0, segs_va0
        else:
            X_tr, y_tr, X_va, y_va, _, _, _, segs_tr, segs_va, _ = load_batch(batch_spots)
        train_dl = DataLoader(SpotSeqDS(X_tr, y_tr, segs_tr, best['slen'], stride=TRAIN_STRIDE),
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=2)
        val_dl   = DataLoader(SpotSeqDS(X_va, y_va, segs_va, best['slen'], stride=VAL_STRIDE),
                              batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=2)
        final_model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(final_model(xb), yb)
            loss.backward(); optimizer.step()
            ep_loss += loss.item() * len(xb); ep_n += len(xb)
        final_model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                loss = combined_loss(final_model(xb), yb)
                ep_val_loss += loss.item() * len(xb); ep_val_n += len(xb)
        if i > 0: del X_tr, y_tr, X_va, y_va; gc.collect()
    train_loss = ep_loss / ep_n; val_loss = ep_val_loss / ep_val_n
    train_losses.append(train_loss); val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    if val_loss < best_val - 1e-5:
        best_val = val_loss; no_improve = 0
        best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= 2: print(f"Early stopping nach Epoch {epoch+1}"); break

if best_state is None: raise RuntimeError("Kein best_state gespeichert.")
final_model.load_state_dict(best_state)

print("\nEvaluiere auf Testset...")
all_preds, all_targets, all_ts_te = [], [], []
for i, batch_spots in enumerate(spot_batches):
    if i == 0:
        X_te, y_te, ts_te, segs_te = X_te0, y_te0, ts_te0, segs_te0
    else:
        _, _, _, _, X_te, y_te, ts_te, _, _, segs_te = load_batch(batch_spots)
    test_dl = DataLoader(SpotSeqDS(X_te, y_te, segs_te, best['slen'], stride=TEST_STRIDE),
                         batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=2)
    all_preds.append(predict_loader(final_model, test_dl))
    all_targets.append(extract_targets(y_te, segs_te, best['slen'], stride=TEST_STRIDE))
    all_ts_te.append(extract_timestamps(ts_te, segs_te, best['slen'], stride=TEST_STRIDE))
    if i > 0: del X_te, y_te, ts_te; gc.collect()

all_preds    = np.concatenate(all_preds, axis=0)
y_te_global  = np.concatenate(all_targets, axis=0)
ts_te_global = np.concatenate(all_ts_te, axis=0)
y_p_point, y_p_q10, y_p_q50, y_p_q90 = all_preds[:,0], all_preds[:,1], all_preds[:,2], all_preds[:,3]

mae  = float(mean_absolute_error(y_te_global, y_p_point))
rmse = float(np.sqrt(mean_squared_error(y_te_global, y_p_point)))
r2   = float(r2_score(y_te_global, y_p_point))
pinball_scores = {
    'q10': pinball_np(y_te_global, y_p_q10, 0.1),
    'q50': pinball_np(y_te_global, y_p_q50, 0.5),
    'q90': pinball_np(y_te_global, y_p_q90, 0.9),
}
coverage = float(np.mean((y_p_q10 <= y_te_global) & (y_te_global <= y_p_q90)))
print(f"\n=== TEST ERGEBNISSE ===\nMAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")
print(f"Pinball: {pinball_scores} | Coverage: {coverage:.3f}")

print("\nBerechne Permutation Importance auf Batch 0...")
baseline_dl = DataLoader(SpotSeqDS(X_te0, y_te0, segs_te0, best['slen'], stride=1),
                         batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=2)
baseline_preds = predict_loader(final_model, baseline_dl)[:, 0]
y_true_b = extract_targets(y_te0, segs_te0, best['slen'], stride=1)
base_mae = float(mean_absolute_error(y_true_b, baseline_preds))
perm_importance = {}
for feat_idx, feat_name in enumerate(FEATURES):
    X_perm = X_te0.copy()
    X_perm[:, feat_idx] = np.random.default_rng(42 + feat_idx).permutation(X_perm[:, feat_idx])
    perm_dl = DataLoader(SpotSeqDS(X_perm, y_te0, segs_te0, best['slen'], stride=1),
                         batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=2)
    y_perm = predict_loader(final_model, perm_dl)[:, 0]
    perm_importance[feat_name] = round(float(mean_absolute_error(y_true_b, y_perm) - base_mae), 6)
perm_importance = dict(sorted(perm_importance.items(), key=lambda x: x[1], reverse=True))

print("\nOOS Evaluation...")
df_oos = dataset.to_table(columns=FEATURES + ['spot_uuid', TARGET, 'ts'],
                          filter=pc.field('spot_uuid').isin(list(oos_spots))).to_pandas()
df_oos = clean_df(df_oos)
segs_oos = build_segments(df_oos)
X_oos = scaler.transform(df_oos[FEATURES].values.astype(np.float32)).astype(np.float32)
y_oos = df_oos[TARGET].values.astype(np.float32)
ts_oos_raw = df_oos['ts'].values
del df_oos; gc.collect()

oos_dl = DataLoader(SpotSeqDS(X_oos, y_oos, segs_oos, best['slen'], stride=OOS_STRIDE),
                    batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=2)
preds_oos    = predict_loader(final_model, oos_dl)
y_oos_t      = extract_targets(y_oos, segs_oos, best['slen'], stride=OOS_STRIDE)
ts_oos_final = extract_timestamps(ts_oos_raw, segs_oos, best['slen'], stride=OOS_STRIDE)
y_oos_p_point, y_oos_p_q10, y_oos_p_q50, y_oos_p_q90 = preds_oos[:,0], preds_oos[:,1], preds_oos[:,2], preds_oos[:,3]

oos_mae      = float(mean_absolute_error(y_oos_t, y_oos_p_point))
oos_rmse     = float(np.sqrt(mean_squared_error(y_oos_t, y_oos_p_point)))
oos_r2       = float(r2_score(y_oos_t, y_oos_p_point))
oos_coverage = float(np.mean((y_oos_p_q10 <= y_oos_t) & (y_oos_t <= y_oos_p_q90)))
oos_pinball  = {
    'q10': pinball_np(y_oos_t, y_oos_p_q10, 0.1),
    'q50': pinball_np(y_oos_t, y_oos_p_q50, 0.5),
    'q90': pinball_np(y_oos_t, y_oos_p_q90, 0.9),
}
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f} | Coverage={oos_coverage:.3f}")

torch.save(final_model.state_dict(), 'itransformer_model.pt')
np.save('itransformer_point_preds.npy', y_p_point); np.save('itransformer_q10_preds.npy', y_p_q10)
np.save('itransformer_q50_preds.npy', y_p_q50);    np.save('itransformer_q90_preds.npy', y_p_q90)
np.save('itransformer_targets.npy', y_te_global);  np.save('itransformer_test_timestamps.npy', ts_te_global)
np.save('itransformer_oos_point_preds.npy', y_oos_p_point); np.save('itransformer_oos_q10_preds.npy', y_oos_p_q10)
np.save('itransformer_oos_q50_preds.npy', y_oos_p_q50);    np.save('itransformer_oos_q90_preds.npy', y_oos_p_q90)
np.save('itransformer_oos_targets.npy', y_oos_t);  np.save('itransformer_oos_timestamps.npy', ts_oos_final)

with open('itransformer_results.json', 'w') as f:
    json.dump({
        'model': 'iTransformer Multi-Output',
        'split': {'train': f'bis {train_end}', 'val': 'Nov–Dez 2025', 'test': 'Jan 2026 – Ende'},
        'n_train_spots': len(all_spots), 'n_oos_spots': len(oos_spots), 'n_batches': len(spot_batches),
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'best_params': best, 'best_val_loss': float(best_val),
        'n_params': sum(p.numel() for p in final_model.parameters()),
        'train_time': round(time.time() - t0, 2),
        'train_losses': [float(x) for x in train_losses], 'val_losses': [float(x) for x in val_losses],
        'permutation_importance': perm_importance,
        'quantiles': {'alphas': QUANTILES, 'pinball_loss': pinball_scores, 'coverage_80pct': coverage},
        'test': {'n_rows': int(len(y_te_global)), 'stride': TEST_STRIDE},
        'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2, 'coverage_80pct': oos_coverage,
                'pinball_loss': oos_pinball, 'n_spots': len(oos_spots), 'n_rows': int(len(y_oos_t)), 'stride': OOS_STRIDE},
    }, f, indent=2)
print("\nFertig! itransformer_results.json gespeichert.")