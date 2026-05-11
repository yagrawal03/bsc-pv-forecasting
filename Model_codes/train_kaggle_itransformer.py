import gc, json, time, torch, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, optuna
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- CONFIG & PATHS ---
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

N_FEATURES = len(FEATURES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); torch.cuda.manual_seed(42)

# --- DATA PREPARATION ---
print("Lade Daten...")
meta = ds.dataset(str(DATA_PATH)).to_table(columns=['spot_uuid', TARGET]).to_pandas()
stats = meta.groupby('spot_uuid', observed=True)[TARGET].agg(['count', lambda x: x.isna().sum()]).reset_index()
stats.columns = ['spot_uuid', 'total', 'nans']
stats['clean'] = 1 - (stats['nans'] / stats['total'])
selected = stats.sort_values(['clean', 'total'], ascending=False).iloc[:406]['spot_uuid'].tolist()
del meta, stats; gc.collect()

df = ds.dataset(str(DATA_PATH)).to_table(columns=FEATURES + ['spot_uuid', TARGET, 'ts'], filter=pc.field('spot_uuid').isin(selected)).to_pandas()
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df = df[df[TARGET] <= 2.0].reset_index(drop=True)
df[FEATURES] = df[FEATURES].ffill().fillna(0.0)
df['ts'] = pd.to_datetime(df['ts'])

train_end, val_end = pd.Timestamp('2025-10-31 23:00:00'), pd.Timestamp('2025-12-31 23:00:00')
_split = np.zeros(len(df), dtype='int8')
_split[df['ts'] > train_end] = 1
_split[df['ts'] > val_end]   = 2

def get_segments(df_orig, mask):
    segs, curr = [], 0
    for _, g in df_orig[mask].groupby('spot_uuid', observed=True):
        n = len(g); segs.append((curr, n)); curr += n
    return segs

segs_tr, segs_va, segs_te = get_segments(df, _split==0), get_segments(df, _split==1), get_segments(df, _split==2)
scaler = StandardScaler()
X_tr = scaler.fit_transform(df.loc[_split==0, FEATURES].values.astype(np.float32))
y_tr = df.loc[_split==0, TARGET].values.astype(np.float32)
X_va = scaler.transform(df.loc[_split==1, FEATURES].values.astype(np.float32))
y_va = df.loc[_split==1, TARGET].values.astype(np.float32)
X_te = scaler.transform(df.loc[_split==2, FEATURES].values.astype(np.float32))
y_te = df.loc[_split==2, TARGET].values.astype(np.float32)
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
del df; gc.collect()

# --- MODEL ARCHITECTURE ---
class SpotSeqDS(Dataset):
    def __init__(self, X, y, segs, slen, stride):
        self.X, self.y, self.slen = X, y, slen
        self.starts = []
        for off, n in segs:
            if n >= slen:
                for i in range(0, n - slen + 1, stride): self.starts.append(off + i)
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(self.X[s:s+self.slen]), torch.tensor(self.y[s+self.slen-1])

def extract_targets(y, segs, slen, stride=1):
    y_t, pos = [], 0
    for off, n in segs:
        if n >= slen:
            for i in range(0, n - slen + 1, stride): y_t.append(y[pos + slen - 1 + i])
        pos += n
    return np.array(y_t)

class ITransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, x):
        res = x; x, _ = self.attn(x, x, x)
        x = self.norm1(x + res)
        x = self.norm2(self.ff(x) + x)
        return x

class ITransformer(nn.Module):
    def __init__(self, seq_len, n_features, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([ITransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(n_features * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers: x = layer(x)
        x = self.norm(x).flatten(1)
        return self.head(x).squeeze(-1)

# --- OPTUNA SEARCH ---
def objective(trial):
    slen, d_model = trial.suggest_categorical('slen', [48, 96, 168]), trial.suggest_categorical('d_model', [32, 64, 128])
    n_heads, n_layers = trial.suggest_categorical('n_heads', [2, 4, 8]), trial.suggest_int('n_layers', 1, 3)
    d_ff, dropout = trial.suggest_categorical('d_ff', [64, 128, 256]), trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 5e-4, 3e-3, log=True)
    if d_model % n_heads != 0: return float('inf')

    m = ITransformer(slen, N_FEATURES, d_model, n_heads, n_layers, d_ff, dropout).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    dl = DataLoader(SpotSeqDS(X_tr, y_tr, segs_tr, slen, stride=192), batch_size=512, shuffle=True)
    dva = DataLoader(SpotSeqDS(X_va, y_va, segs_va, slen, stride=slen), batch_size=1024)

    bv = float('inf')
    for ep in range(10):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); nn.L1Loss()(m(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
        m.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in dva: vl += nn.L1Loss()(m(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(dva.dataset)
        bv = min(bv, vl)
    return bv

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
best = study.best_params

# --- FINAL TRAINING WITH EARLY STOPPING ---  # FIXED
final_model = ITransformer(best['slen'], N_FEATURES, best['d_model'], best['n_heads'], best['n_layers'], best['d_ff'], best['dropout']).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
train_dl = DataLoader(SpotSeqDS(X_tr, y_tr, segs_tr, best['slen'], stride=48), batch_size=512, shuffle=True)
val_dl = DataLoader(SpotSeqDS(X_va, y_va, segs_va, best['slen'], stride=best['slen']), batch_size=1024)

best_val, no_improve, best_state = float('inf'), 0, None
train_losses, val_losses = [], []
t0 = time.time()

for ep in range(50):
    final_model.train(); ep_loss, ep_n = 0.0, 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = nn.L1Loss()(final_model(xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward(); optimizer.step()
        ep_loss += loss.item() * len(xb); ep_n += len(xb)
    train_losses.append(ep_loss / ep_n)
    final_model.eval(); vl = 0.0
    with torch.no_grad():
        for xb, yb in val_dl: vl += nn.L1Loss()(final_model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
    vl /= len(val_dl.dataset); val_losses.append(vl)
    scheduler.step(vl)
    print(f"Epoch {ep+1:02d} | Train: {train_losses[-1]:.6f} | Val: {vl:.6f}")
    if vl < best_val - 1e-5:
        best_val = vl; no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= 5: print(f"Early stopping Epoch {ep+1}"); break

final_model.load_state_dict(best_state)

# --- TEST EVAL ---
test_dl = DataLoader(SpotSeqDS(X_te, y_te, segs_te, best['slen'], stride=1), batch_size=1024)
preds = []
with torch.no_grad():
    for xb, _ in test_dl: preds.append(final_model(xb.to(DEVICE)).cpu().numpy())
y_p = np.maximum(0, np.concatenate(preds))
y_t = extract_targets(y_te, segs_te, best['slen'], stride=1)
mae  = float(mean_absolute_error(y_t, y_p))
rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
r2   = float(r2_score(y_t, y_p))
print(f"\n=== ERGEBNISSE ===\nMAE:  {mae:.6f}\nRMSE: {rmse:.6f}\nR²:   {r2:.6f}")

# --- OOS ---
oos_spots = pd.read_parquet(OOS_PATH)['spot_uuid'].tolist()
df_oos = ds.dataset(str(DATA_PATH)).to_table(columns=FEATURES + ['ts', 'spot_uuid', TARGET], filter=pc.field('spot_uuid').isin(oos_spots)).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
X_oos  = scaler.transform(df_oos[FEATURES].ffill().fillna(0.0).values.astype(np.float32))
y_oos  = df_oos[TARGET].values.astype(np.float32)
ts_oos = df_oos['ts'].values
segs_oos, curr = [], 0
for _, g in df_oos.groupby('spot_uuid', observed=True):
    n = len(g); segs_oos.append((curr, n)); curr += n
del df_oos; gc.collect()

OOS_STRIDE = 1
oos_dl = DataLoader(SpotSeqDS(X_oos, y_oos, segs_oos, best['slen'], stride=OOS_STRIDE), batch_size=512)
preds_oos = []
with torch.no_grad():
    for xb, _ in oos_dl: preds_oos.append(final_model(xb.to(DEVICE)).cpu().numpy())
y_oos_all = np.maximum(0, np.concatenate(preds_oos))  # (N, 4)
y_oos_p   = y_oos_all[:, 0]
y_oos_t   = extract_targets(y_oos,  segs_oos, best['slen'], stride=OOS_STRIDE)
ts_oos_t  = extract_targets(ts_oos, segs_oos, best['slen'], stride=OOS_STRIDE)
oos_mae  = float(mean_absolute_error(y_oos_t, y_oos_p))
oos_rmse = float(np.sqrt(mean_squared_error(y_oos_t, y_oos_p)))
oos_r2   = float(r2_score(y_oos_t, y_oos_p))
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f} | n={len(y_oos_t):,}")

np.save('itransformer_oos_point_preds.npy', y_oos_p)
np.save('itransformer_oos_targets.npy',     y_oos_t)
np.save('itransformer_oos_timestamps.npy',  ts_oos_t)

torch.save(final_model.state_dict(), 'itransformer_model.pt')
with open('itransformer_results.json', 'w') as f:
    json.dump({
        'model': 'iTransformer (Multi-Output, alle Spots)',
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'best_params': best, 'best_val_mae': float(best_val),
        'train_losses': [float(x) for x in train_losses],
        'val_losses':   [float(x) for x in val_losses],
        'train_time': round(time.time() - t0, 2),
        'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2,
                'n_spots': len(oos_spots), 'n_rows': len(y_oos_t), 'stride': OOS_STRIDE}
    }, f, indent=2)