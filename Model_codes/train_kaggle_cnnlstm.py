import gc, json, time, torch, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, optuna
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); torch.cuda.manual_seed(42)
print(f"Device: {DEVICE}")

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
df['ts'] = pd.to_datetime(df['ts'])
train_end = pd.Timestamp('2025-10-31 23:00:00')
val_end   = pd.Timestamp('2025-12-31 23:00:00')
_split = np.zeros(len(df), dtype='int8')
_split[df['ts'] > train_end] = 1
_split[df['ts'] > val_end]   = 2

def get_segments(df, mask):
    segs, curr = [], 0
    for _, g in df[mask].groupby('spot_uuid', observed=True):
        n = len(g); segs.append((curr, n)); curr += n
    return segs

segs_tr = get_segments(df, _split == 0)
segs_va = get_segments(df, _split == 1)
segs_te = get_segments(df, _split == 2)

scaler = StandardScaler()
X_tr = scaler.fit_transform(df.loc[_split==0, FEATURES].values.astype(np.float32))
y_tr = df.loc[_split==0, TARGET].values.astype(np.float32)
X_va = scaler.transform(df.loc[_split==1, FEATURES].values.astype(np.float32))
y_va = df.loc[_split==1, TARGET].values.astype(np.float32)
X_te = scaler.transform(df.loc[_split==2, FEATURES].values.astype(np.float32))
y_te = df.loc[_split==2, TARGET].values.astype(np.float32)
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
del df; gc.collect()

class SpotSeqDS(Dataset):
    def __init__(self, X, y, segs, slen, stride):
        self.X, self.y, self.slen = X, y, slen
        self.starts = []
        for off, n in segs:
            if n >= slen:
                for i in range(0, n - slen + 1, stride):
                    self.starts.append(off + i)
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(self.X[s:s+self.slen]), torch.tensor(self.y[s+self.slen-1])

def extract_targets(y, segs, slen, stride=1):
    y_t, pos = [], 0
    for off, n in segs:
        if n >= slen:
            for i in range(0, n - slen + 1, stride):
                y_t.append(y[pos + slen - 1 + i])
        pos += n
    return np.array(y_t)

class CNNLSTMMod(nn.Module):
    def __init__(self, inp, cnn_ch, hid, n_layers, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inp, cnn_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_ch), nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn_ch, hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, 1))
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

def objective(trial):
    slen = trial.suggest_int('slen', 96, 576, step=96)
    cnn_ch = trial.suggest_int('cnn_ch', 16, 64, step=8)
    hid = trial.suggest_int('hid', 64, 256)
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    m = CNNLSTMMod(len(FEATURES), cnn_ch, hid, n_layers, dropout).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    dl = DataLoader(SpotSeqDS(X_tr, y_tr, segs_tr, slen, stride=192), batch_size=512, shuffle=True)
    dva = DataLoader(SpotSeqDS(X_va, y_va, segs_va, slen, stride=slen), batch_size=1024)
    best_v = float('inf')
    for ep in range(10):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); nn.L1Loss()(m(xb.to(DEVICE)), yb.to(DEVICE)).backward(); opt.step()
        m.eval(); v_l = 0.0
        with torch.no_grad():
            for xb, yb in dva: v_l += nn.L1Loss()(m(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        v_l /= len(dva.dataset)
        best_v = min(best_v, v_l)
    return best_v

print("Starte Optuna (30 Trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
best = study.best_params

final_model = CNNLSTMMod(len(FEATURES), best['cnn_ch'], best['hid'],
                          best['n_layers'], best['dropout']).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

train_dl = DataLoader(SpotSeqDS(X_tr, y_tr, segs_tr, best['slen'], stride=48),
                      batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
val_dl   = DataLoader(SpotSeqDS(X_va, y_va, segs_va, best['slen'], stride=best['slen']),
                      batch_size=512, pin_memory=True, num_workers=2)

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
    final_model.eval(); val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            val_loss += nn.L1Loss()(final_model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
    val_loss /= len(val_dl.dataset); val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Epoch {ep+1:02d} | Train: {train_losses[-1]:.6f} | Val: {val_loss:.6f}")
    if val_loss < best_val - 1e-5:
        best_val = val_loss; no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= 5: break

final_model.load_state_dict(best_state)

print("Evaluiere auf Testset...")
final_model.eval()
TEST_STRIDE = 1
test_dl = DataLoader(SpotSeqDS(X_te, y_te, segs_te, best['slen'], stride=TEST_STRIDE),
                     batch_size=512, pin_memory=True, num_workers=2)
preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        preds.append(final_model(xb.to(DEVICE)).cpu().numpy())

y_p = np.maximum(0, np.concatenate(preds))
y_t = extract_targets(y_te, segs_te, best['slen'], stride=TEST_STRIDE)

mae  = float(mean_absolute_error(y_t, y_p))
rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
r2   = float(r2_score(y_t, y_p))
print(f"\n=== ERGEBNISSE ===\nMAE:  {mae:.6f}\nRMSE: {rmse:.6f}\nR²:   {r2:.6f}")

print("\nOOS Evaluation...")
oos_spots = pd.read_parquet(OOS_PATH)['spot_uuid'].tolist()
df_oos = ds.dataset(str(DATA_PATH)).to_table(
    columns=FEATURES + ['ts', 'spot_uuid', TARGET],
    filter=pc.field('spot_uuid').isin(oos_spots)
).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos[FEATURES] = df_oos[FEATURES].ffill().fillna(0.0)
segs_oos, curr = [], 0
for _, g in df_oos.groupby('spot_uuid', observed=True):
    n = len(g); segs_oos.append((curr, n)); curr += n
X_oos  = scaler.transform(df_oos[FEATURES].values.astype(np.float32))
y_oos  = df_oos[TARGET].values.astype(np.float32)
ts_oos = df_oos['ts'].values
del df_oos; gc.collect()

OOS_STRIDE = 1
oos_dl = DataLoader(SpotSeqDS(X_oos, y_oos, segs_oos, best['slen'], stride=OOS_STRIDE),
                    batch_size=512, pin_memory=True, num_workers=2)
preds_oos = []
with torch.no_grad():
    for xb, _ in oos_dl:
        preds_oos.append(final_model(xb.to(DEVICE)).cpu().numpy())

y_oos_all = np.maximum(0, np.concatenate(preds_oos))  # (N, 4)
y_oos_p   = y_oos_all[:, 0]
y_oos_t   = extract_targets(y_oos,  segs_oos, best['slen'], stride=OOS_STRIDE)
ts_oos_t  = extract_targets(ts_oos, segs_oos, best['slen'], stride=OOS_STRIDE)

oos_mae  = float(mean_absolute_error(y_oos_t, y_oos_p))
oos_rmse = float(np.sqrt(mean_squared_error(y_oos_t, y_oos_p)))
oos_r2   = float(r2_score(y_oos_t, y_oos_p))
print(f"OOS → MAE={oos_mae:.6f} | n={len(y_oos_t):,}")

np.save('cnnlstm_oos_point_preds.npy', y_oos_p)
np.save('cnnlstm_oos_targets.npy',     y_oos_t)
np.save('cnnlstm_oos_timestamps.npy',  ts_oos_t)

torch.save(final_model.state_dict(), 'cnnlstm_model.pt')
with open('cnnlstm_results.json', 'w') as f:
    json.dump({
        'model': 'CNN-LSTM (Multi-Output, alle Spots)',
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'best_params': best,
        'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2,
                'n_spots': len(oos_spots), 'n_rows': len(y_oos_t), 'stride': OOS_STRIDE}
    }, f, indent=2)