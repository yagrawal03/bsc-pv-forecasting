import subprocess
subprocess.run(['pip', 'install', '-q',
    'git+https://github.com/amazon-science/chronos-forecasting.git',
    '--force-reinstall', '--no-deps'
], check=False)
subprocess.run(['pip', 'install', '-q', 'chronos-forecasting'], check=False)

import gc, json, time, os, numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosPipeline
from tqdm.auto import tqdm

DATA_PATH   = Path('/kaggle/input/datasets/yashagrawal511/data-final-parquet/data_final.parquet')
OOS_PATH    = Path('/kaggle/input/datasets/yashagrawal511/unseen/oos_spots.parquet')
TARGET      = 'kwh_norm'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONTEXT_LEN = 512
PRED_LEN    = 144
BATCH_SIZE  = 8
torch.manual_seed(42)
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
    columns=[TARGET, 'spot_uuid', 'ts'], filter=pc.field('spot_uuid').isin(selected)
).to_pandas()
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# ── AUSREISSER ENTFERNEN ──
before = len(df)
df = df[df[TARGET] <= 2.0].reset_index(drop=True)
print(f"Ausreißer entfernt: {before - len(df):,} Zeilen (kwh_norm > 2.0)")

df[TARGET] = df[TARGET].ffill().fillna(0.0)
print(f"Datensatzgröße: {len(df):,} Zeilen")

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

segs_te = get_segments(df, _split == 2)
y_te = df.loc[_split==2, TARGET].values.astype(np.float32)
print(f"Test: {len(y_te):,} Zeilen")
del df; gc.collect()

class ChronosSeqDS(Dataset):
    def __init__(self, y, segs, context_len, pred_len, stride):
        self.y = y
        self.context_len = context_len
        self.pred_len = pred_len
        self.starts = []
        for off, n in segs:
            if n >= context_len + pred_len:
                for i in range(0, n - context_len - pred_len + 1, stride):
                    self.starts.append(off + i)
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        context = torch.tensor(self.y[s:s+self.context_len], dtype=torch.float32)
        target  = torch.tensor(self.y[s+self.context_len:s+self.context_len+self.pred_len], dtype=torch.float32)
        return context, target

print("Lade Chronos-T5-tiny (Zero-Shot)...")
pipeline = ChronosPipeline.from_pretrained(
    'amazon/chronos-t5-tiny',
    device_map=str(DEVICE),
    torch_dtype=torch.float16)
print(f"Parameter: {sum(p.numel() for p in pipeline.model.parameters()):,}")

def run_inference(dl, desc="Inference"):
    pipeline.model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for context, target in tqdm(dl, desc=desc, leave=True):
            try:
                samples = pipeline.predict(
                    context,
                    prediction_length=PRED_LEN,
                    num_samples=5,
                )
                pred = samples.median(dim=1).values
                preds.append(pred.numpy())
                targets.append(target.numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print("OOM — Batch übersprungen"); continue
    return (np.maximum(0, np.concatenate(preds).flatten()),
            np.concatenate(targets).flatten())

# ── EVALUATION TESTSET ──
print("\nEvaluiere auf Testset...")
t0 = time.time()
test_ds = ChronosSeqDS(y_te, segs_te, CONTEXT_LEN, PRED_LEN, stride=PRED_LEN)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=2)
print(f"Test Batches: {len(test_dl)}")
y_p, y_t = run_inference(test_dl, desc="Test Inference")

mae  = float(mean_absolute_error(y_t, y_p))
rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
r2   = float(r2_score(y_t, y_p))
print(f"\n=== ERGEBNISSE ===\nMAE:  {mae:.6f}\nRMSE: {rmse:.6f}\nR²:   {r2:.6f}")

np.save('chronos_preds.npy', y_p)
np.save('chronos_targets.npy', y_t)
with open('chronos_results.json', 'w') as f:
    json.dump({'model': 'Chronos-T5-tiny (zero-shot)',
               'mae': mae, 'rmse': rmse, 'r2': r2,
               'context_len': CONTEXT_LEN, 'pred_len': PRED_LEN,
               'stride': PRED_LEN, 'num_samples': 5,
               'n_params': sum(p.numel() for p in pipeline.model.parameters()),
               'test_rows': int(len(y_t)),
               'eval_time_so_far': round(time.time() - t0, 2)}, f, indent=2)
print("✓ Testset Ergebnisse zwischengespeichert!")

# ── OOS EVALUATION ──
print("\nOOS Evaluation...")
oos_spots = pd.read_parquet(OOS_PATH)['spot_uuid'].tolist()
df_oos = ds.dataset(str(DATA_PATH)).to_table(
    columns=[TARGET, 'spot_uuid'],
    filter=pc.field('spot_uuid').isin(oos_spots)
).to_pandas()
df_oos = df_oos.dropna(subset=[TARGET]).reset_index(drop=True)
df_oos = df_oos[df_oos[TARGET] <= 2.0].reset_index(drop=True)  # gleicher Filter
df_oos[TARGET] = df_oos[TARGET].ffill().fillna(0.0)

segs_oos, curr = [], 0
for _, g in df_oos.groupby('spot_uuid', observed=True):
    n = len(g); segs_oos.append((curr, n)); curr += n
y_oos = df_oos[TARGET].values.astype(np.float32)
del df_oos; gc.collect()

oos_dl = DataLoader(
    ChronosSeqDS(y_oos, segs_oos, CONTEXT_LEN, PRED_LEN, stride=PRED_LEN),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=2)
y_oos_p, y_oos_t = run_inference(oos_dl, desc="OOS Inference")

oos_mae  = float(mean_absolute_error(y_oos_t, y_oos_p))
oos_rmse = float(np.sqrt(mean_squared_error(y_oos_t, y_oos_p)))
oos_r2   = float(r2_score(y_oos_t, y_oos_p))
print(f"OOS → MAE={oos_mae:.6f} | RMSE={oos_rmse:.6f} | R²={oos_r2:.6f}")

# ── SPEICHERN ──
np.save('chronos_preds.npy',       y_p)
np.save('chronos_targets.npy',     y_t)
np.save('chronos_oos_preds.npy',   y_oos_p)
np.save('chronos_oos_targets.npy', y_oos_t)

with open('chronos_results.json', 'w') as f:
    json.dump({'model': 'Chronos-T5-tiny (zero-shot)',
               'mae': mae, 'rmse': rmse, 'r2': r2,
               'context_len': CONTEXT_LEN, 'pred_len': PRED_LEN,
               'stride': PRED_LEN, 'num_samples': 5,
               'n_params': sum(p.numel() for p in pipeline.model.parameters()),
               'test_rows': int(len(y_t)),
               'eval_time': round(time.time() - t0, 2),
               'oos': {'mae': oos_mae, 'rmse': oos_rmse, 'r2': oos_r2,
                       'n_spots': len(oos_spots), 'n_rows': int(len(y_oos_t))}}, f, indent=2)

print("\nDateien gespeichert:")
for fname in ['chronos_results.json', 'chronos_preds.npy', 'chronos_targets.npy',
              'chronos_oos_preds.npy', 'chronos_oos_targets.npy']:
    path = f'/kaggle/working/{fname}'
    if os.path.exists(path):
        print(f"✓ {fname} ({os.path.getsize(path)/1024/1024:.2f} MB)")
    else:
        print(f"✗ {fname} FEHLT!")

print("\nFertig! Klicke 'Save & Run All (Commit)' damit alles permanent gespeichert wird.")