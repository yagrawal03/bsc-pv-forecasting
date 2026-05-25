**Short-Term Solar Power Forecasting
for Residential PV Systems**

Probabilistic short-term photovoltaic feed-in forecasting for ~3,000 residential PV installations at 15-minute resolution. The project compares classical ML models (LightGBM, RandomForest), deep learning models (GRU, CNN-LSTM, iTransformer), and a foundation model (Chronos) with respect to point-forecast accuracy, probabilistic calibration, and inference efficiency.

---

## Repository Structure

```
.
├── 01_data_prep.ipynb              # Data preparation & feature engineering
├── 02_data_analysis.ipynb          # Exploratory data analysis (EDA)
├── 03_model_training.ipynb         # Subset training results & model comparison
├── 04_model_subset.ipynb           # Subset representativeness & feature selection
├── 05_training_dokumentation.ipynb # Full-train documentation (no execution)
├── 06_fullmodel_evaluation.ipynb   # Evaluation of fully-trained models (Stage 2)
│
├── Model_codes/                    # Kaggle GPU training scripts (Stage 1 & 2)
│   ├── train_kaggle_lgbm.py
│   ├── train_kaggle_rf.py
│   ├── train_kaggle_gru.py
│   ├── train_kaggle_cnnlstm.py
│   ├── train_kaggle_itransformer.py
│   ├── train_kaggle_TSMixer.py
│   ├── train_kaggle_chronos.py
│   ├── train_kaggle_linear_regression.py
│   └── train_spt_lgbm.py
│
├── fulltrainedmodels/              # Stage 2: training scripts + results (2,916 spots)
│   ├── gru.py / gru_results/
│   ├── cnn_lstm.py / cnn_lstm_results/
│   ├── i_transformer.py / i_transformer_results/
│   ├── random_forest.py / random_forest_results/
│   └── ligthgbm.py / lightgbm_results/
│
├── modelresults/                   # Outputs from NB3 / NB6 (plots, CSVs, JSONs)
│   ├── results_snapshot.json       # Stage 1 metrics incl. wMAE / wRMSE
│   ├── full_model_ranking.csv      # Stage 2 final ranking table
│   └── *.png                       # Training curves, feature importance, comparisons
│
├── dataprep/
│   └── unseen_dataset_builder.py   # Builds the 100-spot OOS hold-out set
│
└── data/
    ├── processed/                  # data_final.parquet, unseen_data.parquet (not in repo)
    ├── Kaggle_results/             # Stage 1 model outputs (.npy, .json, .pt, .pkl)
    └── raw/                        # Raw input files
```

---

## Dataset

| Property | Value |
|---|---|
| PV installations (spots) | 3,016 |
| Time range | 09 Jul 2024 – 27 Mar 2026 |
| Resolution | 15 minutes |
| Total rows | 67,917,827 |
| Features used for training | 45 |
| Target variable | `kwh_norm` = kWh / kwp_est |
| Train split | until 31 Oct 2025 |
| Validation split | Nov – Dec 2025 |
| Test split | Jan 2026 – end |
| OOS hold-out | 100 spots (completely withheld) |

> The data is **not included** in this repository due to size. The full pipeline for reproducing `data_final.parquet` is documented in `01_data_prep.ipynb`.

---

## Feature Engineering

45 features across 7 groups:

| Group | Features |
|---|---|
| Irradiance (7) | `ghi`, `dhi`, `dni`, `tsi`, `kt`, `dni_frac`, `dhi_frac` |
| Solar position (2) | `solar_elevation`, `solar_azimuth` |
| Weather (8) | `temperature_2m`, `precipitation_mm`, `wind_speed_10m`, `relative_humidity_2m`, `snowfall`, `visibility`, `weather_regime`, `snow_roll6h` |
| Rolling aggregates (3) | `snow_roll24h`, `precip_roll6h`, `temp_roll24h` |
| Cyclic time encoding (6) | `hour_sin/cos`, `month_sin/cos`, `doy_sin/cos` |
| Plant metadata (9) | `latitude`, `longitude`, `panel_tilt`, `max_kwh_est`, `kwp_est`, `panel_unimodal_azimuth`, `panel_tilt_azi_confidence`, `panel_bimodal_east_azimuth`, `panel_bimodal_east_fraction` |
| Analog ensemble (7) | `analog_knn_mean_norm`, `scaled_analog_kwh_1`, `scaled_analog_knn_mean`, `analog_dist_1`, `analog_dist_gap`, `analog_knn_std`, `analog_age_hours` |

**Analog features** are consistently the strongest predictors. For each 15-min timestamp, the k=10 most similar historical production profiles of the same plant are retrieved via weighted kNN (seasonal proximity ±60 days, solar elevation ±3°, azimuth ±30°, min separation 12 h) and scaled by current irradiance.

**Panel tilt & azimuth** are inferred from production profiles following Meng, Loonen & Hensen (2020, *Solar Energy* 211:418–432). Both unimodal (south-facing) and bimodal (east/west split) configurations are detected.

**Weather data** is fetched at native 15-min resolution from the Open-Meteo Historical Forecast API (`minutely_15`, DWD ICON-D2 for Central Europe) and joined to the measurement data by exact timestamp.

---

## Two-Stage Training Strategy

### Stage 1 — Subset (HPO & Model Selection)

- **Data**: Top-406 spots by data completeness (`clean_rate = 1 − NaN/total`; min 98.5%)
- **Purpose**: Hyperparameter optimisation (Optuna TPE-Sampler, 25 trials, seed=42) and model selection
- **Models**: LightGBM, LinearRegression, RandomForest, GRU, CNN-LSTM, TSMixer, iTransformer, Chronos (zero-shot), Analog KNN
- **Scripts**: `Model_codes/train_kaggle_*.py` — designed for Kaggle T4 GPU
- **Point predictions only** — no quantile heads at this stage
- **Evaluation**: `03_model_training.ipynb`, `04_model_subset.ipynb`

### Stage 2 — Full Training (Best 5 Models)

- **Data**: 2,916 spots (all 3,016 minus 100 OOS hold-out spots)
- **Hyperparameters**: transferred from Stage 1, no further HPO
- **Models**: LightGBM, RandomForest, GRU, CNN-LSTM, iTransformer
- **Scripts**: `fulltrainedmodels/*.py`
- **Quantile models**: q10/q50/q90 via Pinball Loss (DL multi-output head) or separate LightGBM models
- **Excluded from Stage 2**: LinearRegression (insufficient accuracy), TSMixer (redundant with GRU), Chronos (zero-shot, no retraining needed)

---

## Model Architectures

### GRU
Input sequence → GRU → LayerNorm → MLP head → `[point, q10, q50, q90]`
```python
gru  = nn.GRU(n_features, hid, n_layers, batch_first=True)
norm = nn.LayerNorm(hid)
head = nn.Sequential(Linear(hid, hid//2), GELU(), Dropout(dropout), Linear(hid//2, 4))
```

### CNN-LSTM
Conv1d feature extraction → LSTM → linear head. Quantiles via separate Pinball-Loss models.
```python
conv = Sequential(Conv1d(n_features, cnn_ch, 3, padding=1), BatchNorm1d(cnn_ch), ReLU())
lstm = LSTM(cnn_ch, hid, n_layers, batch_first=True)
head = Sequential(Linear(hid, hid//2), ReLU(), Linear(hid//2, 1))
```

### iTransformer
Inverted Transformer: self-attention over **features** (not time steps) following Liu et al. (2023).
```python
input_proj = nn.Linear(seq_len, d_model)      # project each feature's time series
layers     = ModuleList([TransformerLayer(...)])
head       = Sequential(Linear(n_features * d_model, d_model), GELU(), Linear(d_model, 4))
def forward(self, x):
    x = self.input_proj(x.transpose(1, 2))    # (B, F, d_model)
    for l in self.layers: x = l(x)
    return self.head(x.flatten(1))             # [point, q10, q50, q90]
```

### LightGBM
Four separate models: `lgbm_point.txt` (MAE), `lgbm_q10/q50/q90.txt` (Pinball Loss).

### RandomForest
`sklearn.RandomForestRegressor` with `warm_start=True` for incremental batch training across 2,916 spots. Quantiles via empirical leaf-node distributions.

---

## Hyperparameters

All Stage 2 models use the best hyperparameters found in Stage 1.

| Model | Key Parameters |
|---|---|
| LightGBM | `lr=0.0266`, `num_leaves=249`, `feature_fraction=0.882`, `bagging_fraction=0.868`, `reg_lambda=0.487` |
| RandomForest | `warm_start`, 80 trees total (8 batches × 10), `n_features=45` |
| GRU | `slen=168`, `hid=128`, `n_layers=2`, `dropout=0.10`, `lr=0.001` |
| CNN-LSTM | `slen=480`, `cnn_ch=52`, `hid=99`, `n_layers=1`, `dropout=0.191`, `lr=0.00112` |
| iTransformer | `slen=168`, `d_model=128`, `n_heads=4`, `n_layers=1`, `d_ff=64`, `dropout=0.049`, `lr=0.000406` |

**Shared DL config**: Adam, MAE (L1Loss) objective, early stopping patience=5, ReduceLROnPlateau patience=3 factor=0.5, Kaggle T4 GPU.

---

## Results

### Stage 1 — Test Set (406 Spots)

| Model | MAE | wMAE | wRMSE | R² |
|---|---|---|---|---|
| **LightGBM SPT** | 0.02812 | **0.0533** | **0.0764** | 0.645 |
| RandomForest | 0.01239 | 0.0935 | 0.1318 | 0.716 |
| LightGBM | 0.01216 | 0.0947 | 0.1328 | 0.724 |
| iTransformer | 0.01148 | 0.0965 | 0.1379 | 0.689 |
| GRU | 0.01131 | 0.0995 | 0.1455 | 0.688 |
| CNN-LSTM | 0.01160 | 0.1015 | 0.1451 | 0.684 |
| TSMixer | 0.01153 | 0.1018 | 0.1438 | 0.699 |
| LinearRegression | 0.01907 | 0.1393 | 0.1635 | 0.237 |
| Chronos (zero-shot) | 0.05472 | 0.2705 | 0.3144 | −0.512 |
| Analog KNN | — | 0.7408 | 0.9572 | −12.82 |

### Stage 2 — OOS Hold-out (100 Spots)

| Model | OOS MAE | OOS wMAE | OOS wRMSE | Coverage 80% | Test wRMSE |
|---|---|---|---|---|---|
| **CNN-LSTM** | **0.01569** | **0.0885** | **0.1250** | 87.3% | 0.13599 |
| GRU | 0.01632 | 0.0893 | 0.1221 | 49.9% ⚠️ | 0.13654 |
| iTransformer | 0.01719 | 0.0908 | 0.1262 | 79.7% | 0.13708 |
| RandomForest | 0.01830 | 0.1009 | 0.1352 | 32.5% ⚠️ | 0.14135 |
| LightGBM | 0.02022 | 0.1052 | 0.1424 | 74.5% | 0.15562 |

**wMAE / wRMSE** are capacity-weighted metrics (w = y / ȳ) — giving higher weight to high-production timesteps.  
**Coverage 80%** measures P(q10 ≤ y ≤ q90); target is 0.80. GRU and RandomForest are undercalibrated.

---

## Metrics Reference

| Metric | Formula | Notes |
|---|---|---|
| MAE | mean\|ŷ − y\| | Primary point metric |
| wMAE | Σ w·\|ŷ−y\| / Σw | Capacity-weighted, w = y/ȳ |
| RMSE | √ mean(ŷ−y)² | |
| wRMSE | √(Σ w·(ŷ−y)² / Σw) | Capacity-weighted RMSE |
| Pinball | α·(y−q) if y≥q, else (α−1)·(y−q) | For q ∈ {0.10, 0.50, 0.90} |
| Coverage 80% | P(q10 ≤ y ≤ q90) | Target: 0.80 |

---

## Setup

```bash
pip install numpy pandas pyarrow lightgbm scikit-learn torch optuna pvlib tqdm matplotlib pillow
```

Training scripts run on **Kaggle GPU** (T4, 30 GB RAM). Data paths in the scripts (`/kaggle/input/...`) need to be adapted for local execution.

---

## References

- Meng, Q., Loonen, R., Hensen, J. (2020). Inferring building energy performance from PV production profiles. *Solar Energy*, 211, 418–432.
- Liu, Y. et al. (2023). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *arXiv:2310.06625*.
- Ansari, A.F. et al. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Perez, R. et al. (1990). Modeling daylight availability and irradiance components from direct and global irradiance. *Solar Energy*, 44(5), 271–289.

