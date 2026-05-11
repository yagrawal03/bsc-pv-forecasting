# BSc Thesis – PV Forecasting: Vollständige Projektübersicht

> **Zweck dieses Dokuments**: Kompakte Gesamtübersicht des Projekts für KI-Assistenten oder Mitarbeiter, die schnell tiefen Kontext benötigen. Stand: April 2026.

---

## 1. Projektüberblick

**Thema**: Probabilistische Kurzfrist-Solarprognose (PV-Forecasting) für ~3000 Haushaltsphotovoltaikanlagen in Europa.

**Ziel der Bachelorarbeit**: Vergleich klassischer ML-Modelle (LightGBM, RandomForest) mit Deep-Learning-Modellen (GRU, CNN-LSTM, iTransformer) und Foundation Models (Chronos, TSMixer) hinsichtlich Prognosegenauigkeit, Kalibrierung und Praxistauglichkeit (Inferenzzeit, UMBRA-Batterieoptimierung).

**Betrieb**: spotmyenergy / ect_forecasts Repository

**Datenbasis**: 15-min Haushaltsdaten | Jul 2024 – Mär 2026 | 3016 Spots | 67,9 Mio. Zeilen

---

## 2. Repository-Struktur

```
ect_forecasts/
├── bscthesis/
│   ├── 01_data_prep.ipynb          # Datenvorbereitung & Feature Engineering
│   ├── 02_data_analysis.ipynb      # Explorative Datenanalyse (EDA)
│   ├── 03_model_training.ipynb     # Subset-Training & Modellvergleich (NB-Modelle)
│   ├── 04_model_subset.ipynb       # Subset-Auswertung & Feature Selection
│   ├── 05_training_dokumentation.ipynb  # Dokumentation Volltraining (kein Run)
│   ├── 06_fullmodel_evaluation.ipynb    # Evaluation volltrainierter Modelle
│   ├── Model_codes/                # Kaggle-Trainings-Skripte (GPU)
│   │   ├── train_kaggle_lgbm.py
│   │   ├── train_kaggle_gru.py
│   │   ├── train_kaggle_cnnlstm.py
│   │   ├── train_kaggle_itransformer.py
│   │   ├── train_kaggle_rf.py
│   │   ├── train_kaggle_TSMixer.py
│   │   ├── train_kaggle_chronos.py
│   │   └── train_kaggle_linear_regression.py
│   ├── data/
│   │   ├── processed/
│   │   │   ├── data_final.parquet      # Hauptdatensatz (56 Spalten, 67.9M Zeilen)
│   │   │   └── unseen_data.parquet     # 100 OOS-Spots (hold-out)
│   │   └── raw/2026_03_10/            # UMBRA-Daten (Verbrauch, DA-Preise)
│   ├── fulltrainedmodels/
│   │   ├── gru_results/
│   │   ├── cnn_lstm_results/
│   │   ├── i_transformer_results/
│   │   ├── random_forest_results/
│   │   └── lightgbm_results/          # In Training (Stand Apr 2026)
│   └── modelresults/                  # Ausgaben von NB6 (PNGs, CSVs, JSON)
├── experiments/
│   └── total_pv_output/data/processed/
│       └── 2026-03-03_pv_with_analogs.parquet  # Batterie-Metadaten (UMBRA)
├── reports/eda_spt/                   # EDA-Plots aus NB2
└── umbra_eval/                        # UMBRA Battery Optimizer Integration
```

---

## 3. Datensatz

### 3.1 Hauptdatensatz (`data_final.parquet`)

| Eigenschaft | Wert |
|---|---|
| Zeilen | 67.917.827 |
| Spots (PV-Anlagen) | 3.016 |
| Zeitraum | 09.07.2024 – 27.03.2026 |
| Auflösung | 15 Minuten |
| Spalten gesamt | 56 |
| Trainingssplit | bis 31.10.2025 |
| Validierung | Nov–Dez 2025 |
| Test | Jan 2026 – Ende |
| OOS (hold-out) | 100 Spots (komplett zurückgehalten) |

### 3.2 Feature-Set (45 Modell-Features)

```python
FEATURES = [
    # Einstrahlung (7)
    'ghi', 'dhi', 'dni', 'tsi', 'kt', 'dni_frac', 'dhi_frac',
    # Solar-Position (2)
    'solar_elevation', 'solar_azimuth',
    # Wetter (8)
    'temperature_2m', 'precipitation_mm', 'wind_speed_10m',
    'relative_humidity_2m', 'snowfall', 'visibility',
    'weather_regime', 'snow_roll6h',
    # Rolling (3)
    'snow_roll24h', 'precip_roll6h', 'temp_roll24h',
    # Zeitkodierung (6)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    # Anlagen-Metadaten (9)
    'latitude', 'longitude', 'panel_tilt', 'max_kwh_est', 'kwp_est',
    'panel_unimodal_azimuth', 'panel_tilt_azi_confidence',
    'panel_bimodal_east_azimuth', 'panel_bimodal_east_fraction',
    # Panel-Geometrie (3)
    'panel_bimodal_west_azimuth', 'panel_is_ew_split', 'irr_mismatch',
    # Analog-Features (7)
    'analog_knn_mean_norm', 'scaled_analog_kwh_1', 'scaled_analog_knn_mean',
    'analog_dist_1', 'analog_dist_gap', 'analog_knn_std', 'analog_age_hours',
]
# N_FEAT = 45
```

### 3.3 Zielvariable

- `kwh_norm` = normierte PV-Produktion (kwh / kwp_est), Wertebereich [0, ~1.5]
- `kwh_norm_filled` = forward-filled Version für Analyse

### 3.4 Analog-Features (wichtigstes Feature-Set)

Für jeden Zeitpunkt werden die K=10 ähnlichsten historischen PV-Profile (Analoge) gesucht und skaliert. Key-Features:
- `analog_knn_mean_norm`: Mittlerer normierter KNN-Analogwert → stärkster Prädiktor
- `scaled_analog_kwh_1`: Bester Analog, auf Anlagenkapazität skaliert
- `analog_knn_std`: Streuung der Analoge = Unsicherheitsindikator
- `analog_age_hours`: Alter des nächsten Analogs

### 3.5 Panel-Geometrie (inferiert, kein Cold-Start-Problem im Training)

Tilt & Azimuth werden aus historischen Produktionsprofilen geschätzt:
- **Unimodal**: Hauptausrichtung (Süd-Anlage)
- **Bimodal**: Ost-West-Split (z.B. 40% Ost, 60% West)
- `irr_mismatch`: Abweichung zwischen theoretischer und tatsächlicher Einstrahlung

**Cold-Start-Problem**: Neue Spots haben keine Analog-History → Fallback: skalierter KNN-Analog aus geografisch ähnlichen Spots. Ist als Future Work notiert.

---

## 4. Notebook-Pipeline (NB1–NB6)

### NB1: `01_data_prep.ipynb` – Datenvorbereitung

**Was es macht:**
1. Rohdaten laden (SPT parquet + Wetterdaten)
2. Wetter-Features berechnen (GHI, DHI, DNI, kt, Schnee-Rolls, etc.)
3. Panel-Tilt/Azimuth inferieren (Algorithmus: Peak-Zeit-Analyse pro Spot)
4. kWp schätzen (`kwp_est = max_production / panel_efficiency_factor`)
5. Analog-KNN berechnen (K=10 nächste historische Profile)
6. `temperature_penalty_rel`: Temperaturkoeffizient-Korrekturfaktor (-0.004/°C über 25°C)
7. `weather_regime`: Klassifikation (klar/bewölkt/Niederschlag/Schnee)
8. Checkpoint & finales Parquet speichern

**Key-Outputs:**
- `data_final.parquet` (56 Spalten, 67.9M Zeilen)
- `unseen_data.parquet` (100 OOS-Spot-UUIDs)

### NB2: `02_data_analysis.ipynb` – Explorative Datenanalyse

**Sektionen:**
- **0**: Daten laden (Checkpoint-System)
- **0b**: Datensatz-Überblick (Monatliche Produktion, aktive Spots, Tagesgang)
- **1**: Datensatz-Übersicht & Vollständigkeit (Spot-Tabelle: NaN-Rate, aktive Tage)
- **1b**: Werte um Null/NaN – isoliert oder strukturell?
- **2**: Verteilungen (kWh-Tagesgang & Histogramm)
- **3**: Saisonalität (Monatlicher Median + Heatmap Stunde × Monat)
- **4**: Strahlungs- & Wetteranalyse (GHI, kt, kwh_norm vs. GHI)
- **5**: Panel-Neigung & Ausrichtung (Tilt/Azimuth-Histogramme)
- **6**: Ausreißeranalyse (Robust Z-Score mit MAD, Schwelle 8.0)
- **7**: Tagesprofil-Analyse (Klare Tage: Ost/Süd/West-Ausrichtung)
- **8**: Anlagenkapazität (kWp-Histogramm)
- **9**: Analog-Features Analyse
- **10**: Korrelationsanalyse (Spearman)
- **10b**: Korrelationsmatrix (alle ~50 Features, Heatmap)
- **10c**: Zeitliche & Lag-Korrelationsanalyse
- **11**: Typische Tagesverläufe
- **12**: Zusammenfassung

**Technische Notizen:**
- Checkpoint-System: Falls `_analysis_checkpoint.parquet` existiert → überspringt Laden
- OOM-Fix in Sektion 1: Vektorisierter Groupby statt Python-Loop über 3016 Spots

### NB3: `03_model_training.ipynb` – Subset-Training & Modellvergleich

**Lädt vorberechnete Ergebnisse aus Kaggle-Trainings-Skripten und vergleicht alle Modelle.**

NB3 trainiert selbst nicht — die eigentlichen Trainings laufen auf Kaggle (GPU). NB3 ist das Auswertungs-Notebook für das Subset.

**Trainiertes Modell-Lineup (Subset) — nur Point-Prediction:**

| Modell | Typ | Output |
|---|---|---|
| LinearRegression | Linear | Point |
| GRU | RNN | Point |
| CNN-LSTM | CNN+RNN | Point |
| TSMixer | MLP-Mixer | Point |
| iTransformer | Transformer | Point |
| Chronos | Foundation Model (Zero-Shot) | Quantile (stochastisch, 20 Samples) |

**Wichtig**: Im Subset-Training wurden **keine Quantile** (q10/q50/q90) trainiert — nur Point-Predictions für den Modellvergleich. Die probabilistische Ausgabe (Quantile) wurde erst beim **Volltraining** der besten 5 Modelle hinzugefügt.

- Inferenzzeit-Messung (UMBRA-Vorbereitung)
- UMBRA-Simulation für alle Subset-Modelle
- Direkter Modellvergleich (Tabelle + Plot)
- Chronos hat eigenen Abschnitt (kein direkter Vergleich, Zero-Shot)

### NB4: `04_model_subset.ipynb` – Subset-Auswertung & Modellselektion

**Zweck**: Analysiert das Trainings-Subset und rechtfertigt die Modellauswahl für das Volltraining.

**Subset-Auswahlmethode:**
- Aus 3016 Spots werden die **Top-406 Spots** nach höchster Datenvollständigkeit ausgewählt
- Vollständigkeit = `clean_rate = 1 − NaN/total` (Anteil gültiger Messwerte)
- Diese 406 Spots haben die wenigsten Messlücken und sind damit für das Training am wertvollsten
- Die Auswahl erfolgt **vor** der OOS-Trennung

**OOS-Spots (100 ungesehene Zeitreihen):**
- Die **Top-100 Spots außerhalb des Trainings-Subsets** nach clean rate (≥ 98.2%)
- D.h. OOS-Spots sind ebenfalls sehr vollständige Zeitreihen — worst-case für die Modelle
- Gespeichert in `Model_codes/oos_spots.json` und `data/processed/unseen_data.parquet`
- Diese 100 Spots werden beim Volltraining **komplett zurückgehalten** — kein Datenleck

**Volltraining-Datenbasis:**
- Alle 3016 Spots **minus die 100 OOS-Spots** = 2916 Trainings-Spots
- Das Subset (406 Spots) war nur für HPO und Modellselektion — das Volltraining nutzt alle 2916

**Weitere Analysen in NB4:**
- Repräsentativität des Subsets (Geo-Verteilung, kWp, Saisonalität vs. Gesamtdatensatz)
- Feature Importance (LightGBM Gain + RF Gini + DL Permutation)
- Finales Feature-Set: 45 Features (10 entfernt aus 55)

### NB5: `05_training_dokumentation.ipynb` – Volltraining-Dokumentation

**Nur Dokumentation, kein Code-Run.**

**Inhalt:**
1. OOS-Datensatz erstellen (100 Spots zurückhalten)
2. Entfernte Features (10 aus 55 → 45)
3. Gemeinsame Infrastruktur (Datenladen, Scaler, DataLoader)
4. Klassische ML (LightGBM + RandomForest Volltraining)
5. Deep Learning (GRU, CNN-LSTM, iTransformer Volltraining)

**Sequenzlängen DL:**
- GRU: 168 Zeitschritte (= 42 Stunden)
- CNN-LSTM: 480 Zeitschritte (= 120 Stunden)
- iTransformer: 168 Zeitschritte (= 42 Stunden)
- Chronos Context: 512 Zeitschritte (= 128 Stunden, ca. 5.3 Tage)
- Stochastische Samples: 20 (Chronos)
- Quantil-Prediction-Horizon: 144 Zeitschritte (= 36 Stunden)

### NB6: `06_fullmodel_evaluation.ipynb` – Vollmodell-Evaluation

**Evaluiert 5 volltrainierte Modelle auf 100 OOS-Spots:**

**Sektionen:**
1. Punktvorhersage-Metriken (MAE, wMAE, RMSE, wRMSE, R²)
2. Quantil-Metriken (Pinball-Loss q10/q50/q90, Coverage 80%, Intervallbreite)
3. Pinball-Loss Visualisierung
4. Kalibrierungsplot (Reliability Diagram)
5. Vorhersageintervall-Beispiel (12h-Fenster = 48 Timesteps)
6. Feature Importance (Gini/Gain für Baummodelle, Permutation für DL)
7. Horizont-Analyse (MAE vs. Lead-Time + Zeitreihenplot)
8. Inferenzzeit & Stromkosten (65W CPU, 37 ct/kWh, 1M Predictions)
9. UMBRA Batterieoptimierung (80 Spots)
10. MLflow Logging
11. Zusammenfassung-Tabelle

---

## 5. Modell-Details & Ergebnisse

### 5.1 Modell-Registry (Volltraining)

| Modell | Typ | Params | Trainingszeit | Sequenzlänge |
|---|---|---|---|---|
| LightGBM-Full | Gradient Boosting | – | laufend | – (tabular) |
| RandomForest-Full | Ensemble | – | 3.221 s (~54 min) | – (tabular) |
| GRU-Full | RNN | 175.044 | 6.130 s (~102 min) | 168 TS (42h) |
| CNN-LSTM-Full | CNN+LSTM | 72.864 | 8.709 s (~145 min) | 480 TS (120h) |
| iTransformer-Full | Transformer | 842.948 | 5.726 s (~95 min) | 168 TS (42h) |

Training: Kaggle (GPU), 2916 Trainings-Spots, 100 OOS-Spots

### 5.2 OOS-Ergebnisse (100 Hold-out-Spots)

| Modell | OOS MAE | OOS R² | Coverage 80% | Pinball q50 |
|---|---|---|---|---|
| **CNN-LSTM-Full** | **0.01569** | **0.7623** | 0.919 | 0.00787 |
| GRU-Full | 0.01632 | 0.7593 | 0.499 ⚠️ | 0.00817 |
| iTransformer-Full | 0.01719 | 0.7274 | 0.925 | 0.00861 |
| RandomForest-Full | 0.01830 | 0.7649 | 0.325 ⚠️ | 0.00818 |
| LightGBM-Full | – (Training läuft) | – | – | – |

**Wichtige Beobachtungen:**
- CNN-LSTM bestes MAE; GRU & RF schlecht kalibriert (Coverage ~50% statt 80%)
- iTransformer beste Kalibrierung (Coverage 92.5%)
- R² alle Modelle ~0.73–0.76

### 5.3 Hyperparameter (Best)

**GRU:**
```python
{'slen': 168, 'hid': 128, 'n_layers': 2, 'dropout': 0.1, 'lr': 0.001}
```

**CNN-LSTM:**
```python
{'slen': 480, 'cnn_ch': 52, 'hid': 99, 'n_layers': 1,
 'dropout': 0.191, 'lr': 0.00112}
```

**iTransformer:**
```python
{'slen': 168, 'd_model': 128, 'n_heads': 4, 'n_layers': 1,
 'd_ff': 64, 'dropout': 0.049, 'lr': 0.000406}
```

### 5.4 Feature Importance (GRU Permutation)

Top-Features nach Δ MAE:
1. `analog_knn_mean_norm` (0.00431) — stärkster Prädiktor
2. `scaled_analog_knn_mean` (0.00192)
3. `kwp_est` (0.00115)
4. `analog_knn_std` (0.00105)
5. `temperature_2m` (0.00050)
6. `dni` (0.00024)

→ Analog-Features dominieren klar vor Strahlungsdaten.

---

## 6. Modell-Architekturen (Code)

### GRU (`train_kaggle_gru.py`)
```python
class GRUModel(nn.Module):
    def __init__(self, n_features, hid, n_layers, dropout):
        self.gru  = nn.GRU(n_features, hid, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hid)
        self.head = nn.Sequential(Linear(hid, hid//2), GELU(),
                                  Dropout(dropout), Linear(hid//2, 4))
    def forward(self, x):
        _, h = self.gru(x); h = self.norm(h[-1])
        return self.head(h)  # [point, q10, q50, q90]
```

### CNN-LSTM (`train_kaggle_cnnlstm.py`)
```python
class CNNLSTMModel(nn.Module):
    conv = Sequential(Conv1d(n_features, cnn_ch, kernel_size=3, padding=1),
                      BatchNorm1d(cnn_ch), ReLU())
    lstm = LSTM(cnn_ch, hid, n_layers, batch_first=True)
    fc   = Sequential(Linear(hid, hid//2), ReLU(), Linear(hid//2, 1))
    # Output: point nur; Quantile via separaten Köpfen
```

### iTransformer (`train_kaggle_itransformer.py`)
```python
class iTransformerModel(nn.Module):
    # Invertierter Transformer: Attention über Features (nicht Zeit)
    self.input_proj = nn.Linear(seq_len, d_model)  # ← Layer-Name wichtig!
    self.layers = ModuleList([TransformerLayer(d_model, n_heads, d_ff, dropout)])
    self.head   = Sequential(Linear(n_features * d_model, d_model), GELU(),
                             Linear(d_model, 4))  # [point, q10, q50, q90]
    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))  # (B, F, d_model)
        for l in self.layers: x = l(x)
        return self.head(x.flatten(1))
```

**Wichtig**: `input_proj` (nicht `proj`) — state_dict-Schlüssel muss übereinstimmen!

### LightGBM (quantile regression)
```python
# 4 separate Modelle: point (mse), q10, q50, q90 (quantile loss)
lgb.train(params={'objective': 'quantile', 'alpha': 0.1, ...})
# Dateien: lgbm_point.txt, lgbm_q10.txt, lgbm_q50.txt, lgbm_q90.txt
```

### RandomForest (quantile via quantile regression forests)
```python
# sklearn RandomForestRegressor mit warm_start
# Quantile via predict_quantile (eigene Implementierung)
```

---

## 7. Trainings-Infrastruktur

### Zweistufige Trainings-Strategie

**Stufe 1 – Subset-Training (HPO & Modellselektion):**
- Datenbasis: Top-406 Spots nach Datenvollständigkeit (clean_rate = 1 − NaN/total)
- Ziel: Hyperparameter-Optimierung (Optuna, 50–100 Trials) + Modellvergleich
- Modelle: LightGBM, LinearRegression, GRU, CNN-LSTM, TSMixer, iTransformer, Chronos
- Training: Kaggle GPU (P100/T4), Auswertung in NB3/NB4
- Ergebnis: Beste 5 Modelle + optimale Hyperparameter identifiziert

**Stufe 2 – Volltraining (beste 5 Modelle):**
- Datenbasis: Alle 3016 Spots **minus 100 OOS-Spots** = **2916 Trainings-Spots**
- Die 100 OOS-Spots (Top-100 nach clean rate außerhalb des Subsets, ≥98.2%) werden komplett zurückgehalten
- Hyperparameter: übernommen aus Stufe 1 (kein weiteres HPO)
- Modelle: LightGBM, RandomForest, GRU, CNN-LSTM, iTransformer
- Training: Kaggle GPU, Skripte: `bscthesis/Model_codes/train_kaggle_*.py`
- Output: `.pt`-Dateien (DL), `.txt`/`.pkl` (ML), `.npy` (Predictions), `.json` (Metriken)

**Ausgeschlossene Modelle im Volltraining:**
- LinearRegression: zu schwach im Subset-Vergleich
- TSMixer: ähnliche Leistung wie GRU, höhere Komplexität
- Chronos: Zero-Shot Foundation Model, kein Volltraining nötig

**Quantile nur im Volltraining**: Die Subset-Modelle trainieren nur Point-Predictions. Erst die 5 Vollmodelle haben Multi-Output-Köpfe für q10/q50/q90 — bei DL als gemeinsamer Pinball-Loss-Kopf, bei LightGBM als 4 separate Modelle (mse + quantile).

### Output-Dateistruktur pro Modell

**GRU** (`gru_results/`):
- `gru_model.pt` — Gewichte
- `gru_results.json` — Metriken, HPO-Ergebnis, Permutation Importance
- `gru_oos_point_preds.npy`, `gru_oos_q10_preds.npy`, `gru_oos_q50_preds.npy`, `gru_oos_q90_preds.npy`
- `gru_oos_targets.npy`, `gru_oos_timestamps.npy`

**LightGBM** (`lightgbm_results/`) — erwartet nach Training:
- `lgbm_point.txt`, `lgbm_q10.txt`, `lgbm_q50.txt`, `lgbm_q90.txt`
- `lgbm_oos_preds.npy`, `lgbm_oos_targets.npy`
- `lgbm_oos_q10.npy`, `lgbm_oos_q50.npy`, `lgbm_oos_q90.npy`
- `lgbm_results.json`

**RandomForest** (`random_forest_results/`):
- `rf_model.pkl`, `rf_scaler.pkl`
- `rf_oos_preds.npy`, `rf_oos_targets.npy`
- `rf_oos_q10.npy`, `rf_oos_q50.npy`, `rf_oos_q90.npy`
- `rf_oos_timestamps.npy`, `rf_results.json`

---

## 8. Metriken & Verlustfunktionen

### Punkt-Metriken
- **MAE**: Mean Absolute Error
- **wMAE**: Gewichtetes MAE (w = y_true / mean(y_true)) — betont hohe Produktion
- **RMSE**, **wRMSE**: Analoge gewichtete Version
- **R²**: Erklärte Varianz

### Quantil-Metriken
- **Pinball-Loss** (α ∈ {0.1, 0.5, 0.9}): `L = α·(y-q)` wenn `y≥q`, sonst `(α-1)·(y-q)`
- **Coverage 80%**: Anteil echter Werte im [q10, q90]-Band → Ziel: 0.80
- **IntervalWidth**: Mittlere Breite q90-q10

### Verlustfunktion Training
- DL: Multi-Output-Pinball-Loss (gleichgewichtet Point+q10+q50+q90)
- LightGBM: 4 separate Modelle (mse + quantile)
- RF: MSE (Quantile post-hoc via empirische Verteilung)

---

## 9. UMBRA Battery Optimizer Integration

UMBRA optimiert Lade-/Entladestrategie einer Hausbatterie basierend auf PV-Prognose + Day-Ahead-Preisen + Verbrauchsprofil.

**Evaluationssetup:**
- 80 Spots (OOS + Training, seed=42)
- Metrik: kWh/Tag (selbst verbrauchte Energie aus Batterie)
- Vergleich: Modell-Prognose vs. Ground-Truth-Prognose (Δ GT)

**Technisch:**
- Subprocess-basiert (ein Python-Prozess pro Spot, Timeout 180s)
- Input: PV-Prognose als Zeitreihe (`spot_uuid`, `ts`, `kwh`)
- Batterie-Metadaten aus `2026-03-03_pv_with_analogs.parquet`

---

## 10. Bekannte Probleme & Lösungen

| Problem | Ursache | Lösung |
|---|---|---|
| NB2 OOM (Kernel stirbt) | Python-Loop über 3016 Spots mit `reset_index` | Vektorisierter Groupby + `gc.collect()` |
| NB6 iTransformer state_dict-Fehler | `proj` vs `input_proj` Layer-Name | `self.input_proj` in NB6-Architekturdefinition |
| LGB OOS Quantile falsche Dateinamen | `lgbm_q10_oos_preds.npy` ≠ `lgbm_oos_q10.npy` | Dateinamen in NB6 Cell 2 korrigiert |
| GRU Coverage 80% nur ~50% | Modell unterschätzt Unsicherheit | Bekanntes Problem, notiert |
| RF Coverage 80% nur ~33% | Quantile-Schätzung zu schmal | Bekanntes Problem, notiert |
| Cold-Start neue Spots | Keine Analog-History | Future Work: skalierter KNN-Fallback |

---

## 11. Nicht verwendete Features (56 Spalten → 45 Trainings-Features)

Der Datensatz hat **56 Spalten**, davon sind `ts`, `spot_uuid`, `kwh`, `kwh_norm` keine Modell-Features (Identifikatoren/Zielvariable). Von den verbleibenden ~52 Features wurden **45 zum Training genutzt** — die anderen wurden ausgeschlossen weil sie eine (nahezu) 1:1-Korrelation mit bereits enthaltenen Features hatten oder keinerlei zusätzliche Erklärungskraft lieferten:

- `analog_kwh_1`, `analog_kwh_1_norm` → 1:1 mit skalierten Versionen (`scaled_analog_kwh_1`, `scaled_analog_kwh_1_norm`)
- `analog_knn_mean` → 1:1 mit `analog_knn_mean_norm`
- `scaled_analog_kwh_1_norm` → redundant zu `scaled_analog_kwh_1`
- `analog_available` → im Training konstant True (kein Informationsgehalt)
- `ghi_magnitude` → nahezu 1:1-Korrelation mit `ghi`
- `temperature_penalty_rel` → sehr geringer Beitrag in Permutation Importance
- `panel_bimodal_east_fraction` → Information durch `panel_is_ew_split` abgedeckt
- weitere Features mit nahezu 1:1-Korrelation nach Analyse in NB4

---

## 12. MLflow Tracking

- **Experiment**: `BSc_PV_Forecasting_FullTrain`
- **Server**: `http://mlflow.cluster.spotmyenergy.systems`
- **Geloggte Metriken**: alle OOS-Metriken, UMBRA-Ergebnis, Inferenzzeit
- **Geloggte Parameter**: n_params, train_time, n_oos_spots, train_spots

---

## 13. Wichtige Code-Konventionen

- Alle temporären Variablen beginnen mit `_` (z.B. `_df_tmp`, `_mask80`)
- Pandas-Spalten nach Analyse löschen: `del _var; gc.collect()`
- Modell-Ausgabe immer: `[point, q10, q50, q90]` (Index 0 = Point)
- `observed=True` bei allen Groupby auf kategorischen Spalten (spot_uuid)
- Scaler: StandardScaler, auf 500k Zufallszeilen gefittet
- Outlier-Filter: kwh_norm ≤ 2.0 auf allen Splits

---

## 14. Nächste Schritte (Stand Apr 2026)

- [ ] LightGBM Volltraining abwarten → NB6 komplett ausführen
- [ ] GRU/RF Kalibrierung diskutieren (Coverage-Problem in Thesis)
- [ ] UMBRA-Evaluation aller 5 Modelle
- [ ] Thesis schreiben: Kapitel Methoden, Ergebnisse, Diskussion
- [ ] Cold-Start-Problem als Future Work dokumentieren

---

*Generiert: April 2026 | Repository: ect_forecasts | Branch: bscthesis*
