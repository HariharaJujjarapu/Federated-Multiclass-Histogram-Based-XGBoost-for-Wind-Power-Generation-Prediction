# ⚡ Federated Multiclass Histogram-Based XGBoost for Wind Power Generation Prediction

A privacy-preserving, distributed gradient boosting framework implemented from scratch in Python to predict multi-tier wind power generation using meteorological and power telemetry data (`50Hertz`). 

This project simulates a **federated learning setting across 5 edge client nodes**, training a 5-class XGBoost model using distributed gradient and Hessian histogram aggregation while keeping raw client data local and private. It includes a baseline comparison against a standard **centralized XGBoost** implementation.

---

## 📌 Project Highlights

- **Decentralized Split Finding:** Edge nodes compute local first- and second-order Taylor expansion statistics (Gradients $G$ and Hessians $H$) over shared quantile bins. The central server aggregates these histograms to select optimal tree splits with zero raw data transmission.
- **Privacy-Preserving Quantile Binning:** Approximates global feature bins from client-reported percentile summaries using weighted quantiles, maintaining feature space uniformity without sharing tabular records.
- **Accuracy Parity:** Achieves **93.08% test accuracy** and a **0.9309 Macro-F1 score**, matching the centralized XGBoost baseline ceiling (93.15% evaluation accuracy, `eval-merror: 0.0685`) within a <0.07% delta.

---

## 📊 Dataset & Preprocessing

- **Dataset:** Berlin Wind Power Generation & Weather Telemetry (`Berlin_wind_multiclass 1(in).csv`, 70,080 rows).
- **Target Variable Discretization:** The continuous wind power target (`X50Hertz..MW.`) is discretized into 5 balanced quantile tiers (`0, 1, 2, 3, 4`) using `pd.qcut` (14,016 samples per class).
- **Feature Selection:** Filtered out low-correlation features ($\vert{}r\vert{} \le 0.05$ with target), dropping `['Year', 'Day', 'Hour', 'Minute', 'Fill.Flag', 'Surface.Albedo']` and retaining **17 meteorological/telemetry predictors** (`Month`, `Temperature`, `DHI`, `DNI`, `GHI`, `Wind.Speed`, `Wind.Direction`, `Pressure`, `Relative.Humidity`, etc.).
- **Imputation & Splits:** Stratified split into Train (70%), Validation (15%), and Test (15%) with median imputation fit strictly on the training partition.

---

## ⚙️ Architecture & Hyperparameters

### Federated System Design
- **Number of Simulated Clients:** 5 (Stratified, balanced class distribution across clients)
- **Feature Binning:** Global weighted quantiles with up to 64 bins (`MAX_BINS = 64`), 101 percentile checkpoints (`N_PERCENTILES = 101`)

### Gradient Boosting Engine
- **Objective Function:** Multiclass Softmax with diagonal Hessian approximation
- **Base Estimators / Rounds:** 750 rounds (`N_ESTIMATORS = 750`)
- **Max Tree Depth:** 5 (`MAX_DEPTH = 5`)
- **Learning Rate ($\eta$):** 0.3
- **Regularization:** $\lambda = 1.0$ (L2 leaf penalty), $\gamma = 0.0$ (Min split loss)
- **Minimum Child Weight ($H_{\text{min}}$):** 1.0

$$\text{Weight } w^* = -\frac{G}{H + \lambda}, \quad \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{(H_L + H_R) + \lambda} \right] - \gamma$$

---

## 📈 Experimental Results

### Centralized vs. Federated Performance Comparison

| Model Architecture | Rounds / Estimators | Validation Accuracy | Test Accuracy | Test Macro-F1 | Data Privacy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized XGBoost** | 2,000 | 93.15% (`eval-merror: 0.0685`) | ~93.14% | ~0.9312 | ❌ Centralized Storage |
| **Federated Histogram XGBoost** | 750 | **93.02%** | **93.08%** | **0.9309** | ✅ Zero Data Leakage |

### Confusion Matrix (Test Set — 10,512 Samples)

```text
               Predicted Class
             0      1      2      3      4
Class 0 |  1981    117      4      1      0
Class 1 |   105   1878    115      4      0
Class 2 |     4    100   1911     87      0
Class 3 |     3      1     90   1967     42
Class 4 |     0      0      0     54   2048
