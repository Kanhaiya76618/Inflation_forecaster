# 📈 Inflation Forecasting Using Food and Energy Price Indicators

> Predicting India's Consumer Price Index (CPI) using macroeconomic time series indicators via classical and deep learning methods.

---

## 📌 Project Overview

This project builds a complete **end-to-end Time Series Forecasting pipeline** to predict **India's Consumer Price Index (CPI)** — a key measure of inflation — using global macroeconomic signals such as crude oil prices, food price indices, energy indices, and agricultural commodity prices.

The analysis is structured as a clean Jupyter Notebook with 11 well-documented sections, covering everything from data sourcing to model comparison and 12-month forward forecasting.

---

## 🎯 Objective

- Forecast **India's monthly CPI** using leading macroeconomic indicators
- Compare **5 different forecasting models** (statistical, ML, and deep learning)
- Identify **which indicators have the most predictive power** for CPI using Granger causality tests
- Provide a **12-month forward forecast** with 95% confidence intervals

---

## 📦 Dataset

All data is fetched **automatically** from the [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/) API — no manual downloads required.

| Variable | FRED Series ID | Description |
|---|---|---|
| **CPI (India)** | `INDCPIALLMINMEI` | Consumer Price Index, All Items, India (OECD) |
| **Crude Oil** | `DCOILBRENTEU` | Brent Crude Oil Price (USD/barrel) |
| **Food Price Index** | `PFOODINDEXM` | Global Food Price Index (World Bank) |
| **Energy Index** | `PNRGINDEXM` | Global Energy Price Index |
| **Agricultural Index** | `PRAWMINDEXM` | Agricultural Raw Materials Index |
| **Metals (Copper)** | `PCOPPUSDM` | Copper Price (USD/metric ton) |

> **Multi-Country Support**: Change `ACTIVE_COUNTRY = "India"` at the top of the notebook to switch to **United States, Brazil, Germany, or China** — the entire pipeline adapts automatically.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Time series plots for all variables
- Correlation heatmap
- ADF stationarity tests
- Seasonal decomposition (trend, seasonality, residuals)
- ACF / PACF plots

### 2. Data Preprocessing
- Forward-fill for missing values
- First-order differencing for stationarity
- Lag feature engineering (1, 3, 6-month lags)
- Chronological 80/20 train-test split

### 3. Models Built & Compared

| Model | Type | Description |
|---|---|---|
| **SARIMA** | Statistical | Seasonal ARIMA, univariate baseline |
| **VAR** | Statistical | Vector Autoregression, multivariate |
| **SARIMAX** | Statistical | ARIMA with exogenous variables |
| **XGBoost** | Machine Learning | Gradient boosting with lag features |
| **LightGBM** | Machine Learning | Fast gradient boosting |
| **LSTM** | Deep Learning | Sliding-window (look_back=12) recurrent network |

### 4. Evaluation Metrics
- RMSE, MAE, MAPE, R²
- Predicted vs Actual plots for each model
- Side-by-side model comparison table

### 5. Forecasting
- 12-month forward CPI forecast using the best-performing model
- 95% Confidence intervals

### 6. Insights
- Granger Causality Tests (does oil/food price Granger-cause CPI?)
- XGBoost feature importance (which lag features matter most?)

---

## 🗂️ Project Structure

```
📁 time-series-inflation-forecasting/
│
├── 📓 inflation_forecasting.ipynb   # Main Jupyter Notebook (11 sections)
├── 🐍 generate_notebook.py          # Script to regenerate the notebook
├── 📄 requirements.txt              # Python dependencies
└── 📝 README.md                     # This file
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/inflation-forecasting.git
cd inflation-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook inflation_forecasting.ipynb
```

### 4. Run all cells
In Jupyter: **Kernel → Restart & Run All**

> ⚠️ The notebook auto-downloads all data from FRED on first run. An internet connection is required.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `statsmodels` | SARIMA, VAR, SARIMAX, ADF tests |
| `pmdarima` | Auto-ARIMA model selection |
| `scikit-learn` | Metrics, preprocessing |
| `xgboost`, `lightgbm` | Gradient boosting models |
| `tensorflow / keras` | LSTM deep learning model |

---

## 📊 Sample Results

After running the notebook on India CPI (2000–2024):

| Model | RMSE ↓ | MAPE% ↓ | R² ↑ |
|---|---|---|---|
| SARIMA | ~2.1 | ~1.5% | ~0.93 |
| SARIMAX | ~1.8 | ~1.3% | ~0.95 |
| XGBoost | ~1.2 | ~0.9% | ~0.98 |
| LightGBM | ~1.3 | ~1.0% | ~0.97 |
| LSTM | ~2.5 | ~1.8% | ~0.88 |
| VAR | ~3.1 | ~2.2% | ~0.80 |

> *Exact values depend on data vintage at time of download.*

---

## 🌍 Extending to Other Countries

At the top of the notebook, simply change:

```python
ACTIVE_COUNTRY = "India"   # ← Change to: United States, Brazil, Germany, China
```

The entire pipeline — data download, modeling, and forecasting — adapts automatically.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Akash**  
Time Series Forecasting & Applied Machine Learning Project  
March 2026
