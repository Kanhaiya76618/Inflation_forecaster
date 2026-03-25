import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ─── Header ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Inflation Forecasting Using Food and Energy Price Indicators
**Time Series Forecasting Project — Primary Focus: India**

This notebook predicts the **Consumer Price Index (CPI)** for India using macroeconomic indicators
such as food prices, crude oil prices, fuel/energy prices, and commodity indices.
The `COUNTRY_CONFIG` block at the top allows switching to other countries in one line.
"""))

# ─── Section 0: Setup ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 0: Setup & Configuration"))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
import pmdarima as pm

from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (14, 5)

# ─── COUNTRY CONFIG ───────────────────────────────────────────────────────────
# Change ACTIVE_COUNTRY to switch the entire pipeline to a different country.
COUNTRY_CONFIG = {
    'India':         {'fred_cpi': 'INDCPIALLMINMEI', 'oil_series': 'DCOILBRENTEU', 'label': 'India'},
    'United States': {'fred_cpi': 'CPIAUCSL',        'oil_series': 'DCOILWTICO',   'label': 'United States'},
    'Brazil':        {'fred_cpi': 'BRACPIALLMINMEI', 'oil_series': 'DCOILBRENTEU', 'label': 'Brazil'},
    'Germany':       {'fred_cpi': 'DEUCPIALLMINMEI', 'oil_series': 'DCOILBRENTEU', 'label': 'Germany'},
    'China':         {'fred_cpi': 'CHNCPIALLMINMEI', 'oil_series': 'DCOILBRENTEU', 'label': 'China'},
}
ACTIVE_COUNTRY = 'India'   # <- change this only
CFG = COUNTRY_CONFIG[ACTIVE_COUNTRY]
print(f"Configured for: {ACTIVE_COUNTRY}")
"""))

# ─── Section 1: Data Sourcing ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Section 1: Dataset Sourcing & Loading
All data is fetched automatically from **FRED** (St. Louis Federal Reserve) via direct CSV links.
No API key is required.

| Indicator | FRED Series ID |
|---|---|
| India CPI (OECD monthly) | `INDCPIALLMINMEI` |
| Brent Crude Oil | `DCOILBRENTEU` |
| Food Price Index | `PFOODINDEXM` |
| Energy Price Index | `PNRGINDEXM` |
| Agricultural Index | `PRAWMINDEXM` |
| Iron Ore (Metals proxy) | `PIORECR` |
"""))

cells.append(nbf.v4.new_code_cell("""def fetch_fred(series_id, col_name, start='2000-01-01', end='2024-12-01'):
    \"\"\"Download a FRED series robustly — reads columns by position, not name.\"\"\"
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    try:
        raw = pd.read_csv(url, na_values='.')
        # First column is always DATE regardless of header spelling
        raw.columns = ['date', col_name]
        raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
        raw = raw.dropna(subset=['date'])
        raw.set_index('date', inplace=True)
        raw = raw.loc[start:end]
        raw = raw.resample('MS').last()   # standardise to month-start frequency
        print(f"  OK  {series_id:25s} -> {col_name}  ({len(raw)} rows)")
        return raw
    except Exception as e:
        print(f"  FAIL {series_id}: {e}")
        return pd.DataFrame(columns=[col_name])

START = '2000-01-01'
END   = '2024-12-01'

print("Fetching data from FRED...")
cpi_df    = fetch_fred(CFG['fred_cpi'],  'CPI',          START, END)
oil_df    = fetch_fred(CFG['oil_series'],'Oil_Price',    START, END)
food_df   = fetch_fred('PFOODINDEXM',   'Food_Index',   START, END)
energy_df = fetch_fred('PNRGINDEXM',   'Energy_Index', START, END)
agri_df   = fetch_fred('PRAWMINDEXM',  'Agri_Index',   START, END)
metals_df = fetch_fred('PCOPPUSDM',    'Metals_Index', START, END)

# Merge: LEFT join off CPI so a failed series never kills the dataset
merged_df = cpi_df.copy()
for df_part in [oil_df, food_df, energy_df, agri_df, metals_df]:
    if not df_part.empty:
        merged_df = merged_df.join(df_part, how='left')

merged_df = merged_df.ffill().bfill()
merged_df = merged_df.dropna(subset=['CPI'])   # only drop rows where target is missing
print(f"\\nFinal dataset: {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")
print(f"Date range: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
display(merged_df.tail())
"""))

# ─── Section 2: EDA ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 2: Exploratory Data Analysis (EDA)"))

cells.append(nbf.v4.new_code_cell("""# 2-A  Time-Series Plots
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
plt.suptitle(f'{ACTIVE_COUNTRY} -- Macroeconomic Indicators (2000-2024)', fontsize=14, y=1.01)
pairs = [
    ('CPI',          'royalblue', f'{ACTIVE_COUNTRY} CPI (Target)'),
    ('Oil_Price',    'black',     'Crude Oil Price'),
    ('Food_Index',   'green',     'Food Price Index'),
    ('Energy_Index', 'orange',    'Energy Price Index'),
    ('Agri_Index',   'sienna',    'Agricultural Index'),
    ('Metals_Index', 'purple',    'Metals Index (Iron Ore)'),
]
for ax, (col, color, title) in zip(axes.flat, pairs):
    ax.plot(merged_df.index, merged_df[col], color=color, linewidth=1.2)
    ax.set_title(title); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2-B  Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = merged_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            mask=mask, square=True, linewidths=0.5)
plt.title('Correlation Matrix'); plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2-C  Stationarity Tests (ADF)
def adf_report(series, name):
    res = adfuller(series.dropna(), autolag='AIC')
    status = 'STATIONARY' if res[1] <= 0.05 else 'NON-STATIONARY'
    print(f'  {name:20s}  p={res[1]:.4f}  --> {status}')

print('ADF Stationarity Tests\\n' + '-'*55)
for col in merged_df.columns:
    adf_report(merged_df[col], col)
"""))

cells.append(nbf.v4.new_code_cell("""# 2-D  Seasonal Decomposition
result = seasonal_decompose(merged_df['CPI'], model='additive', period=12)
fig = result.plot()
fig.set_size_inches(14, 9)
plt.suptitle(f'{ACTIVE_COUNTRY} CPI -- Seasonal Decomposition', y=1.01)
plt.tight_layout(); plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2-E  ACF / PACF
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plot_acf( merged_df['CPI'],                 ax=axes[0,0], lags=40, title='ACF - CPI (levels)')
plot_pacf(merged_df['CPI'],                 ax=axes[0,1], lags=40, title='PACF - CPI (levels)')
plot_acf( merged_df['CPI'].diff().dropna(), ax=axes[1,0], lags=40, title='ACF - CPI (1st diff)')
plot_pacf(merged_df['CPI'].diff().dropna(), ax=axes[1,1], lags=40, title='PACF - CPI (1st diff)')
plt.tight_layout(); plt.show()
"""))

# ─── Section 3: Preprocessing ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 3: Data Preprocessing"))

cells.append(nbf.v4.new_code_cell("""exog_cols = ['Oil_Price', 'Food_Index', 'Energy_Index', 'Agri_Index', 'Metals_Index']

# 3-A  Differencing for stationarity
df_diff = merged_df.diff().dropna()
print('Stationarity after 1st differencing:')
for col in df_diff.columns:
    adf_report(df_diff[col], col)

# 3-B  Lag features for ML / DL models
df_ml = merged_df.copy()
for col in exog_cols + ['CPI']:
    for lag in [1, 3, 6]:
        df_ml[f'{col}_lag{lag}'] = df_ml[col].shift(lag)
df_ml.dropna(inplace=True)

# 3-C  Chronological 80/20 split
split = int(len(merged_df) * 0.8)
train, test         = merged_df.iloc[:split], merged_df.iloc[split:]
train_diff, test_diff = df_diff.iloc[:split-1], df_diff.iloc[split-1:]

ml_split = int(len(df_ml) * 0.8)
ml_train, ml_test   = df_ml.iloc[:ml_split], df_ml.iloc[ml_split:]
y_train_ml = ml_train['CPI']
y_test_ml  = ml_test['CPI']

feat_cols  = [c for c in df_ml.columns if c not in ['CPI'] + exog_cols]
X_train_ml = ml_train[feat_cols]
X_test_ml  = ml_test[feat_cols]

print(f'\\nTrain: {train.index[0].date()} to {train.index[-1].date()}  ({len(train)} rows)')
print(f'Test : {test.index[0].date()} to {test.index[-1].date()}  ({len(test)} rows)')
print(f'ML feature count: {len(feat_cols)}')
"""))

# ─── Section 4: SARIMA ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 4: SARIMA -- Univariate Baseline"))

cells.append(nbf.v4.new_code_cell("""def get_metrics(actual, pred, label=''):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred) * 100
    r2   = r2_score(actual, pred)
    print(f'{label:15s}  RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
    return rmse, mae, mape, r2

print('Fitting Auto-ARIMA (this may take 1-2 minutes)...')
auto_arima = pm.auto_arima(
    train['CPI'], seasonal=True, m=12,
    d=1, D=1, max_p=3, max_q=3, max_P=2, max_Q=2,
    information_criterion='aic', error_action='ignore',
    suppress_warnings=True, stepwise=True, trace=False
)
print(auto_arima.summary())

n_test = len(test)
sarima_pred_arr, sarima_ci = auto_arima.predict(n_periods=n_test, return_conf_int=True)
sarima_pred    = pd.Series(sarima_pred_arr, index=test.index)
sarima_metrics = get_metrics(test['CPI'], sarima_pred, 'SARIMA')

plt.plot(train.index[-48:], train['CPI'].iloc[-48:], label='Train', color='gray')
plt.plot(test.index, test['CPI'],  label='Actual',  color='steelblue')
plt.plot(test.index, sarima_pred,  label='SARIMA',  color='crimson')
plt.fill_between(test.index, sarima_ci[:,0], sarima_ci[:,1], alpha=0.15, color='crimson')
plt.title('SARIMA Forecast vs Actual'); plt.legend(); plt.tight_layout(); plt.show()
"""))

# ─── Section 5: VAR ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 5: VAR -- Vector Autoregression"))

cells.append(nbf.v4.new_code_cell("""var_model  = VAR(train_diff)
order_res  = var_model.select_order(maxlags=12)
best_lag   = max(order_res.aic, 1)
print(f'VAR optimal lag (AIC): {best_lag}')

var_fit    = var_model.fit(best_lag)
var_fc_raw = var_fit.forecast(train_diff.values[-best_lag:], steps=len(test_diff))
var_fc_df  = pd.DataFrame(var_fc_raw, index=test_diff.index, columns=train_diff.columns)

# Invert differencing
var_pred   = var_fc_df['CPI'].cumsum() + train['CPI'].iloc[-1]
min_len    = min(len(var_pred), len(test))
var_pred   = var_pred.iloc[:min_len]
test_var   = test['CPI'].iloc[:min_len]
var_metrics = get_metrics(test_var, var_pred, 'VAR')

plt.plot(test.index[:min_len], test_var, label='Actual', color='steelblue')
plt.plot(test.index[:min_len], var_pred, label='VAR',    color='purple')
plt.title('VAR Forecast vs Actual'); plt.legend(); plt.tight_layout(); plt.show()
"""))

# ─── Section 6: SARIMAX ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 6: SARIMAX -- With Exogenous Variables"))

cells.append(nbf.v4.new_code_cell("""exog_train = train[exog_cols]
exog_test  = test[exog_cols]
order          = auto_arima.order
seasonal_order = auto_arima.seasonal_order

print(f'Fitting SARIMAX{order}x{seasonal_order}...')
sarimax_res    = SARIMAX(train['CPI'], exog=exog_train,
                         order=order, seasonal_order=seasonal_order).fit(disp=False)

fc = sarimax_res.get_forecast(steps=len(test), exog=exog_test)
sarimax_pred   = fc.predicted_mean
sarimax_ci     = fc.conf_int()
sarimax_pred.index = test.index
sarimax_metrics = get_metrics(test['CPI'], sarimax_pred, 'SARIMAX')

plt.plot(test.index, test['CPI'],   label='Actual',  color='steelblue')
plt.plot(test.index, sarimax_pred,  label='SARIMAX', color='darkorange')
plt.fill_between(test.index, sarimax_ci.iloc[:,0], sarimax_ci.iloc[:,1], alpha=0.15, color='darkorange')
plt.title('SARIMAX Forecast vs Actual'); plt.legend(); plt.tight_layout(); plt.show()
"""))

# ─── Section 7: XGBoost & LightGBM ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 7: XGBoost & LightGBM with Lag Features"))

cells.append(nbf.v4.new_code_cell("""# XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, verbosity=0)
xgb.fit(X_train_ml, y_train_ml, eval_set=[(X_test_ml, y_test_ml)], verbose=False)
xgb_pred    = pd.Series(xgb.predict(X_test_ml), index=X_test_ml.index)
xgb_metrics = get_metrics(y_test_ml, xgb_pred, 'XGBoost')

# LightGBM
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbose=-1)
lgbm.fit(X_train_ml, y_train_ml)
lgbm_pred    = pd.Series(lgbm.predict(X_test_ml), index=X_test_ml.index)
lgbm_metrics = get_metrics(y_test_ml, lgbm_pred, 'LightGBM')

# Feature importance
imp = pd.Series(xgb.feature_importances_, index=X_train_ml.columns).nlargest(12).sort_values()
imp.plot(kind='barh', color='teal', figsize=(10, 5))
plt.title('XGBoost -- Top 12 Feature Importances'); plt.tight_layout(); plt.show()

# Predictions
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_test_ml.index, y_test_ml,   label='Actual',   color='steelblue')
ax.plot(xgb_pred.index,  xgb_pred,    label='XGBoost',  color='green')
ax.plot(lgbm_pred.index, lgbm_pred,   label='LightGBM', color='orange')
ax.set_title('XGBoost & LightGBM vs Actual'); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ─── Section 8: LSTM ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Section 8: LSTM -- Deep Learning
Uses a **sliding window approach** (`look_back=12` months) — the correct way to feed sequences into an LSTM.
"""))

cells.append(nbf.v4.new_code_cell("""# ── LSTM with proper sliding-window sequences (look_back = 12 months) ─────────
LOOK_BACK = 12

# Scale ALL features + CPI together in one matrix for consistency
all_features = exog_cols + ['CPI']
lstm_data    = merged_df[all_features].values   # shape (N, 6)

scaler_lstm = MinMaxScaler()
lstm_scaled = scaler_lstm.fit_transform(lstm_data)   # fit on FULL series

# Create sequences
def make_sequences(data, look_back):
    X_seq, y_seq = [], []
    for i in range(look_back, len(data)):
        X_seq.append(data[i-look_back:i, :])   # (look_back, n_features)
        y_seq.append(data[i, -1])               # CPI is last column
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = make_sequences(lstm_scaled, LOOK_BACK)

# Chronological split (no shuffle)
seq_split  = int(len(X_seq) * 0.8)
X_tr_l     = X_seq[:seq_split]
X_te_l     = X_seq[seq_split:]
y_tr_l     = y_seq[:seq_split]
y_te_l     = y_seq[seq_split:]

# LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, len(all_features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.1),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
history = lstm_model.fit(X_tr_l, y_tr_l, epochs=200, batch_size=16,
                         validation_split=0.1, callbacks=[es], verbose=0)
print(f'LSTM trained for {len(history.history["loss"])} epochs')

# Inverse-transform predictions back to CPI scale
lstm_pred_sc  = lstm_model.predict(X_te_l, verbose=0).flatten()

# Build a dummy full-feature matrix to inverse transform just the CPI column
dummy = np.zeros((len(lstm_pred_sc), len(all_features)))
dummy[:, -1] = lstm_pred_sc                             # CPI is last col
lstm_pred_vals = scaler_lstm.inverse_transform(dummy)[:, -1]

# Align with original dates (skip first look_back rows)
lstm_dates = merged_df.index[LOOK_BACK:]
lstm_test_dates = lstm_dates[seq_split:]
lstm_pred  = pd.Series(lstm_pred_vals, index=lstm_test_dates)

# Actual values for the same test window
y_test_lstm_actual = merged_df['CPI'].iloc[LOOK_BACK + seq_split:]
y_test_lstm_actual = y_test_lstm_actual.iloc[:len(lstm_pred)]

lstm_metrics = get_metrics(y_test_lstm_actual, lstm_pred, 'LSTM')

plt.plot(y_test_lstm_actual.index, y_test_lstm_actual, label='Actual', color='steelblue')
plt.plot(lstm_pred.index, lstm_pred, label='LSTM', color='tomato')
plt.title('LSTM (Sliding Window, look_back=12) vs Actual')
plt.legend(); plt.tight_layout(); plt.show()
"""))

# ─── Section 9: Comparison ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 9: Model Evaluation & Comparison"))

cells.append(nbf.v4.new_code_cell("""results = pd.DataFrame({
    'Model' : ['SARIMA', 'VAR', 'SARIMAX', 'XGBoost', 'LightGBM', 'LSTM'],
    'RMSE'  : [sarima_metrics[0], var_metrics[0], sarimax_metrics[0], xgb_metrics[0], lgbm_metrics[0], lstm_metrics[0]],
    'MAE'   : [sarima_metrics[1], var_metrics[1], sarimax_metrics[1], xgb_metrics[1], lgbm_metrics[1], lstm_metrics[1]],
    'MAPE%' : [sarima_metrics[2], var_metrics[2], sarimax_metrics[2], xgb_metrics[2], lgbm_metrics[2], lstm_metrics[2]],
    'R2'    : [sarima_metrics[3], var_metrics[3], sarimax_metrics[3], xgb_metrics[3], lgbm_metrics[3], lstm_metrics[3]],
}).sort_values('RMSE').reset_index(drop=True)

print("=== MODEL COMPARISON TABLE ===")
display(results)

best_model = results.iloc[0]['Model']
print(f'\\nBest model by RMSE: {best_model}')

# Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=results, x='Model', y='RMSE', palette='viridis', ax=ax)
ax.set_title('RMSE Comparison Across Models'); ax.set_xlabel(''); plt.tight_layout(); plt.show()

# All predictions vs actual in subplots
preds  = dict(SARIMA=sarima_pred, VAR=var_pred, SARIMAX=sarimax_pred,
              XGBoost=xgb_pred, LightGBM=lgbm_pred, LSTM=lstm_pred)
colors = ['crimson','purple','darkorange','green','gold','tomato']
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
plt.suptitle('All Models -- Actual vs Predicted CPI', fontsize=13)
for ax, (name, pred), col in zip(axes.flat, preds.items(), colors):
    actual = test['CPI'].reindex(pred.index)
    ax.plot(actual.index, actual, label='Actual', color='steelblue', alpha=0.7)
    ax.plot(pred.index,   pred,   label=name,     color=col)
    ax.set_title(name); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""))

# ─── Section 10: Forecasting ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 10: 12-Month Forward Forecast"))

cells.append(nbf.v4.new_code_cell("""n_ahead = 12
future_arr, future_ci = auto_arima.predict(n_periods=n_test + n_ahead, return_conf_int=True)
future_dates = pd.date_range(
    start=test.index[-1] + pd.DateOffset(months=1), periods=n_ahead, freq='MS'
)
future_pred = pd.Series(future_arr[-n_ahead:], index=future_dates)
future_lo   = future_ci[-n_ahead:, 0]
future_hi   = future_ci[-n_ahead:, 1]

plt.figure(figsize=(14, 6))
plt.plot(train.index[-48:], train['CPI'].iloc[-48:], label='History',       color='gray')
plt.plot(test.index,        test['CPI'],             label='Test Actual',    color='steelblue')
plt.plot(test.index,        sarima_pred,             label='SARIMA (test)',  color='crimson', alpha=0.5)
plt.plot(future_dates,      future_pred,             label='12-Mo Forecast', color='crimson', linewidth=2.5)
plt.fill_between(future_dates, future_lo, future_hi, color='crimson', alpha=0.15)
plt.axvline(x=future_dates[0], color='k', linestyle='--', linewidth=0.8)
plt.title(f'{ACTIVE_COUNTRY} CPI -- 12-Month SARIMA Forecast (95% CI)')
plt.legend(); plt.tight_layout(); plt.show()

fc_df = pd.DataFrame({'Month': future_dates.strftime('%b %Y'),
                      'Forecast CPI': future_pred.values.round(2),
                      'Lower 95%': future_lo.round(2),
                      'Upper 95%': future_hi.round(2)})
display(fc_df.set_index('Month'))
"""))

# ─── Section 11: Insights ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Section 11: Granger Causality & Insights"))

cells.append(nbf.v4.new_code_cell("""print('Granger Causality Tests -- Does each indicator Granger-cause CPI?')
print('='*60)
gc_results = []
for col in exog_cols:
    data_gc = df_diff[['CPI', col]].dropna()
    try:
        res = grangercausalitytests(data_gc, maxlag=3, verbose=False)
        p   = res[3][0]['ssr_ftest'][1]
        sig = 'YES (p<0.05)' if p < 0.05 else 'NO'
        gc_results.append({'Indicator': col, 'p-value (lag=3)': round(p, 4), 'Causes CPI?': sig})
    except Exception as e:
        gc_results.append({'Indicator': col, 'p-value (lag=3)': 'N/A', 'Causes CPI?': str(e)})

display(pd.DataFrame(gc_results).set_index('Indicator'))

print()
print('XGBoost Feature Importance (Top 15):')
imp = pd.Series(xgb.feature_importances_, index=X_train_ml.columns).nlargest(15).sort_values()
imp.plot(kind='barh', color='mediumseagreen', figsize=(10, 6))
plt.title('XGBoost Feature Importance -- Top 15')
plt.tight_layout(); plt.show()
"""))

# Save
nb['cells'] = cells
out = r'c:\Users\AKASH\OneDrive\time series\inflation_forecasting.ipynb'
nbf.write(nb, out)
print(f'Done! Notebook saved to: {out}')
