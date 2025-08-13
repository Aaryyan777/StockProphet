import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import joblib

tkr = "AAPL"
df_raw = yf.download(tkr, start="2020-01-01", end="2024-01-01")

df_proc = df_raw.reset_index()
df_proc = df_proc[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
df_proc.columns = ['ds', 'y', 'Open', 'High', 'Low', 'Volume']

df_proc['SMA_10'] = df_proc['y'].rolling(window=10).mean()
df_proc['SMA_30'] = df_proc['y'].rolling(window=30).mean()

def calc_rsi(d, window=14):
    dif = d.diff(1)
    gn = dif.where(dif > 0, 0)
    ls = -dif.where(dif < 0, 0)
    ag = gn.ewm(com=window - 1, min_periods=window).mean()
    al = ls.ewm(com=window - 1, min_periods=window).mean()
    rs_val = ag / al
    rsi_val = 100 - (100 / (1 + rs_val))
    return rsi_val

df_proc['RSI'] = calc_rsi(df_proc['y'])
df_proc.dropna(inplace=True)

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

best_params = {
    'changepoint_prior_scale': 0.001,
    'seasonality_prior_scale': 0.01
}

init_train_sz = int(len(df_proc) * 0.7)
hrz = 30
stp = 30

maes = []
rmses = []
mapes = []
r2s = []
dir_accs = []

print("Starting Rolling Window Cross-Validation...")

for i in range(init_train_sz, len(df_proc) - hrz + 1, stp):
    df_train = df_proc.iloc[:i].copy()
    df_test = df_proc.iloc[i:i + hrz].copy()

    if len(df_test) == 0:
        continue

    print(f"Fold: Training on {df_train['ds'].min()} to {df_train['ds'].max()}, Testing on {df_test['ds'].min()} to {df_test['ds'].max()}")

    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale']
    )
    model.add_regressor('SMA_10')
    model.add_regressor('SMA_30')
    model.add_regressor('RSI')
    model.fit(df_train)

    future = df_test[['ds', 'SMA_10', 'SMA_30', 'RSI']]
    forecast = model.predict(future)

    df_test_align = df_test.set_index('ds')
    forecast_align = forecast.set_index('ds')

    idx_common = df_test_align.index.intersection(forecast_align.index)
    if len(idx_common) == 0:
        print("No common dates for evaluation in this fold. Skipping.")
        continue

    y_true = df_test_align.loc[idx_common, 'y']
    y_pred = forecast_align.loc[idx_common, 'yhat']

    maes.append(mean_absolute_error(y_true, y_pred))
    rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    mapes.append(calc_mape(y_true, y_pred))
    r2s.append(r2_score(y_true, y_pred))

    dir_true = np.sign(y_true.diff().dropna())
    dir_pred = np.sign(y_pred.diff().dropna())

    idx_dir_common = dir_true.index.intersection(dir_pred.index)
    if len(idx_dir_common) > 0:
        dir_true_align = dir_true.loc[idx_dir_common]
        dir_pred_align = dir_pred.loc[idx_dir_common]
        dir_acc = np.mean(dir_true_align == dir_pred_align) * 100
        dir_accs.append(dir_acc)
    else:
        print("Not enough data for directional accuracy in this fold.")

print("\n--- Cross-Validation Results (Prophet Model) ---")
print(f"Average MAE: {np.mean(maes):.2f} (Std: {np.std(maes):.2f})")
print(f"Average RMSE: {np.mean(rmses):.2f} (Std: {np.std(rmses):.2f})")
print(f"Average MAPE: {np.mean(mapes):.2f}% (Std: {np.std(mapes):.2f}%)")
print(f"Average R-squared (R²): {np.mean(r2s):.2f} (Std: {np.std(r2s):.2f})")
print(f"Average Directional Accuracy: {np.mean(dir_accs):.2f}% (Std: {np.std(dir_accs):.2f}%)")

# Retrain the best model on the full dataset for deployment
print("\nRetraining best model on full dataset for deployment...")
model_final = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale']
)
model_final.add_regressor('SMA_10')
model_final.add_regressor('SMA_30')
model_final.add_regressor('RSI')
model_final.fit(df_proc)

model_file = 'prophet_model.joblib'
joblib.dump(model_final, model_file)
print(f"Model saved as {model_file}")

# Plotting the final forecast (optional, can be removed for production script)
# future_full = model_final.make_future_dataframe(periods=30)
# forecast_full = model_final.predict(future_full)

# plt.figure(figsize=(12, 6))
# model_final.plot(forecast_full, ax=plt.gca())
# plt.title(f'{tkr} Stock Price Forecast (Full Data)')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.grid(True)
# plt.show()

# model_final.plot_components(forecast_full)
# plt.show()
