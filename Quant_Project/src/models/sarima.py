import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


file_path = 'E:/Quant_Project/data/features_data.csv' 
data = pd.read_csv(file_path)


data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = data.sort_values(by=['Date'])


company = 'META'  
company_data = data[data['Company'] == company].copy()


company_data['Close/Last'] = pd.to_numeric(company_data['Close/Last'], errors='coerce')
company_data[['Volume', 'EMA_20', 'MACD']] = company_data[['Volume', 'EMA_20', 'MACD']].apply(pd.to_numeric, errors='coerce')


company_data.set_index('Date', inplace=True)


company_data = company_data.dropna(subset=['Close/Last', 'Volume', 'EMA_20', 'MACD'])


train_size = int(len(company_data) * 0.8)  
train, test = company_data.iloc[:train_size], company_data.iloc[train_size:]
train_exog = train[['Volume', 'EMA_20', 'MACD']]
test_exog = test[['Volume', 'EMA_20', 'MACD']]

print(f"Training set size: {len(train)}, Testing set size: {len(test)}")

# ADF test for stationarity on the target variable
def adf_test(series):
    result = adfuller(series)
    print("ADF Test Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing may be needed.")

print(f"\nADF Test for {company} Training Data:")
adf_test(train['Close/Last'])


print("\nRunning auto_arima to find optimal parameters...")
auto_arima_model = auto_arima(
    train['Close/Last'],
    exogenous=train_exog,
    seasonal=True,
    m=252,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\nOptimal SARIMA Model Parameters:")
print("Order:", auto_arima_model.order)
print("Seasonal Order:", auto_arima_model.seasonal_order)

sarimax_model = SARIMAX(
    train['Close/Last'],
    exog=train_exog,
    order=auto_arima_model.order,
    seasonal_order=auto_arima_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_result = sarimax_model.fit(disp=False)
print("\nSARIMAX Model Summary:")
print(sarimax_result.summary())


forecast_steps = len(test)
forecast = sarimax_result.get_forecast(steps=forecast_steps, exog=test_exog)
forecast_mean = pd.Series(forecast.predicted_mean.values, index=test.index)
forecast_conf_int = forecast.conf_int()


mse = mean_squared_error(test['Close/Last'], forecast_mean)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train['Close/Last'], label='Training Data', color='blue')
plt.plot(test['Close/Last'], label='Testing Data', color='green')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast with Exog', color='red')
plt.fill_between(
    forecast_mean.index,
    forecast_conf_int.iloc[:, 0],
    forecast_conf_int.iloc[:, 1],
    color='red',
    alpha=0.2,
    label='Confidence Interval with Exog'
)
plt.title(f"SARIMAX Forecast for {company} Closing Prices with Exogenous Variables")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid()
plt.show()
