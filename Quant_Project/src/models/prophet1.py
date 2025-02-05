from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler


file_path = 'E:/Quant_Project/data/u_dataset.csv'  
data = pd.read_csv(file_path)


company = 'META'                                
company_data = data[data['Company'] == company].copy()


company_data['Close/Last'] = pd.to_numeric(company_data['Close/Last'], errors='coerce')
company_data = company_data.dropna(subset=['Close/Last'])

# Selecting aditional regressors
regressors = ['Volume', 'Daily_Return', 'VWAP', 'MACD', 'RSI', 'Day', 'Month', 'Year', 'Weekday']


company_data = company_data.dropna(subset=regressors)

# Prepare the dataset for Prophet
prophet_data = company_data[['Close/Last'] + regressors].rename(columns={'Close/Last': 'y'})
prophet_data['ds'] = pd.to_datetime(
    company_data[['Year', 'Month', 'Day']]
) 

# Normalize additional regressors
scaler = StandardScaler()
prophet_data[regressors] = scaler.fit_transform(prophet_data[regressors])


train_size = int(len(prophet_data) * 0.8)
train, test = prophet_data.iloc[:train_size], prophet_data.iloc[train_size:]


model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.add_seasonality(name='monthly', period=30.5, fourier_order=4)  

for regressor in regressors:
    model.add_regressor(regressor)


model.fit(train)

# Make future predictions
future = model.make_future_dataframe(periods=len(test)) 
for regressor in regressors:
   
    future[regressor] = list(train[regressor]) + list(test[regressor])

forecast = model.predict(future)

aligned_test = pd.merge(test, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')

# Evaluate the model
mse = mean_squared_error(aligned_test['y'], aligned_test['yhat'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(aligned_test['y'], aligned_test['yhat'])
mape = np.mean(np.abs((aligned_test['y'] - aligned_test['yhat']) / aligned_test['y'])) * 100  # MAPE formula
accuracy = 100 - mape  

# Print metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy: {accuracy:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(aligned_test['ds'], aligned_test['y'], label='Actual', color='blue')
plt.plot(aligned_test['ds'], aligned_test['yhat'], label='Predicted', color='orange')
plt.fill_between(
    aligned_test['ds'],
    aligned_test['yhat_lower'],
    aligned_test['yhat_upper'],
    color='orange',
    alpha=0.3,
    label='Confidence Interval'
)
plt.title(f"Prophet Model - Actual vs Predicted ({company})")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid()
plt.show()
