import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


file_path = 'E:/Quant_Project/data/u_dataset.csv'
data = pd.read_csv(file_path)


company = 'META'                                
company_data = data[data['Company'] == company].copy()


company_data['Close/Last'] = pd.to_numeric(company_data['Close/Last'], errors='coerce')
company_data = company_data.dropna(subset=['Close/Last'])

# Create lagged features
lags = 5
for lag in range(1, lags + 1):
    company_data[f'lag_{lag}'] = company_data['Close/Last'].shift(lag)

# Droping rows with NaN values caused by lagging
company_data = company_data.dropna()


X = company_data[[f'lag_{lag}' for lag in range(1, lags + 1)]]
y = company_data['Close/Last']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")


predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
print("\nSample Predictions:")
print(predictions.head())
  