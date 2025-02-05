import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path =  'E:/Quant_Project/data/u_dataset.csv'
data = pd.read_csv(file_path)


company = 'QCOM'  
company_data = data[data['Company'] == company].copy()


company_data['Close/Last'] = pd.to_numeric(company_data['Close/Last'], errors='coerce')
company_data = company_data.dropna(subset=['Close/Last'])


lags = 5
for lag in range(1, lags + 1):
    company_data[f'lag_{lag}'] = company_data['Close/Last'].shift(lag)

# Adding more features to improve model
company_data['Price_Change'] = company_data['Close/Last'].pct_change() 
company_data['Volume_Change'] = company_data['Volume'].pct_change()  
company_data['Cumulative_Return'] = (1 + company_data['Price_Change']).cumprod()  

# Drop rows with NaN values caused by lagging or new features
company_data = company_data.dropna()

# Define features (lagged values + new features)
features = (
    [f'lag_{lag}' for lag in range(1, lags + 1)] + 
    ['VWAP', 'Volume', 'Price_Change', 'Volume_Change', 'Cumulative_Return', 'Day', 'Month', 'Weekday']
)
X = company_data[features]  
y = company_data['Close/Last'] 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [200, 300, 400],  
    'max_depth': [10, 15, 20, None],  
    'min_samples_split': [5, 10, 15],  
    'min_samples_leaf': [1, 2, 4],     
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)


best_rf_model = grid_search.best_estimator_


y_pred = best_rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)
