import pandas as pd
import numpy as np


file_path = 'E:/Quant_Project/data/cleaned_data.csv'
data = pd.read_csv(file_path)



data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# Price-Based Features
data['Daily_Price_Change'] = data['Close/Last'].diff()
data['Daily_Return'] = data['Close/Last'].pct_change() * 100  # Percentage
data['High_Low_Range'] = data['High'] - data['Low']

# Technical Indicators

data['SMA_10'] = data['Close/Last'].rolling(window=10).mean()
data['SMA_20'] = data['Close/Last'].rolling(window=20).mean()

# Exponential Moving Averages (EMA)
data['EMA_10'] = data['Close/Last'].ewm(span=10, adjust=False).mean()
data['EMA_20'] = data['Close/Last'].ewm(span=20, adjust=False).mean()

# Moving Average Convergence Divergence (MACD)
data['MACD'] = data['Close/Last'].ewm(span=12, adjust=False).mean() - data['Close/Last'].ewm(span=26, adjust=False).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Relative Strength Index (RSI)
change = data['Close/Last'].diff()
gain = change.where(change > 0, 0)
loss = -change.where(change < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

#  Volume Features
data['Volume_Change'] = data['Volume'].diff()
data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
data['VWAP'] = (data['Close/Last'] * data['Volume']).cumsum() / data['Volume'].cumsum()

# Volatility Measures
data['Historical_Volatility'] = data['Daily_Return'].rolling(window=10).std()

# Momentum Indicators
data['Price_Momentum_10'] = data['Close/Last'] - data['Close/Last'].shift(10)
data['Volume_Momentum_10'] = data['Volume'] - data['Volume'].shift(10)


data.to_csv('E:/Quant_Project/data/features_data.csv', index=False)


print(data.head())
